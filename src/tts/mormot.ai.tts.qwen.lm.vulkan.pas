/// Qwen3-TTS - a stack of Qwen3 layers (TQwenLm) on the GPU, through Vulkan
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - follows TQwen3Vulkan.Chunk of mormot2-ai (model.qwen3.vulkan), with its
// shaders: the embeddings and the RoPE factors go up, a whole pass is
// recorded (norms, Q8_0 projections, qkrope, attention on a K/V cache in
// float32 or half floats which stays on the GPU, SwiGLU, head), the GPU runs
// it, and only the logits and the state of the last row come back
// - one pass per call: a talker frame, a code predictor group - passes of a
// few milliseconds rather than a long peak (the GPU may render a game at the
// same time)
// - the weights are sent once, at creation; TQwenVulkan (the device and the
// shaders) is shared by the talker and the code predictor
unit mormot.ai.tts.qwen.lm.vulkan;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.vulkan,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.tts.qwen.lm;

type
  /// one GPU buffer per layer
  TQwenVkBuffers = array of TInferVkBuffer;

  /// the Vulkan device and the shaders of mormot2-ai, shared by the stacks
  TQwenVulkan = class
  protected
    fBackend: TInferVulkanBackend;
    fVk: TInferVulkan;
    fNorm, fSwiGlu, fAttention, fAttention16, fMerge, fAdd, fQkRope, fQkRope16,
      fAddNorm: TInferVkPipeline;
    fDeviceBytes: Int64;
  public
    /// open the first Vulkan GPU; aThreads: the CPU threads of the backend
    // (0 = two thirds of the cores)
    constructor Create(aThreads: integer = 0);
    destructor Destroy; override;
    /// true when a Vulkan GPU is there (tried once)
    class function Available: boolean;
    /// the name of the GPU
    function DeviceName: RawUtf8;
    property Backend: TInferVulkanBackend
      read fBackend;
    /// the bytes allocated on the GPU by the stacks: weights, caches, work
    property DeviceBytes: Int64
      read fDeviceBytes;
  end;

  /// TQwenLm on the GPU of aVulkan
  TQwenLmVulkan = class(TQwenLm)
  protected
    fVulkan: TQwenVulkan;
    fVk: TInferVulkan;
    fKvF16: boolean;
    fCap, fRows, fOutRows: PtrInt;
    fKeys, fValues: TQwenVkBuffers;
    fDevX, fDevH, fDevQ, fDevK, fDevV, fDevAttn, fDevTmp, fDevGate, fDevUp,
      fDevCos, fPart, fLogits: TInferVkBuffer;
    fIn, fOut: TInferVkBuffer;
    fHostCos: TSingleDynArray;
    fCacheBytes, fScratchBytes: Int64;
    procedure Upload;
    function Vector(const v: TSingleDynArray): TInferVkBuffer;
    procedure Need(var aBuffer: TInferVkBuffer; aBytes: Int64);
    procedure Prepare(aTokens, aRows: PtrInt);
    procedure Chunk(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
      aLogits, aHidden: PSingleArray);
  public
    /// send the weights of aWeights to the GPU of aVulkan (neither is owned);
    // aKvF16: the K/V cache in half floats
    // - the backend keeps the device copies keyed by their host address until
    // aVulkan is freed or aVulkan.Backend.ClearWeights: free aWeights only
    // after one of them, or other weights reusing the addresses would be
    // served stale
    constructor CreateVulkan(aWeights: TQwenLmWeights; aVulkan: TQwenVulkan;
      aKvF16: boolean = true);
    destructor Destroy; override;
    procedure Forward(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
      aLogits, aHidden: PSingleArray); override;
    /// the bytes of the K/V cache and of the work buffers on the GPU
    property CacheBytes: Int64
      read fCacheBytes;
    property ScratchBytes: Int64
      read fScratchBytes;
  end;


implementation

type
  // the push constants of the mormot2-ai shaders (model.qwen3.vulkan)
  TNormPush = record
    Rows, Cols: cardinal;
    Eps: single;
  end;

  TAttentionPush = record
    Rows, Heads, HeadDim, KvHeads, Past, Splits, HeadBlock, RowBlock: cardinal;
  end;

  TMergePush = record
    Rows, Heads, HeadDim, Splits: cardinal;
  end;

  TQkRopePush = record
    Rows, Heads, KvHeads, HeadDim, Past: cardinal;
    Eps: single;
  end;

const
  /// rows per pass at most (the prefill): each dispatch stays below 65535
  // groups (rows x Mlp / 128 for SwiGLU)
  CHUNK_MAX = 256;
  /// the queries of one attention.comp group at most (MAXQ)
  ATTENTION_QUERIES = 8;
  /// attention.comp splits the keys in slices of about SPLIT_KEYS when there
  // are few groups (one row), SPLIT_GROUPS groups in all at most
  SPLIT_KEYS = 64;
  SPLIT_GROUPS = 2048;


{ TQwenVulkan }

class function TQwenVulkan.Available: boolean;
begin
  result := TInferVulkan.Available;
end;

constructor TQwenVulkan.Create(aThreads: integer);
begin
  inherited Create;
  fBackend := TInferVulkanBackend.Create(aThreads);
  fVk := fBackend.Vulkan;
  fNorm := TInferVkPipeline.Create(fVk, @SPV_RMSNORM, length(SPV_RMSNORM), 3,
    SizeOf(TNormPush));
  fSwiGlu := TInferVkPipeline.Create(fVk, @SPV_SWIGLU, length(SPV_SWIGLU), 2, 4);
  fAttention := TInferVkPipeline.Create(fVk, @SPV_ATTENTION, length(SPV_ATTENTION),
    5, SizeOf(TAttentionPush));
  fAttention16 := TInferVkPipeline.Create(fVk, @SPV_ATTENTION_F16,
    length(SPV_ATTENTION_F16), 5, SizeOf(TAttentionPush));
  fMerge := TInferVkPipeline.Create(fVk, @SPV_ATTNMERGE, length(SPV_ATTNMERGE), 2,
    SizeOf(TMergePush));
  fAdd := TInferVkPipeline.Create(fVk, @SPV_ADD, length(SPV_ADD), 3, 4);
  fQkRope := TInferVkPipeline.Create(fVk, @SPV_QKROPE, length(SPV_QKROPE), 8,
    SizeOf(TQkRopePush));
  fQkRope16 := TInferVkPipeline.Create(fVk, @SPV_QKROPE_F16,
    length(SPV_QKROPE_F16), 8, SizeOf(TQkRopePush));
  fAddNorm := TInferVkPipeline.Create(fVk, @SPV_ADDNORM, length(SPV_ADDNORM), 4,
    SizeOf(TNormPush));
  fNorm.Name := 'rmsnorm';
  fSwiGlu.Name := 'swiglu';
  fAttention.Name := 'attention';
  fAttention16.Name := 'attention';
  fMerge.Name := 'attnmerge';
  fAdd.Name := 'add';
  fQkRope.Name := 'qkrope';
  fQkRope16.Name := 'qkrope';
  fAddNorm.Name := 'addnorm';
end;

destructor TQwenVulkan.Destroy;
begin
  fAddNorm.Free;
  fQkRope16.Free;
  fQkRope.Free;
  fAdd.Free;
  fMerge.Free;
  fAttention16.Free;
  fAttention.Free;
  fSwiGlu.Free;
  fNorm.Free;
  fBackend.Free;
  inherited Destroy;
end;

function TQwenVulkan.DeviceName: RawUtf8;
begin
  result := fVk.DeviceName;
end;


{ TQwenLmVulkan }

constructor TQwenLmVulkan.CreateVulkan(aWeights: TQwenLmWeights;
  aVulkan: TQwenVulkan; aKvF16: boolean);
begin
  inherited Create(aWeights);
  if (aWeights.HeadDim > 256) or (aWeights.HeadDim and 3 <> 0) then
    EQwenLm.RaiseUtf8('%: head dimension % not supported', [self, aWeights.HeadDim]);
  fVulkan := aVulkan;
  fVk := aVulkan.fVk;
  fKvF16 := aKvF16;
  SetLength(fKeys, aWeights.LayerCount);
  SetLength(fValues, aWeights.LayerCount);
  fVulkan.fBackend.GpuLock;
  try
    Upload;
  finally
    fVulkan.fBackend.GpuUnLock;
  end;
end;

destructor TQwenLmVulkan.Destroy;
var
  i: PtrInt;
begin
  for i := 0 to high(fKeys) do
  begin
    fKeys[i].Free;
    fValues[i].Free;
  end;
  fDevX.Free;
  fDevH.Free;
  fDevQ.Free;
  fDevK.Free;
  fDevV.Free;
  fDevAttn.Free;
  fDevTmp.Free;
  fDevGate.Free;
  fDevUp.Free;
  fDevCos.Free;
  fPart.Free;
  fLogits.Free;
  fIn.Free;
  fOut.Free;
  if fVulkan <> nil then
    dec(fVulkan.fDeviceBytes, fCacheBytes + fScratchBytes);
  inherited Destroy;
end;

function TQwenLmVulkan.Vector(const v: TSingleDynArray): TInferVkBuffer;
begin
  result := fVulkan.fBackend.DeviceWeight(pointer(v), idtF32,
    Int64(system.length(v)) * 4);
end;

procedure TQwenLmVulkan.Upload;
var
  w: TQwenLmWeights;
  l, q, kv, o: PtrInt;
  lw: PQwenLmLayer;
  b: TInferVulkanBackend;

  procedure Matrix(const aW: TInferWeight; aRows, aCols: PtrInt);
  begin
    if not b.GpuLinear(aW.DType, aCols) or
       (b.LinearPart(aW.DType, aCols, aRows) < aRows) then
      EQwenLm.RaiseUtf8('%: no shader for a % x % % matrix', [self, aRows, aCols,
        INFER_DTYPE_NAME[aW.DType]]);
    b.UploadLinear(aW, aCols, aRows);
  end;

begin
  // everything in advance: a recorded pass cannot upload anything
  b := fVulkan.fBackend;
  w := fWeights;
  q := w.Heads * w.HeadDim;
  kv := w.KvHeads * w.HeadDim;
  for l := 0 to w.LayerCount - 1 do
  begin
    lw := w.Layer(l);
    Matrix(lw^.WQ, q, w.Hidden);
    Matrix(lw^.WK, kv, w.Hidden);
    Matrix(lw^.WV, kv, w.Hidden);
    Matrix(lw^.WO, w.Hidden, q);
    Matrix(lw^.WGate, w.Mlp, w.Hidden);
    Matrix(lw^.WUp, w.Mlp, w.Hidden);
    Matrix(lw^.WDown, w.Hidden, w.Mlp);
    Vector(lw^.InNorm);
    Vector(lw^.PostNorm);
    Vector(lw^.QNorm);
    Vector(lw^.KNorm);
  end;
  Vector(w.Norm);
  for o := 0 to w.OutputCount - 1 do
    Matrix(w.Output(o), w.OutputRows(o), w.Hidden);
  inc(fVulkan.fDeviceBytes, w.Bytes);
end;

procedure TQwenLmVulkan.Need(var aBuffer: TInferVkBuffer; aBytes: Int64);
begin
  if (aBuffer <> nil) and (aBuffer.Size >= aBytes) then
    exit;
  if aBuffer <> nil then
  begin
    dec(fScratchBytes, aBuffer.Size);
    dec(fVulkan.fDeviceBytes, aBuffer.Size);
    FreeAndNil(aBuffer);
  end;
  aBuffer := TInferVkBuffer.Create(fVk, aBytes, true);
  inc(fScratchBytes, aBuffer.Size);
  inc(fVulkan.fDeviceBytes, aBuffer.Size);
end;

procedure TQwenLmVulkan.Prepare(aTokens, aRows: PtrInt);
var
  w: TQwenLmWeights;
  cap, kvw, l, half, es, vocab, o: PtrInt;
  keys, values: TQwenVkBuffers;
begin
  w := fWeights;
  kvw := w.KvHeads * w.HeadDim;
  half := w.HeadDim shr 1;
  es := 4;
  if fKvF16 then
    es := 2;
  // the K/V cache: doubles, copied on the GPU when it grows
  if aTokens > fCap then
  begin
    cap := fCap * 2;
    if cap < aTokens + aTokens shr 2 then
      cap := aTokens + aTokens shr 2;
    if cap < 64 then
      cap := 64;
    if Int64(cap) * kvw * es > fVk.MaxStorageRange then
      cap := fVk.MaxStorageRange div (Int64(kvw) * es);
    if cap < aTokens then
      EQwenLm.RaiseUtf8('%: a KV cache of % rows is too large for this device',
        [self, aTokens]);
    SetLength(keys, system.length(fKeys));
    SetLength(values, system.length(fKeys));
    try
      for l := 0 to high(keys) do
      begin
        keys[l] := TInferVkBuffer.Create(fVk, Int64(cap) * kvw * es, true);
        values[l] := TInferVkBuffer.Create(fVk, Int64(cap) * kvw * es, true);
      end;
      if fLen > 0 then
      begin
        fVk.BatchBegin;
        try
          for l := 0 to high(keys) do
          begin
            fVk.BatchCopy(fKeys[l], 0, keys[l], 0, Int64(fLen) * kvw * es);
            fVk.BatchCopy(fValues[l], 0, values[l], 0, Int64(fLen) * kvw * es);
          end;
        except
          fVk.BatchAbort;
          raise;
        end;
        fVk.BatchSubmit;
      end;
    except
      for l := 0 to high(keys) do
      begin
        keys[l].Free;
        values[l].Free;
      end;
      raise;
    end;
    for l := 0 to high(fKeys) do
    begin
      fKeys[l].Free;
      fValues[l].Free;
    end;
    fKeys := keys;
    fValues := values;
    dec(fVulkan.fDeviceBytes, fCacheBytes);
    fCacheBytes := 2 * Int64(system.length(fKeys)) * cap * kvw * es;
    inc(fVulkan.fDeviceBytes, fCacheBytes);
    fCap := cap;
  end;
  if aRows > fRows then
  begin
    Need(fDevX, Int64(aRows) * w.Hidden * 4);
    Need(fDevH, Int64(aRows) * w.Hidden * 4);
    Need(fDevTmp, Int64(aRows) * w.Hidden * 4);
    Need(fDevQ, Int64(aRows) * w.Heads * w.HeadDim * 4);
    Need(fDevAttn, Int64(aRows) * w.Heads * w.HeadDim * 4);
    Need(fDevK, Int64(aRows) * kvw * 4);
    Need(fDevV, Int64(aRows) * kvw * 4);
    Need(fDevGate, Int64(aRows) * w.Mlp * 4);
    Need(fDevUp, Int64(aRows) * w.Mlp * 4);
    Need(fDevCos, Int64(aRows) * half * 2 * 4); // cos, then sin
    // visible to the host: the embeddings, then cos, then sin
    FreeAndNil(fIn);
    fIn := TInferVkBuffer.Create(fVk, Int64(aRows) * (w.Hidden + 2 * half) * 4);
    fRows := aRows;
  end;
  if fOut = nil then
  begin
    // the widest head, then the normed state of the last row
    vocab := 0;
    for o := 0 to w.OutputCount - 1 do
      if w.OutputRows(o) > vocab then
        vocab := w.OutputRows(o);
    fOutRows := vocab;
    Need(fLogits, Int64(vocab + 1) * 4);
    fOut := TInferVkBuffer.CreateReadback(fVk, Int64(vocab + w.Hidden) * 4);
  end;
end;

procedure TQwenLmVulkan.Chunk(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
  aLogits, aHidden: PSingleArray);
var
  w: TQwenLmWeights;
  v: TQwenVulkan;
  b: TInferVulkanBackend;
  n, h, f, qw, kvw, half, past, l, splits, most, hb, rb, blocks, vocab: PtrInt;
  attention, qkrope: TInferVkPipeline;
  lw: PQwenLmLayer;
  p: PSingleArray;
  qk: TQkRopePush;
  norm: TNormPush;
  att: TAttentionPush;
  merge: TMergePush;
  part: TInferVkBuffer;
  count: cardinal;
  want: boolean;
begin
  w := fWeights;
  v := fVulkan;
  b := v.fBackend;
  n := aRows;
  h := w.Hidden;
  f := w.Mlp;
  qw := w.Heads * w.HeadDim;
  kvw := w.KvHeads * w.HeadDim;
  half := w.HeadDim shr 1;
  past := fLen;
  want := (aLogits <> nil) or (aHidden <> nil);
  vocab := 0;
  Prepare(past + n, n);
  attention := v.fAttention;
  qkrope := v.fQkRope;
  if fKvF16 then
  begin
    attention := v.fAttention16;
    qkrope := v.fQkRope16;
  end;
  // the attention groups: hb heads of one KV head times rb rows, the keys of
  // one row split in splits slices when there are few groups
  hb := w.Heads div w.KvHeads;
  while (hb > ATTENTION_QUERIES) or ((w.Heads div w.KvHeads) mod hb <> 0) do
    dec(hb);
  rb := ATTENTION_QUERIES div hb;
  blocks := (n + rb - 1) div rb;
  splits := (past + n + SPLIT_KEYS - 1) div SPLIT_KEYS;
  most := SPLIT_GROUPS div ((w.Heads div hb) * blocks);
  if splits > most then
    splits := most;
  if splits < 1 then
    splits := 1;
  if splits > 1 then
  begin
    Need(fPart, Int64(w.Heads) * n * splits * (w.HeadDim + 2) * 4);
    part := fPart;
  end
  else
    part := fDevAttn; // bound, but not read by attention.comp
  // the inputs, written into the visible buffer before the pass
  p := fIn.Mapped;
  MoveFast(x^[0], p^[0], n * h * SizeOf(single));
  QwenGrow(fHostCos, n * half * 2);
  Rotary(n, pointer(fHostCos), @fHostCos[n * half]);
  MoveFast(fHostCos[0], p^[n * h], n * half * 2 * SizeOf(single));
  fVk.BatchBegin;
  try
    fVk.BatchCopy(fIn, 0, fDevX, 0, Int64(n) * h * 4);
    fVk.BatchCopy(fIn, Int64(n) * h * 4, fDevCos, 0, Int64(n) * half * 2 * 4);
    qk.Rows := n;
    qk.Heads := w.Heads;
    qk.KvHeads := w.KvHeads;
    qk.HeadDim := w.HeadDim;
    qk.Past := past;
    qk.Eps := w.Eps;
    norm.Rows := n;
    norm.Cols := h;
    norm.Eps := w.Eps;
    att.Rows := n;
    att.Heads := w.Heads;
    att.HeadDim := w.HeadDim;
    att.KvHeads := w.KvHeads;
    att.Past := past;
    att.Splits := splits;
    att.HeadBlock := hb;
    att.RowBlock := rb;
    merge.Rows := n;
    merge.Heads := w.Heads;
    merge.HeadDim := w.HeadDim;
    merge.Splits := splits;
    // the input norm of layer 0; the next ones come with the residual of the
    // layer before (addnorm)
    fVk.BatchDispatch(v.fNorm, [fDevX, Vector(w.Layer(0)^.InNorm), fDevH], @norm, n);
    for l := 0 to w.LayerCount - 1 do
    begin
      lw := w.Layer(l);
      // the three projections side by side (they only read fDevH), then per
      // head the norms, the RoPE and k, v into the cache
      b.RecordLinear(fDevH, n, h, lw^.WQ, qw, fDevQ, false);
      b.RecordLinear(fDevH, n, h, lw^.WK, kvw, fDevK, false);
      b.RecordLinear(fDevH, n, h, lw^.WV, kvw, fDevV);
      fVk.BatchDispatch(qkrope, [fDevQ, fDevK, fDevV, Vector(lw^.QNorm),
        Vector(lw^.KNorm), fDevCos, fKeys[l], fValues[l]], @qk,
        w.Heads + w.KvHeads, n);
      fVk.BatchDispatch(attention, [fDevQ, fKeys[l], fValues[l], fDevAttn, part],
        @att, w.Heads div hb, blocks, splits);
      if splits > 1 then
        fVk.BatchDispatch(v.fMerge, [part, fDevAttn], @merge, w.Heads, n);
      b.RecordLinear(fDevAttn, n, qw, lw^.WO, h, fDevTmp);
      // the residual and the norm before the MLP, gate and up side by side,
      // SwiGLU, down
      fVk.BatchDispatch(v.fAddNorm, [fDevX, fDevTmp, Vector(lw^.PostNorm), fDevH],
        @norm, n);
      b.RecordLinear(fDevH, n, h, lw^.WGate, f, fDevGate, false);
      b.RecordLinear(fDevH, n, h, lw^.WUp, f, fDevUp);
      count := n * f;
      fVk.BatchDispatch(v.fSwiGlu, [fDevGate, fDevUp], @count, (count + 127) div 128);
      b.RecordLinear(fDevGate, n, f, lw^.WDown, h, fDevTmp);
      // the residual, and the input norm of the next layer
      if l < w.LayerCount - 1 then
        fVk.BatchDispatch(v.fAddNorm, [fDevX, fDevTmp,
          Vector(w.Layer(l + 1)^.InNorm), fDevH], @norm, n)
      else
      begin
        count := n * h;
        fVk.BatchDispatch(v.fAdd, [fDevX, fDevTmp, fDevX], @count,
          (count + 127) div 128);
      end;
      // the GPU starts on the first layer, then receives the next ones by four
      // while the host records the rest
      if (l = 0) or (l and 3 = 3) then
        fVk.BatchFlush;
    end;
    if want then
    begin
      // the last row: its norm, the head
      fVk.BatchCopy(fDevX, Int64(n - 1) * h * 4, fDevTmp, 0, Int64(h) * 4);
      norm.Rows := 1;
      fVk.BatchDispatch(v.fNorm, [fDevTmp, Vector(w.Norm), fDevH], @norm, 1);
      vocab := 0;
      if aLogits <> nil then
      begin
        vocab := w.OutputRows(aOutput);
        b.RecordLinear(fDevH, 1, h, w.Output(aOutput), vocab, fLogits);
        fVk.BatchCopy(fLogits, 0, fOut, 0, Int64(vocab) * 4);
      end;
      if aHidden <> nil then
        fVk.BatchCopy(fDevH, 0, fOut, Int64(fOutRows) * 4, Int64(h) * 4);
    end;
  except
    fVk.BatchAbort;
    raise;
  end;
  fVk.BatchSubmit;
  if aLogits <> nil then
    MoveFast(fOut.Mapped^, aLogits^, vocab * SizeOf(single));
  if aHidden <> nil then
    MoveFast(PSingleArray(fOut.Mapped)^[fOutRows], aHidden^, h * SizeOf(single));
  inc(fLen, n);
end;

procedure TQwenLmVulkan.Forward(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
  aLogits, aHidden: PSingleArray);
var
  done, n: PtrInt;
begin
  if aRows <= 0 then
    exit;
  fVulkan.fBackend.GpuLock;
  try
    done := 0;
    while done < aRows do
    begin
      n := aRows - done;
      if n > CHUNK_MAX then
        n := CHUNK_MAX;
      if done + n = aRows then
        Chunk(@x^[done * fWeights.Hidden], n, aOutput, aLogits, aHidden)
      else
        Chunk(@x^[done * fWeights.Hidden], n, aOutput, nil, nil);
      inc(done, n);
    end;
  finally
    fVulkan.fBackend.GpuUnLock;
  end;
end;

end.
