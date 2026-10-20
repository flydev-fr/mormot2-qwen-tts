/// Qwen3-TTS - a stack of Qwen3 layers: its weights and its CPU computation
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the talker (28 layers) or the code predictor (5 layers)
// - takes embeddings, not ids (the input of the talker is a sum of
// embeddings), and returns the logits of one of its heads for the last row,
// with the normed hidden state of this row; the K/V cache follows the calls
// until Reset - mormot.ai.tts.qwen.lm.vulkan runs the same on the GPU
// - matrices are float32 (the BF16 weights of the file, widened without
// loss) or Q8_0 (quantized at load time as ggml writes them); norms stay
// float32
unit mormot.ai.tts.qwen.lm;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.core.variants,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.threads,
  mormot.ai.tts.qwen.codec;

type
  EQwenLm = class(ESynException);

  /// the weights of one Qwen3 layer
  TQwenLmLayer = record
    InNorm, PostNorm, QNorm, KNorm: TSingleDynArray;
    WQ, WK, WV, WO, WGate, WUp, WDown: TInferWeight;
  end;
  PQwenLmLayer = ^TQwenLmLayer;

  /// the weights of a Qwen3 stack and of its output heads
  TQwenLmWeights = class
  protected
    fName: RawUtf8;
    fDType: TInferDType;
    fHidden, fHeads, fKvHeads, fHeadDim, fMlp: PtrInt;
    fEps, fTheta: double;
    fLayers: array of TQwenLmLayer;
    fNorm: TSingleDynArray;
    fOutputs: array of TInferWeight;
    fOutputRows: TIntegerDynArray;
    // the memory of the matrices (F32 or Q8_0)
    fStore: TRawByteStringDynArray;
    fStoreCount: PtrInt;
    fBytes: Int64;
    // the running conversion (Matrix): blocks of rows, on fPool if any, during
    // Create or AddOutput
    fPool: TInferThreadPool;
    fJobTensor: PInferTensor;
    fJobData: PByteArray;
    fJobRows, fJobCols, fJobRowBytes: PtrInt;
    function Matrix(aFile: TSafeTensorsFile; const aName: RawUtf8;
      aRows, aCols: PtrInt): TInferWeight;
    procedure MatrixRows(aItem: integer);
  public
    /// aPrefix: 'talker.model.' or 'talker.code_predictor.model.'; aConfig:
    // talker_config or code_predictor_config of config.json; aDType: idtF32
    // or idtQ8_0; aPool: the threads of the conversion (nil: the calling
    // thread) - the bytes are the same
    constructor Create(aFile: TSafeTensorsFile; const aName, aPrefix: RawUtf8;
      const aConfig: TDocVariantData; aDType: TInferDType;
      aPool: TInferThreadPool = nil);
    /// add an output head [aRows][Hidden]; returns its index
    function AddOutput(aFile: TSafeTensorsFile; const aTensor: RawUtf8;
      aRows: PtrInt; aPool: TInferThreadPool = nil): PtrInt;
    /// the crc32c of the matrices, in their load order (tests)
    function Digest: cardinal;
    function LayerCount: PtrInt;
    function Layer(aIndex: PtrInt): PQwenLmLayer;
    function OutputCount: PtrInt;
    function Output(aIndex: PtrInt): TInferWeight;
    function OutputRows(aIndex: PtrInt): PtrInt;
    /// the final norm
    property Norm: TSingleDynArray
      read fNorm;
    property Name: RawUtf8
      read fName;
    property DType: TInferDType
      read fDType;
    property Hidden: PtrInt
      read fHidden;
    property Heads: PtrInt
      read fHeads;
    property KvHeads: PtrInt
      read fKvHeads;
    property HeadDim: PtrInt
      read fHeadDim;
    property Mlp: PtrInt
      read fMlp;
    property Eps: double
      read fEps;
    property Theta: double
      read fTheta;
    /// the bytes of the matrices (layers and heads)
    property Bytes: Int64
      read fBytes;
  end;

  /// the computation of a Qwen3 stack, on the CPU (TQwenLmCpu) or the GPU
  // (TQwenLmVulkan)
  TQwenLm = class
  protected
    fWeights: TQwenLmWeights;
    fLen: PtrInt;
    fProbe: TOnQwenProbe;
    fInvFreq: TSingleDynArray;
    /// cos and sin of positions Length .. Length + aRows - 1: [aRows][HeadDim / 2]
    // each, in float32 as PyTorch
    procedure Rotary(aRows: PtrInt; aCos, aSin: PSingleArray);
  public
    /// aWeights is not owned and must live longer
    constructor Create(aWeights: TQwenLmWeights); virtual;
    /// empty the cache: the next row is at position 0
    procedure Reset; virtual;
    /// add aRows rows of embeddings x[aRows][Hidden] at positions
    // Length..Length + aRows - 1; for the last row, return its normed state
    // in aHidden and the logits of head aOutput in aLogits (nil: nothing)
    procedure Forward(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
      aLogits, aHidden: PSingleArray); virtual; abstract;
    property Weights: TQwenLmWeights
      read fWeights;
    /// the positions already in the cache
    property Length: PtrInt
      read fLen;
    /// the intermediate outputs (tests): name.layerN, name.out; nil by default
    property Probe: TOnQwenProbe
      read fProbe write fProbe;
  end;

  /// TQwenLm on the CPU, with the thread pool and the AVX2 kernels of mormot2-ai
  TQwenLmCpu = class(TQwenLm)
  protected
    fCpu: TInferCpuFastBackend;
    fKeys, fValues: array of TSingleDynArray;
    fCap: PtrInt;
    fH, fN, fQ, fK, fV, fAtt, fO, fG, fU, fCos, fSin: TSingleDynArray;
    procedure Room(aRows: PtrInt);
  public
    /// aCpu is not owned
    constructor CreateCpu(aWeights: TQwenLmWeights; aCpu: TInferCpuFastBackend);
    procedure Forward(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
      aLogits, aHidden: PSingleArray); override;
  end;

/// the tensor aName [aRows][aCols] in float32
function QwenWiden(aFile: TSafeTensorsFile; const aName: RawUtf8;
  aRows, aCols: PtrInt): TSingleDynArray;

/// an array of at least aCount items (the content is lost if it grows)
procedure QwenGrow(var a: TSingleDynArray; aCount: PtrInt);


implementation

function QwenWiden(aFile: TSafeTensorsFile; const aName: RawUtf8;
  aRows, aCols: PtrInt): TSingleDynArray;
var
  t: PInferTensor;
  r: PtrInt;
begin
  t := aFile.Tensor(aName);
  if t^.Count <> Int64(aRows) * aCols then
    EQwenLm.RaiseUtf8('%: % is % instead of [%,%]',
      [aFile, aName, InferShapeText(t^.Shape), aRows, aCols]);
  SetLength(result, aRows * aCols);
  for r := 0 to aRows - 1 do
    InferRowToF32(t^.Data, t^.DType, r, aCols, @result[r * aCols]);
end;

procedure QwenGrow(var a: TSingleDynArray; aCount: PtrInt);
begin
  if system.length(a) < aCount then
  begin
    a := nil;
    SetLength(a, aCount);
  end;
end;


{ TQwenLmWeights }

constructor TQwenLmWeights.Create(aFile: TSafeTensorsFile;
  const aName, aPrefix: RawUtf8; const aConfig: TDocVariantData;
  aDType: TInferDType; aPool: TInferThreadPool);
var
  l, q, kv: PtrInt;
  lw: PQwenLmLayer;
  p: RawUtf8;
begin
  inherited Create;
  if not (aDType in [idtF32, idtQ8_0]) then
    EQwenLm.RaiseUtf8('%: % weights not supported', [self, INFER_DTYPE_NAME[aDType]]);
  fPool := aPool;
  fName := aName;
  fDType := aDType;
  fHidden := aConfig.I['hidden_size'];
  fHeads := aConfig.I['num_attention_heads'];
  fKvHeads := aConfig.I['num_key_value_heads'];
  fHeadDim := aConfig.I['head_dim'];
  fMlp := aConfig.I['intermediate_size'];
  fEps := aConfig.D['rms_norm_eps'];
  fTheta := aConfig.D['rope_theta'];
  if (fHidden <= 0) or (fHeads <= 0) or (fKvHeads <= 0) or (fHeadDim <= 0) or
     (fHeads mod fKvHeads <> 0) or (fMlp <= 0) or (fEps <= 0) or (fTheta <= 0) or
     (fHidden and 31 <> 0) or (fMlp and 31 <> 0) or (fHeadDim and 31 <> 0) then
    EQwenLm.RaiseUtf8('%: invalid % configuration', [self, aName]);
  q := fHeads * fHeadDim;
  kv := fKvHeads * fHeadDim;
  SetLength(fLayers, aConfig.I['num_hidden_layers']);
  for l := 0 to high(fLayers) do
  begin
    lw := @fLayers[l];
    p := FormatUtf8('%layers.%.', [aPrefix, l]);
    lw^.InNorm := QwenWiden(aFile, p + 'input_layernorm.weight', 1, fHidden);
    lw^.PostNorm := QwenWiden(aFile, p + 'post_attention_layernorm.weight', 1, fHidden);
    lw^.QNorm := QwenWiden(aFile, p + 'self_attn.q_norm.weight', 1, fHeadDim);
    lw^.KNorm := QwenWiden(aFile, p + 'self_attn.k_norm.weight', 1, fHeadDim);
    lw^.WQ := Matrix(aFile, p + 'self_attn.q_proj.weight', q, fHidden);
    lw^.WK := Matrix(aFile, p + 'self_attn.k_proj.weight', kv, fHidden);
    lw^.WV := Matrix(aFile, p + 'self_attn.v_proj.weight', kv, fHidden);
    lw^.WO := Matrix(aFile, p + 'self_attn.o_proj.weight', fHidden, q);
    lw^.WGate := Matrix(aFile, p + 'mlp.gate_proj.weight', fMlp, fHidden);
    lw^.WUp := Matrix(aFile, p + 'mlp.up_proj.weight', fMlp, fHidden);
    lw^.WDown := Matrix(aFile, p + 'mlp.down_proj.weight', fHidden, fMlp);
  end;
  fNorm := QwenWiden(aFile, aPrefix + 'norm.weight', 1, fHidden);
  fPool := nil;
end;

const
  /// the rows of one conversion block: a few hundred microseconds
  MATRIX_ROWS = 32;

procedure TQwenLmWeights.MatrixRows(aItem: integer);
var
  r, last, b, blocks: PtrInt;
  dst: PInferBlockQ8_0Array;
  block: array[0..31] of single;
begin
  r := aItem * MATRIX_ROWS;
  last := r + MATRIX_ROWS;
  if last > fJobRows then
    last := fJobRows;
  blocks := fJobCols shr 5;
  while r < last do
  begin
    if fDType = idtQ8_0 then
    begin
      // the bytes of ggml (quantize_row_q8_0_ref), those of a GGUF: each block
      // of 32 depends only on its values, wherever it is computed; on the
      // stack (a heap shared between the threads would make them wait)
      dst := @fJobData^[r * fJobRowBytes];
      for b := 0 to blocks - 1 do
      begin
        InferRowToF32(fJobTensor^.Data, fJobTensor^.DType, r * blocks + b, 32,
          @block);
        InferQuantizeRowQ8_0Ref(@block, 32, @dst^[b]);
      end;
    end
    else
      InferRowToF32(fJobTensor^.Data, fJobTensor^.DType, r, fJobCols,
        @fJobData^[r * fJobRowBytes]);
    inc(r);
  end;
end;

function TQwenLmWeights.Matrix(aFile: TSafeTensorsFile; const aName: RawUtf8;
  aRows, aCols: PtrInt): TInferWeight;
var
  t: PInferTensor;
  i, items, rowBytes: PtrInt;
  data: PByteArray;
begin
  t := aFile.Tensor(aName);
  if t^.Count <> Int64(aRows) * aCols then
    EQwenLm.RaiseUtf8('%: % is % instead of [%,%]',
      [self, aName, InferShapeText(t^.Shape), aRows, aCols]);
  if fDType = idtQ8_0 then
    rowBytes := (aCols shr 5) * SizeOf(TInferBlockQ8_0)
  else
    rowBytes := aCols * SizeOf(single);
  if fStoreCount = system.length(fStore) then
    SetLength(fStore, fStoreCount * 2 + 16);
  SetLength(fStore[fStoreCount], Int64(aRows) * rowBytes);
  data := pointer(fStore[fStoreCount]);
  inc(fStoreCount);
  inc(fBytes, Int64(aRows) * rowBytes);
  fJobTensor := t;
  fJobData := data;
  fJobRows := aRows;
  fJobCols := aCols;
  fJobRowBytes := rowBytes;
  items := (aRows + MATRIX_ROWS - 1) div MATRIX_ROWS;
  if fPool = nil then
    for i := 0 to items - 1 do
      MatrixRows(i)
  else
    fPool.Run(MatrixRows, items);
  result := InferWeight(data, fDType);
end;

function TQwenLmWeights.AddOutput(aFile: TSafeTensorsFile; const aTensor: RawUtf8;
  aRows: PtrInt; aPool: TInferThreadPool): PtrInt;
begin
  result := system.length(fOutputs);
  SetLength(fOutputs, result + 1);
  SetLength(fOutputRows, result + 1);
  fPool := aPool;
  try
    fOutputs[result] := Matrix(aFile, aTensor, aRows, fHidden);
  finally
    fPool := nil;
  end;
  fOutputRows[result] := aRows;
end;

function TQwenLmWeights.Digest: cardinal;
var
  i: PtrInt;
begin
  result := 0;
  for i := 0 to fStoreCount - 1 do
    result := crc32c(result, pointer(fStore[i]), system.length(fStore[i]));
end;

function TQwenLmWeights.LayerCount: PtrInt;
begin
  result := system.length(fLayers);
end;

function TQwenLmWeights.Layer(aIndex: PtrInt): PQwenLmLayer;
begin
  result := @fLayers[aIndex];
end;

function TQwenLmWeights.OutputCount: PtrInt;
begin
  result := system.length(fOutputs);
end;

function TQwenLmWeights.Output(aIndex: PtrInt): TInferWeight;
begin
  result := fOutputs[aIndex];
end;

function TQwenLmWeights.OutputRows(aIndex: PtrInt): PtrInt;
begin
  result := fOutputRows[aIndex];
end;


{ TQwenLm }

constructor TQwenLm.Create(aWeights: TQwenLmWeights);
var
  i: PtrInt;
begin
  inherited Create;
  fWeights := aWeights;
  // inv_freq = 1 / theta^(2i / head_dim), in float32 as PyTorch
  SetLength(fInvFreq, aWeights.HeadDim shr 1);
  for i := 0 to high(fInvFreq) do
    fInvFreq[i] := 1 / Exp(Ln(aWeights.Theta) * (2 * i / aWeights.HeadDim));
end;

procedure TQwenLm.Reset;
begin
  fLen := 0;
end;

procedure TQwenLm.Rotary(aRows: PtrInt; aCos, aSin: PSingleArray);
var
  r, i, half: PtrInt;
  f: single;
begin
  half := system.length(fInvFreq);
  for r := 0 to aRows - 1 do
    for i := 0 to half - 1 do
    begin
      f := (fLen + r) * fInvFreq[i]; // float32 product, as PyTorch
      aCos^[r * half + i] := Cos(f);
      aSin^[r * half + i] := Sin(f);
    end;
end;


{ TQwenLmCpu }

constructor TQwenLmCpu.CreateCpu(aWeights: TQwenLmWeights; aCpu: TInferCpuFastBackend);
begin
  inherited Create(aWeights);
  fCpu := aCpu;
  SetLength(fKeys, system.length(aWeights.fLayers));
  SetLength(fValues, system.length(aWeights.fLayers));
end;

procedure TQwenLmCpu.Room(aRows: PtrInt);
var
  l, kv: PtrInt;
begin
  if fLen + aRows <= fCap then
    exit;
  fCap := (fLen + aRows) * 2 + 64;
  kv := fWeights.KvHeads * fWeights.HeadDim;
  for l := 0 to high(fKeys) do
  begin
    // SetLength keeps the rows already computed
    SetLength(fKeys[l], fCap * kv);
    SetLength(fValues[l], fCap * kv);
  end;
end;

procedure TQwenLmCpu.Forward(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
  aLogits, aHidden: PSingleArray);
var
  w: TQwenLmWeights;
  l, i, hid, q, kv, mlp, heads, kvHeads, hd: PtrInt;
  lw: PQwenLmLayer;
  h, last: PSingleArray;
begin
  if aRows <= 0 then
    exit;
  w := fWeights;
  hid := w.Hidden;
  heads := w.Heads;
  kvHeads := w.KvHeads;
  hd := w.HeadDim;
  mlp := w.Mlp;
  q := heads * hd;
  kv := kvHeads * hd;
  QwenGrow(fH, aRows * hid);
  QwenGrow(fN, aRows * hid);
  QwenGrow(fQ, aRows * q);
  QwenGrow(fK, aRows * kv);
  QwenGrow(fV, aRows * kv);
  QwenGrow(fAtt, aRows * q);
  QwenGrow(fO, aRows * hid);
  QwenGrow(fG, aRows * mlp);
  QwenGrow(fU, aRows * mlp);
  QwenGrow(fCos, aRows * (hd shr 1));
  QwenGrow(fSin, aRows * (hd shr 1));
  h := pointer(fH);
  MoveFast(x^[0], h^[0], aRows * hid * SizeOf(single));
  Rotary(aRows, pointer(fCos), pointer(fSin));
  Room(aRows);
  for l := 0 to high(w.fLayers) do
  begin
    lw := @w.fLayers[l];
    fCpu.RmsNorm(h, aRows, hid, pointer(lw^.InNorm), w.Eps, pointer(fN));
    fCpu.Linear(pointer(fN), aRows, hid, lw^.WQ, q, nil, pointer(fQ));
    fCpu.Linear(pointer(fN), aRows, hid, lw^.WK, kv, nil, pointer(fK));
    fCpu.Linear(pointer(fN), aRows, hid, lw^.WV, kv, nil, pointer(fV));
    // q_norm / k_norm: one RMSNorm per head
    fCpu.RmsNorm(pointer(fQ), aRows * heads, hd, pointer(lw^.QNorm), w.Eps,
      pointer(fQ));
    fCpu.RmsNorm(pointer(fK), aRows * kvHeads, hd, pointer(lw^.KNorm), w.Eps,
      pointer(fK));
    fCpu.Rope(pointer(fQ), aRows, heads, hd, pointer(fCos), pointer(fSin));
    fCpu.Rope(pointer(fK), aRows, kvHeads, hd, pointer(fCos), pointer(fSin));
    MoveFast(fK[0], fKeys[l][fLen * kv], aRows * kv * SizeOf(single));
    MoveFast(fV[0], fValues[l][fLen * kv], aRows * kv * SizeOf(single));
    fCpu.CausalAttention(pointer(fQ), aRows, heads, hd, pointer(fKeys[l]),
      pointer(fValues[l]), fLen, kvHeads, pointer(fAtt));
    fCpu.Linear(pointer(fAtt), aRows, q, lw^.WO, hid, nil, pointer(fO));
    for i := 0 to aRows * hid - 1 do
      h^[i] := h^[i] + fO[i];
    fCpu.RmsNorm(h, aRows, hid, pointer(lw^.PostNorm), w.Eps, pointer(fN));
    fCpu.Linear(pointer(fN), aRows, hid, lw^.WGate, mlp, nil, pointer(fG));
    fCpu.Linear(pointer(fN), aRows, hid, lw^.WUp, mlp, nil, pointer(fU));
    fCpu.SwiGlu(pointer(fG), pointer(fU), aRows * mlp);
    fCpu.Linear(pointer(fG), aRows, mlp, lw^.WDown, hid, nil, pointer(fO));
    for i := 0 to aRows * hid - 1 do
      h^[i] := h^[i] + fO[i];
    if Assigned(fProbe) then
      fProbe(FormatUtf8('%.layer%', [w.Name, l]), h, aRows, hid);
  end;
  fCpu.RmsNorm(h, aRows, hid, pointer(w.fNorm), w.Eps, pointer(fN));
  if Assigned(fProbe) then
    fProbe(w.Name + '.out', pointer(fN), aRows, hid);
  last := @fN[(aRows - 1) * hid];
  if aHidden <> nil then
    MoveFast(last^[0], aHidden^[0], hid * SizeOf(single));
  if aLogits <> nil then
    fCpu.Linear(last, 1, hid, w.fOutputs[aOutput], w.fOutputRows[aOutput], nil,
      aLogits);
  inc(fLen, aRows);
end;

end.
