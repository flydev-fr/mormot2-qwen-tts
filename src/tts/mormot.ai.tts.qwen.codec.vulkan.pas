/// Qwen3-TTS - the 12 Hz decoder with its heavy part on the GPU (Vulkan)
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the front part (quantizer, pre_conv, sliding window transformer: a few
// percent of the work) stays on the CPU, in TQwenCodec; the back part
// (upsampling, ConvNeXt, input conv, 4 blocks of dilated convolutions and
// SnakeBeta, output conv: ~2.4 GMAC per frame) runs on the GPU as one
// recorded pass per call: the transformer output goes up, the PCM comes back
// - the shaders: tools/tts/vk (convgemm, snake, dwconv, layernorm), compiled into
// mormot.ai.tts.qwen.spv.inc; the weights are float32, laid out for them
// - as on the CPU, everything is causal: each convolution input keeps its last
// rows on the GPU (Hist) between two calls, copied in front of the new ones;
// each output sums its terms in the same order whatever the number of frames:
// decoding 1 + 3 frames or 4 at once gives the same sound
// - the GPU is the one of the talker (TQwenVulkan), under the same lock
unit mormot.ai.tts.qwen.codec.vulkan;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.vulkan,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.lm.vulkan;

type
  /// a convolution input on the GPU: Buf = [Past history rows ; new rows],
  // Hist = the Past last rows of the previous call
  TQwenVkSeq = record
    Buf, Hist: TInferVkBuffer;
    Past, Cols: PtrInt;
  end;
  PQwenVkSeq = ^TQwenVkSeq;

  /// a convolution on the GPU: weights [Out][Taps * In], bias [Out]
  TQwenVkConv = record
    W, B: TInferVkBuffer;
    InCols, OutCols, Taps, Dil: PtrInt;
  end;

  /// SnakeBeta on the GPU: exp(alpha) and 1 / (exp(beta) + 1e-9)
  TQwenVkSnake = record
    Alpha, InvBeta: TInferVkBuffer;
  end;

  TQwenVkNext = record
    DwW, DwB, NormG, NormB, Gamma: TInferVkBuffer;
    Pw1, Pw2: TQwenVkConv;
  end;

  TQwenVkRes = record
    Act1, Act2: TQwenVkSnake;
    Conv1, Conv2: TQwenVkConv;
    Input: TQwenVkSeq;
  end;

  TQwenVkBlock = record
    Act: TQwenVkSnake;
    Up: TQwenVkConv;
    Rate, InCols, OutCols: PtrInt;
    Input: TQwenVkSeq;
    Res: array[0..2] of TQwenVkRes;
    Output: TInferVkBuffer;
  end;

  /// TQwenCodec whose back part runs on the GPU of aVulkan
  TQwenCodecVulkan = class(TQwenCodec)
  protected
    fVulkan: TQwenVulkan;
    fVk: TInferVulkan;
    fConvGemm, fSnake, fDwConv, fLayerNorm: TInferVkPipeline;
    fOwned: array of TInferVkBuffer;
    fDeviceBytes: Int64;
    fCapFrames: PtrInt;
    // the layers
    fUp: array[0..1] of TQwenVkConv;
    fNext: array[0..1] of TQwenVkNext;
    fConv0: TQwenVkConv;
    fBlocks: array[0..3] of TQwenVkBlock;
    fActOut: TQwenVkSnake;
    fConvOut: TQwenVkConv;
    // the sequences with history
    fS1, fS2, fS3, fFinal: TQwenVkSeq;
    // the work buffers
    fIn, fOut: TInferVkBuffer; // visible to the host: upload, read back
    fU0, fU1, fN, fH, fX0, fD, fPcm: TInferVkBuffer;
    function NewBuffer(aBytes: Int64): TInferVkBuffer;
    function Upload(aData: pointer; aCount: PtrInt): TInferVkBuffer;
    function UploadConv(const aName: RawUtf8; aIn, aOut, aTaps, aDil: PtrInt): TQwenVkConv;
    function UploadTransConv(const aName: RawUtf8; aIn, aOut, aKernel,
      aStride: PtrInt): TQwenVkConv;
    function UploadLinear(const aName: RawUtf8; aIn, aOut: PtrInt): TQwenVkConv;
    function UploadSnake(const aName: RawUtf8; aCount: PtrInt): TQwenVkSnake;
    procedure NewSeq(var aSeq: TQwenVkSeq; aPast, aCols: PtrInt);
    procedure Room(aFrames: PtrInt);
    procedure FreeWork;
    // the recording of a pass
    /// aRows rows of aConv: x read from element aXOff (history included),
    // y written from element aYOff, the residual r read from element aROff
    procedure Gemm(const aConv: TQwenVkConv; x: TInferVkBuffer; aXOff: PtrInt;
      aRows: PtrInt; y: TInferVkBuffer; aYOff: PtrInt; aFlags: cardinal = 0;
      r: TInferVkBuffer = nil; aROff: PtrInt = 0; s: TInferVkBuffer = nil);
    procedure Snake(const aSnake: TQwenVkSnake; x: TInferVkBuffer; aXOff: PtrInt;
      y: TInferVkBuffer; aYOff, aCount, aCols: PtrInt);
    procedure Restore(const aSeq: TQwenVkSeq);
    procedure Save(const aSeq: TQwenVkSeq; aRows: PtrInt);
    procedure NextBlock(const aNext: TQwenVkNext; const aIn: TQwenVkSeq;
      aRows: PtrInt; y: TInferVkBuffer; aYOff: PtrInt);
    procedure Chunk(x: PSingleArray; aFrames: PtrInt; aPcm: PSingleArray);
    procedure Back(x: PSingleArray; aFrames: PtrInt; aPcm: PSingleArray); override;
  public
    /// load aFolder/model.safetensors as TQwenCodec, then send the weights of
    // the back part to the GPU of aVulkan (not owned)
    constructor CreateVulkan(const aFolder: TFileName; aVulkan: TQwenVulkan;
      aThreads: integer = 0);
    destructor Destroy; override;
    procedure Reset; override;
    /// the bytes allocated on the GPU: weights, histories, work buffers
    property DeviceBytes: Int64
      read fDeviceBytes;
  end;


implementation

{$I mormot.ai.tts.qwen.spv.inc}

type
  TGemmPush = record
    Rows, Nin, Nout, Cin, Dil, XOff, YOff, YStride, ROff, Flags: cardinal;
  end;

  TRowsPush = record
    Count, Cols, XOff, YOff: cardinal;
  end;

  TNormPush = record
    Rows, Cols, XOff, YOff: cardinal;
    Eps: single;
  end;

const
  GEMM_GELU = 1;
  GEMM_RESIDUAL = 2;
  GEMM_SCALED = 4;
  GEMM_CLAMP = 8;

  LATENT = 1024;
  DECODER_DIM = 1536;
  /// the frames of one pass at most: the dispatches stay below 65535 groups
  // (1920 x 96 values per frame for snake, by groups of 128)
  CHUNK_FRAMES = 16;
  RATES: array[0..3] of PtrInt = (8, 5, 4, 3);
  DILATIONS: array[0..2] of PtrInt = (1, 3, 9);


{ TQwenCodecVulkan }

constructor TQwenCodecVulkan.CreateVulkan(const aFolder: TFileName;
  aVulkan: TQwenVulkan; aThreads: integer);
var
  i, j, cin, cout: PtrInt;
  r: ^TQwenVkRes;
  p: RawUtf8;
begin
  inherited Create(aFolder, aThreads);
  fVulkan := aVulkan;
  fVk := aVulkan.Backend.Vulkan;
  fConvGemm := TInferVkPipeline.Create(fVk, @SPV_QWEN_CONVGEMM,
    length(SPV_QWEN_CONVGEMM), 6, SizeOf(TGemmPush));
  fSnake := TInferVkPipeline.Create(fVk, @SPV_QWEN_SNAKE, length(SPV_QWEN_SNAKE), 4,
    SizeOf(TRowsPush));
  fDwConv := TInferVkPipeline.Create(fVk, @SPV_QWEN_DWCONV, length(SPV_QWEN_DWCONV), 4,
    SizeOf(TRowsPush));
  fLayerNorm := TInferVkPipeline.Create(fVk, @SPV_QWEN_LAYERNORM,
    length(SPV_QWEN_LAYERNORM), 4, SizeOf(TNormPush));
  fConvGemm.Name := 'convgemm';
  fSnake.Name := 'snake';
  fDwConv.Name := 'dwconv';
  fLayerNorm.Name := 'layernorm';
  aVulkan.Backend.GpuLock;
  try
    for i := 0 to 1 do
    begin
      fUp[i] := UploadTransConv(FormatUtf8('decoder.upsample.%.0.conv', [i]),
        LATENT, LATENT, 2, 2);
      p := FormatUtf8('decoder.upsample.%.1.', [i]);
      fNext[i].DwW := Upload(F32(p + 'dwconv.conv.weight', [LATENT, 1, 7]), LATENT * 7);
      fNext[i].DwB := Upload(F32(p + 'dwconv.conv.bias', [LATENT]), LATENT);
      fNext[i].NormG := Upload(F32(p + 'norm.weight', [LATENT]), LATENT);
      fNext[i].NormB := Upload(F32(p + 'norm.bias', [LATENT]), LATENT);
      fNext[i].Gamma := Upload(F32(p + 'gamma', [LATENT]), LATENT);
      fNext[i].Pw1 := UploadLinear(p + 'pwconv1', LATENT, 4 * LATENT);
      fNext[i].Pw2 := UploadLinear(p + 'pwconv2', 4 * LATENT, LATENT);
    end;
    fConv0 := UploadConv('decoder.decoder.0.conv', LATENT, DECODER_DIM, 7, 1);
    for i := 0 to 3 do
    begin
      cin := DECODER_DIM shr i;
      cout := DECODER_DIM shr (i + 1);
      p := FormatUtf8('decoder.decoder.%.block.', [i + 1]);
      fBlocks[i].Rate := RATES[i];
      fBlocks[i].InCols := cin;
      fBlocks[i].OutCols := cout;
      fBlocks[i].Act := UploadSnake(p + '0', cin);
      fBlocks[i].Up := UploadTransConv(p + '1.conv', cin, cout, 2 * RATES[i], RATES[i]);
      NewSeq(fBlocks[i].Input, fBlocks[i].Up.Taps - 1, cin);
      for j := 0 to 2 do
      begin
        r := @fBlocks[i].Res[j];
        r^.Act1 := UploadSnake(FormatUtf8('%%.act1', [p, j + 2]), cout);
        r^.Conv1 := UploadConv(FormatUtf8('%%.conv1.conv', [p, j + 2]), cout, cout, 7,
          DILATIONS[j]);
        r^.Act2 := UploadSnake(FormatUtf8('%%.act2', [p, j + 2]), cout);
        r^.Conv2 := UploadConv(FormatUtf8('%%.conv2.conv', [p, j + 2]), cout, cout, 1, 1);
        NewSeq(r^.Input, 6 * DILATIONS[j], cout);
      end;
    end;
    fActOut := UploadSnake('decoder.decoder.5', DECODER_DIM shr 4);
    fConvOut := UploadConv('decoder.decoder.6.conv', DECODER_DIM shr 4, 1, 7, 1);
    NewSeq(fS1, 6, LATENT);
    NewSeq(fS2, 6, LATENT);
    NewSeq(fS3, 6, LATENT);
    NewSeq(fFinal, 6, DECODER_DIM shr 4);
    Room(8);
  finally
    aVulkan.Backend.GpuUnLock;
  end;
  Reset;
end;

destructor TQwenCodecVulkan.Destroy;
var
  i: PtrInt;
begin
  FreeWork;
  for i := 0 to high(fOwned) do
    fOwned[i].Free;
  fLayerNorm.Free;
  fDwConv.Free;
  fSnake.Free;
  fConvGemm.Free;
  inherited Destroy;
end;

function TQwenCodecVulkan.NewBuffer(aBytes: Int64): TInferVkBuffer;
begin
  result := TInferVkBuffer.Create(fVk, aBytes, true);
  SetLength(fOwned, length(fOwned) + 1);
  fOwned[high(fOwned)] := result;
  inc(fDeviceBytes, result.Size);
end;

function TQwenCodecVulkan.Upload(aData: pointer; aCount: PtrInt): TInferVkBuffer;
begin
  result := NewBuffer(Int64(aCount) * 4);
  fVk.Upload(result, 0, aData, Int64(aCount) * 4);
end;

function TQwenCodecVulkan.UploadConv(const aName: RawUtf8;
  aIn, aOut, aTaps, aDil: PtrInt): TQwenVkConv;
var
  w: PSingleArray;
  g: TSingleDynArray;
  o, k, c: PtrInt;
begin
  // the file: [Out][In][Taps]; the shader: [Out][Taps][In]
  w := F32(aName + '.weight', [aOut, aIn, aTaps]);
  SetLength(g, aOut * aTaps * aIn);
  for o := 0 to aOut - 1 do
    for k := 0 to aTaps - 1 do
      for c := 0 to aIn - 1 do
        g[(o * aTaps + k) * aIn + c] := w^[(o * aIn + c) * aTaps + k];
  result.W := Upload(pointer(g), length(g));
  result.B := Upload(F32(aName + '.bias', [aOut]), aOut);
  result.InCols := aIn;
  result.OutCols := aOut;
  result.Taps := aTaps;
  result.Dil := aDil;
end;

function TQwenCodecVulkan.UploadTransConv(const aName: RawUtf8;
  aIn, aOut, aKernel, aStride: PtrInt): TQwenVkConv;
var
  w, b: PSingleArray;
  g, gb: TSingleDynArray;
  taps, j, o, k, c, cols: PtrInt;
begin
  // the file: [In][Out][Kernel]; output j of step t receives
  // x[t - i] . w[:, :, j + i * s]; the shader reads the taps from the oldest
  // (k = 0, x[t - taps + 1]) to the most recent: i = taps - 1 - k
  if aKernel mod aStride <> 0 then
    EQwenCodec.RaiseUtf8('%: kernel % is not a multiple of stride %',
      [aName, aKernel, aStride]);
  taps := aKernel div aStride;
  w := F32(aName + '.weight', [aIn, aOut, aKernel]);
  b := F32(aName + '.bias', [aOut]);
  cols := taps * aIn;
  SetLength(g, aStride * aOut * cols);
  SetLength(gb, aStride * aOut);
  for j := 0 to aStride - 1 do
    for o := 0 to aOut - 1 do
    begin
      gb[j * aOut + o] := b^[o];
      for k := 0 to taps - 1 do
        for c := 0 to aIn - 1 do
          g[(j * aOut + o) * cols + k * aIn + c] :=
            w^[(c * aOut + o) * aKernel + j + (taps - 1 - k) * aStride];
    end;
  result.W := Upload(pointer(g), length(g));
  result.B := Upload(pointer(gb), length(gb));
  result.InCols := aIn;
  result.OutCols := aStride * aOut;
  result.Taps := taps;
  result.Dil := 1;
end;

function TQwenCodecVulkan.UploadLinear(const aName: RawUtf8; aIn, aOut: PtrInt): TQwenVkConv;
begin
  result.W := Upload(F32(aName + '.weight', [aOut, aIn]), aOut * aIn);
  result.B := Upload(F32(aName + '.bias', [aOut]), aOut);
  result.InCols := aIn;
  result.OutCols := aOut;
  result.Taps := 1;
  result.Dil := 1;
end;

function TQwenCodecVulkan.UploadSnake(const aName: RawUtf8; aCount: PtrInt): TQwenVkSnake;
var
  a, b: PSingleArray;
  alpha, inv: TSingleDynArray;
  c: PtrInt;
  e: single;
begin
  // the same values as TQwenSnake, in float32
  a := F32(aName + '.alpha', [aCount]);
  b := F32(aName + '.beta', [aCount]);
  SetLength(alpha, aCount);
  SetLength(inv, aCount);
  for c := 0 to aCount - 1 do
  begin
    alpha[c] := Exp(a^[c]);
    e := Exp(b^[c]);
    inv[c] := 1 / (e + 1e-9);
  end;
  result.Alpha := Upload(pointer(alpha), aCount);
  result.InvBeta := Upload(pointer(inv), aCount);
end;

procedure TQwenCodecVulkan.NewSeq(var aSeq: TQwenVkSeq; aPast, aCols: PtrInt);
begin
  aSeq.Past := aPast;
  aSeq.Cols := aCols;
  aSeq.Buf := nil; // allocated by Room
  aSeq.Hist := nil;
  if aPast > 0 then
    aSeq.Hist := NewBuffer(Int64(aPast) * aCols * 4);
end;

procedure TQwenCodecVulkan.FreeWork;
var
  i, j: PtrInt;

  procedure Drop(var b: TInferVkBuffer);
  begin
    if b <> nil then
    begin
      dec(fDeviceBytes, b.Size);
      FreeAndNil(b);
    end;
  end;

begin
  Drop(fIn);
  Drop(fOut);
  Drop(fU0);
  Drop(fU1);
  Drop(fN);
  Drop(fH);
  Drop(fX0);
  Drop(fD);
  Drop(fPcm);
  Drop(fS1.Buf);
  Drop(fS2.Buf);
  Drop(fS3.Buf);
  Drop(fFinal.Buf);
  for i := 0 to 3 do
  begin
    Drop(fBlocks[i].Input.Buf);
    Drop(fBlocks[i].Output);
    for j := 0 to 2 do
      Drop(fBlocks[i].Res[j].Input.Buf);
  end;
  fCapFrames := 0;
end;

procedure TQwenCodecVulkan.Room(aFrames: PtrInt);
var
  i, j, rows, maxD: PtrInt;

  function Work(aBytes: Int64): TInferVkBuffer;
  begin
    result := TInferVkBuffer.Create(fVk, aBytes, true);
    inc(fDeviceBytes, result.Size);
  end;

  procedure SeqBuf(var aSeq: TQwenVkSeq; aRows: PtrInt);
  begin
    aSeq.Buf := Work(Int64(aSeq.Past + aRows) * aSeq.Cols * 4);
  end;

begin
  if aFrames <= fCapFrames then
    exit;
  // the content of the work buffers is rebuilt at each call: the histories
  // (Hist) do not move
  FreeWork;
  fIn := TInferVkBuffer.Create(fVk, Int64(aFrames) * LATENT * 4);
  inc(fDeviceBytes, fIn.Size);
  fOut := TInferVkBuffer.CreateReadback(fVk, Int64(aFrames) * QWEN_FRAME_SAMPLES * 4);
  inc(fDeviceBytes, fOut.Size);
  fU0 := Work(Int64(aFrames) * LATENT * 4);
  fU1 := Work(Int64(aFrames) * 2 * LATENT * 4);
  fN := Work(Int64(aFrames) * 4 * LATENT * 4);
  fH := Work(Int64(aFrames) * 4 * 4 * LATENT * 4);
  fX0 := Work(Int64(aFrames) * 4 * DECODER_DIM * 4);
  fPcm := Work(Int64(aFrames) * QWEN_FRAME_SAMPLES * 4);
  SeqBuf(fS1, 2 * aFrames);
  SeqBuf(fS2, 4 * aFrames);
  SeqBuf(fS3, 4 * aFrames);
  SeqBuf(fFinal, aFrames * QWEN_FRAME_SAMPLES);
  rows := 4 * aFrames;
  maxD := 0;
  for i := 0 to 3 do
  begin
    SeqBuf(fBlocks[i].Input, rows);
    rows := rows * fBlocks[i].Rate;
    fBlocks[i].Output := Work(Int64(rows) * fBlocks[i].OutCols * 4);
    for j := 0 to 2 do
      SeqBuf(fBlocks[i].Res[j].Input, rows);
    if rows * fBlocks[i].OutCols > maxD then
      maxD := rows * fBlocks[i].OutCols;
  end;
  fD := Work(Int64(maxD) * 4);
  fCapFrames := aFrames;
end;

procedure TQwenCodecVulkan.Reset;
var
  zero: TSingleDynArray;
  i, j: PtrInt;

  procedure Clear(const aSeq: TQwenVkSeq);
  begin
    if aSeq.Hist <> nil then
      fVk.Upload(aSeq.Hist, 0, pointer(zero), Int64(aSeq.Past) * aSeq.Cols * 4);
  end;

begin
  inherited Reset;
  if fVulkan = nil then
    exit; // called by TQwenCodec.Create, before the buffers
  // the history of each input: zeros, the causal padding; the largest one is
  // the third residual unit of block 0: 6 x 9 rows of 768
  SetLength(zero, 6 * DILATIONS[2] * (DECODER_DIM shr 1));
  fVulkan.Backend.GpuLock;
  try
    Clear(fS1);
    Clear(fS2);
    Clear(fS3);
    Clear(fFinal);
    for i := 0 to 3 do
    begin
      Clear(fBlocks[i].Input);
      for j := 0 to 2 do
        Clear(fBlocks[i].Res[j].Input);
    end;
  finally
    fVulkan.Backend.GpuUnLock;
  end;
end;

procedure TQwenCodecVulkan.Gemm(const aConv: TQwenVkConv; x: TInferVkBuffer;
  aXOff: PtrInt; aRows: PtrInt; y: TInferVkBuffer; aYOff: PtrInt;
  aFlags: cardinal; r: TInferVkBuffer; aROff: PtrInt; s: TInferVkBuffer);
var
  push: TGemmPush;
begin
  push.Rows := aRows;
  push.Nin := aConv.Taps * aConv.InCols;
  push.Nout := aConv.OutCols;
  push.Cin := aConv.InCols;
  push.Dil := aConv.Dil;
  push.XOff := aXOff;
  push.YOff := aYOff;
  push.YStride := aConv.OutCols;
  push.ROff := aROff;
  push.Flags := aFlags;
  if r = nil then
    r := y; // bound, not read
  if s = nil then
    s := aConv.B;
  fVk.BatchDispatch(fConvGemm, [aConv.W, x, aConv.B, y, r, s], @push,
    (aConv.OutCols + 63) shr 6, (aRows + 63) shr 6);
end;

procedure TQwenCodecVulkan.Snake(const aSnake: TQwenVkSnake; x: TInferVkBuffer;
  aXOff: PtrInt; y: TInferVkBuffer; aYOff, aCount, aCols: PtrInt);
var
  push: TRowsPush;
begin
  push.Count := aCount;
  push.Cols := aCols;
  push.XOff := aXOff;
  push.YOff := aYOff;
  fVk.BatchDispatch(fSnake, [x, y, aSnake.Alpha, aSnake.InvBeta], @push,
    (aCount + 127) shr 7);
end;

procedure TQwenCodecVulkan.Restore(const aSeq: TQwenVkSeq);
begin
  // the history in front of the new rows
  if aSeq.Past > 0 then
    fVk.BatchCopy(aSeq.Hist, 0, aSeq.Buf, 0, Int64(aSeq.Past) * aSeq.Cols * 4);
end;

procedure TQwenCodecVulkan.Save(const aSeq: TQwenVkSeq; aRows: PtrInt);
begin
  // the Past last rows of [history ; new rows] for the next call
  if aSeq.Past > 0 then
    fVk.BatchCopy(aSeq.Buf, Int64(aRows) * aSeq.Cols * 4, aSeq.Hist, 0,
      Int64(aSeq.Past) * aSeq.Cols * 4);
end;

procedure TQwenCodecVulkan.NextBlock(const aNext: TQwenVkNext; const aIn: TQwenVkSeq;
  aRows: PtrInt; y: TInferVkBuffer; aYOff: PtrInt);
var
  rows: TRowsPush;
  norm: TNormPush;
begin
  // dwconv (7, causal) -> LayerNorm -> pwconv1 + GELU -> pwconv2, then
  // y = input + gamma * output
  rows.Count := aRows * LATENT;
  rows.Cols := LATENT;
  rows.XOff := 0;
  rows.YOff := 0;
  fVk.BatchDispatch(fDwConv, [aIn.Buf, aNext.DwW, aNext.DwB, fN], @rows,
    (rows.Count + 127) shr 7);
  norm.Rows := aRows;
  norm.Cols := LATENT;
  norm.XOff := 0;
  norm.YOff := 0;
  norm.Eps := 1e-6;
  fVk.BatchDispatch(fLayerNorm, [fN, aNext.NormG, aNext.NormB, fN], @norm, aRows);
  Gemm(aNext.Pw1, fN, 0, aRows, fH, 0, GEMM_GELU);
  Gemm(aNext.Pw2, fH, 0, aRows, y, aYOff, GEMM_SCALED, aIn.Buf, aIn.Past * LATENT,
    aNext.Gamma);
  Save(aIn, aRows);
end;

procedure TQwenCodecVulkan.Chunk(x: PSingleArray; aFrames: PtrInt; aPcm: PSingleArray);
var
  i, j, rows, cols: PtrInt;
  input: TInferVkBuffer;
  b: ^TQwenVkBlock;
  r: ^TQwenVkRes;
begin
  MoveFast(x^[0], fIn.Mapped^, aFrames * LATENT * SizeOf(single));
  fVk.BatchBegin;
  try
    fVk.BatchCopy(fIn, 0, fU0, 0, Int64(aFrames) * LATENT * 4);
    // upsampling x2 and ConvNeXt, twice; the output [F][2 x 1024] of a
    // transposed convolution already is [2F][1024], written after the history
    Restore(fS1);
    Gemm(fUp[0], fU0, 0, aFrames, fS1.Buf, fS1.Past * LATENT);
    rows := 2 * aFrames;
    NextBlock(fNext[0], fS1, rows, fU1, 0);
    Restore(fS2);
    Restore(fS3);
    Gemm(fUp[1], fU1, 0, rows, fS2.Buf, fS2.Past * LATENT);
    rows := 2 * rows;
    NextBlock(fNext[1], fS2, rows, fS3.Buf, fS3.Past * LATENT);
    Gemm(fConv0, fS3.Buf, 0, rows, fX0, 0);
    Save(fS3, rows);
    fVk.BatchFlush;
    // the 4 blocks
    input := fX0;
    cols := DECODER_DIM;
    for i := 0 to 3 do
    begin
      b := @fBlocks[i];
      Restore(b^.Input);
      Snake(b^.Act, input, 0, b^.Input.Buf, b^.Input.Past * cols, rows * cols, cols);
      Gemm(b^.Up, b^.Input.Buf, 0, rows, b^.Output, 0);
      Save(b^.Input, rows);
      rows := rows * b^.Rate;
      cols := b^.OutCols;
      for j := 0 to 2 do
      begin
        r := @b^.Res[j];
        Restore(r^.Input);
        Snake(r^.Act1, b^.Output, 0, r^.Input.Buf, r^.Input.Past * cols,
          rows * cols, cols);
        Gemm(r^.Conv1, r^.Input.Buf, 0, rows, fD, 0);
        Save(r^.Input, rows);
        Snake(r^.Act2, fD, 0, fD, 0, rows * cols, cols);
        Gemm(r^.Conv2, fD, 0, rows, b^.Output, 0, GEMM_RESIDUAL, b^.Output, 0);
      end;
      input := b^.Output;
      fVk.BatchFlush;
    end;
    // the output: snake, conv 96 -> 1, clamp
    Restore(fFinal);
    Snake(fActOut, input, 0, fFinal.Buf, fFinal.Past * cols, rows * cols, cols);
    Gemm(fConvOut, fFinal.Buf, 0, rows, fPcm, 0, GEMM_CLAMP);
    Save(fFinal, rows);
    fVk.BatchCopy(fPcm, 0, fOut, 0, Int64(rows) * 4);
  except
    fVk.BatchAbort;
    raise;
  end;
  fVk.BatchSubmit;
  MoveFast(fOut.Mapped^, aPcm^[0], aFrames * QWEN_FRAME_SAMPLES * SizeOf(single));
end;

procedure TQwenCodecVulkan.Back(x: PSingleArray; aFrames: PtrInt; aPcm: PSingleArray);
var
  done, n: PtrInt;
begin
  fVulkan.Backend.GpuLock;
  try
    Room(MinPtrInt(aFrames, CHUNK_FRAMES));
    done := 0;
    while done < aFrames do
    begin
      n := aFrames - done;
      if n > CHUNK_FRAMES then
        n := CHUNK_FRAMES;
      Chunk(@x^[done * LATENT], n, @aPcm^[done * QWEN_FRAME_SAMPLES]);
      inc(done, n);
    end;
  finally
    fVulkan.Backend.GpuUnLock;
  end;
end;

end.
