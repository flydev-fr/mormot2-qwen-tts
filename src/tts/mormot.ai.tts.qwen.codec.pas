/// Qwen3-TTS - the 12 Hz audio decoder (speech_tokenizer), frame by frame
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - one frame = 16 codes (12.5 frames per second) -> 1920 samples at 24 kHz
// - everything is causal: causal convolutions, attention limited to the last
//   72 frames; decoding one frame after another therefore gives exactly the
//   full decode - each layer keeps only what it needs of the past (the last
//   rows of its input, the KV cache of the attention)
// - reference: qwen_tts 0.1.1, Qwen3TTSTokenizerV2Decoder in
//   core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
// - tensors are stored as rows = time steps, columns = channels (the opposite
//   of PyTorch): a convolution becomes a matrix product (im2col) on the AVX2
//   kernel of mormot2-ai, spread over a thread pool
// - mormot.ai.tts.qwen.codec.vulkan moves the heavy back part to the GPU
unit mormot.ai.tts.qwen.codec;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.kernels,
  mormot.ai.infer.threads,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.tts.common;

const
  /// the code groups of one frame
  QWEN_CODE_GROUPS = 16;
  /// the values of one code: 0..2047
  QWEN_CODEBOOK_SIZE = 2048;
  /// the rate of the PCM output
  QWEN_SAMPLE_RATE = 24000;
  /// the samples of one frame: 2 x 2 (upsampling) x 8 x 5 x 4 x 3
  QWEN_FRAME_SAMPLES = 1920;

type
  EQwenCodec = class(ESynException);

  /// an intermediate output, for tests: aRows new rows (time steps) of aCols
  // columns (channels)
  TOnQwenProbe = procedure(const aName: RawUtf8; aData: PSingleArray;
    aRows, aCols: PtrInt) of object;

  /// a job split by blocks of rows: rows aFirst..aLast
  TOnQwenRows = procedure(aFirst, aLast: PtrInt) of object;

  TQwenCodec = class;

  /// causal 1D convolution, stride 1: weights [Cout][Cin][K] as in the file
  // (the cin * K + k order is the one of the im2col columns); keeps the last
  // (K-1) * dilation input rows
  TQwenConv = class
  protected
    fCodec: TQwenCodec;
    fIn, fOut, fK, fDil, fPast: PtrInt;
    fW, fB: PSingleArray;
    fHist: TSingleDynArray;
    // the running im2col
    fBuf, fCol: PSingleArray;
    fFirst: PtrInt;
    procedure Im2Col(aFirst, aLast: PtrInt);
  public
    constructor Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
      aIn, aOut, aK, aDil: PtrInt);
    procedure Reset;
    /// y[aRows][Cout] := conv(x[aRows][Cin]); y must not overlap x
    procedure Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
  end;

  /// causal transposed convolution (stride s, kernel K a multiple of s), cut
  // on the right by K - s as Qwen3TTSTokenizerV2CausalTransConvNet: row
  // t * s + j receives x[t - i] . W[:, :, j + i * s] for i < K / s
  TQwenTransConv = class
  protected
    fCodec: TQwenCodec;
    fIn, fOut, fStride, fTaps: PtrInt;
    // [s * Cout][taps * Cin] and [s * Cout], copied from [Cin][Cout][K]
    fW, fB: TSingleDynArray;
    fHist: TSingleDynArray;
  public
    constructor Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
      aIn, aOut, aK, aStride: PtrInt);
    procedure Reset;
    /// y[aRows * s][Cout] := convtranspose(x[aRows][Cin])
    procedure Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
  end;

  /// SnakeBeta: x + 1 / (exp(beta) + 1e-9) * sin(x * exp(alpha))^2, per channel
  TQwenSnake = class
  protected
    fCodec: TQwenCodec;
    fCount: PtrInt;
    fAlpha, fInvBeta: TSingleDynArray;
    fX: PSingleArray;
    procedure Apply(aFirst, aLast: PtrInt);
  public
    constructor Create(aCodec: TQwenCodec; const aPrefix: RawUtf8; aCount: PtrInt);
    /// x[aRows][Count] in place
    procedure Forward(x: PSingleArray; aRows: PtrInt);
  end;

  /// ConvNeXt block: depthwise convolution (7), LayerNorm, 4x GELU MLP,
  // gamma, residual
  TQwenConvNeXt = class
  protected
    fCodec: TQwenCodec;
    fDim: PtrInt;
    fDwW, fDwB, fNormW, fNormB, fPw1W, fPw1B, fPw2W, fPw2B, fGamma: PSingleArray;
    fHist: TSingleDynArray;
  public
    constructor Create(aCodec: TQwenCodec; const aPrefix: RawUtf8; aDim: PtrInt);
    procedure Reset;
    /// y[aRows][Dim] := x + block(x); y must not overlap x
    procedure Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
  end;

  /// residual unit: snake, dilated conv 7, snake, conv 1, + input
  TQwenResUnit = class
  protected
    fCodec: TQwenCodec;
    fDim: PtrInt;
    fAct1, fAct2: TQwenSnake;
    fConv1, fConv2: TQwenConv;
    fX, fH: PSingleArray;
    procedure AddInput(aFirst, aLast: PtrInt);
  public
    constructor Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
      aDim, aDilation: PtrInt);
    destructor Destroy; override;
    procedure Reset;
    /// x[aRows][Dim] in place
    procedure Forward(x: PSingleArray; aRows: PtrInt);
  end;

  /// decoder block: snake, transposed convolution (x rate), 3 residual units
  // (dilations 1, 3, 9)
  TQwenBlock = class
  protected
    fIn, fOut, fRate: PtrInt;
    fAct: TQwenSnake;
    fUp: TQwenTransConv;
    fRes: array[0..2] of TQwenResUnit;
  public
    constructor Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
      aIn, aOut, aRate: PtrInt);
    destructor Destroy; override;
    procedure Reset;
    /// y[aRows * rate][Out] := block(x[aRows][In]); x is modified
    procedure Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
  end;

  /// the weights of one transformer layer (8 layers, 512, 16 heads of 64)
  TQwenTfLayer = record
    InNorm, WQ, WK, WV, WO, PostNorm, WGate, WUp, WDown, AttnScale,
      MlpScale: PSingleArray;
  end;

  /// the 12 Hz decoder: codes -> 24 kHz PCM, frame by frame, on the CPU
  // - not to be shared between threads: one Decode at a time
  TQwenCodec = class
  protected
    fFile: TSafeTensorsFile;
    fPool: TInferThreadPool;
    fCpu: TInferCpuFastBackend;
    fThreads: integer;
    fProbe: TOnQwenProbe;
    fFrames: Int64;
    fScratch: array[0..23] of TSingleDynArray;
    // quantizer: 16 codebooks [2048][256], projections [512][256]
    fCodebook: TSingleDynArray;
    fFirstProj, fRestProj: PSingleArray;
    fPreConv: TQwenConv;
    // transformer
    fTfIn, fTfInB, fTfOut, fTfOutB, fTfNorm: PSingleArray;
    fTf: array[0..7] of TQwenTfLayer;
    fKeys, fValues: array[0..7] of TSingleDynArray;
    fCacheCap: PtrInt;
    fCacheBase: Int64;
    fInvFreq: array[0..31] of single;
    // upsampling, then the decoder
    fUp: array[0..1] of TQwenTransConv;
    fNext: array[0..1] of TQwenConvNeXt;
    fConv0: TQwenConv;
    fBlocks: array[0..3] of TQwenBlock;
    fActOut: TQwenSnake;
    fConvOut: TQwenConv;
    // the running matrix product
    fGx, fGw, fGb, fGy: PSingleArray;
    fGRows, fGIn, fGOut, fGTiles: PtrInt;
    fGFast: boolean;
    // the running job by rows
    fRowsJob: TOnQwenRows;
    fRowsCount, fRowsStep: PtrInt;
    procedure GemmItem(aItem: integer);
    procedure RowsItem(aItem: integer);
    procedure Emit(const aName: RawUtf8; aData: PSingleArray; aRows, aCols: PtrInt);
    procedure Quantize(aCodes: PIntegerArray; aFrames: PtrInt; y: PSingleArray);
    procedure CacheRoom(aRows: PtrInt);
    procedure Transformer(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
    /// the back part: the transformer output x[aFrames][1024] -> aFrames *
    // 1920 samples; on the CPU here, on the GPU in
    // mormot.ai.tts.qwen.codec.vulkan (the light front part stays on the CPU)
    procedure Back(x: PSingleArray; aFrames: PtrInt; aPcm: PSingleArray); virtual;
  public
    /// load aFolder/model.safetensors (the speech_tokenizer folder of the model)
    // - aThreads: the threads to compute on, the caller included; 0 = two
    // thirds of the cores, as mormot2-ai does
    constructor Create(const aFolder: TFileName; aThreads: integer = 0);
    destructor Destroy; override;
    /// forget the past: the next frame starts a new sentence
    procedure Reset; virtual;
    /// decode aFrames frames of 16 codes (aCodes[f * 16 + g]) and write
    // aFrames * 1920 samples into aPcm, between -1 and 1
    // - frames of successive calls follow each other: decoding 1 + 3 frames or
    // 4 at once gives the same samples
    procedure Decode(aCodes: PIntegerArray; aFrames: PtrInt; aPcm: PSingleArray);
    /// y[aRows][aOut] := x[aRows][aIn] . w[aOut][aIn]^T + bias, on the pool
    procedure Gemm(x: PSingleArray; aRows, aIn: PtrInt; w: PSingleArray;
      aOut: PtrInt; bias, y: PSingleArray);
    /// aJob over rows 0..aRows-1, by blocks of at least aMinRows, on the pool
    procedure ParallelRows(aRows, aMinRows: PtrInt; const aJob: TOnQwenRows);
    /// a work buffer of at least aCount singles; the pointer stays valid
    // until the next larger call with the same aIndex
    function Scratch(aIndex, aCount: PtrInt): PSingleArray;
    /// the F32 weights aName, of shape aDims
    function F32(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
    /// the frames decoded since Reset
    property Frames: Int64
      read fFrames;
    property Threads: integer
      read fThreads;
    /// called with each intermediate output (tests); nil by default
    property Probe: TOnQwenProbe
      read fProbe write fProbe;
  end;

/// the folder of a model in the Hugging Face cache (aHfHome/hub): the
// revision refs/main points to; '' if missing
function QwenHfSnapshot(const aHfHome: TFileName; const aRepo: RawUtf8): TFileName;


implementation

const
  // work buffers (TQwenCodec.Scratch)
  S_A = 0;      // the chain, ping-pong with S_B
  S_B = 1;
  S_C = 2;      // residual units
  S_D = 3;
  S_BUF = 4;    // history + input of a convolution
  S_COL = 5;    // im2col
  S_NX1 = 6;    // ConvNeXt
  S_NX2 = 7;
  S_H = 8;      // transformer
  S_N = 9;
  S_Q = 10;
  S_K = 11;
  S_V = 12;
  S_ATT = 13;
  S_O = 14;
  S_G = 15;
  S_U = 16;
  S_COS = 17;
  S_SIN = 18;
  S_Q1 = 19;    // quantizer
  S_Q2 = 20;
  S_T = 21;

  GEMM_TILE = 16;           // outputs per block (the weight tile stays in cache)
  GEMM_ROWS = 32;           // rows per block
  GEMM_PARALLEL = 1 shl 16; // below (rows x inputs x outputs): no pool
  CONV_CHUNK = 1024;        // im2col rows at once

  TF_DIM = 512;
  TF_HEADS = 16;
  TF_HEAD_DIM = 64;
  TF_QKV = TF_HEADS * TF_HEAD_DIM;
  TF_MLP = 1024;
  TF_WINDOW = 72;           // keys seen: the frame and the 71 before
  TF_EPS = 1e-5;
  LATENT = 1024;
  CODE_DIM = 256;
  QUANT_DIM = 512;
  DECODER_DIM = 1536;


{ TQwenConv }

constructor TQwenConv.Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
  aIn, aOut, aK, aDil: PtrInt);
begin
  inherited Create;
  fCodec := aCodec;
  fIn := aIn;
  fOut := aOut;
  fK := aK;
  fDil := aDil;
  fPast := (aK - 1) * aDil;
  fW := aCodec.F32(aPrefix + '.weight', [aOut, aIn, aK]);
  fB := aCodec.F32(aPrefix + '.bias', [aOut]);
  Reset;
end;

procedure TQwenConv.Reset;
begin
  fHist := nil;
  SetLength(fHist, fPast * fIn); // zeros: the causal padding
end;

procedure TQwenConv.Im2Col(aFirst, aLast: PtrInt);
var
  t, c, k, w: PtrInt;
  src, dst: PSingleArray;
begin
  w := fIn * fK;
  for t := aFirst to aLast do
  begin
    dst := @fCol^[t * w];
    for k := 0 to fK - 1 do
    begin
      // row t + k * dil of [history ; input]: time step t - (K-1-k) * dil
      src := @fBuf^[(fFirst + t + k * fDil) * fIn];
      for c := 0 to fIn - 1 do
        dst^[c * fK + k] := src^[c];
    end;
  end;
end;

procedure TQwenConv.Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
var
  s, n: PtrInt;
begin
  if fK = 1 then
  begin
    fCodec.Gemm(x, aRows, fIn, fW, fOut, fB, y);
    exit;
  end;
  fBuf := fCodec.Scratch(S_BUF, (fPast + aRows) * fIn);
  MoveFast(pointer(fHist)^, fBuf^[0], fPast * fIn * SizeOf(single));
  MoveFast(x^[0], fBuf^[fPast * fIn], aRows * fIn * SizeOf(single));
  s := 0;
  while s < aRows do
  begin
    n := aRows - s;
    if n > CONV_CHUNK then
      n := CONV_CHUNK;
    fCol := fCodec.Scratch(S_COL, n * fIn * fK);
    fFirst := s;
    fCodec.ParallelRows(n, 64, Im2Col);
    fCodec.Gemm(fCol, n, fIn * fK, fW, fOut, fB, @y^[s * fOut]);
    inc(s, n);
  end;
  MoveFast(fBuf^[aRows * fIn], pointer(fHist)^, fPast * fIn * SizeOf(single));
end;


{ TQwenTransConv }

constructor TQwenTransConv.Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
  aIn, aOut, aK, aStride: PtrInt);
var
  w, b: PSingleArray;
  j, i, o, c, cols: PtrInt;
begin
  inherited Create;
  if aK mod aStride <> 0 then
    EQwenCodec.RaiseUtf8('%: kernel % is not a multiple of stride %',
      [aPrefix, aK, aStride]);
  fCodec := aCodec;
  fIn := aIn;
  fOut := aOut;
  fStride := aStride;
  fTaps := aK div aStride;
  w := aCodec.F32(aPrefix + '.weight', [aIn, aOut, aK]);
  b := aCodec.F32(aPrefix + '.bias', [aOut]);
  cols := fTaps * aIn;
  SetLength(fW, aStride * aOut * cols);
  SetLength(fB, aStride * aOut);
  for j := 0 to aStride - 1 do
    for o := 0 to aOut - 1 do
    begin
      fB[j * aOut + o] := b^[o];
      for i := 0 to fTaps - 1 do
        for c := 0 to aIn - 1 do
          fW[(j * aOut + o) * cols + i * aIn + c] :=
            w^[(c * aOut + o) * aK + j + i * aStride];
    end;
  Reset;
end;

procedure TQwenTransConv.Reset;
begin
  fHist := nil;
  SetLength(fHist, (fTaps - 1) * fIn);
end;

procedure TQwenTransConv.Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
var
  buf, x2: PSingleArray;
  past, t, i: PtrInt;
begin
  if fTaps = 1 then
  begin
    fCodec.Gemm(x, aRows, fIn, pointer(fW), fStride * fOut, pointer(fB), y);
    exit;
  end;
  past := fTaps - 1;
  buf := fCodec.Scratch(S_BUF, (past + aRows) * fIn);
  MoveFast(pointer(fHist)^, buf^[0], past * fIn * SizeOf(single));
  MoveFast(x^[0], buf^[past * fIn], aRows * fIn * SizeOf(single));
  // row t: [x_t, x_t-1, ...]
  x2 := fCodec.Scratch(S_COL, aRows * fTaps * fIn);
  for t := 0 to aRows - 1 do
    for i := 0 to fTaps - 1 do
      MoveFast(buf^[(past + t - i) * fIn], x2^[(t * fTaps + i) * fIn],
        fIn * SizeOf(single));
  fCodec.Gemm(x2, aRows, fTaps * fIn, pointer(fW), fStride * fOut, pointer(fB), y);
  MoveFast(buf^[aRows * fIn], pointer(fHist)^, past * fIn * SizeOf(single));
end;


{ TQwenSnake }

constructor TQwenSnake.Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
  aCount: PtrInt);
var
  a, b: PSingleArray;
  c: PtrInt;
  e: single;
begin
  inherited Create;
  fCodec := aCodec;
  fCount := aCount;
  a := aCodec.F32(aPrefix + '.alpha', [aCount]);
  b := aCodec.F32(aPrefix + '.beta', [aCount]);
  SetLength(fAlpha, aCount);
  SetLength(fInvBeta, aCount);
  for c := 0 to aCount - 1 do
  begin
    // in float32 as PyTorch does: exp, then 1 / (beta + 1e-9)
    fAlpha[c] := Exp(a^[c]);
    e := Exp(b^[c]);
    fInvBeta[c] := 1 / (e + 1e-9);
  end;
end;

procedure TQwenSnake.Apply(aFirst, aLast: PtrInt);
var
  i, c: PtrInt;
  row: PSingleArray;
  v, s: single;
begin
  for i := aFirst to aLast do
  begin
    row := @fX^[i * fCount];
    for c := 0 to fCount - 1 do
    begin
      v := row^[c] * fAlpha[c];
      s := Sin(v);
      row^[c] := row^[c] + fInvBeta[c] * (s * s);
    end;
  end;
end;

procedure TQwenSnake.Forward(x: PSingleArray; aRows: PtrInt);
begin
  fX := x;
  fCodec.ParallelRows(aRows, 32, Apply);
end;


{ TQwenConvNeXt }

constructor TQwenConvNeXt.Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
  aDim: PtrInt);
begin
  inherited Create;
  fCodec := aCodec;
  fDim := aDim;
  fDwW := aCodec.F32(aPrefix + '.dwconv.conv.weight', [aDim, 1, 7]);
  fDwB := aCodec.F32(aPrefix + '.dwconv.conv.bias', [aDim]);
  fNormW := aCodec.F32(aPrefix + '.norm.weight', [aDim]);
  fNormB := aCodec.F32(aPrefix + '.norm.bias', [aDim]);
  fPw1W := aCodec.F32(aPrefix + '.pwconv1.weight', [4 * aDim, aDim]);
  fPw1B := aCodec.F32(aPrefix + '.pwconv1.bias', [4 * aDim]);
  fPw2W := aCodec.F32(aPrefix + '.pwconv2.weight', [aDim, 4 * aDim]);
  fPw2B := aCodec.F32(aPrefix + '.pwconv2.bias', [aDim]);
  fGamma := aCodec.F32(aPrefix + '.gamma', [aDim]);
  Reset;
end;

procedure TQwenConvNeXt.Reset;
begin
  fHist := nil;
  SetLength(fHist, 6 * fDim);
end;

procedure TQwenConvNeXt.Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
var
  buf, n, h: PSingleArray;
  t, c, k: PtrInt;
  acc: single;
begin
  buf := fCodec.Scratch(S_BUF, (6 + aRows) * fDim);
  MoveFast(pointer(fHist)^, buf^[0], 6 * fDim * SizeOf(single));
  MoveFast(x^[0], buf^[6 * fDim], aRows * fDim * SizeOf(single));
  // depthwise convolution, kernel 7, causal
  n := fCodec.Scratch(S_NX1, aRows * fDim);
  for t := 0 to aRows - 1 do
    for c := 0 to fDim - 1 do
    begin
      acc := fDwB^[c];
      for k := 0 to 6 do
        acc := acc + fDwW^[c * 7 + k] * buf^[(t + k) * fDim + c];
      n^[t * fDim + c] := acc;
    end;
  MoveFast(buf^[aRows * fDim], pointer(fHist)^, 6 * fDim * SizeOf(single));
  fCodec.fCpu.LayerNorm(n, aRows, fDim, fNormW, fNormB, 1e-6, n);
  h := fCodec.Scratch(S_NX2, aRows * 4 * fDim);
  fCodec.Gemm(n, aRows, fDim, fPw1W, 4 * fDim, fPw1B, h);
  fCodec.fCpu.Gelu(h, aRows * 4 * fDim);
  fCodec.Gemm(h, aRows, 4 * fDim, fPw2W, fDim, fPw2B, n);
  for t := 0 to aRows - 1 do
    for c := 0 to fDim - 1 do
      y^[t * fDim + c] := x^[t * fDim + c] + fGamma^[c] * n^[t * fDim + c];
end;


{ TQwenResUnit }

constructor TQwenResUnit.Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
  aDim, aDilation: PtrInt);
begin
  inherited Create;
  fCodec := aCodec;
  fDim := aDim;
  fAct1 := TQwenSnake.Create(aCodec, aPrefix + '.act1', aDim);
  fConv1 := TQwenConv.Create(aCodec, aPrefix + '.conv1.conv', aDim, aDim, 7, aDilation);
  fAct2 := TQwenSnake.Create(aCodec, aPrefix + '.act2', aDim);
  fConv2 := TQwenConv.Create(aCodec, aPrefix + '.conv2.conv', aDim, aDim, 1, 1);
end;

destructor TQwenResUnit.Destroy;
begin
  fAct1.Free;
  fConv1.Free;
  fAct2.Free;
  fConv2.Free;
  inherited Destroy;
end;

procedure TQwenResUnit.Reset;
begin
  fConv1.Reset;
  fConv2.Reset;
end;

procedure TQwenResUnit.AddInput(aFirst, aLast: PtrInt);
var
  i: PtrInt;
begin
  for i := aFirst * fDim to (aLast + 1) * fDim - 1 do
    fX^[i] := fH^[i] + fX^[i];
end;

procedure TQwenResUnit.Forward(x: PSingleArray; aRows: PtrInt);
var
  c, d: PSingleArray;
begin
  c := fCodec.Scratch(S_C, aRows * fDim);
  d := fCodec.Scratch(S_D, aRows * fDim);
  MoveFast(x^[0], c^[0], aRows * fDim * SizeOf(single));
  fAct1.Forward(c, aRows);
  fConv1.Forward(c, aRows, d);
  fAct2.Forward(d, aRows);
  fConv2.Forward(d, aRows, c);
  fX := x;
  fH := c;
  fCodec.ParallelRows(aRows, 64, AddInput);
end;


{ TQwenBlock }

constructor TQwenBlock.Create(aCodec: TQwenCodec; const aPrefix: RawUtf8;
  aIn, aOut, aRate: PtrInt);
const
  DILATIONS: array[0..2] of PtrInt = (1, 3, 9);
var
  i: PtrInt;
begin
  inherited Create;
  fIn := aIn;
  fOut := aOut;
  fRate := aRate;
  fAct := TQwenSnake.Create(aCodec, aPrefix + '.block.0', aIn);
  fUp := TQwenTransConv.Create(aCodec, aPrefix + '.block.1.conv', aIn, aOut,
    2 * aRate, aRate);
  for i := 0 to 2 do
    fRes[i] := TQwenResUnit.Create(aCodec,
      FormatUtf8('%.block.%', [aPrefix, i + 2]), aOut, DILATIONS[i]);
end;

destructor TQwenBlock.Destroy;
var
  i: PtrInt;
begin
  fAct.Free;
  fUp.Free;
  for i := 0 to 2 do
    fRes[i].Free;
  inherited Destroy;
end;

procedure TQwenBlock.Reset;
var
  i: PtrInt;
begin
  fUp.Reset;
  for i := 0 to 2 do
    fRes[i].Reset;
end;

procedure TQwenBlock.Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
var
  i: PtrInt;
begin
  fAct.Forward(x, aRows);
  fUp.Forward(x, aRows, y);
  for i := 0 to 2 do
    fRes[i].Forward(y, aRows * fRate);
end;


{ TQwenCodec }

constructor TQwenCodec.Create(const aFolder: TFileName; aThreads: integer);
const
  RATES: array[0..3] of PtrInt = (8, 5, 4, 3);
var
  g, l, i, c: PtrInt;
  sum, usage: PSingleArray;
  u: single;
  p: RawUtf8;
begin
  inherited Create;
  if aThreads <= 0 then
    aThreads := (SystemInfo.dwNumberOfProcessors * 2 + 2) div 3;
  if aThreads < 1 then
    aThreads := 1;
  fThreads := aThreads;
  if fThreads > 1 then
    fPool := TInferThreadPool.Create(fThreads - 1);
  // the small operations (norms, attention, GELU): on the calling thread, in
  // the arithmetic of the reference
  fCpu := TInferCpuFastBackend.Create(1);
  fCpu.Exact := true;
  fFile := TSafeTensorsFile.Create(IncludeTrailingPathDelimiter(aFolder) +
    'model.safetensors');
  // quantizer: codebook = embedding_sum / max(cluster_usage, 1e-5)
  SetLength(fCodebook, QWEN_CODE_GROUPS * QWEN_CODEBOOK_SIZE * CODE_DIM);
  for g := 0 to QWEN_CODE_GROUPS - 1 do
  begin
    if g = 0 then
      p := 'decoder.quantizer.rvq_first.vq.layers.0._codebook'
    else
      p := FormatUtf8('decoder.quantizer.rvq_rest.vq.layers.%._codebook', [g - 1]);
    sum := F32(p + '.embedding_sum', [QWEN_CODEBOOK_SIZE, CODE_DIM]);
    usage := F32(p + '.cluster_usage', [QWEN_CODEBOOK_SIZE]);
    for i := 0 to QWEN_CODEBOOK_SIZE - 1 do
    begin
      u := usage^[i];
      if u < 1e-5 then
        u := 1e-5;
      for c := 0 to CODE_DIM - 1 do
        fCodebook[(g * QWEN_CODEBOOK_SIZE + i) * CODE_DIM + c] :=
          sum^[i * CODE_DIM + c] / u;
    end;
  end;
  fFirstProj := F32('decoder.quantizer.rvq_first.output_proj.weight',
    [QUANT_DIM, CODE_DIM, 1]);
  fRestProj := F32('decoder.quantizer.rvq_rest.output_proj.weight',
    [QUANT_DIM, CODE_DIM, 1]);
  fPreConv := TQwenConv.Create(self, 'decoder.pre_conv.conv', QUANT_DIM, LATENT, 3, 1);
  // transformer
  p := 'decoder.pre_transformer.';
  fTfIn := F32(p + 'input_proj.weight', [TF_DIM, LATENT]);
  fTfInB := F32(p + 'input_proj.bias', [TF_DIM]);
  fTfOut := F32(p + 'output_proj.weight', [LATENT, TF_DIM]);
  fTfOutB := F32(p + 'output_proj.bias', [LATENT]);
  fTfNorm := F32(p + 'norm.weight', [TF_DIM]);
  for l := 0 to high(fTf) do
  begin
    p := FormatUtf8('decoder.pre_transformer.layers.%.', [l]);
    fTf[l].InNorm := F32(p + 'input_layernorm.weight', [TF_DIM]);
    fTf[l].WQ := F32(p + 'self_attn.q_proj.weight', [TF_QKV, TF_DIM]);
    fTf[l].WK := F32(p + 'self_attn.k_proj.weight', [TF_QKV, TF_DIM]);
    fTf[l].WV := F32(p + 'self_attn.v_proj.weight', [TF_QKV, TF_DIM]);
    fTf[l].WO := F32(p + 'self_attn.o_proj.weight', [TF_DIM, TF_QKV]);
    fTf[l].PostNorm := F32(p + 'post_attention_layernorm.weight', [TF_DIM]);
    fTf[l].WGate := F32(p + 'mlp.gate_proj.weight', [TF_MLP, TF_DIM]);
    fTf[l].WUp := F32(p + 'mlp.up_proj.weight', [TF_MLP, TF_DIM]);
    fTf[l].WDown := F32(p + 'mlp.down_proj.weight', [TF_DIM, TF_MLP]);
    fTf[l].AttnScale := F32(p + 'self_attn_layer_scale.scale', [TF_DIM]);
    fTf[l].MlpScale := F32(p + 'mlp_layer_scale.scale', [TF_DIM]);
  end;
  // RoPE, base 10000: inv_freq = 1 / 10000^(2i / 64), in float32
  for i := 0 to high(fInvFreq) do
    fInvFreq[i] := 1 / Exp(Ln(10000) * (2 * i / TF_HEAD_DIM));
  for i := 0 to 1 do
  begin
    fUp[i] := TQwenTransConv.Create(self,
      FormatUtf8('decoder.upsample.%.0.conv', [i]), LATENT, LATENT, 2, 2);
    fNext[i] := TQwenConvNeXt.Create(self,
      FormatUtf8('decoder.upsample.%.1', [i]), LATENT);
  end;
  fConv0 := TQwenConv.Create(self, 'decoder.decoder.0.conv', LATENT, DECODER_DIM, 7, 1);
  for i := 0 to 3 do
    fBlocks[i] := TQwenBlock.Create(self, FormatUtf8('decoder.decoder.%', [i + 1]),
      DECODER_DIM shr i, DECODER_DIM shr (i + 1), RATES[i]);
  fActOut := TQwenSnake.Create(self, 'decoder.decoder.5', DECODER_DIM shr 4);
  fConvOut := TQwenConv.Create(self, 'decoder.decoder.6.conv', DECODER_DIM shr 4, 1, 7, 1);
  Reset;
end;

destructor TQwenCodec.Destroy;
var
  i: PtrInt;
begin
  fPreConv.Free;
  for i := 0 to 1 do
  begin
    fUp[i].Free;
    fNext[i].Free;
  end;
  fConv0.Free;
  for i := 0 to 3 do
    fBlocks[i].Free;
  fActOut.Free;
  fConvOut.Free;
  fFile.Free;
  fCpu.Free;
  fPool.Free;
  inherited Destroy;
end;

function TQwenCodec.F32(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
begin
  result := fFile.F32(aName, aDims);
end;

function TQwenCodec.Scratch(aIndex, aCount: PtrInt): PSingleArray;
begin
  if length(fScratch[aIndex]) < aCount then
  begin
    fScratch[aIndex] := nil; // no copy of the old content
    SetLength(fScratch[aIndex], aCount);
  end;
  result := pointer(fScratch[aIndex]);
end;

procedure TQwenCodec.Emit(const aName: RawUtf8; aData: PSingleArray;
  aRows, aCols: PtrInt);
begin
  if Assigned(fProbe) then
    fProbe(aName, aData, aRows, aCols);
end;

procedure TQwenCodec.Reset;
var
  i: PtrInt;
begin
  fFrames := 0;
  fCacheBase := 0;
  fPreConv.Reset;
  for i := 0 to 1 do
  begin
    fUp[i].Reset;
    fNext[i].Reset;
  end;
  fConv0.Reset;
  for i := 0 to 3 do
    fBlocks[i].Reset;
  fConvOut.Reset;
end;

procedure TQwenCodec.GemmItem(aItem: integer);
var
  b, t, r, r1, o0, n, o, i: PtrInt;
  xr, yr, wr: PSingleArray;
  acc: double;
begin
  b := aItem div fGTiles;
  t := aItem - b * fGTiles;
  o0 := t * GEMM_TILE;
  n := fGOut - o0;
  if n > GEMM_TILE then
    n := GEMM_TILE;
  r1 := (b + 1) * GEMM_ROWS;
  if r1 > fGRows then
    r1 := fGRows;
  for r := b * GEMM_ROWS to r1 - 1 do
  begin
    xr := @fGx^[r * fGIn];
    yr := @fGy^[r * fGOut + o0];
    if fGFast then
      InferDotsF32(@xr^[0], @fGw^[o0 * fGIn], fGIn * SizeOf(single), n, fGIn, yr)
    else
      for o := 0 to n - 1 do
      begin
        wr := @fGw^[(o0 + o) * fGIn];
        acc := 0;
        for i := 0 to fGIn - 1 do
          acc := acc + xr^[i] * wr^[i];
        yr^[o] := acc;
      end;
    if fGb <> nil then
      for o := 0 to n - 1 do
        yr^[o] := yr^[o] + fGb^[o0 + o];
  end;
end;

procedure TQwenCodec.Gemm(x: PSingleArray; aRows, aIn: PtrInt; w: PSingleArray;
  aOut: PtrInt; bias, y: PSingleArray);
var
  items, i: PtrInt;
begin
  if aRows <= 0 then
    exit;
  fGx := x;
  fGw := w;
  fGb := bias;
  fGy := y;
  fGRows := aRows;
  fGIn := aIn;
  fGOut := aOut;
  fGTiles := (aOut + GEMM_TILE - 1) div GEMM_TILE;
  // the float kernel wants rows that are multiples of 16
  fGFast := Assigned(InferDotsF32) and (aIn and 15 = 0);
  items := ((aRows + GEMM_ROWS - 1) div GEMM_ROWS) * fGTiles;
  if (fPool = nil) or (Int64(aRows) * aIn * aOut < GEMM_PARALLEL) then
    for i := 0 to items - 1 do
      GemmItem(i)
  else
    fPool.Run(GemmItem, items);
end;

procedure TQwenCodec.RowsItem(aItem: integer);
var
  last: PtrInt;
begin
  last := (aItem + 1) * fRowsStep - 1;
  if last >= fRowsCount then
    last := fRowsCount - 1;
  fRowsJob(aItem * fRowsStep, last);
end;

procedure TQwenCodec.ParallelRows(aRows, aMinRows: PtrInt; const aJob: TOnQwenRows);
var
  items: PtrInt;
begin
  if aRows <= 0 then
    exit;
  if (fPool = nil) or (aRows < 2 * aMinRows) then
  begin
    aJob(0, aRows - 1);
    exit;
  end;
  // four blocks per thread: slow threads do not delay the others
  fRowsStep := (aRows + fThreads * 4 - 1) div (fThreads * 4);
  if fRowsStep < aMinRows then
    fRowsStep := aMinRows;
  items := (aRows + fRowsStep - 1) div fRowsStep;
  fRowsJob := aJob;
  fRowsCount := aRows;
  fPool.Run(RowsItem, items);
end;

procedure TQwenCodec.Quantize(aCodes: PIntegerArray; aFrames: PtrInt; y: PSingleArray);
var
  q1, q2, t: PSingleArray;
  f, g, c, code: PtrInt;
  e: PSingleArray;
begin
  q1 := Scratch(S_Q1, aFrames * CODE_DIM);
  q2 := Scratch(S_Q2, aFrames * CODE_DIM);
  for f := 0 to aFrames - 1 do
    for g := 0 to QWEN_CODE_GROUPS - 1 do
    begin
      code := aCodes^[f * QWEN_CODE_GROUPS + g];
      if code < 0 then
        code := 0 // as torch.clamp(min=0) before decoding
      else if code >= QWEN_CODEBOOK_SIZE then
        EQwenCodec.RaiseUtf8('%.Decode: code % out of range in group %',
          [self, code, g]);
      e := @fCodebook[(g * QWEN_CODEBOOK_SIZE + code) * CODE_DIM];
      if g = 0 then
        MoveFast(e^[0], q1^[f * CODE_DIM], CODE_DIM * SizeOf(single))
      else if g = 1 then
        MoveFast(e^[0], q2^[f * CODE_DIM], CODE_DIM * SizeOf(single))
      else
        for c := 0 to CODE_DIM - 1 do
          q2^[f * CODE_DIM + c] := q2^[f * CODE_DIM + c] + e^[c];
    end;
  t := Scratch(S_T, aFrames * QUANT_DIM);
  Gemm(q1, aFrames, CODE_DIM, fFirstProj, QUANT_DIM, nil, y);
  Gemm(q2, aFrames, CODE_DIM, fRestProj, QUANT_DIM, nil, t);
  for c := 0 to aFrames * QUANT_DIM - 1 do
    y^[c] := y^[c] + t^[c];
end;

procedure TQwenCodec.CacheRoom(aRows: PtrInt);
var
  keep, from, l: PtrInt;
begin
  // the cache rows: positions fCacheBase .. fFrames - 1, then the new ones
  if fFrames - fCacheBase + aRows <= fCacheCap then
    exit;
  keep := fFrames - fCacheBase;
  if keep > TF_WINDOW - 1 then
    keep := TF_WINDOW - 1;
  from := fFrames - fCacheBase - keep;
  if keep + aRows > fCacheCap then
  begin
    fCacheCap := keep + aRows + 4 * TF_WINDOW;
    for l := 0 to high(fKeys) do
    begin
      SetLength(fKeys[l], fCacheCap * TF_QKV);
      SetLength(fValues[l], fCacheCap * TF_QKV);
    end;
  end;
  if (keep > 0) and (from > 0) then
    for l := 0 to high(fKeys) do
    begin
      MoveFast(fKeys[l][from * TF_QKV], fKeys[l][0], keep * TF_QKV * SizeOf(single));
      MoveFast(fValues[l][from * TF_QKV], fValues[l][0], keep * TF_QKV * SizeOf(single));
    end;
  fCacheBase := fFrames - keep;
end;

procedure TQwenCodec.Transformer(x: PSingleArray; aRows: PtrInt; y: PSingleArray);
var
  h, n, q, k, v, att, o, g, u, cs, sn: PSingleArray;
  lw: ^TQwenTfLayer;
  r, i, l, row0: PtrInt;
  p, first: Int64;
  f: single;
begin
  h := Scratch(S_H, aRows * TF_DIM);
  n := Scratch(S_N, aRows * TF_DIM);
  q := Scratch(S_Q, aRows * TF_QKV);
  k := Scratch(S_K, aRows * TF_QKV);
  v := Scratch(S_V, aRows * TF_QKV);
  att := Scratch(S_ATT, aRows * TF_QKV);
  o := Scratch(S_O, aRows * TF_DIM);
  g := Scratch(S_G, aRows * TF_MLP);
  u := Scratch(S_U, aRows * TF_MLP);
  cs := Scratch(S_COS, aRows * (TF_HEAD_DIM shr 1));
  sn := Scratch(S_SIN, aRows * (TF_HEAD_DIM shr 1));
  Gemm(x, aRows, LATENT, fTfIn, TF_DIM, fTfInB, h);
  Emit('dec.tf.in', h, aRows, TF_DIM);
  for r := 0 to aRows - 1 do
    for i := 0 to (TF_HEAD_DIM shr 1) - 1 do
    begin
      f := (fFrames + r) * fInvFreq[i]; // float32 product, as PyTorch
      cs^[r * (TF_HEAD_DIM shr 1) + i] := Cos(f);
      sn^[r * (TF_HEAD_DIM shr 1) + i] := Sin(f);
    end;
  CacheRoom(aRows);
  row0 := fFrames - fCacheBase;
  for l := 0 to high(fTf) do
  begin
    lw := @fTf[l];
    fCpu.RmsNorm(h, aRows, TF_DIM, lw^.InNorm, TF_EPS, n);
    Gemm(n, aRows, TF_DIM, lw^.WQ, TF_QKV, nil, q);
    Gemm(n, aRows, TF_DIM, lw^.WK, TF_QKV, nil, k);
    Gemm(n, aRows, TF_DIM, lw^.WV, TF_QKV, nil, v);
    fCpu.Rope(q, aRows, TF_HEADS, TF_HEAD_DIM, cs, sn);
    fCpu.Rope(k, aRows, TF_HEADS, TF_HEAD_DIM, cs, sn);
    MoveFast(k^[0], fKeys[l][row0 * TF_QKV], aRows * TF_QKV * SizeOf(single));
    MoveFast(v^[0], fValues[l][row0 * TF_QKV], aRows * TF_QKV * SizeOf(single));
    for r := 0 to aRows - 1 do
    begin
      // sliding window: frame p sees p - 71 .. p
      p := fFrames + r;
      first := p - (TF_WINDOW - 1);
      if first < 0 then
        first := 0;
      fCpu.CausalAttention(@q^[r * TF_QKV], 1, TF_HEADS, TF_HEAD_DIM,
        @fKeys[l][(first - fCacheBase) * TF_QKV],
        @fValues[l][(first - fCacheBase) * TF_QKV], p - first, TF_HEADS,
        @att^[r * TF_QKV]);
    end;
    Gemm(att, aRows, TF_QKV, lw^.WO, TF_DIM, nil, o);
    for i := 0 to aRows * TF_DIM - 1 do
      h^[i] := h^[i] + lw^.AttnScale^[i mod TF_DIM] * o^[i];
    fCpu.RmsNorm(h, aRows, TF_DIM, lw^.PostNorm, TF_EPS, n);
    Gemm(n, aRows, TF_DIM, lw^.WGate, TF_MLP, nil, g);
    Gemm(n, aRows, TF_DIM, lw^.WUp, TF_MLP, nil, u);
    fCpu.SwiGlu(g, u, aRows * TF_MLP);
    Gemm(g, aRows, TF_MLP, lw^.WDown, TF_DIM, nil, o);
    for i := 0 to aRows * TF_DIM - 1 do
      h^[i] := h^[i] + lw^.MlpScale^[i mod TF_DIM] * o^[i];
    if Assigned(fProbe) then
      Emit(FormatUtf8('dec.tf.layer%', [l]), h, aRows, TF_DIM);
  end;
  fCpu.RmsNorm(h, aRows, TF_DIM, fTfNorm, TF_EPS, n);
  Gemm(n, aRows, TF_DIM, fTfOut, LATENT, fTfOutB, y);
  Emit('dec.tf.out', y, aRows, LATENT);
end;

procedure TQwenCodec.Decode(aCodes: PIntegerArray; aFrames: PtrInt; aPcm: PSingleArray);
var
  a, b: PSingleArray;
  size: PtrInt;
begin
  if aFrames <= 0 then
    exit;
  // the largest step: the output of the last block, 1920 rows of 96 per
  // frame; Back reuses S_A and S_B without growing them
  size := aFrames * QWEN_FRAME_SAMPLES * (DECODER_DIM shr 4);
  a := Scratch(S_A, size);
  b := Scratch(S_B, size);
  Quantize(aCodes, aFrames, a);
  Emit('dec.quant', a, aFrames, QUANT_DIM);
  fPreConv.Forward(a, aFrames, b);
  Emit('dec.preconv', b, aFrames, LATENT);
  Transformer(b, aFrames, a);
  Back(a, aFrames, aPcm);
  inc(fFrames, aFrames);
end;

procedure TQwenCodec.Back(x: PSingleArray; aFrames: PtrInt; aPcm: PSingleArray);
var
  a, b, t: PSingleArray;
  rows, size, i, dim: PtrInt;
  v: single;
begin
  size := aFrames * QWEN_FRAME_SAMPLES * (DECODER_DIM shr 4);
  a := x;
  b := Scratch(S_B, size);
  rows := aFrames;
  for i := 0 to 1 do
  begin
    fUp[i].Forward(a, rows, b);
    rows := rows * 2;
    fNext[i].Forward(b, rows, a);
    if Assigned(fProbe) then
      Emit(FormatUtf8('dec.up%', [i]), a, rows, LATENT);
  end;
  fConv0.Forward(a, rows, b);
  Emit('dec.dec0', b, rows, DECODER_DIM);
  for i := 0 to 3 do
  begin
    fBlocks[i].Forward(b, rows, a);
    rows := rows * fBlocks[i].fRate;
    dim := fBlocks[i].fOut;
    if Assigned(fProbe) then
      Emit(FormatUtf8('dec.dec%', [i + 1]), a, rows, dim);
    t := a;
    a := b;
    b := t;
  end;
  fActOut.Forward(b, rows);
  fConvOut.Forward(b, rows, a);
  Emit('dec.dec6', a, rows, 1);
  for i := 0 to rows - 1 do
  begin
    v := a^[i];
    if v > 1 then
      v := 1
    else if v < -1 then
      v := -1;
    aPcm^[i] := v;
  end;
end;


function QwenHfSnapshot(const aHfHome: TFileName; const aRepo: RawUtf8): TFileName;
begin
  result := TtsHfSnapshot(aHfHome, aRepo);
end;

end.
