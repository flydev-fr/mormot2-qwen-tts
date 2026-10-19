/// Text-to-speech - the building blocks of the Kokoro-82M network on the CPU
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - matrix product spread over a thread pool (the micro-kernel of
// mormot.ai.tts.kokoro.kernels), 1D convolutions (read in place, or im2col
// for a stride > 1), transposed convolutions, bidirectional LSTM,
// normalizations (layer, instance, AdaIN), activations
// - layout: rows = time steps, columns = channels, the reverse of PyTorch;
// a convolution becomes a matrix product
// - the convolution weights are "weight_norm" in the file (g, v):
// w = v * (g / |v|), computed once at load time, then packed in panels of
// 16 outputs for the kernel
unit mormot.ai.tts.kokoro.nn;

{$I mormot.defines.inc}

{$ifndef FPC}
  {$EXCESSPRECISION OFF} // single in single, as PyTorch's float32
{$endif FPC}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.ai.infer.kernels,
  mormot.ai.infer.threads,
  mormot.ai.tts.torchfile,
  mormot.ai.tts.kokoro.kernels;

type
  EKokoroNet = class(ESynException);

  /// a job by blocks of rows: rows aFirst..aLast
  TOnKokoroRows = procedure(aFirst, aLast: PtrInt) of object;
  /// a job by fixed blocks: the block aBlock, rows aFirst..aLast
  TOnKokoroBlock = procedure(aBlock, aFirst, aLast: PtrInt) of object;

  /// the thread pool and the matrix product
  TKokoroCpu = class
  protected
    fPool: TInferThreadPool;
    fThreads: integer;
    // the running product: x and y as addresses, the strides in bytes
    fGm: PKPanels;
    fGx, fGy, fGXStride, fGTapStride, fGYStride, fGFlags: PtrInt;
    fGRows, fGRowStep, fGRowItems, fGPanelStep: PtrInt;
    fGPairs: boolean;
    fScratch: array[0..2] of TSingleDynArray;
    fGemmMicroSec, fGemmMac: Int64;
    fCx, fCy: PSingleArray;
    fCCols, fCXStride, fCYStride: PtrInt;
    fRowsJob: TOnKokoroRows;
    fRowsCount, fRowsStep: PtrInt;
    fBlockJob: TOnKokoroBlock;
    fBlockRows, fBlockSize: PtrInt;
    fEx, fEy, fEGamma, fEBeta: PSingleArray;
    fESlope: single;
    fECount, fECols, fEStride: PtrInt;
    procedure GemmItem(aItem: integer);
    procedure RowsItem(aItem: integer);
    procedure BlockItem(aItem: integer);
    procedure AddRows(aFirst, aLast: PtrInt);
    procedure LeakyRows(aFirst, aLast: PtrInt);
    procedure DivideRows(aFirst, aLast: PtrInt);
    procedure AddScaleRows(aFirst, aLast: PtrInt);
    procedure LayerNormRows(aFirst, aLast: PtrInt);
    procedure CopyRowsJob(aFirst, aLast: PtrInt);
    procedure DoGemm(x: pointer; aRows, aXStride, aTapStride: PtrInt;
      const m: TKPanels; y: PSingleArray; aYStride: PtrInt; aFlags: PtrInt);
  public
    /// aThreads: the compute threads, the caller included; 0 = two thirds of
    // the cores
    constructor Create(aThreads: integer = 0);
    destructor Destroy; override;
    /// y[r * aYStride + o] := bias[o] + the sum over k and c of
    // x[r * aXStride + k * aTapStride + c] * w[o][c][k]; the strides in singles
    // - aFlags = KGEMM_ADD: y := y + the product
    procedure Gemm(x: pointer; aRows, aXStride, aTapStride: PtrInt;
      const m: TKPanels; y: PSingleArray; aYStride: PtrInt; aFlags: PtrInt = 0);
    /// a shared work buffer of at least aCount singles (0 .. 2), kept from
    // one call to the next; its content is lost if it grows
    function Scratch(aIndex, aCount: PtrInt): PSingleArray;
    /// aRows rows of aCols singles from x to y, spread
    procedure CopyRows(x: PSingleArray; aXStride: PtrInt; y: PSingleArray;
      aYStride, aRows, aCols: PtrInt);
    /// aJob over the rows 0..aRows-1, by blocks of at least aMinRows
    procedure ParallelRows(aRows, aMinRows: PtrInt; const aJob: TOnKokoroRows);
    /// aJob over blocks of aBlock rows, always the same whatever the number of
    // threads (reproducible partial sums); returns their
    // count
    function ParallelBlocks(aRows, aBlock: PtrInt; const aJob: TOnKokoroBlock): PtrInt;
    /// y[i] := x[i] + y[i] over aCount values, spread
    procedure Add(x, y: PSingleArray; aCount: PtrInt);
    /// LeakyReLU in place, spread
    procedure Leaky(x: PSingleArray; aCount: PtrInt; aSlope: single);
    /// x[i] := x[i] / aDivisor over aCount values, spread
    procedure Divide(x: PSingleArray; aCount: PtrInt; aDivisor: single);
    /// y[i] := (y[i] + x[i]) * aScale over aCount values, spread
    procedure AddScale(x, y: PSingleArray; aCount: PtrInt; aScale: single);
    /// KokoroLayerNorm spread by rows
    procedure LayerNorm(x: PSingleArray; aRows, aCols, aStride: PtrInt;
      gamma, beta: PSingleArray; aEps: single);
    property Threads: integer
      read fThreads;
    /// the time spent in Gemm and its multiply-adds, accumulated
    // (the model's profile resets them)
    property GemmMicroSec: Int64
      read fGemmMicroSec write fGemmMicroSec;
    property GemmMac: Int64
      read fGemmMac write fGemmMac;
  end;

  /// Linear: y = x W^T + b
  TKLinear = class
  protected
    fCpu: TKokoroCpu;
    fM: TKPanels;
  public
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aIn, aOut: PtrInt; aBias: boolean = true);
    /// y[aRows][Out] := x[aRows][In] W^T + b; x has aXStride singles per row
    procedure Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray;
      aXStride: PtrInt = 0);
    property InCount: PtrInt
      read fM.Cin;
    property OutCount: PtrInt
      read fM.OutCount;
  end;

  /// Conv1d (groups = 1): stride, dilation, symmetric padding
  // - stride 1: read in place by the kernel, tap by tap; stride > 1: im2col
  TKConv = class
  protected
    fCpu: TKokoroCpu;
    fM: TKPanels;
    fIn, fOut, fK, fStride, fDil, fPad: PtrInt;
    fX, fCol: PSingleArray;
    fRows, fFirst, fLen: PtrInt;
    fColBuf: TSingleDynArray;
    procedure Im2Col(aFirst, aLast: PtrInt);
  public
    /// aWeightNorm: weight_g / weight_v (otherwise weight); aBias: bias
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aIn, aOut, aK: PtrInt; aStride: PtrInt = 1; aDil: PtrInt = 1;
      aPad: PtrInt = -1; aWeightNorm: boolean = true; aBias: boolean = true);
    function OutLen(aLen: PtrInt): PtrInt;
    /// y[OutLen(aLen)][Out] := conv(x[aLen][In])
    procedure Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
    /// the same, stride 1, over an x already bordered: Pad rows of zeros
    // before x^[0] and after its last row; aFlags = KGEMM_ADD: y += conv
    procedure ForwardPadded(x: PSingleArray; aLen: PtrInt; y: PSingleArray;
      aFlags: PtrInt = 0);
    property InCount: PtrInt
      read fIn;
    property OutCount: PtrInt
      read fOut;
    /// the rows of zeros on each side
    property Pad: PtrInt
      read fPad;
  end;

  /// ConvTranspose1d: groups = 1, or per channel (groups = In = Out)
  TKConvT = class
  protected
    fCpu: TKokoroCpu;
    fIn, fOut, fK, fStride, fPad, fOutPad: PtrInt;
    fDepthwise: boolean;
    fM: TKPanels;          // [(k, o)][In] for the product, then scatter
    fW, fB: TSingleDynArray;  // per channel: [C][K], [C]
    // the running scatter: the product [aLen][K][Out] -> y
    fJFull, fJy: PSingleArray;
    fJLen: PtrInt;
    procedure GatherRows(aFirst, aLast: PtrInt);
  public
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aIn, aOut, aK, aStride, aPad: PtrInt; aOutPad: PtrInt = 0;
      aDepthwise: boolean = false);
    function OutLen(aLen: PtrInt): PtrInt;
    procedure Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
  end;

  /// one-layer bidirectional LSTM (PyTorch's gates i, f, g, o)
  TKLstm = class
  protected
    fCpu: TKokoroCpu;
    fIn, fH: PtrInt;
    fIh: TKPanels;                 // [8H][In]: forward then backward, b_ih
    fHh: array[0..1] of TSingleDynArray;   // [4H][H]
    fBhh: array[0..1] of TSingleDynArray;  // [4H]
    fG, fY: PSingleArray;
    fLen: PtrInt;
    procedure Direction(aFirst, aLast: PtrInt);
  public
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aIn, aHidden: PtrInt);
    /// y[aLen][2H] := [forward | backward] of x[aLen][In]
    procedure Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
    property Hidden: PtrInt
      read fH;
  end;

  /// AdaIN1d: InstanceNorm1d (over time, per channel) then
  // (1 + gamma) * x + beta, gamma and beta drawn from the style by fc
  TKAdaIN = class
  protected
    fCpu: TKokoroCpu;
    fFc: TKLinear;
    fC: PtrInt;
    fGamma, fBeta: TSingleDynArray;
    // the running job: sums by blocks of rows, then application
    fJx, fJy: PSingleArray;
    fPart, fMean: array of double;
    fScale, fShift: TSingleDynArray;
    procedure SumBlock(aBlock, aFirst, aLast: PtrInt);
    procedure VarBlock(aBlock, aFirst, aLast: PtrInt);
    procedure ApplyRows(aFirst, aLast: PtrInt);
  public
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aStyle, aChannels: PtrInt);
    destructor Destroy; override;
    /// gamma and beta for the style s[Style]
    procedure Prepare(s: PSingleArray);
    /// the mean and the variance of each channel of x[aLen][C]: Scale and
    // Shift, such that AdaIN(x) = x * Scale + Shift
    procedure Stats(x: PSingleArray; aLen: PtrInt);
    /// y := AdaIN(x[aLen][C]); y may be x
    procedure Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
    property Scale: TSingleDynArray
      read fScale;
    property Shift: TSingleDynArray
      read fShift;
  end;

  /// AdaLayerNorm: LayerNorm over the channels, without affine, then
  // (1 + gamma) * x + beta
  TKAdaLayerNorm = class
  protected
    fFc: TKLinear;
    fC: PtrInt;
    fGamma, fBeta: TSingleDynArray;
  public
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aStyle, aChannels: PtrInt);
    destructor Destroy; override;
    procedure Prepare(s: PSingleArray);
    /// x[aLen][C] in place, rows of aStride singles
    procedure Forward(x: PSingleArray; aLen, aStride: PtrInt);
  end;

  /// StyleTTS2's AdainResBlk1d: AdaIN, LeakyReLU 0.2, (per-channel
  // transposed convolution x2), conv 3, AdaIN, LeakyReLU, conv 3; plus the
  // shortcut (repeat x2, 1x1 conv without bias); all of it / sqrt(2)
  TKAdainResBlk = class
  protected
    fCpu: TKokoroCpu;
    fIn, fOut: PtrInt;
    fUpsample: boolean;
    fNorm1, fNorm2: TKAdaIN;
    fConv1, fConv2: TKConv;
    fPool: TKConvT;
    fShort: TKConv;
  public
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aIn, aOut, aStyle: PtrInt; aUpsample: boolean = false);
    destructor Destroy; override;
    procedure Prepare(s: PSingleArray);
    function OutLen(aLen: PtrInt): PtrInt;
    /// y[OutLen][Out] := block(x[aLen][In])
    procedure Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
    property InCount: PtrInt
      read fIn;
    property OutCount: PtrInt
      read fOut;
  end;

  /// AdaINResBlock1 of the vocoder: three times AdaIN, Snake, dilated conv,
  // AdaIN, Snake, conv, + the input
  TKAdaINResBlock1 = class
  protected
    fCpu: TKokoroCpu;
    fC: PtrInt;
    fAdain1, fAdain2: array[0..2] of TKAdaIN;
    fConv1, fConv2: array[0..2] of TKConv;
    fAlpha1, fAlpha2, fInv1, fInv2: array[0..2] of TSingleDynArray;
    // running AdaIN and Snake: x -> y
    fSx, fSy, fSScale, fSShift, fSa, fSinv: PSingleArray;
    procedure SnakeRows(aFirst, aLast: PtrInt);
    procedure AdaSnake(aNorm: TKAdaIN; const aAlpha, aInv: TSingleDynArray;
      x: PSingleArray; aLen: PtrInt; y: PSingleArray);
  public
    constructor Create(aCpu: TKokoroCpu; aFile: TTorchFile; const aPrefix: RawUtf8;
      aChannels, aK: PtrInt; const aDilations: array of PtrInt; aStyle: PtrInt);
    destructor Destroy; override;
    procedure Prepare(s: PSingleArray);
    /// y[aLen][C] := block(x[aLen][C]); y must not be x
    procedure Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
  end;

/// the weights of a "weight_norm" convolution: v * (g / |v|), |v| per
// output over all the other dimensions; aRows outputs of aCols weights
function KokoroWeightNorm(aFile: TTorchFile; const aPrefix: RawUtf8;
  aRows, aCols: PtrInt): TSingleDynArray;

/// LeakyReLU in place
procedure KokoroLeaky(x: PSingleArray; aCount: PtrInt; aSlope: single);

/// LayerNorm of each row x[aRows][aCols] (in place), with gamma and beta
// (nil: without), eps
procedure KokoroLayerNorm(x: PSingleArray; aRows, aCols, aStride: PtrInt;
  gamma, beta: PSingleArray; aEps: single);

/// an array of at least aCount singles (the content is lost if it grows)
procedure KokoroGrow(var a: TSingleDynArray; aCount: PtrInt);


implementation

const
  /// the multiply-adds under which the product stays on one thread
  GEMM_PARALLEL = 1 shl 18;
  /// the blocks of 6 rows of a product job, at most and at least
  GEMM_TILES_MAX = 16;
  GEMM_TILES_MIN = 4;
  /// the rows of one im2col pass
  CONV_CHUNK = 1024;

procedure KokoroGrow(var a: TSingleDynArray; aCount: PtrInt);
begin
  if length(a) < aCount then
  begin
    a := nil;
    SetLength(a, aCount);
  end;
end;

function KokoroWeightNorm(aFile: TTorchFile; const aPrefix: RawUtf8;
  aRows, aCols: PtrInt): TSingleDynArray;
var
  g, v: PSingleArray;
  t: PTorchTensor;
  r, c: PtrInt;
  norm: double;
  scale: single;
begin
  t := aFile.Tensor(aPrefix + '.weight_v');
  if (t^.DType <> tdF32) or (t^.Count <> Int64(aRows) * aCols) then
    raise EKokoroNet.CreateUtf8('%.weight_v is %, expected % x %',
      [aPrefix, TorchShapeText(t^.Shape), aRows, aCols]);
  v := t^.Data;
  t := aFile.Tensor(aPrefix + '.weight_g');
  if (t^.DType <> tdF32) or (t^.Count <> aRows) then
    raise EKokoroNet.CreateUtf8('%.weight_g is %', [aPrefix, TorchShapeText(t^.Shape)]);
  g := t^.Data;
  SetLength(result, aRows * aCols);
  for r := 0 to aRows - 1 do
  begin
    norm := 0;
    for c := 0 to aCols - 1 do
      norm := norm + sqr(double(v^[r * aCols + c]));
    // torch._weight_norm: v * (g / norm), in float32
    scale := g^[r] / single(Sqrt(norm));
    for c := 0 to aCols - 1 do
      result[r * aCols + c] := v^[r * aCols + c] * scale;
  end;
end;

procedure KokoroLeaky(x: PSingleArray; aCount: PtrInt; aSlope: single);
var
  i: PtrInt;
begin
  for i := 0 to aCount - 1 do
    if x^[i] < 0 then
      x^[i] := x^[i] * aSlope;
end;

procedure KokoroLayerNorm(x: PSingleArray; aRows, aCols, aStride: PtrInt;
  gamma, beta: PSingleArray; aEps: single);
var
  r, c: PtrInt;
  row: PSingleArray;
  mean, v, d: double;
  inv: single;
begin
  for r := 0 to aRows - 1 do
  begin
    row := @x^[r * aStride];
    mean := 0;
    for c := 0 to aCols - 1 do
      mean := mean + row^[c];
    mean := mean / aCols;
    v := 0;
    for c := 0 to aCols - 1 do
    begin
      d := row^[c] - mean;
      v := v + d * d;
    end;
    inv := 1 / Sqrt(v / aCols + aEps);
    for c := 0 to aCols - 1 do
    begin
      row^[c] := (row^[c] - single(mean)) * inv;
      if gamma <> nil then
        row^[c] := row^[c] * gamma^[c] + beta^[c];
    end;
  end;
end;


{ TKokoroCpu }

constructor TKokoroCpu.Create(aThreads: integer);
begin
  inherited Create;
  if aThreads <= 0 then
    aThreads := (SystemInfo.dwNumberOfProcessors * 2 + 2) div 3;
  if aThreads < 1 then
    aThreads := 1;
  fThreads := aThreads;
  if fThreads > 1 then
    fPool := TInferThreadPool.Create(fThreads - 1);
end;

destructor TKokoroCpu.Destroy;
begin
  fPool.Free;
  inherited Destroy;
end;

procedure TKokoroCpu.GemmItem(aItem: integer);
var
  ri, pg, r0, n, p, p1: PtrInt;
begin
  // the jobs: (group of panels, block of rows), rows first: the threads
  // running together read the same weights
  pg := aItem div fGRowItems;
  ri := aItem - pg * fGRowItems;
  r0 := ri * fGRowStep;
  n := fGRows - r0;
  if n > fGRowStep then
    n := fGRowStep;
  p := pg * fGPanelStep;
  p1 := p + fGPanelStep;
  if p1 > fGm^.Panels then
    p1 := fGm^.Panels;
  // panel by panel (two at a time with AVX-512): its weights stay in
  // cache for all the blocks
  while p < p1 do
    if fGPairs and (p + 1 < p1) then
    begin
      KokoroGemmPair(pointer(fGx + r0 * fGXStride), fGXStride, fGTapStride, fGm^, p,
        pointer(fGy + r0 * fGYStride + p * (KGEMM_NR * SizeOf(single))), fGYStride,
        n, fGFlags);
      inc(p, 2);
    end
    else
    begin
      KokoroGemmPanel(pointer(fGx + r0 * fGXStride), fGXStride, fGTapStride, fGm^, p,
        pointer(fGy + r0 * fGYStride + p * (KGEMM_NR * SizeOf(single))), fGYStride,
        n, fGFlags);
      inc(p);
    end;
end;

procedure TKokoroCpu.Gemm(x: pointer; aRows, aXStride, aTapStride: PtrInt;
  const m: TKPanels; y: PSingleArray; aYStride: PtrInt; aFlags: PtrInt);
var
  t0, t1: Int64;
begin
  // the time and the work, for the profile
  QueryPerformanceMicroSeconds(t0);
  DoGemm(x, aRows, aXStride, aTapStride, m, y, aYStride, aFlags);
  QueryPerformanceMicroSeconds(t1);
  inc(fGemmMicroSec, t1 - t0);
  inc(fGemmMac, Int64(aRows) * m.Ktot * m.OutCount);
end;

procedure TKokoroCpu.DoGemm(x: pointer; aRows, aXStride, aTapStride: PtrInt;
  const m: TKPanels; y: PSingleArray; aYStride: PtrInt; aFlags: PtrInt);
var
  tiles, per, want, groups: PtrInt;
begin
  if (aRows <= 0) or (m.OutCount <= 0) then
    exit;
  fGm := @m;
  fGx := PtrInt(x);
  fGy := PtrInt(y);
  fGXStride := aXStride * SizeOf(single);
  fGTapStride := aTapStride * SizeOf(single);
  fGYStride := aYStride * SizeOf(single);
  fGFlags := aFlags;
  fGRows := aRows;
  fGPairs := KokoroGemmPairs;
  if (fPool = nil) or (Int64(aRows) * m.Ktot * m.OutCount < GEMM_PARALLEL) then
  begin
    fGRowStep := aRows;
    fGRowItems := 1;
    fGPanelStep := m.Panels;
    GemmItem(0);
    exit;
  end;
  // enough jobs to balance the threads, big enough to keep the weights of
  // a panel in cache over several blocks of 6 rows
  want := fThreads * 8;
  tiles := (aRows + KGEMM_MR - 1) div KGEMM_MR;
  per := GEMM_TILES_MAX;
  while (per > GEMM_TILES_MIN) and ((tiles + per - 1) div per < want) do
    per := per shr 1;
  fGRowStep := per * KGEMM_MR;
  fGRowItems := (aRows + fGRowStep - 1) div fGRowStep;
  // even groups of panels for the two-panel kernel
  fGPanelStep := m.Panels;
  while (fGPanelStep > 1 + ord(fGPairs)) and
        (fGRowItems * ((m.Panels + fGPanelStep - 1) div fGPanelStep) < want) do
    fGPanelStep := (fGPanelStep + 1) shr 1;
  groups := (m.Panels + fGPanelStep - 1) div fGPanelStep;
  fPool.Run(GemmItem, fGRowItems * groups);
end;

function TKokoroCpu.Scratch(aIndex, aCount: PtrInt): PSingleArray;
begin
  KokoroGrow(fScratch[aIndex], aCount);
  result := pointer(fScratch[aIndex]);
end;

procedure TKokoroCpu.CopyRowsJob(aFirst, aLast: PtrInt);
var
  r: PtrInt;
begin
  for r := aFirst to aLast do
    MoveFast(fCx^[r * fCXStride], fCy^[r * fCYStride], fCCols * SizeOf(single));
end;

procedure TKokoroCpu.CopyRows(x: PSingleArray; aXStride: PtrInt; y: PSingleArray;
  aYStride, aRows, aCols: PtrInt);
begin
  fCx := x;
  fCy := y;
  fCXStride := aXStride;
  fCYStride := aYStride;
  fCCols := aCols;
  ParallelRows(aRows, 256, CopyRowsJob);
end;

procedure TKokoroCpu.RowsItem(aItem: integer);
var
  last: PtrInt;
begin
  last := (aItem + 1) * fRowsStep - 1;
  if last >= fRowsCount then
    last := fRowsCount - 1;
  fRowsJob(aItem * fRowsStep, last);
end;

procedure TKokoroCpu.BlockItem(aItem: integer);
var
  last: PtrInt;
begin
  last := (aItem + 1) * fBlockSize - 1;
  if last >= fBlockRows then
    last := fBlockRows - 1;
  fBlockJob(aItem, aItem * fBlockSize, last);
end;

function TKokoroCpu.ParallelBlocks(aRows, aBlock: PtrInt;
  const aJob: TOnKokoroBlock): PtrInt;
var
  i: PtrInt;
begin
  result := 0;
  if aRows <= 0 then
    exit;
  result := (aRows + aBlock - 1) div aBlock;
  fBlockJob := aJob;
  fBlockRows := aRows;
  fBlockSize := aBlock;
  if (fPool = nil) or (result = 1) then
    for i := 0 to result - 1 do
      BlockItem(i)
  else
    fPool.Run(BlockItem, result);
end;

const
  ELEMENTS_BLOCK = 1 shl 14;

procedure TKokoroCpu.AddRows(aFirst, aLast: PtrInt);
var
  i, last: PtrInt;
begin
  last := (aLast + 1) * ELEMENTS_BLOCK - 1;
  if last >= fECount then
    last := fECount - 1;
  for i := aFirst * ELEMENTS_BLOCK to last do
    fEy^[i] := fEx^[i] + fEy^[i];
end;

procedure TKokoroCpu.LeakyRows(aFirst, aLast: PtrInt);
var
  i, last: PtrInt;
begin
  last := (aLast + 1) * ELEMENTS_BLOCK - 1;
  if last >= fECount then
    last := fECount - 1;
  for i := aFirst * ELEMENTS_BLOCK to last do
    if fEx^[i] < 0 then
      fEx^[i] := fEx^[i] * fESlope;
end;

procedure TKokoroCpu.DivideRows(aFirst, aLast: PtrInt);
var
  i, last: PtrInt;
begin
  last := (aLast + 1) * ELEMENTS_BLOCK - 1;
  if last >= fECount then
    last := fECount - 1;
  for i := aFirst * ELEMENTS_BLOCK to last do
    fEx^[i] := fEx^[i] / fESlope;
end;

procedure TKokoroCpu.AddScaleRows(aFirst, aLast: PtrInt);
var
  i, last: PtrInt;
begin
  last := (aLast + 1) * ELEMENTS_BLOCK - 1;
  if last >= fECount then
    last := fECount - 1;
  for i := aFirst * ELEMENTS_BLOCK to last do
    fEy^[i] := (fEy^[i] + fEx^[i]) * fESlope;
end;

procedure TKokoroCpu.LayerNormRows(aFirst, aLast: PtrInt);
begin
  KokoroLayerNorm(@fEx^[aFirst * fEStride], aLast - aFirst + 1, fECols, fEStride,
    fEGamma, fEBeta, fESlope);
end;

procedure TKokoroCpu.Divide(x: PSingleArray; aCount: PtrInt; aDivisor: single);
begin
  fEx := x;
  fESlope := aDivisor;
  fECount := aCount;
  ParallelRows((aCount + ELEMENTS_BLOCK - 1) div ELEMENTS_BLOCK, 1, DivideRows);
end;

procedure TKokoroCpu.AddScale(x, y: PSingleArray; aCount: PtrInt; aScale: single);
begin
  fEx := x;
  fEy := y;
  fESlope := aScale;
  fECount := aCount;
  ParallelRows((aCount + ELEMENTS_BLOCK - 1) div ELEMENTS_BLOCK, 1, AddScaleRows);
end;

procedure TKokoroCpu.LayerNorm(x: PSingleArray; aRows, aCols, aStride: PtrInt;
  gamma, beta: PSingleArray; aEps: single);
begin
  fEx := x;
  fECols := aCols;
  fEStride := aStride;
  fEGamma := gamma;
  fEBeta := beta;
  fESlope := aEps;
  ParallelRows(aRows, 8, LayerNormRows);
end;

procedure TKokoroCpu.Add(x, y: PSingleArray; aCount: PtrInt);
begin
  fEx := x;
  fEy := y;
  fECount := aCount; // the elements, by packets of ELEMENTS_BLOCK
  ParallelRows((aCount + ELEMENTS_BLOCK - 1) div ELEMENTS_BLOCK, 1, AddRows);
end;

procedure TKokoroCpu.Leaky(x: PSingleArray; aCount: PtrInt; aSlope: single);
begin
  fEx := x;
  fESlope := aSlope;
  fECount := aCount;
  ParallelRows((aCount + ELEMENTS_BLOCK - 1) div ELEMENTS_BLOCK, 1, LeakyRows);
end;

procedure TKokoroCpu.ParallelRows(aRows, aMinRows: PtrInt; const aJob: TOnKokoroRows);
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
  fRowsStep := (aRows + fThreads * 4 - 1) div (fThreads * 4);
  if fRowsStep < aMinRows then
    fRowsStep := aMinRows;
  items := (aRows + fRowsStep - 1) div fRowsStep;
  fRowsJob := aJob;
  fRowsCount := aRows;
  fPool.Run(RowsItem, items);
end;


{ TKLinear }

constructor TKLinear.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aIn, aOut: PtrInt; aBias: boolean);
var
  b: PSingleArray;
begin
  inherited Create;
  fCpu := aCpu;
  b := nil;
  if aBias then
    b := aFile.F32(aPrefix + '.bias', [aOut]);
  KokoroPack(fM, aFile.F32(aPrefix + '.weight', [aOut, aIn]), aOut, aIn, 1, b);
end;

procedure TKLinear.Forward(x: PSingleArray; aRows: PtrInt; y: PSingleArray;
  aXStride: PtrInt);
begin
  if aXStride = 0 then
    aXStride := fM.Cin;
  fCpu.Gemm(x, aRows, aXStride, 0, fM, y, fM.OutCount);
end;


{ TKConv }

constructor TKConv.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aIn, aOut, aK, aStride, aDil, aPad: PtrInt;
  aWeightNorm, aBias: boolean);
var
  w: TSingleDynArray;
  b: PSingleArray;
begin
  inherited Create;
  fCpu := aCpu;
  fIn := aIn;
  fOut := aOut;
  fK := aK;
  fStride := aStride;
  fDil := aDil;
  if aPad < 0 then
    aPad := (aK * aDil - aDil) div 2;
  fPad := aPad;
  if aWeightNorm then
    w := KokoroWeightNorm(aFile, aPrefix, aOut, aIn * aK)
  else
  begin
    SetLength(w, aOut * aIn * aK);
    MoveFast(aFile.F32(aPrefix + '.weight', [aOut, aIn, aK])^, w[0],
      length(w) * SizeOf(single));
  end;
  b := nil;
  if aBias then
    b := aFile.F32(aPrefix + '.bias', [aOut]);
  if aStride = 1 then
    // tap by tap, read in place
    KokoroPack(fM, pointer(w), aOut, aIn, aK, b)
  else
    // the im2col columns, in PyTorch's order: c * K + k
    KokoroPack(fM, pointer(w), aOut, aIn * aK, 1, b);
end;

function TKConv.OutLen(aLen: PtrInt): PtrInt;
begin
  result := (aLen + 2 * fPad - fDil * (fK - 1) - 1) div fStride + 1;
end;

procedure TKConv.Im2Col(aFirst, aLast: PtrInt);
var
  t, c, k, src: PtrInt;
  dst, row: PSingleArray;
begin
  for t := aFirst to aLast do
  begin
    dst := @fCol^[t * fM.Cin];
    for k := 0 to fK - 1 do
    begin
      src := (fFirst + t) * fStride - fPad + k * fDil;
      if (src < 0) or (src >= fLen) then
        for c := 0 to fIn - 1 do
          dst^[c * fK + k] := 0
      else
      begin
        row := @fX^[src * fIn];
        for c := 0 to fIn - 1 do
          dst^[c * fK + k] := row^[c];
      end;
    end;
  end;
end;

procedure TKConv.Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
var
  rows, s, n: PtrInt;
  p: PSingleArray;
begin
  rows := OutLen(aLen);
  if rows <= 0 then
    exit;
  if fStride = 1 then
  begin
    if fPad = 0 then
      fCpu.Gemm(x, rows, fIn, fDil * fIn, fM, y, fOut)
    else
    begin
      // a copy bordered with zeros, then read in place
      p := fCpu.Scratch(2, (aLen + 2 * fPad) * fIn);
      FillCharFast(p^[0], fPad * fIn * SizeOf(single), 0);
      FillCharFast(p^[(aLen + fPad) * fIn], fPad * fIn * SizeOf(single), 0);
      fCpu.CopyRows(x, fIn, @p^[fPad * fIn], fIn, aLen, fIn);
      ForwardPadded(@p^[fPad * fIn], aLen, y);
    end;
    exit;
  end;
  fX := x;
  fLen := aLen;
  s := 0;
  while s < rows do
  begin
    n := rows - s;
    if n > CONV_CHUNK then
      n := CONV_CHUNK;
    KokoroGrow(fColBuf, n * fM.Cin);
    fCol := pointer(fColBuf);
    fFirst := s;
    fCpu.ParallelRows(n, 32, Im2Col);
    fCpu.Gemm(fCol, n, fM.Cin, 0, fM, @y^[s * fOut], fOut);
    inc(s, n);
  end;
end;

procedure TKConv.ForwardPadded(x: PSingleArray; aLen: PtrInt; y: PSingleArray;
  aFlags: PtrInt);
begin
  if fStride <> 1 then
    raise EKokoroNet.CreateUtf8('%.ForwardPadded: stride %', [self, fStride]);
  fCpu.Gemm(@x^[-fPad * fIn], OutLen(aLen), fIn, fDil * fIn, fM, y, fOut, aFlags);
end;


{ TKConvT }

constructor TKConvT.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aIn, aOut, aK, aStride, aPad, aOutPad: PtrInt;
  aDepthwise: boolean);
var
  w: TSingleDynArray;
  g: TSingleDynArray;
  c, o, k: PtrInt;
begin
  inherited Create;
  fCpu := aCpu;
  fIn := aIn;
  fOut := aOut;
  fK := aK;
  fStride := aStride;
  fPad := aPad;
  fOutPad := aOutPad;
  fDepthwise := aDepthwise;
  SetLength(fB, aOut);
  MoveFast(aFile.F32(aPrefix + '.bias', [aOut])^, fB[0], aOut * SizeOf(single));
  if aDepthwise then
  begin
    // [C][1][K], weight_norm per channel
    fW := KokoroWeightNorm(aFile, aPrefix, aIn, aK);
    exit;
  end;
  // [In][Out][K], weight_norm per input; for the product: [(k, o)][In],
  // each tap k gives a row of Out contiguous outputs
  w := KokoroWeightNorm(aFile, aPrefix, aIn, aOut * aK);
  SetLength(g, aOut * aK * aIn);
  for c := 0 to aIn - 1 do
    for o := 0 to aOut - 1 do
      for k := 0 to aK - 1 do
        g[(k * aOut + o) * aIn + c] := w[(c * aOut + o) * aK + k];
  KokoroPack(fM, pointer(g), aOut * aK, aIn, 1, nil);
end;

function TKConvT.OutLen(aLen: PtrInt): PtrInt;
begin
  result := (aLen - 1) * fStride - 2 * fPad + fK + fOutPad;
end;

procedure TKConvT.GatherRows(aFirst, aLast: PtrInt);
var
  t, j, j1, k, o, n: PtrInt;
  src, dst: PSingleArray;
begin
  // output t receives the tap k = t + pad - j * stride of each input j:
  // the bias, then the inputs j in increasing order, as the per-input loop
  for t := aFirst to aLast do
  begin
    dst := @fJy^[t * fOut];
    MoveFast(fB[0], dst^[0], fOut * SizeOf(single));
    n := t + fPad - fK + 1;
    if n <= 0 then
      j := 0
    else
      j := (n + fStride - 1) div fStride;
    j1 := (t + fPad) div fStride;
    if j1 >= fJLen then
      j1 := fJLen - 1;
    while j <= j1 do
    begin
      k := t + fPad - j * fStride;
      src := @fJFull^[(j * fK + k) * fOut];
      for o := 0 to fOut - 1 do
        dst^[o] := dst^[o] + src^[o];
      inc(j);
    end;
  end;
end;

procedure TKConvT.Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
var
  total, j, k, t, c: PtrInt;
begin
  total := OutLen(aLen);
  if not fDepthwise then
  begin
    // each input gives K rows of outputs, stored at j * stride - pad + k
    fJFull := fCpu.Scratch(2, aLen * fK * fOut);
    fCpu.Gemm(x, aLen, fIn, 0, fM, fJFull, fK * fOut);
    fJy := y;
    fJLen := aLen;
    fCpu.ParallelRows(total, 64, GatherRows);
    exit;
  end;
  // per channel
  for t := 0 to total - 1 do
    MoveFast(fB[0], y^[t * fOut], fOut * SizeOf(single));
  for j := 0 to aLen - 1 do
    for k := 0 to fK - 1 do
    begin
      t := j * fStride - fPad + k;
      if (t < 0) or (t >= total) then
        continue;
      for c := 0 to fIn - 1 do
        y^[t * fOut + c] := y^[t * fOut + c] + x^[j * fIn + c] * fW[c * fK + k];
    end;
end;


{ TKLstm }

constructor TKLstm.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aIn, aHidden: PtrInt);
var
  w, b: TSingleDynArray;
  d, g: PtrInt;
  suffix: RawUtf8;
begin
  inherited Create;
  fCpu := aCpu;
  fIn := aIn;
  fH := aHidden;
  g := 4 * aHidden;
  SetLength(w, 2 * g * aIn);
  SetLength(b, 2 * g);
  for d := 0 to 1 do
  begin
    if d = 0 then
      suffix := ''
    else
      suffix := '_reverse';
    MoveFast(aFile.F32(aPrefix + '.weight_ih_l0' + suffix, [g, aIn])^,
      w[d * g * aIn], g * aIn * SizeOf(single));
    MoveFast(aFile.F32(aPrefix + '.bias_ih_l0' + suffix, [g])^, b[d * g],
      g * SizeOf(single));
    SetLength(fHh[d], g * aHidden);
    MoveFast(aFile.F32(aPrefix + '.weight_hh_l0' + suffix, [g, aHidden])^,
      fHh[d][0], g * aHidden * SizeOf(single));
    SetLength(fBhh[d], g);
    MoveFast(aFile.F32(aPrefix + '.bias_hh_l0' + suffix, [g])^, fBhh[d][0],
      g * SizeOf(single));
  end;
  KokoroPack(fIh, pointer(w), 2 * g, aIn, 1, pointer(b));
  if aHidden and 15 <> 0 then
    raise EKokoroNet.CreateUtf8('%: hidden size % not supported', [self, aHidden]);
end;

procedure TKLstm.Direction(aFirst, aLast: PtrInt);
var
  d, s, t, j, k, g4: PtrInt;
  h, c, gates: TSingleDynArray;
  i: single;
begin
  g4 := 4 * fH;
  SetLength(h, fH);
  SetLength(c, fH);
  SetLength(gates, g4);
  for d := aFirst to aLast do
  begin
    FillCharFast(h[0], fH * SizeOf(single), 0);
    FillCharFast(c[0], fH * SizeOf(single), 0);
    for s := 0 to fLen - 1 do
    begin
      if d = 0 then
        t := s
      else
        t := fLen - 1 - s;
      // h W_hh^T: 4H products of length H
      if Assigned(InferDotsF32) then
        InferDotsF32(@h[0], @fHh[d][0], fH * SizeOf(single), g4, fH, pointer(gates))
      else
        for j := 0 to g4 - 1 do
        begin
          i := 0;
          for k := 0 to fH - 1 do
            i := i + h[k] * fHh[d][j * fH + k];
          gates[j] := i;
        end;
      // the gates, then c and h
      KokoroLstmCell(@fG^[t * 2 * g4 + d * g4], pointer(gates), pointer(fBhh[d]),
        pointer(c), pointer(h), fH);
      MoveFast(h[0], fY^[t * 2 * fH + d * fH], fH * SizeOf(single));
    end;
  end;
end;

procedure TKLstm.Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
var
  g: TSingleDynArray;
begin
  if aLen <= 0 then
    exit;
  // the inputs of both directions at once: x W_ih^T + b_ih
  SetLength(g, aLen * 8 * fH);
  fCpu.Gemm(x, aLen, fIn, 0, fIh, pointer(g), 8 * fH);
  fG := pointer(g);
  fY := y;
  fLen := aLen;
  // then the recurrence, one direction per thread
  fCpu.ParallelRows(2, 1, Direction);
end;


{ TKAdaIN }

constructor TKAdaIN.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aStyle, aChannels: PtrInt);
begin
  inherited Create;
  fCpu := aCpu;
  fC := aChannels;
  fFc := TKLinear.Create(aCpu, aFile, aPrefix + '.fc', aStyle, 2 * aChannels);
end;

destructor TKAdaIN.Destroy;
begin
  fFc.Free;
  inherited Destroy;
end;

procedure TKAdaIN.Prepare(s: PSingleArray);
var
  h: TSingleDynArray;
begin
  SetLength(h, 2 * fC);
  fFc.Forward(s, 1, pointer(h));
  fGamma := copy(h, 0, fC);
  fBeta := copy(h, fC, fC);
end;

const
  /// the rows of a block of partial sums (reproducible)
  STATS_BLOCK = 256;

procedure TKAdaIN.SumBlock(aBlock, aFirst, aLast: PtrInt);
begin
  KokoroColSums(@fJx^[aFirst * fC], aLast - aFirst + 1, fC, fC, @fPart[aBlock * fC]);
end;

procedure TKAdaIN.VarBlock(aBlock, aFirst, aLast: PtrInt);
begin
  KokoroColSquares(@fJx^[aFirst * fC], aLast - aFirst + 1, fC, fC, pointer(fMean),
    @fPart[aBlock * fC]);
end;

procedure TKAdaIN.ApplyRows(aFirst, aLast: PtrInt);
begin
  KokoroAffineSnake(@fJx^[aFirst * fC], aLast - aFirst + 1, fC, fC,
    @fJy^[aFirst * fC], fC, pointer(fScale), pointer(fShift), nil, nil);
end;

procedure TKAdaIN.Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
begin
  if aLen <= 0 then
    exit;
  Stats(x, aLen);
  fJx := x;
  fJy := y;
  fCpu.ParallelRows(aLen, 64, ApplyRows);
end;

procedure TKAdaIN.Stats(x: PSingleArray; aLen: PtrInt);
var
  blocks, b, c: PtrInt;
  sum: double;
  inv: single;
begin
  fJx := x;
  blocks := (aLen + STATS_BLOCK - 1) div STATS_BLOCK;
  // the mean and the (population) variance of each channel over time
  fPart := nil;
  SetLength(fPart, blocks * fC);
  SetLength(fMean, fC);
  fCpu.ParallelBlocks(aLen, STATS_BLOCK, SumBlock);
  for c := 0 to fC - 1 do
  begin
    sum := 0;
    for b := 0 to blocks - 1 do
      sum := sum + fPart[b * fC + c];
    fMean[c] := sum / aLen;
  end;
  FillCharFast(fPart[0], length(fPart) * SizeOf(double), 0);
  fCpu.ParallelBlocks(aLen, STATS_BLOCK, VarBlock);
  SetLength(fScale, fC);
  SetLength(fShift, fC);
  for c := 0 to fC - 1 do
  begin
    sum := 0;
    for b := 0 to blocks - 1 do
      sum := sum + fPart[b * fC + c];
    inv := 1 / Sqrt(sum / aLen + 1e-5);
    // (1 + gamma) * (x - m) * inv + beta
    fScale[c] := (1 + fGamma[c]) * inv;
    fShift[c] := fBeta[c] - fScale[c] * single(fMean[c]);
  end;
end;


{ TKAdaLayerNorm }

constructor TKAdaLayerNorm.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aStyle, aChannels: PtrInt);
begin
  inherited Create;
  fC := aChannels;
  fFc := TKLinear.Create(aCpu, aFile, aPrefix + '.fc', aStyle, 2 * aChannels);
end;

destructor TKAdaLayerNorm.Destroy;
begin
  fFc.Free;
  inherited Destroy;
end;

procedure TKAdaLayerNorm.Prepare(s: PSingleArray);
var
  h: TSingleDynArray;
begin
  SetLength(h, 2 * fC);
  fFc.Forward(s, 1, pointer(h));
  fGamma := copy(h, 0, fC);
  fBeta := copy(h, fC, fC);
end;

procedure TKAdaLayerNorm.Forward(x: PSingleArray; aLen, aStride: PtrInt);
var
  t, c: PtrInt;
  row: PSingleArray;
begin
  KokoroLayerNorm(x, aLen, fC, aStride, nil, nil, 1e-5);
  for t := 0 to aLen - 1 do
  begin
    row := @x^[t * aStride];
    for c := 0 to fC - 1 do
      row^[c] := (1 + fGamma[c]) * row^[c] + fBeta[c];
  end;
end;


{ TKAdainResBlk }

constructor TKAdainResBlk.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aIn, aOut, aStyle: PtrInt; aUpsample: boolean);
begin
  inherited Create;
  fCpu := aCpu;
  fIn := aIn;
  fOut := aOut;
  fUpsample := aUpsample;
  fNorm1 := TKAdaIN.Create(aCpu, aFile, aPrefix + '.norm1', aStyle, aIn);
  fNorm2 := TKAdaIN.Create(aCpu, aFile, aPrefix + '.norm2', aStyle, aOut);
  fConv1 := TKConv.Create(aCpu, aFile, aPrefix + '.conv1', aIn, aOut, 3);
  fConv2 := TKConv.Create(aCpu, aFile, aPrefix + '.conv2', aOut, aOut, 3);
  if aUpsample then
    fPool := TKConvT.Create(aCpu, aFile, aPrefix + '.pool', aIn, aIn, 3, 2, 1, 1, true);
  if aIn <> aOut then
    fShort := TKConv.Create(aCpu, aFile, aPrefix + '.conv1x1', aIn, aOut, 1, 1, 1,
      0, true, false);
end;

destructor TKAdainResBlk.Destroy;
begin
  fNorm1.Free;
  fNorm2.Free;
  fConv1.Free;
  fConv2.Free;
  fPool.Free;
  fShort.Free;
  inherited Destroy;
end;

procedure TKAdainResBlk.Prepare(s: PSingleArray);
begin
  fNorm1.Prepare(s);
  fNorm2.Prepare(s);
end;

function TKAdainResBlk.OutLen(aLen: PtrInt): PtrInt;
begin
  if fUpsample then
    result := 2 * aLen
  else
    result := aLen;
end;

procedure TKAdainResBlk.Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
var
  n, t: PtrInt;
  a, b, sc, up: TSingleDynArray;
  xs: PSingleArray;
  r2: single;
begin
  n := OutLen(aLen);
  // the residual
  SetLength(a, aLen * fIn);
  fNorm1.Forward(x, aLen, pointer(a));
  fCpu.Leaky(pointer(a), aLen * fIn, 0.2);
  if fUpsample then
  begin
    SetLength(b, n * fIn);
    fPool.Forward(pointer(a), aLen, pointer(b));
    a := b;
    b := nil;
  end;
  SetLength(b, n * fOut);
  fConv1.Forward(pointer(a), n, pointer(b));
  fNorm2.Forward(pointer(b), n, pointer(b));
  fCpu.Leaky(pointer(b), n * fOut, 0.2);
  fConv2.Forward(pointer(b), n, y);
  // the shortcut
  xs := x;
  if fUpsample then
  begin
    SetLength(up, n * fIn);
    for t := 0 to aLen - 1 do
    begin
      MoveFast(x^[t * fIn], up[2 * t * fIn], fIn * SizeOf(single));
      MoveFast(x^[t * fIn], up[(2 * t + 1) * fIn], fIn * SizeOf(single));
    end;
    xs := pointer(up);
  end;
  if fShort <> nil then
  begin
    SetLength(sc, n * fOut);
    fShort.Forward(xs, n, pointer(sc));
    xs := pointer(sc);
  end;
  // (residual + shortcut) * rsqrt(2), in float32 as torch
  r2 := 1 / Sqrt(single(2));
  fCpu.AddScale(xs, y, n * fOut, r2);
end;


{ TKAdaINResBlock1 }

constructor TKAdaINResBlock1.Create(aCpu: TKokoroCpu; aFile: TTorchFile;
  const aPrefix: RawUtf8; aChannels, aK: PtrInt;
  const aDilations: array of PtrInt; aStyle: PtrInt);
var
  i, c: PtrInt;
  a: PSingleArray;
begin
  inherited Create;
  fCpu := aCpu;
  fC := aChannels;
  for i := 0 to 2 do
  begin
    fAdain1[i] := TKAdaIN.Create(aCpu, aFile, FormatUtf8('%.adain1.%', [aPrefix, i]),
      aStyle, aChannels);
    fAdain2[i] := TKAdaIN.Create(aCpu, aFile, FormatUtf8('%.adain2.%', [aPrefix, i]),
      aStyle, aChannels);
    fConv1[i] := TKConv.Create(aCpu, aFile, FormatUtf8('%.convs1.%', [aPrefix, i]),
      aChannels, aChannels, aK, 1, aDilations[i]);
    fConv2[i] := TKConv.Create(aCpu, aFile, FormatUtf8('%.convs2.%', [aPrefix, i]),
      aChannels, aChannels, aK, 1, 1);
    SetLength(fAlpha1[i], aChannels);
    SetLength(fAlpha2[i], aChannels);
    SetLength(fInv1[i], aChannels);
    SetLength(fInv2[i], aChannels);
    a := aFile.F32(FormatUtf8('%.alpha1.%', [aPrefix, i]), [1, aChannels, 1]);
    MoveFast(a^[0], fAlpha1[i][0], aChannels * SizeOf(single));
    a := aFile.F32(FormatUtf8('%.alpha2.%', [aPrefix, i]), [1, aChannels, 1]);
    MoveFast(a^[0], fAlpha2[i][0], aChannels * SizeOf(single));
    for c := 0 to aChannels - 1 do
    begin
      fInv1[i][c] := 1 / fAlpha1[i][c];
      fInv2[i][c] := 1 / fAlpha2[i][c];
    end;
  end;
end;

destructor TKAdaINResBlock1.Destroy;
var
  i: PtrInt;
begin
  for i := 0 to 2 do
  begin
    fAdain1[i].Free;
    fAdain2[i].Free;
    fConv1[i].Free;
    fConv2[i].Free;
  end;
  inherited Destroy;
end;

procedure TKAdaINResBlock1.Prepare(s: PSingleArray);
var
  i: PtrInt;
begin
  for i := 0 to 2 do
  begin
    fAdain1[i].Prepare(s);
    fAdain2[i].Prepare(s);
  end;
end;

procedure TKAdaINResBlock1.SnakeRows(aFirst, aLast: PtrInt);
begin
  // AdaIN then Snake1D: x + (1 / a) * sin(a * x)^2, per channel
  KokoroAffineSnake(@fSx^[aFirst * fC], aLast - aFirst + 1, fC, fC,
    @fSy^[aFirst * fC], fC, fSScale, fSShift, fSa, fSinv);
end;

procedure TKAdaINResBlock1.AdaSnake(aNorm: TKAdaIN; const aAlpha,
  aInv: TSingleDynArray; x: PSingleArray; aLen: PtrInt; y: PSingleArray);
begin
  // the statistics of x, then a single pass: y := Snake(AdaIN(x))
  aNorm.Stats(x, aLen);
  fSx := x;
  fSy := y;
  fSScale := pointer(aNorm.Scale);
  fSShift := pointer(aNorm.Shift);
  fSa := pointer(aAlpha);
  fSinv := pointer(aInv);
  fCpu.ParallelRows(aLen, 32, SnakeRows);
end;

procedure TKAdaINResBlock1.Forward(x: PSingleArray; aLen: PtrInt; y: PSingleArray);
var
  i, m: PtrInt;
  a, b: PSingleArray;
begin
  if y <> x then
    fCpu.CopyRows(x, fC, y, fC, aLen, fC);
  // the inputs of the convolutions, bordered with m rows of zeros on each
  // side: read in place by the kernel
  m := 0;
  for i := 0 to 2 do
  begin
    if fConv1[i].Pad > m then
      m := fConv1[i].Pad;
    if fConv2[i].Pad > m then
      m := fConv2[i].Pad;
  end;
  a := fCpu.Scratch(0, (aLen + 2 * m) * fC);
  b := fCpu.Scratch(1, (aLen + 2 * m) * fC);
  FillCharFast(a^[0], m * fC * SizeOf(single), 0);
  FillCharFast(a^[(aLen + m) * fC], m * fC * SizeOf(single), 0);
  FillCharFast(b^[0], m * fC * SizeOf(single), 0);
  FillCharFast(b^[(aLen + m) * fC], m * fC * SizeOf(single), 0);
  a := @a^[m * fC];
  b := @b^[m * fC];
  for i := 0 to 2 do
  begin
    AdaSnake(fAdain1[i], fAlpha1[i], fInv1[i], y, aLen, a);
    fConv1[i].ForwardPadded(a, aLen, b);
    AdaSnake(fAdain2[i], fAlpha2[i], fInv2[i], b, aLen, b);
    // the residual: y := y + conv2(b), by the kernel
    fConv2[i].ForwardPadded(b, aLen, y, KGEMM_ADD);
  end;
end;

end.
