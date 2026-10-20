/// Qwen3-TTS - the two Qwen3 stacks (talker, code predictor) on the mormot2-ai decoder
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the talker (28 layers) or the code predictor (5 layers): a stack of the
// checkpoint without embedding nor head (TDecoderModel.CreateStack), fed with
// embedding rows (the input of the talker is a sum of embeddings) through
// ForwardEmbeds, on the CPU or on a Vulkan GPU (TQwen3Vulkan); the normed
// state of the last row comes back (aNormed), the K/V cache follows the calls
// until Reset
// - the heads are ours (TQwenLmHeads), applied on the CPU to that state: the
// talker has one, the code predictor 15, a different one at each step
// - weights: float32 (the BF16 of the file widened without loss) or Q8_0
// (quantized at load as ggml writes it), converted on the thread pool of the
// backend
// - TQwenVulkan: the Vulkan device shared by both stacks and the 12 Hz decoder
unit mormot.ai.tts.qwen.lm;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.threads,
  mormot.ai.infer.vulkan,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.infer.decoder,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.model.qwen3.vulkan,
  mormot.ai.tts.qwen.codec;

type
  EQwenLm = class(ESynException);

  /// the Vulkan device, shared by the stacks and the 12 Hz decoder
  TQwenVulkan = class
  protected
    fBackend: TInferVulkanBackend;
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
    /// the bytes of the weights of the stacks on the GPU
    property DeviceBytes: Int64
      read fDeviceBytes;
  end;

  /// the output heads of a stack, [Rows][Hidden] each: float32 or Q8_0
  TQwenLmHeads = class
  protected
    fName: RawUtf8;
    fDType: TInferDType;
    fHidden: PtrInt;
    fHeads: array of TInferWeight;
    fRows: TIntegerDynArray;
    fStore: TRawByteStringDynArray;
    fBytes: Int64;
    // the running conversion: blocks of rows, on a pool if any
    fJobTensor: PInferTensor;
    fJobData: PByteArray;
    fJobRows, fJobRowBytes: PtrInt;
    procedure MatrixRows(aItem: integer);
  public
    /// aDType: idtF32 or idtQ8_0
    constructor Create(const aName: RawUtf8; aHidden: PtrInt; aDType: TInferDType);
    /// add the head aTensor [aRows][Hidden]; returns its index - aPool: the
    // threads of the conversion (nil: the calling thread), the same bytes
    function Add(aFile: TSafeTensorsFile; const aTensor: RawUtf8; aRows: PtrInt;
      aPool: TInferThreadPool = nil): PtrInt;
    /// the crc32c of the heads, in their load order (tests)
    function Digest: cardinal;
    function Count: PtrInt;
    function Head(aIndex: PtrInt): TInferWeight;
    function Rows(aIndex: PtrInt): PtrInt;
    property Name: RawUtf8
      read fName;
    property DType: TInferDType
      read fDType;
    property Hidden: PtrInt
      read fHidden;
    /// the bytes of the heads
    property Bytes: Int64
      read fBytes;
  end;

  /// the computation of a Qwen3 stack: its layers on the CPU or on the GPU,
  // its heads on the CPU
  TQwenLm = class
  protected
    fModel: TDecoderModel;
    fDevice: TQwen3Vulkan;
    fVulkan: TQwenVulkan;
    fCtx: TDecoderContext;
    fHeads: TQwenLmHeads;
    fCpu: TInferBackend;
    fLast: TSingleDynArray;
    fLen, fDeviceBytes: PtrInt;
    fProbe: TOnQwenProbe;
    procedure OnState(const aName: RawUtf8; aValues: PSingleArray;
      aRows, aCols: PtrInt);
  public
    /// aModel (QwenLmStack) is owned; aHeads, aCpu (where the heads run) and
    // aVulkan are not, and must live longer; aVulkan: nil for the CPU, or the
    // GPU where the layers run, with the K/V cache in half floats if aKvF16
    // (it drifts in this loop, which feeds floats back: float32 by default)
    constructor Create(aModel: TDecoderModel; aHeads: TQwenLmHeads;
      aCpu: TInferBackend; aVulkan: TQwenVulkan = nil; aKvF16: boolean = false);
    destructor Destroy; override;
    /// empty the cache: the next row is at position 0
    procedure Reset;
    /// add aRows rows of embeddings x[aRows][Hidden] at positions
    // Length..Length + aRows - 1; for the last row, return its normed state
    // in aHidden and the logits of head aOutput in aLogits (nil: nothing)
    procedure Forward(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
      aLogits, aHidden: PSingleArray);
    property Model: TDecoderModel
      read fModel;
    property Heads: TQwenLmHeads
      read fHeads;
    /// the positions already in the cache
    property Length: PtrInt
      read fLen;
    /// the intermediate outputs (tests, CPU only): name.layerN, name.out;
    // nil by default
    property Probe: TOnQwenProbe
      read fProbe write fProbe;
  end;

/// a stack of the checkpoint in aSnapshot, without embedding nor head:
// aConfigPath 'talker_config' or 'talker_config.code_predictor_config',
// aPrefix 'talker.model.' or 'talker.code_predictor.model.', aDType idtF32 or
// idtQ8_0; aBackend converts the weights on its pool and is not owned
function QwenLmStack(const aSnapshot: TFileName; aBackend: TInferBackend;
  const aConfigPath, aPrefix: RawUtf8; aDType: TInferDType): TDecoderModel;

/// the tensor aName [aRows][aCols] in float32
function QwenWiden(aFile: TSafeTensorsFile; const aName: RawUtf8;
  aRows, aCols: PtrInt): TSingleDynArray;


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

function QwenLmStack(const aSnapshot: TFileName; aBackend: TInferBackend;
  const aConfigPath, aPrefix: RawUtf8; aDType: TInferDType): TDecoderModel;
begin
  result := TQwen3Model.CreateStack(IncludeTrailingPathDelimiter(aSnapshot),
    aBackend, aConfigPath, aPrefix, '', '', aDType);
end;

/// the bytes of the layers of aModel: the matrices and the norms
function StackBytes(aModel: TDecoderModel): Int64;
var
  c: TDecoderConfig;
  q, kv, cells: Int64;
begin
  c := aModel.Config;
  q := c.Heads * c.HeadDim;
  kv := c.KvHeads * c.HeadDim;
  cells := Int64(c.Hidden) * (2 * q + 2 * kv + 3 * c.Intermediate);
  if aModel.Weights = idtQ8_0 then
    result := (cells shr 5) * SizeOf(TInferBlockQ8_0)
  else
    result := cells * SizeOf(single);
  result := (result + (2 * c.Hidden + 2 * c.HeadDim) * SizeOf(single)) * c.Layers +
    c.Hidden * SizeOf(single);
end;


{ TQwenVulkan }

class function TQwenVulkan.Available: boolean;
begin
  result := TInferVulkan.Available;
end;

constructor TQwenVulkan.Create(aThreads: integer);
begin
  inherited Create;
  fBackend := TInferVulkanBackend.Create(aThreads);
end;

destructor TQwenVulkan.Destroy;
begin
  fBackend.Free;
  inherited Destroy;
end;

function TQwenVulkan.DeviceName: RawUtf8;
begin
  result := fBackend.Vulkan.DeviceName;
end;


{ TQwenLmHeads }

constructor TQwenLmHeads.Create(const aName: RawUtf8; aHidden: PtrInt;
  aDType: TInferDType);
begin
  inherited Create;
  if not (aDType in [idtF32, idtQ8_0]) then
    EQwenLm.RaiseUtf8('%: % weights not supported', [self, INFER_DTYPE_NAME[aDType]]);
  if (aHidden <= 0) or (aHidden and 31 <> 0) then
    EQwenLm.RaiseUtf8('%: invalid width % for %', [self, aHidden, aName]);
  fName := aName;
  fHidden := aHidden;
  fDType := aDType;
end;

const
  /// the rows of one conversion block: a few hundred microseconds
  MATRIX_ROWS = 32;

procedure TQwenLmHeads.MatrixRows(aItem: integer);
var
  r, last, b, blocks: PtrInt;
  dst: PInferBlockQ8_0Array;
  block: array[0..31] of single;
begin
  r := aItem * MATRIX_ROWS;
  last := r + MATRIX_ROWS;
  if last > fJobRows then
    last := fJobRows;
  blocks := fHidden shr 5;
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
      InferRowToF32(fJobTensor^.Data, fJobTensor^.DType, r, fHidden,
        @fJobData^[r * fJobRowBytes]);
    inc(r);
  end;
end;

function TQwenLmHeads.Add(aFile: TSafeTensorsFile; const aTensor: RawUtf8;
  aRows: PtrInt; aPool: TInferThreadPool): PtrInt;
var
  t: PInferTensor;
  i, items: PtrInt;
begin
  t := aFile.Tensor(aTensor);
  if t^.Count <> Int64(aRows) * fHidden then
    EQwenLm.RaiseUtf8('%: % is % instead of [%,%]',
      [self, aTensor, InferShapeText(t^.Shape), aRows, fHidden]);
  if fDType = idtQ8_0 then
    fJobRowBytes := (fHidden shr 5) * SizeOf(TInferBlockQ8_0)
  else
    fJobRowBytes := fHidden * SizeOf(single);
  result := system.length(fHeads);
  SetLength(fHeads, result + 1);
  SetLength(fRows, result + 1);
  SetLength(fStore, result + 1);
  SetLength(fStore[result], Int64(aRows) * fJobRowBytes);
  inc(fBytes, Int64(aRows) * fJobRowBytes);
  fJobTensor := t;
  fJobData := pointer(fStore[result]);
  fJobRows := aRows;
  items := (aRows + MATRIX_ROWS - 1) div MATRIX_ROWS;
  if aPool = nil then
    for i := 0 to items - 1 do
      MatrixRows(i)
  else
    aPool.Run(MatrixRows, items);
  fHeads[result] := InferWeight(fJobData, fDType);
  fRows[result] := aRows;
end;

function TQwenLmHeads.Digest: cardinal;
var
  i: PtrInt;
begin
  result := 0;
  for i := 0 to high(fStore) do
    result := crc32c(result, pointer(fStore[i]), system.length(fStore[i]));
end;

function TQwenLmHeads.Count: PtrInt;
begin
  result := system.length(fHeads);
end;

function TQwenLmHeads.Head(aIndex: PtrInt): TInferWeight;
begin
  result := fHeads[aIndex];
end;

function TQwenLmHeads.Rows(aIndex: PtrInt): PtrInt;
begin
  result := fRows[aIndex];
end;


{ TQwenLm }

constructor TQwenLm.Create(aModel: TDecoderModel; aHeads: TQwenLmHeads;
  aCpu: TInferBackend; aVulkan: TQwenVulkan; aKvF16: boolean);
begin
  inherited Create;
  fModel := aModel;
  fHeads := aHeads;
  fCpu := aCpu;
  if aModel.Config.Hidden <> aHeads.Hidden then
    EQwenLm.RaiseUtf8('%: % has a width of %, its heads %', [self, aHeads.Name,
      aModel.Config.Hidden, aHeads.Hidden]);
  if aVulkan <> nil then
  begin
    // the weights go up once, here
    fVulkan := aVulkan;
    fDevice := TQwen3Vulkan.Create(aModel, aVulkan.Backend);
    fDevice.KvF16 := aKvF16;
    fCtx := fDevice.NewContext;
    fDeviceBytes := StackBytes(aModel);
    inc(aVulkan.fDeviceBytes, fDeviceBytes);
  end
  else
    fCtx := aModel.NewContext;
  SetLength(fLast, aHeads.Hidden);
end;

destructor TQwenLm.Destroy;
begin
  // the context, then the device (it forgets the weights on the GPU), then
  // the model: each before what it uses
  fCtx.Free;
  fDevice.Free;
  fModel.Free;
  if fVulkan <> nil then
    dec(fVulkan.fDeviceBytes, fDeviceBytes);
  inherited Destroy;
end;

procedure TQwenLm.Reset;
begin
  fLen := 0;
  fCtx.Reset;
end;

procedure TQwenLm.OnState(const aName: RawUtf8; aValues: PSingleArray;
  aRows, aCols: PtrInt);
begin
  if aName = 'norm' then
    fProbe(fHeads.Name + '.out', aValues, aRows, aCols)
  else if (system.length(aName) = 7) and IdemPChar(pointer(aName), 'LAYER') then
    // layer00 .. layer99: the output of a layer
    fProbe(FormatUtf8('%.layer%', [fHeads.Name,
      Utf8ToInteger(copy(aName, 6, 2), 0)]), aValues, aRows, aCols);
end;

procedure TQwenLm.Forward(x: PSingleArray; aRows: PtrInt; aOutput: PtrInt;
  aLogits, aHidden: PSingleArray);
begin
  if aRows <= 0 then
    exit;
  if fDevice <> nil then
    fDevice.ForwardEmbeds(fCtx, x, aRows, nil, false, pointer(fLast))
  else if Assigned(fProbe) then
    fModel.ForwardEmbeds(fCtx, x, aRows, nil, false, OnState, pointer(fLast))
  else
    fModel.ForwardEmbeds(fCtx, x, aRows, nil, false, nil, pointer(fLast));
  if aHidden <> nil then
    MoveFast(fLast[0], aHidden^[0], fHeads.Hidden * SizeOf(single));
  if aLogits <> nil then
    fCpu.Linear(pointer(fLast), 1, fHeads.Hidden, fHeads.Head(aOutput),
      fHeads.Rows(aOutput), nil, aLogits);
  inc(fLen, aRows);
end;

end.
