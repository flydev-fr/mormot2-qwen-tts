/// Qwen3-TTS tests - the model, the PyTorch references and their helpers
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the model: QWEN_TTS_MODEL (a snapshot folder), or the Hugging Face cache
// (QwenTtsModelFolder); the references of tools/tts/qwen_ref.py:
// QWEN_TTS_REF/<name>/manifest.json, or ref/<name> next to the executable;
// without them, the tests are skipped
// - the small embeddings -> logits dataset (tests/data/infer-ref/qwen3-tts-lm)
// is in the repository: only the model is needed for it
unit test.tts.qwen.ref;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.ai.tts.wave,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen;

type
  /// one reference: its manifest, its 16 x N codes
  TQwenRef = record
    Folder: TFileName;
    Doc: TDocVariantData;
    Frames: PtrInt;
    Codes: TIntegerDynArray;
  end;

  /// the outputs collected by the Probe of a component, as rows = time steps,
  // the successive calls of one name appended
  TQwenProbeLog = class
  public
    Names: TRawUtf8DynArray;
    Data: array of TSingleDynArray;
    Cols: TIntegerDynArray;
    procedure Clear;
    procedure Collect(const aName: RawUtf8; aData: PSingleArray; aRows, aCols: PtrInt);
    /// the index of aName, -1 if it received nothing
    function Find(const aName: RawUtf8): PtrInt;
  end;

/// the model snapshot folder, with its final delimiter; '' if none
function QwenTestModel: TFileName;

/// the folder of the references, with its final delimiter
function QwenTestRefRoot: TFileName;

/// a console line for a skipped test without model
function QwenNoModelText: string;

/// the CPU threads of the decoder and the engine: QWEN_THREADS, to measure
// them; 0 without it (two thirds of the cores)
function QwenTestThreads: integer;

/// load <root>/aName/manifest.json and its codes; false if missing
function QwenLoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;

/// the tensor aName of the reference (float32), nil if missing
function QwenRefTensor(const aRef: TQwenRef; const aName: RawUtf8;
  out aShape: TInt64DynArray): TSingleDynArray;

/// max |a - b| and max |b| over aCount values
procedure QwenDiff(a, b: PSingleArray; aCount: PtrInt; out aDiff, aPeak: double);

/// write a 16-bit PCM mono WAV at 24 kHz (to listen to a result)
procedure QwenSaveWav(const aFileName: TFileName; const aPcm: TSingleDynArray);


implementation

function QwenTestModel: TFileName;
begin
  result := GetEnvironmentVariable('QWEN_TTS_MODEL');
  if result = '' then
    result := QwenTtsModelFolder;
  if result <> '' then
    result := IncludeTrailingPathDelimiter(result);
end;

function QwenTestRefRoot: TFileName;
begin
  result := GetEnvironmentVariable('QWEN_TTS_REF');
  if result = '' then
    result := Executable.ProgramFilePath + 'ref';
  result := IncludeTrailingPathDelimiter(result);
end;

function QwenNoModelText: string;
begin
  result := Utf8ToString(FormatUtf8('skipped: no % model (QWEN_TTS_MODEL or HF_HOME)',
    [QWEN_TTS_REPO]));
end;

function QwenTestThreads: integer;
begin
  result := Utf8ToInteger(StringToUtf8(GetEnvironmentVariable('QWEN_THREADS')), 0);
end;

function QwenLoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;
var
  raw: RawByteString;
  e: PDocVariantData;
begin
  result := false;
  aRef.Folder := QwenTestRefRoot + Utf8ToString(aName) + PathDelim;
  if not aRef.Doc.InitJsonFromFile(aRef.Folder + 'manifest.json', JSON_FAST) then
    exit;
  e := aRef.Doc.O['tensors']^.O['codes'];
  aRef.Frames := e^.A['shape']^.Values[0];
  raw := StringFromFile(aRef.Folder + Utf8ToString(e^.U['file']));
  if length(raw) <> aRef.Frames * QWEN_CODE_GROUPS * 4 then
    exit;
  SetLength(aRef.Codes, aRef.Frames * QWEN_CODE_GROUPS);
  MoveFast(pointer(raw)^, pointer(aRef.Codes)^, length(raw));
  result := true;
end;

function QwenRefTensor(const aRef: TQwenRef; const aName: RawUtf8;
  out aShape: TInt64DynArray): TSingleDynArray;
var
  e, s: PDocVariantData;
  raw: RawByteString;
  i: PtrInt;
begin
  result := nil;
  aShape := nil;
  e := aRef.Doc.O['tensors']^.O[aName];
  if e^.Count = 0 then
    exit;
  s := e^.A['shape'];
  SetLength(aShape, s^.Count);
  for i := 0 to s^.Count - 1 do
    aShape[i] := s^.Values[i];
  raw := StringFromFile(aRef.Folder + Utf8ToString(e^.U['file']));
  SetLength(result, length(raw) shr 2);
  MoveFast(pointer(raw)^, pointer(result)^, length(raw));
end;

procedure QwenDiff(a, b: PSingleArray; aCount: PtrInt; out aDiff, aPeak: double);
var
  i: PtrInt;
  d: double;
begin
  aDiff := 0;
  aPeak := 0;
  for i := 0 to aCount - 1 do
  begin
    d := abs(a^[i] - b^[i]);
    if d > aDiff then
      aDiff := d;
    d := abs(b^[i]);
    if d > aPeak then
      aPeak := d;
  end;
end;

procedure QwenSaveWav(const aFileName: TFileName; const aPcm: TSingleDynArray);
begin
  FileFromString(TtsWav16(pointer(aPcm), length(aPcm), QWEN_SAMPLE_RATE), aFileName);
end;


{ TQwenProbeLog }

procedure TQwenProbeLog.Clear;
begin
  Names := nil;
  Data := nil;
  Cols := nil;
end;

function TQwenProbeLog.Find(const aName: RawUtf8): PtrInt;
begin
  for result := 0 to high(Names) do
    if Names[result] = aName then
      exit;
  result := -1;
end;

procedure TQwenProbeLog.Collect(const aName: RawUtf8; aData: PSingleArray;
  aRows, aCols: PtrInt);
var
  i, n: PtrInt;
begin
  i := Find(aName);
  if i < 0 then
  begin
    i := length(Names);
    SetLength(Names, i + 1);
    SetLength(Data, i + 1);
    SetLength(Cols, i + 1);
    Names[i] := aName;
    Cols[i] := aCols;
  end;
  n := length(Data[i]);
  SetLength(Data[i], n + aRows * aCols);
  MoveFast(aData^[0], Data[i][n], aRows * aCols * SizeOf(single));
end;

end.
