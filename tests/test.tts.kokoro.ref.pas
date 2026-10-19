/// Kokoro TTS tests - the model, espeak-ng, the PyTorch references
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the model: KOKORO_TTS_MODEL (a snapshot folder), or the Hugging Face
// cache (KokoroTtsModelFolder); espeak-ng: KOKORO_ESPEAK (the folder of
// espeak-ng.dll and espeak-ng-data), or espeak-ng next to the executable;
// the references of tools/tts/kokoro_ref.py: KOKORO_TTS_REF/<name>/
// manifest.json, or ref-kokoro/<name> next to the executable; without them,
// the tests are skipped
unit test.tts.kokoro.ref;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.ai.tts.espeak,
  mormot.ai.tts.kokoro;

const
  /// the reference sentences, in the order kokoro_ref.py writes them
  KOKORO_REFS: array[0..3] of RawUtf8 = (
    'fr-siwis', 'fr-long', 'fr-accents', 'fr-numbers');

type
  /// one reference: its manifest
  TKokoroRef = record
    Folder: TFileName;
    Doc: TDocVariantData;
  end;

/// the model snapshot folder, with its final delimiter; '' if none
function KokoroTestSnapshot: TFileName;

/// the espeak-ng folder (espeak-ng.dll, espeak-ng-data), with its final
// delimiter; '' if missing
function KokoroTestEspeak: TFileName;

/// the folder of the references, with its final delimiter
function KokoroTestRefRoot: TFileName;

/// a console line for a skipped test without model
function KokoroNoModelText: string;

/// a console line for a skipped test without espeak-ng
function KokoroNoEspeakText: string;

/// load <root>/aName/manifest.json; false if missing
function KokoroLoadRef(const aName: RawUtf8; out aRef: TKokoroRef): boolean;

/// the tensor aName of the reference (float32), nil if missing
function KokoroRefTensor(const aRef: TKokoroRef; const aName: RawUtf8;
  out aShape: TInt64DynArray): TSingleDynArray;

/// the integer tensor aName of the reference, nil if missing
function KokoroRefInts(const aRef: TKokoroRef; const aName: RawUtf8): TIntegerDynArray;


implementation

function KokoroTestSnapshot: TFileName;
begin
  result := GetEnvironmentVariable('KOKORO_TTS_MODEL');
  if result = '' then
    result := KokoroTtsModelFolder;
  if result <> '' then
    result := IncludeTrailingPathDelimiter(result);
end;

function KokoroTestEspeak: TFileName;
begin
  result := GetEnvironmentVariable('KOKORO_ESPEAK');
  if result = '' then
    result := Executable.ProgramFilePath + 'espeak-ng';
  result := IncludeTrailingPathDelimiter(result);
  if not FileExists(result + ESPEAK_LIBRARY) then
    result := '';
end;

function KokoroTestRefRoot: TFileName;
begin
  result := GetEnvironmentVariable('KOKORO_TTS_REF');
  if result = '' then
    result := Executable.ProgramFilePath + 'ref-kokoro';
  result := IncludeTrailingPathDelimiter(result);
end;

function KokoroNoModelText: string;
begin
  result := Utf8ToString(FormatUtf8('skipped: no % model (KOKORO_TTS_MODEL or HF_HOME)',
    [KOKORO_TTS_REPO]));
end;

function KokoroNoEspeakText: string;
begin
  result := Utf8ToString(FormatUtf8('skipped: no % (KOKORO_ESPEAK)', [ESPEAK_LIBRARY]));
end;

function KokoroLoadRef(const aName: RawUtf8; out aRef: TKokoroRef): boolean;
begin
  aRef.Folder := KokoroTestRefRoot + Utf8ToString(aName) + PathDelim;
  result := aRef.Doc.InitJsonFromFile(aRef.Folder + 'manifest.json', JSON_FAST);
end;

function RefRaw(const aRef: TKokoroRef; const aName: RawUtf8;
  out aShape: TInt64DynArray): RawByteString;
var
  e, s: PDocVariantData;
  i: PtrInt;
begin
  result := '';
  aShape := nil;
  e := aRef.Doc.O['tensors']^.O[aName];
  if e^.Count = 0 then
    exit;
  s := e^.A['shape'];
  SetLength(aShape, s^.Count);
  for i := 0 to s^.Count - 1 do
    aShape[i] := s^.Values[i];
  result := StringFromFile(aRef.Folder + Utf8ToString(e^.U['file']));
end;

function KokoroRefTensor(const aRef: TKokoroRef; const aName: RawUtf8;
  out aShape: TInt64DynArray): TSingleDynArray;
var
  raw: RawByteString;
begin
  result := nil;
  raw := RefRaw(aRef, aName, aShape);
  SetLength(result, length(raw) shr 2);
  MoveFast(pointer(raw)^, pointer(result)^, length(raw));
end;

function KokoroRefInts(const aRef: TKokoroRef; const aName: RawUtf8): TIntegerDynArray;
var
  raw: RawByteString;
  shape: TInt64DynArray;
begin
  result := nil;
  raw := RefRaw(aRef, aName, shape);
  SetLength(result, length(raw) shr 2);
  MoveFast(pointer(raw)^, pointer(result)^, length(raw));
end;

end.
