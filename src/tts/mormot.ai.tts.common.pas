/// Text-to-speech - what the engines share: text chunks, model folders
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
unit mormot.ai.tts.common;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text;

/// split a text into chunks said one after the other: a line or a sentence,
// grouped up to aMax bytes (aFirstMax for the first one, short so that the
// sound starts soon); bullets (- * +) are dropped; a line without final
// punctuation gets one, for the pause
function TtsSplitText(const aText: RawUtf8; aFirstMax, aMax: integer): TRawUtf8DynArray;

/// the folder of a model in a Hugging Face cache (aHfHome/hub): the revision
// refs/main points to; '' if missing
function TtsHfSnapshot(const aHfHome: TFileName; const aRepo: RawUtf8): TFileName;

/// the snapshot of aRepo in a Hugging Face cache: '.hf' next to the
// executable, then HF_HOME, then the user's ~/.cache/huggingface; '' if none
// has it
function TtsModelFolder(const aRepo: RawUtf8): TFileName;


implementation

function TtsSplitText(const aText: RawUtf8; aFirstMax, aMax: integer): TRawUtf8DynArray;
var
  sentences: TRawUtf8DynArray;
  n, count, i, j, k, L, limit: PtrInt;
  line, chunk: RawUtf8;

  procedure AddSentence(const aSentence: RawUtf8);
  begin
    if aSentence = '' then
      exit;
    SetLength(sentences, n + 1);
    sentences[n] := aSentence;
    inc(n);
  end;

  procedure SplitLine(const aLine: RawUtf8);
  var
    p, start, len: PtrInt;
  begin
    len := length(aLine);
    start := 1;
    for p := 1 to len do
      if (aLine[p] in ['.', '!', '?', ';']) and ((p = len) or (aLine[p + 1] = ' ')) then
      begin
        AddSentence(TrimU(copy(aLine, start, p - start + 1)));
        start := p + 1;
      end;
    if start <= len then
      AddSentence(TrimU(copy(aLine, start, len - start + 1)));
  end;

  procedure AddChunk;
  begin
    SetLength(result, count + 1);
    result[count] := chunk;
    inc(count);
    chunk := '';
  end;

begin
  result := nil;
  sentences := nil;
  n := 0;
  L := length(aText);
  i := 1;
  while i <= L do
  begin
    j := i;
    while (j <= L) and (aText[j] <> #10) do
      inc(j);
    line := TrimU(copy(aText, i, j - i));
    i := j + 1;
    if (length(line) > 1) and (line[1] in ['-', '*', '+']) and (line[2] = ' ') then
      line := TrimU(copy(line, 3, maxInt));
    if line = '' then
      continue;
    if not (line[length(line)] in ['.', '!', '?', ';', ':', ',']) then
      line := line + '.';
    SplitLine(line);
  end;
  count := 0;
  chunk := '';
  for k := 0 to n - 1 do
  begin
    if count = 0 then
      limit := aFirstMax
    else
      limit := aMax;
    if (chunk <> '') and (length(chunk) + 1 + length(sentences[k]) > limit) then
      AddChunk;
    if chunk = '' then
      chunk := sentences[k]
    else
      chunk := chunk + ' ' + sentences[k];
  end;
  if chunk <> '' then
    AddChunk;
end;

function TtsHfSnapshot(const aHfHome: TFileName; const aRepo: RawUtf8): TFileName;
var
  folder: TFileName;
  rev: RawUtf8;
begin
  result := '';
  folder := IncludeTrailingPathDelimiter(aHfHome) + 'hub' + PathDelim +
    Utf8ToString('models--' + StringReplaceAll(aRepo, '/', '--')) + PathDelim;
  rev := TrimU(StringFromFile(folder + 'refs' + PathDelim + 'main'));
  if rev = '' then
    exit;
  folder := folder + 'snapshots' + PathDelim + Utf8ToString(rev) + PathDelim;
  if DirectoryExists(folder) then
    result := folder;
end;

function TtsModelFolder(const aRepo: RawUtf8): TFileName;
var
  home: TFileName;
begin
  result := TtsHfSnapshot(Executable.ProgramFilePath + '.hf', aRepo);
  if result <> '' then
    exit;
  home := GetEnvironmentVariable('HF_HOME');
  if home <> '' then
  begin
    result := TtsHfSnapshot(home, aRepo);
    if result <> '' then
      exit;
  end;
  home := GetEnvironmentVariable({$ifdef OSWINDOWS}'USERPROFILE'{$else}'HOME'{$endif});
  if home <> '' then
    result := TtsHfSnapshot(IncludeTrailingPathDelimiter(home) + '.cache' +
      PathDelim + 'huggingface', aRepo);
end;

end.
