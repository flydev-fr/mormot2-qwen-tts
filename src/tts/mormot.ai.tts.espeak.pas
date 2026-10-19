/// Text-to-speech - text to IPA phonemes through espeak-ng, as Kokoro wants them
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - replays the Python chain Kokoro uses outside English: misaki's
// EspeakG2P (fr-fr...), which calls phonemizer (EspeakBackend,
// preserve_punctuation, with_stress, tie '^', language_switch remove-flags)
// then replaces a few phoneme sequences
// - phonemizer hides punctuation from espeak: it cuts the text at each run
// of marks (;:,.!?...), phonemizes the pieces and puts the marks back
// (Punctuation.preserve / restore, ported here as they are)
// - espeak_TextToPhonemes returns one clause per call, in IPA, the phonemes
// of one sound tied by U+0361 (replaced by '^', then by misaki's letters:
// 'a^I' -> 'I'...)
// - espeak-ng (GPL) is loaded at runtime: espeak-ng.dll and espeak-ng-data,
// the ones the espeakng-loader Python package ships, in one folder
// - not to be shared between threads: espeak keeps a global state
unit mormot.ai.tts.espeak;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text;

const
  /// the espeak-ng library looked for in the folder given to TTtsEspeak
  ESPEAK_LIBRARY =
    {$ifdef OSWINDOWS} 'espeak-ng.dll' {$else}
    {$ifdef OSDARWIN} 'libespeak-ng.dylib' {$else}
    'libespeak-ng.so.1' {$endif} {$endif};

type
  ETtsEspeak = class(ESynException);

  /// espeak-ng loaded from a folder, one language chosen
  TTtsEspeak = class
  protected
    fLib: TLibHandle;
    fFolder: TFileName;
    fLanguage: RawUtf8;
    fInitialize: function(output, buflength: integer; path: PAnsiChar;
      options: integer): integer; cdecl;
    fListVoices: function(spec: pointer): PPointerArray; cdecl;
    fSetVoiceByName: function(name: PAnsiChar): integer; cdecl;
    fTextToPhonemes: function(textptr: PPointer; textmode,
      phonememode: integer): PAnsiChar; cdecl;
    fTerminate: function: integer; cdecl;
    function VoiceOf(const aLanguage: RawUtf8): RawUtf8;
  public
    /// loads aFolder/espeak-ng.dll and its data aFolder/espeak-ng-data,
    // then the voice of aLanguage (fr-fr, en-us, es...)
    constructor Create(const aFolder: TFileName; const aLanguage: RawUtf8 = 'fr-fr');
    destructor Destroy; override;
    /// the raw espeak output for a piece without punctuation: the clauses
    // joined by spaces, the phonemes of one sound tied by U+0361
    function TextToPhonemes(const aText: RawUtf8): RawUtf8;
    /// what phonemizer returns (before misaki): the punctuation put back,
    // '^' between the phonemes of one sound, a space after each word
    function Phonemize(const aText: RawUtf8): RawUtf8;
    /// Kokoro's phonemes: misaki's EspeakG2P (version 1.0)
    function Kokoro(const aText: RawUtf8): RawUtf8;
    property Language: RawUtf8
      read fLanguage;
  end;


implementation

const
  /// espeakCHARS_UTF8
  ESPEAK_UTF8 = 1;
  /// AUDIO_OUTPUT_SYNCHRONOUS: no sound, the caller waits
  ESPEAK_SYNCHRONOUS = 2;
  /// espeakINITIALIZE_DONT_EXIT: an error returns a code, no exit()
  ESPEAK_DONT_EXIT = $8000;
  /// IPA (2), the phonemes of one sound tied (bit 7) by U+0361 (bits 8..23)
  ESPEAK_IPA_TIE = $02 or (1 shl 7) or ($0361 shl 8);
  TIE = #$0361;

  /// the marks phonemizer hides from espeak (_DEFAULT_MARKS)
  MARKS: UnicodeString = ';:,.!?'#$00A1#$00BF#$2014#$2026'"'#$00AB#$00BB#$201C#$201D'(){}[]';

type
  /// a mark taken out of the text, and where it was: B start, I middle,
  // E end, A alone
  TMark = record
    Text: UnicodeString;
    Position: AnsiChar;
  end;
  TMarks = array of TMark;

function IsMark(c: WideChar): boolean;
begin
  result := Pos(c, MARKS) > 0;
end;

function IsSpace(c: WideChar): boolean;
begin
  // Python's \s on a str: ASCII and the common Unicode spaces
  case ord(c) of
    9..13, 32, $85, $A0, $1680, $2000..$200A, $2028, $2029, $202F, $205F, $3000:
      result := true;
  else
    result := false;
  end;
end;

/// the next match of (\s*[marks]+\s*)+ from aFrom (1-based): true with its
// position and length
function NextMarks(const s: UnicodeString; aFrom: integer;
  out aStart, aLen: integer): boolean;
var
  i, j, k, n: integer;
begin
  n := length(s);
  for i := aFrom to n do
  begin
    j := i;
    while (j <= n) and IsSpace(s[j]) do
      inc(j);
    if (j > n) or not IsMark(s[j]) then
      continue;
    // at least one mark: the longest (\s* marks+ \s*)+ run
    k := i;
    repeat
      j := k;
      while (j <= n) and IsSpace(s[j]) do
        inc(j);
      if (j > n) or not IsMark(s[j]) then
        break;
      while (j <= n) and IsMark(s[j]) do
        inc(j);
      while (j <= n) and IsSpace(s[j]) do
        inc(j);
      k := j;
    until false;
    aStart := i;
    aLen := k - i;
    result := true;
    exit;
  end;
  result := false;
end;

/// Punctuation._preserve_line: the pieces without marks, and the marks
procedure Preserve(const aLine: UnicodeString; out aChunks: TUnicodeStringDynArray;
  out aMarks: TMarks);
var
  starts, lens: TIntegerDynArray;
  n, p, s, l, i, at: integer;
  line: UnicodeString;
begin
  aChunks := nil;
  aMarks := nil;
  n := 0;
  p := 1;
  while NextMarks(aLine, p, s, l) do
  begin
    SetLength(starts, n + 1);
    SetLength(lens, n + 1);
    starts[n] := s;
    lens[n] := l;
    inc(n);
    p := s + l;
  end;
  if n = 0 then
  begin
    aChunks := TUnicodeStringDynArray.Create(aLine);
    exit;
  end;
  if (n = 1) and (lens[0] = length(aLine)) then
  begin
    SetLength(aMarks, 1);
    aMarks[0].Text := aLine;
    aMarks[0].Position := 'A';
    exit;
  end;
  SetLength(aMarks, n);
  for i := 0 to n - 1 do
  begin
    aMarks[i].Text := copy(aLine, starts[i], lens[i]);
    aMarks[i].Position := 'I';
    if (i = 0) and (starts[i] = 1) then
      aMarks[i].Position := 'B'
    else if (i = n - 1) and (starts[i] + lens[i] = length(aLine) + 1) then
      aMarks[i].Position := 'E';
  end;
  // line.split(mark): before the first occurrence, then the rest
  line := aLine;
  SetLength(aChunks, n + 1);
  for i := 0 to n - 1 do
  begin
    at := Pos(aMarks[i].Text, line);
    aChunks[i] := copy(line, 1, at - 1);
    line := copy(line, at + length(aMarks[i].Text), maxInt);
  end;
  aChunks[n] := line;
end;

function EndsWithSpace(const s: UnicodeString): boolean;
begin
  result := (s <> '') and (s[length(s)] = ' ');
end;

/// Punctuation.restore for a single line, word separator ' ', strip false;
// aText: the phonemized pieces (the empty ones already removed)
function Restore(aText: TUnicodeStringDynArray; aMarks: TMarks): UnicodeString;
var
  parts: TUnicodeStringDynArray;
  mark, all: UnicodeString;
  m: TMark;
  i: integer;

  procedure Add(const s: UnicodeString);
  begin
    SetLength(parts, length(parts) + 1);
    parts[high(parts)] := s;
  end;

  procedure Drop(var a: TUnicodeStringDynArray);
  begin
    a := copy(a, 1, maxInt);
  end;

begin
  parts := nil;
  // a single line: every mark at index 0 (pos stays 0 until the end)
  while (aText <> nil) or (aMarks <> nil) do
    if aMarks = nil then
    begin
      for i := 0 to high(aText) do
        if EndsWithSpace(aText[i]) then
          Add(aText[i])
        else
          Add(aText[i] + ' ');
      aText := nil;
    end
    else if aText = nil then
    begin
      all := '';
      for i := 0 to high(aMarks) do
        all := all + aMarks[i].Text;
      Add(all); // spaces stay spaces
      aMarks := nil;
    end
    else
    begin
      m := aMarks[0];
      aMarks := copy(aMarks, 1, maxInt);
      mark := m.Text;
      if EndsWithSpace(aText[0]) then
        SetLength(aText[0], length(aText[0]) - 1);
      case m.Position of
        'B':
          aText[0] := mark + aText[0];
        'E':
          begin
            if EndsWithSpace(mark) then
              Add(aText[0] + mark)
            else
              Add(aText[0] + mark + ' ');
            Drop(aText);
          end;
        'A':
          if EndsWithSpace(mark) then
            Add(mark)
          else
            Add(mark + ' ');
      else
        if length(aText) = 1 then
          aText[0] := aText[0] + mark
        else
        begin
          mark := aText[0] + mark + aText[1];
          Drop(aText);
          aText[0] := mark;
        end;
      end;
    end;
  // phonemize() returns one line per input line: here, the pieces of the
  // only line put end to end
  result := '';
  for i := 0 to high(parts) do
    result := result + parts[i];
end;

/// EspeakBackend._postprocess_line, tie '^', with stress, strip false
function PostProcess(const aLine: UnicodeString): UnicodeString;
var
  line, w: UnicodeString;
  words: TUnicodeStringDynArray;
  i, p, q: integer;
begin
  line := Trim(aLine);
  line := StringReplace(line, #10, ' ', [rfReplaceAll]);
  line := StringReplace(line, '  ', ' ', [rfReplaceAll]);
  while Pos('__', line) > 0 do
    line := StringReplace(line, '__', '_', [rfReplaceAll]);
  line := StringReplace(line, '_ ', ' ', [rfReplaceAll]);
  // remove-flags: the (en), (fr)... espeak writes on a language switch
  repeat
    p := Pos('(', line);
    if p = 0 then
      break;
    q := p + 1;
    while (q <= length(line)) and (line[q] <> ')') do
      inc(q);
    if q > length(line) then
      break;
    if q = p + 1 then
      break; // '()': (.+?) wants at least one character
    Delete(line, p, q - p + 1);
  until false;
  result := '';
  if line = '' then
    exit;
  words := nil;
  p := 1;
  for i := 1 to length(line) + 1 do
    if (i > length(line)) or (line[i] = ' ') then
    begin
      SetLength(words, length(words) + 1);
      words[high(words)] := copy(line, p, i - p);
      p := i + 1;
    end;
  for i := 0 to high(words) do
  begin
    w := Trim(words[i]);
    w := StringReplace(w, TIE, '^', [rfReplaceAll]);
    result := result + w + ' ';
  end;
end;


{ TTtsEspeak }

constructor TTtsEspeak.Create(const aFolder: TFileName; const aLanguage: RawUtf8);
var
  data: RawUtf8;
  voice: RawUtf8;

  procedure Bind(var aProc; const aName: PAnsiChar);
  begin
    PPointer(@aProc)^ := LibraryResolve(fLib, aName);
    if PPointer(@aProc)^ = nil then
      raise ETtsEspeak.CreateUtf8('%: no % in %', [self, aName, ESPEAK_LIBRARY]);
  end;

begin
  inherited Create;
  fFolder := IncludeTrailingPathDelimiter(aFolder);
  fLib := LibraryOpen(fFolder + ESPEAK_LIBRARY);
  if fLib = TLibHandle(0) then
    raise ETtsEspeak.CreateUtf8('%: cannot load %%', [self, fFolder, ESPEAK_LIBRARY]);
  Bind(fInitialize, 'espeak_Initialize');
  Bind(fListVoices, 'espeak_ListVoices');
  Bind(fSetVoiceByName, 'espeak_SetVoiceByName');
  Bind(fTextToPhonemes, 'espeak_TextToPhonemes');
  Bind(fTerminate, 'espeak_Terminate');
  data := StringToUtf8(fFolder + 'espeak-ng-data');
  if fInitialize(ESPEAK_SYNCHRONOUS, 0, pointer(data), ESPEAK_DONT_EXIT) <= 0 then
    raise ETtsEspeak.CreateUtf8('%: espeak_Initialize failed on %', [self, data]);
  // as phonemizer does: the first voice announcing this language
  voice := VoiceOf(aLanguage);
  if voice = '' then
    raise ETtsEspeak.CreateUtf8('%: no espeak voice for %', [self, aLanguage]);
  if fSetVoiceByName(pointer(voice)) <> 0 then
    raise ETtsEspeak.CreateUtf8('%: espeak_SetVoiceByName(%) failed', [self, voice]);
  fLanguage := aLanguage;
end;

destructor TTtsEspeak.Destroy;
begin
  if fLib <> TLibHandle(0) then
  begin
    if Assigned(fTerminate) then
      fTerminate;
    LibraryClose(fLib);
  end;
  inherited Destroy;
end;

function TTtsEspeak.VoiceOf(const aLanguage: RawUtf8): RawUtf8;
type
  // espeak_VOICE from speak_lib.h
  TEspeakVoice = record
    Name, Languages, Identifier: PAnsiChar;
    Gender, Age, Variant, Xx1: byte;
    Score: integer;
    Spare: pointer;
  end;
  PEspeakVoice = ^TEspeakVoice;
var
  list: PPointerArray;
  v: PEspeakVoice;
  p: PAnsiChar;
  i: integer;
begin
  result := '';
  list := fListVoices(nil);
  if list = nil then
    exit;
  i := 0;
  while list^[i] <> nil do
  begin
    v := list^[i];
    // languages: a priority byte then the name, ..., a zero byte
    p := v^.Languages;
    if p <> nil then
    begin
      // phonemizer only reads the first language of each voice
      inc(p);
      if p^ <> #0 then
        if IdemPropNameU(aLanguage, PUtf8Char(p), StrLen(p)) then
        begin
          result := v^.Identifier;
          exit;
        end;
    end;
    inc(i);
  end;
end;

function TTtsEspeak.TextToPhonemes(const aText: RawUtf8): RawUtf8;
var
  text: RawUtf8;
  p: pointer;
  r: PAnsiChar;
begin
  result := '';
  text := aText; // espeak moves through the text: our own copy
  UniqueRawUtf8(text);
  p := pointer(text);
  while p <> nil do
  begin
    r := fTextToPhonemes(@p, ESPEAK_UTF8, ESPEAK_IPA_TIE);
    if (r <> nil) and (r^ <> #0) then
    begin
      if result <> '' then
        result := result + ' ';
      result := result + RawUtf8(r);
    end;
  end;
end;

function TTtsEspeak.Phonemize(const aText: RawUtf8): RawUtf8;
var
  chunks, done: TUnicodeStringDynArray;
  marks: TMarks;
  line: UnicodeString;
  i: integer;
begin
  Preserve(Utf8ToString(aText), chunks, marks);
  done := nil;
  for i := 0 to high(chunks) do
    if chunks[i] <> '' then
    begin
      // one phonemizer line per piece, empty if espeak returns nothing
      line := PostProcess(Utf8ToString(TextToPhonemes(StringToUtf8(chunks[i]))));
      SetLength(done, length(done) + 1);
      done[high(done)] := line;
    end;
  result := StringToUtf8(Restore(done, marks));
end;

function TTtsEspeak.Kokoro(const aText: RawUtf8): RawUtf8;
const
  // misaki's EspeakG2P.e2m, sorted by key as sorted(dict.items())
  E2M: array[0..10, 0..1] of UnicodeString = (
    ('a^'#$026A, 'I'), ('a^'#$028A, 'W'), ('d^z', #$02A3), ('d^'#$0292, #$02A4),
    ('e^'#$026A, 'A'), ('o^'#$028A, 'O'), ('s^s', 'S'), ('t^s', #$02A6),
    ('t^'#$0283, #$02A7), (#$0254'^'#$026A, 'Y'), (#$0259'^'#$028A, 'Q'));
var
  text, ps: UnicodeString;
  i: integer;
begin
  text := Utf8ToString(aText);
  // angle quotes become curly ones, parentheses become angle quotes:
  // espeak reads parentheses, phonemizer keeps angle quotes
  text := StringReplace(text, #$00AB, #$201C, [rfReplaceAll]);
  text := StringReplace(text, #$00BB, #$201D, [rfReplaceAll]);
  text := StringReplace(text, '(', #$00AB, [rfReplaceAll]);
  text := StringReplace(text, ')', #$00BB, [rfReplaceAll]);
  ps := Trim(Utf8ToString(Phonemize(StringToUtf8(text))));
  for i := 0 to high(E2M) do
    ps := StringReplace(ps, E2M[i, 0], E2M[i, 1], [rfReplaceAll]);
  ps := StringReplace(ps, '^', '', [rfReplaceAll]);
  ps := StringReplace(ps, '-', '', [rfReplaceAll]);
  ps := StringReplace(ps, #$00AB, '(', [rfReplaceAll]);
  ps := StringReplace(ps, #$00BB, ')', [rfReplaceAll]);
  result := StringToUtf8(ps);
end;

end.
