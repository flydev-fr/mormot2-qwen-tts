/// Kokoro TTS - the native Kokoro-82M engine, no Python
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the text is cut into chunks (TtsSplitText, the first one short so that
// the sound starts soon); each one becomes phonemes through espeak-ng
// (mormot.ai.tts.espeak, as misaki does), then goes through the model
// (mormot.ai.tts.kokoro.model, on the CPU) with the style of the voice; its
// 24 kHz PCM sound goes to the audio output queue (TTtsPcmStream) and plays
// while the next chunk is computed
// - the voice tells the language: the first letter of its name (f = French,
// e = Spanish, i = Italian, p = Brazilian Portuguese, h = Hindi; a and b,
// English, go through espeak-ng here, not through misaki's dictionary)
unit mormot.ai.tts.kokoro;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.log,
  mormot.core.perf,
  mormot.core.search,
  mormot.ai.tts.common,
  mormot.ai.tts.wave,
  mormot.ai.tts.espeak,
  mormot.ai.tts.kokoro.model;

const
  KOKORO_TTS_REPO = 'hexgrad/Kokoro-82M';
  KOKORO_TTS_DEFAULT_VOICE = 'ff_siwis';
  /// the text chunks, in bytes: the first one short, for the first sound
  KOKORO_TTS_FIRST_CHUNK = 100;
  KOKORO_TTS_CHUNK = 250;
  /// beyond it, the first chunk is cut at its last comma before this limit,
  // after at least KOKORO_TTS_FIRST_MIN bytes
  KOKORO_TTS_FIRST_CUT = 50;
  KOKORO_TTS_FIRST_MIN = 6;

type
  EKokoroTts = class(ESynException);

  /// the settings of the engine
  TKokoroTtsSettings = record
    /// the model folder (config.json, kokoro-v1_0.pth, voices); empty:
    // KokoroTtsModelFolder
    ModelFolder: TFileName;
    /// the espeak-ng folder (espeak-ng.dll, espeak-ng-data); empty:
    // espeak-ng next to the executable
    EspeakFolder: TFileName;
    /// ff_siwis, af_heart... when the text does not say
    Voice: RawUtf8;
    /// 1: normal; higher, faster
    Speed: double;
    /// the CPU threads, 0 = two thirds of the cores
    Threads: integer;
  end;

  /// receives the sound as it comes: aCount samples at 24 kHz, between -1 and
  // 1; returns false to cut
  TOnKokoroPcm = function(aPcm: PSingleArray; aCount: PtrInt): boolean of object;

  /// receives the measure lines (load, each text said)
  TOnKokoroTtsLog = procedure(const aLine: RawUtf8);

  /// the measures of a text said
  TKokoroTtsStats = record
    Chunks: PtrInt;
    Phonemes: PtrInt;
    AudioSeconds: double;
    /// espeak-ng + the model, without waiting for the playback
    ComputeSeconds: double;
    /// from the start to the first sound returned
    FirstSoundMs: double;
    Aborted: boolean;
  end;

  /// the text-to-speech engine: Open, then Speak (Windows audio output) or
  // Synthesize (any sink)
  // - one text at a time: call it from a single thread
  TKokoroTts = class
  protected
    fSettings: TKokoroTtsSettings;
    fSnapshot: TFileName;
    fModel: TKokoroModel;
    fVoices: array of TKokoroVoice;
    fEspeaks: array of TTtsEspeak;
    fLast: TKokoroTtsStats;
    fLoadSeconds: double;
    fOnLog: TOnKokoroTtsLog;
    {$ifdef OSWINDOWS}
    fStream: TTtsPcmStream;
    fPcm16: array of SmallInt;
    function Play(aPcm: PSingleArray; aCount: PtrInt): boolean;
    {$endif OSWINDOWS}
    procedure Log(const aLine: RawUtf8);
    function VoiceOf(const aName: RawUtf8): TKokoroVoice;
    function EspeakOf(const aVoice: RawUtf8): TTtsEspeak;
  public
    constructor Create(const aSettings: TKokoroTtsSettings);
    destructor Destroy; override;
    /// kokoro:cpu
    function Name: RawUtf8;
    /// load the model, espeak-ng and the default voice; called by the first
    // text if needed
    procedure Open;
    procedure Close;
    {$ifdef OSWINDOWS}
    /// say aText on the default audio output, and return at its end, or as
    // soon as aAbort returns true; aVoice: '' for the voice of the settings
    procedure Speak(const aText: RawUtf8; const aVoice: RawUtf8 = '';
      const aAbort: TOnTtsAbort = nil);
    {$endif OSWINDOWS}
    /// say aText into aSink, chunk by chunk; returns the measures
    function Synthesize(const aText, aVoice: RawUtf8; const aSink: TOnKokoroPcm;
      const aAbort: TOnTtsAbort = nil): TKokoroTtsStats;
    /// the voices of the model (voices/*.pt), once opened
    function Voices: TRawUtf8DynArray;
    property Settings: TKokoroTtsSettings
      read fSettings;
    property Model: TKokoroModel
      read fModel;
    /// the measures of the last text
    property Last: TKokoroTtsStats
      read fLast;
    /// the duration of the last Open
    property LoadSeconds: double
      read fLoadSeconds;
    /// the measure lines, in addition to TSynLog; nil by default
    property OnLog: TOnKokoroTtsLog
      read fOnLog write fOnLog;
  end;

/// the default settings: ff_siwis, speed 1
function KokoroTtsDefaults: TKokoroTtsSettings;

/// the model snapshot in a Hugging Face cache: '.hf' next to the executable,
// then HF_HOME, then the user's ~/.cache/huggingface; '' if none has it
function KokoroTtsModelFolder: TFileName;

/// the espeak-ng language of a Kokoro voice: fr-fr for ff_siwis...; '' for
// a language not supported here (Japanese, Chinese)
function KokoroTtsLanguage(const aVoice: RawUtf8): RawUtf8;


implementation

const
  ABORTED_TEXT: array[boolean] of RawUtf8 = ('', ' (cut)');

function Fixed2(aValue: double): RawUtf8;
begin
  result := FormatUtf8('%', [round(aValue * 100) / 100]);
end;

function KokoroTtsDefaults: TKokoroTtsSettings;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.Voice := KOKORO_TTS_DEFAULT_VOICE;
  result.Speed := 1;
end;

function KokoroTtsModelFolder: TFileName;
begin
  result := TtsModelFolder(KOKORO_TTS_REPO);
end;

/// the first chunk cut at a comma if it is long: the first sound waits for
// the whole computation of the first chunk (the model does not stream)
procedure SplitFirst(var aChunks: TRawUtf8DynArray);
var
  first: RawUtf8;
  p, cut, i: PtrInt;
  r: TRawUtf8DynArray;
begin
  if aChunks = nil then
    exit;
  first := aChunks[0];
  if length(first) <= KOKORO_TTS_FIRST_CUT then
    exit;
  cut := 0;
  for p := KOKORO_TTS_FIRST_MIN to KOKORO_TTS_FIRST_CUT do
    if (first[p] = ',') and (p < length(first)) and (first[p + 1] = ' ') then
      cut := p; // the last comma before the limit
  if cut = 0 then
    exit;
  SetLength(r, length(aChunks) + 1);
  r[0] := copy(first, 1, cut);
  r[1] := TrimU(copy(first, cut + 1, maxInt));
  for i := 1 to high(aChunks) do
    r[i + 1] := aChunks[i];
  aChunks := r;
end;

function KokoroTtsLanguage(const aVoice: RawUtf8): RawUtf8;
begin
  // the lang_code values of kokoro.pipeline (LANG_CODES)
  if aVoice = '' then
    result := 'fr-fr'
  else
    case aVoice[1] of
      'a':
        result := 'en-us';
      'b':
        result := 'en-gb';
      'e':
        result := 'es';
      'f':
        result := 'fr-fr';
      'h':
        result := 'hi';
      'i':
        result := 'it';
      'p':
        result := 'pt-br';
    else
      result := '';
    end;
end;


{ TKokoroTts }

constructor TKokoroTts.Create(const aSettings: TKokoroTtsSettings);
begin
  inherited Create;
  fSettings := aSettings;
  if fSettings.Voice = '' then
    fSettings.Voice := KOKORO_TTS_DEFAULT_VOICE;
  if fSettings.Speed <= 0 then
    fSettings.Speed := 1;
end;

destructor TKokoroTts.Destroy;
begin
  Close;
  inherited Destroy;
end;

function TKokoroTts.Name: RawUtf8;
begin
  result := 'kokoro:cpu';
end;

procedure TKokoroTts.Log(const aLine: RawUtf8);
begin
  TSynLog.Add.Log(sllInfo, aLine, self);
  if Assigned(fOnLog) then
    fOnLog(aLine);
end;

function TKokoroTts.VoiceOf(const aName: RawUtf8): TKokoroVoice;
var
  i: PtrInt;
begin
  for i := 0 to high(fVoices) do
    if fVoices[i].Name = aName then
    begin
      result := fVoices[i];
      exit;
    end;
  if not FileExists(fSnapshot + 'voices' + PathDelim + Utf8ToString(aName) + '.pt') then
    raise EKokoroTts.CreateUtf8('unknown Kokoro voice "%"', [aName]);
  result := TKokoroVoice.Create(fSnapshot, aName);
  SetLength(fVoices, length(fVoices) + 1);
  fVoices[high(fVoices)] := result;
end;

function TKokoroTts.EspeakOf(const aVoice: RawUtf8): TTtsEspeak;
var
  lang: RawUtf8;
  folder: TFileName;
  i: PtrInt;
begin
  lang := KokoroTtsLanguage(aVoice);
  if lang = '' then
    raise EKokoroTts.CreateUtf8('Kokoro voice "%": language not supported here', [aVoice]);
  for i := 0 to high(fEspeaks) do
    if fEspeaks[i].Language = lang then
    begin
      result := fEspeaks[i];
      exit;
    end;
  // espeak-ng has one voice at a time: instances per language would share
  // the library, the last one chosen winning; so the voice is switched
  folder := fSettings.EspeakFolder;
  if folder = '' then
    folder := Executable.ProgramFilePath + 'espeak-ng';
  if fEspeaks <> nil then
  begin
    // a single instance: the same library, another language
    for i := 0 to high(fEspeaks) do
      fEspeaks[i].Free;
    fEspeaks := nil;
  end;
  result := TTtsEspeak.Create(folder, lang);
  SetLength(fEspeaks, 1);
  fEspeaks[0] := result;
end;

function TKokoroTts.Voices: TRawUtf8DynArray;
var
  files: TFileNameDynArray;
  i: PtrInt;
begin
  result := nil;
  if fSnapshot = '' then
    exit;
  files := FileNames(fSnapshot + 'voices', '*.pt', [ffoSortByName]);
  SetLength(result, length(files));
  for i := 0 to high(files) do
    result[i] := StringToUtf8(ChangeFileExt(ExtractFileName(files[i]), ''));
end;

procedure TKokoroTts.Open;
var
  timer: TPrecisionTimer;
  folder: TFileName;
begin
  if fModel <> nil then
    exit;
  timer.Start;
  folder := fSettings.ModelFolder;
  if folder = '' then
    folder := KokoroTtsModelFolder;
  if (folder = '') or not FileExists(IncludeTrailingPathDelimiter(folder) +
     'kokoro-v1_0.pth') then
    raise EKokoroTts.CreateUtf8('no % model found (set ModelFolder or HF_HOME)',
      [KOKORO_TTS_REPO]);
  fSnapshot := IncludeTrailingPathDelimiter(folder);
  fModel := TKokoroModel.Create(fSnapshot, fSettings.Threads);
  VoiceOf(fSettings.Voice);
  EspeakOf(fSettings.Voice);
  fLoadSeconds := timer.StopInMicroSec / 1e6;
  Log(FormatUtf8('% loaded in %s, voice %, % threads', [Name, Fixed2(fLoadSeconds),
    fSettings.Voice, fModel.Cpu.Threads]));
end;

procedure TKokoroTts.Close;
var
  i: PtrInt;
begin
  {$ifdef OSWINDOWS}
  FreeAndNil(fStream);
  {$endif OSWINDOWS}
  for i := 0 to high(fVoices) do
    fVoices[i].Free;
  fVoices := nil;
  for i := 0 to high(fEspeaks) do
    fEspeaks[i].Free;
  fEspeaks := nil;
  FreeAndNil(fModel);
end;

function TKokoroTts.Synthesize(const aText, aVoice: RawUtf8;
  const aSink: TOnKokoroPcm; const aAbort: TOnTtsAbort): TKokoroTtsStats;
var
  st: TKokoroTtsStats;
  chunks: TRawUtf8DynArray;
  voiceName, ps: RawUtf8;
  voice: TKokoroVoice;
  espeak: TTtsEspeak;
  ids: TIntegerDynArray;
  pcm: TSingleDynArray;
  count: PtrInt;
  start, now: Int64;
  c: PtrInt;
  u: UnicodeString;
begin
  FillCharFast(st, SizeOf(st), 0);
  if fModel = nil then
    Open;
  voiceName := aVoice;
  if voiceName = '' then
    voiceName := fSettings.Voice;
  voice := VoiceOf(voiceName);
  espeak := EspeakOf(voiceName);
  chunks := TtsSplitText(aText, KOKORO_TTS_FIRST_CHUNK, KOKORO_TTS_CHUNK);
  SplitFirst(chunks);
  QueryPerformanceMicroSeconds(start);
  for c := 0 to high(chunks) do
  begin
    if Assigned(aAbort) and aAbort() then
    begin
      st.Aborted := true;
      break;
    end;
    ps := espeak.Kokoro(chunks[c]);
    u := Utf8ToString(ps);
    if u = '' then
      continue;
    if length(u) > KOKORO_MAX_PHONEMES then
    begin
      // ALBERT's context: TtsSplitText usually cuts shorter
      TSynLog.Add.Log(sllWarning, 'kokoro: % phonemes, cut to %',
        [length(u), KOKORO_MAX_PHONEMES], self);
      SetLength(u, KOKORO_MAX_PHONEMES);
      ps := StringToUtf8(u);
    end;
    ids := fModel.Ids(ps);
    pcm := fModel.Forward(ids, voice.Style(length(u)), fSettings.Speed);
    count := length(pcm);
    if st.Chunks = 0 then
    begin
      QueryPerformanceMicroSeconds(now);
      st.FirstSoundMs := (now - start) / 1000;
    end;
    inc(st.Chunks);
    inc(st.Phonemes, length(u));
    st.AudioSeconds := st.AudioSeconds + count / KOKORO_SAMPLE_RATE;
    if not aSink(pointer(pcm), count) then
    begin
      st.Aborted := true;
      break;
    end;
  end;
  QueryPerformanceMicroSeconds(now);
  st.ComputeSeconds := (now - start) / 1e6;
  fLast := st;
  result := st;
end;

{$ifdef OSWINDOWS}

function TKokoroTts.Play(aPcm: PSingleArray; aCount: PtrInt): boolean;
begin
  if length(fPcm16) < aCount then
    SetLength(fPcm16, aCount);
  TtsFloatToPcm16(aPcm, aCount, pointer(fPcm16));
  fStream.Write(pointer(fPcm16), aCount * SizeOf(SmallInt));
  result := true;
end;

procedure TKokoroTts.Speak(const aText, aVoice: RawUtf8; const aAbort: TOnTtsAbort);
var
  stats: TKokoroTtsStats;
begin
  if fModel = nil then
    Open;
  if fStream = nil then
    fStream := TTtsPcmStream.Create(KOKORO_SAMPLE_RATE);
  try
    stats := Synthesize(aText, aVoice, Play, aAbort);
    if not stats.Aborted then
      fStream.Drain(aAbort);
  finally
    fStream.Stop;
  end;
  Log(FormatUtf8('%s of audio in %s (x% real time), first sound % ms, % chunk(s)%: %',
    [Fixed2(stats.AudioSeconds), Fixed2(stats.ComputeSeconds),
     Fixed2(stats.AudioSeconds / (stats.ComputeSeconds + 1e-9)),
     round(stats.FirstSoundMs), stats.Chunks, ABORTED_TEXT[stats.Aborted],
     copy(aText, 1, 60)]));
end;

{$endif OSWINDOWS}

end.
