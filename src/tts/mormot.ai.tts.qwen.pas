/// Qwen3-TTS - the native Qwen3-TTS-12Hz-0.6B-CustomVoice engine, no Python
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the talker and the code predictor (mormot.ai.tts.qwen.talker, on the GPU
// through Vulkan or on the CPU) give a frame of 16 codes per 80 ms of audio;
// the 12 Hz decoder (mormot.ai.tts.qwen.codec, its heavy part on the GPU with
// mormot.ai.tts.qwen.codec.vulkan when the talker is there) turns it into
// 1920 samples at 24 kHz, played at once (TTtsPcmStream): the sound starts
// after the first frame, the rest is computed while it plays
// - both work at the same time: the talker on its own thread puts the frames
// into a small queue, the decoder empties it on the speaking thread
// - a long text is said sentence by sentence (QwenSplitText), in the same
// stream; each text logs its audio duration, its compute time and the delay
// of the first sound
unit mormot.ai.tts.qwen;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  classes,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.log,
  mormot.core.perf,
  mormot.ai.infer.tensor,
  mormot.ai.tts.common,
  mormot.ai.tts.wave,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.codec.vulkan,
  mormot.ai.tts.qwen.lm.vulkan,
  mormot.ai.tts.qwen.talker;

const
  QWEN_TTS_REPO = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice';
  QWEN_TTS_DEFAULT_VOICE = 'vivian';
  /// the text chunks, in bytes: sentences, grouped
  QWEN_TTS_FIRST_CHUNK = 200;
  QWEN_TTS_CHUNK = 300;
  /// the frame cap of a chunk, if it never ends (no eos): 50 (4 s of sound),
  // plus 3 per byte of text (10 to 15 bytes are said per second)
  QWEN_TTS_MAX_FRAMES = 50;
  QWEN_TTS_FRAMES_PER_BYTE = 3;

type
  EQwenTts = class(ESynException);

  /// the settings of the engine
  TQwenTtsSettings = record
    /// the model folder (config.json, model.safetensors, speech_tokenizer);
    // empty: QwenTtsModelFolder
    ModelFolder: TFileName;
    /// vivian, ryan, aiden... when the text does not say
    Voice: RawUtf8;
    /// Auto, French, English...
    Language: RawUtf8;
    /// true: the talker on the GPU (Vulkan, Q8_0); false or no GPU: CPU
    UseGpu: boolean;
    /// CPU: false = Q8_0 (fast), true = float32 (the codes of PyTorch)
    CpuFloat32: boolean;
    /// the CPU threads, 0 = two thirds of the cores
    Threads: integer;
    /// 0: one seed per sentence; otherwise always the same voice for a text
    Seed: Int64;
    /// the temperature of the code sampling, talker and code predictor: lower
    // gives a calmer and steadier delivery, higher a more expressive but less
    // reliable one; 0: the one of generation_config.json (0.9)
    Temperature: double;
    /// true: the 12 Hz decoder on the CPU even when the talker is on the GPU
    CpuDecoder: boolean;
  end;

  /// receives the sound as it comes: aCount samples at 24 kHz, between -1 and
  // 1; returns false to cut
  TOnQwenPcm = function(aPcm: PSingleArray; aCount: PtrInt): boolean of object;

  /// receives the measure lines (load, each text said), e.g. for the console;
  // called from the speaking thread
  TOnQwenTtsLog = procedure(const aLine: RawUtf8);

  /// the measures of a text said
  TQwenTtsStats = record
    Frames: PtrInt;
    AudioSeconds: double;
    /// talker + decoder, without waiting for the playback
    ComputeSeconds: double;
    /// from the start to the first sound returned
    FirstSoundMs: double;
    /// the chunks cut at the frame cap, without eos
    Capped: PtrInt;
    Aborted: boolean;
  end;

  /// the text-to-speech engine: Open, then Speak (Windows audio output) or
  // Synthesize (any sink)
  // - one text at a time: call it from a single thread
  TQwenTts = class
  protected
    fSettings: TQwenTtsSettings;
    fTalker: TQwenTalker;
    fCodec: TQwenCodec;
    fLast: TQwenTtsStats;
    fLoadSeconds: double;
    fFramesPerByte: integer;
    fOnLog: TOnQwenTtsLog;
    {$ifdef OSWINDOWS}
    fStream: TTtsPcmStream;
    fPcm16: array of SmallInt;
    function Play(aPcm: PSingleArray; aCount: PtrInt): boolean;
    {$endif OSWINDOWS}
    procedure Log(const aLine: RawUtf8);
  public
    constructor Create(const aSettings: TQwenTtsSettings);
    destructor Destroy; override;
    /// qwen:vulkan, qwen:cpu or qwen:cpu-f32
    function Name: RawUtf8;
    /// load the model (one or two seconds); called by the first text if needed
    procedure Open;
    procedure Close;
    {$ifdef OSWINDOWS}
    /// say aText on the default audio output, and return at its end, or as
    // soon as aAbort returns true; aVoice: '' for the voice of the settings
    procedure Speak(const aText: RawUtf8; const aVoice: RawUtf8 = '';
      const aAbort: TOnTtsAbort = nil);
    {$endif OSWINDOWS}
    /// say aText into aSink, frame by frame; returns the measures
    function Synthesize(const aText, aVoice: RawUtf8; const aSink: TOnQwenPcm;
      const aAbort: TOnTtsAbort = nil): TQwenTtsStats;
    property Settings: TQwenTtsSettings
      read fSettings;
    property Talker: TQwenTalker
      read fTalker;
    property Codec: TQwenCodec
      read fCodec;
    /// the measures of the last text
    property Last: TQwenTtsStats
      read fLast;
    /// the duration of the last Open
    property LoadSeconds: double
      read fLoadSeconds;
    /// the cap of a chunk: QWEN_TTS_MAX_FRAMES + FramesPerByte frames per
    // byte; QWEN_TTS_FRAMES_PER_BYTE by default
    property FramesPerByte: integer
      read fFramesPerByte write fFramesPerByte;
    /// the measure lines, in addition to TSynLog; nil by default
    property OnLog: TOnQwenTtsLog
      read fOnLog write fOnLog;
  end;

/// the default settings: GPU, vivian, Auto
function QwenTtsDefaults: TQwenTtsSettings;

/// the model snapshot in a Hugging Face cache: '.hf' next to the executable,
// then HF_HOME, then the user's ~/.cache/huggingface; '' if none has it
function QwenTtsModelFolder: TFileName;

/// split a text into chunks said one after the other: a line or a sentence,
// grouped up to aMax bytes (aFirstMax for the first one, short so that the
// sound starts soon); bullets (- * +) are dropped; a line without final
// punctuation gets one, for the pause
function QwenSplitText(const aText: RawUtf8; aFirstMax, aMax: integer): TRawUtf8DynArray;


implementation

const
  ABORTED_TEXT: array[boolean] of RawUtf8 = ('', ' (cut)');
  CAPPED_TEXT: array[boolean] of RawUtf8 = ('', ' (no end, capped)');
  /// the frames the talker can be ahead of the decoder: 2.5 s of sound
  QWEN_TTS_QUEUE = 32;

type
  /// a frame waiting for the decoder: its text chunk and its codes
  TQwenTtsQueued = record
    Chunk: PtrInt;
    Codes: array[0..QWEN_CODE_GROUPS - 1] of integer;
  end;

  /// the talker on its own thread, for one text: it puts the frames of each
  // chunk into a small queue that the decoder empties on the speaking thread;
  // the GPU computes the next frame while the CPU decodes
  TQwenTtsProducer = class(TThread)
  protected
    fTalker: TQwenTalker;
    fChunks: TRawUtf8DynArray;
    fVoice, fLanguage: RawUtf8;
    fFramesPerByte: integer;
    fCapped: PtrInt;
    fLock: TLightLock;
    fQueue: array[0..QWEN_TTS_QUEUE - 1] of TQwenTtsQueued;
    // the frames put and taken since the start, the finished chunks
    fTail, fHead, fClosed: PtrInt;
    fFinished, fStopped: boolean;
    fFailure: TObject;
    // wake up the decoder (a frame was put) and the talker (a slot is free)
    fFilled, fFreed: TSynEvent;
    procedure Execute; override;
    function Push(const aFrame: TQwenTtsQueued): boolean;
  public
    /// start the thread; aTalker is not owned, and not to be touched before
    // Finish
    constructor Create(aTalker: TQwenTalker; const aChunks: TRawUtf8DynArray;
      const aVoice, aLanguage: RawUtf8; aFramesPerByte: integer);
    destructor Destroy; override;
    /// aWant frames of the same chunk into aCodes, fewer at the end of the
    // chunk; returns their number and their chunk (aChunk), 0 when all is
    // said, -1 if nothing is ready after aWaitMs
    function Take(aWant: PtrInt; aCodes: PIntegerArray; out aChunk: PtrInt;
      aWaitMs: cardinal): PtrInt;
    /// stop the talker after its running frame, wait for the end of the
    // thread and raise its exception if it had one
    procedure Finish;
    /// the chunks cut at the cap (after Finish)
    property Capped: PtrInt
      read fCapped;
  end;

/// two decimals, for the measures
function Fixed2(aValue: double): RawUtf8;
begin
  result := FormatUtf8('%', [round(aValue * 100) / 100]);
end;

function QwenTtsDefaults: TQwenTtsSettings;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.Voice := QWEN_TTS_DEFAULT_VOICE;
  result.Language := 'Auto';
  result.UseGpu := true;
end;

function QwenTtsModelFolder: TFileName;
begin
  result := TtsModelFolder(QWEN_TTS_REPO);
end;

function QwenSplitText(const aText: RawUtf8; aFirstMax, aMax: integer): TRawUtf8DynArray;
begin
  result := TtsSplitText(aText, aFirstMax, aMax);
end;


{ TQwenTtsProducer }

constructor TQwenTtsProducer.Create(aTalker: TQwenTalker;
  const aChunks: TRawUtf8DynArray; const aVoice, aLanguage: RawUtf8;
  aFramesPerByte: integer);
begin
  fTalker := aTalker;
  fChunks := aChunks;
  fVoice := aVoice;
  fLanguage := aLanguage;
  fFramesPerByte := aFramesPerByte;
  fFilled := TSynEvent.Create;
  fFreed := TSynEvent.Create;
  inherited Create({suspended=}false);
end;

destructor TQwenTtsProducer.Destroy;
begin
  // without Finish (an exception of the decoder): the talker stops too
  fLock.Lock;
  fStopped := true;
  fLock.UnLock;
  fFreed.SetEvent;
  inherited Destroy; // waits for the end of Execute
  fFailure.Free;
  fFilled.Free;
  fFreed.Free;
end;

procedure TQwenTtsProducer.Execute;
var
  c, n, limit: PtrInt;
  frame: TQwenTtsQueued;
begin
  SetCurrentThreadName('qwen talker');
  try
    for c := 0 to high(fChunks) do
    begin
      if fStopped then
        break; // no prefill for nothing
      fTalker.Start(fChunks[c], fVoice, fLanguage);
      frame.Chunk := c;
      limit := QWEN_TTS_MAX_FRAMES + fFramesPerByte * length(fChunks[c]);
      n := 0;
      while fTalker.NextFrame(@frame.Codes) do
      begin
        if not Push(frame) then
          break; // stopped by Finish
        inc(n);
        if n < limit then
          continue;
        // no eos: the talker would speak for minutes for nothing
        inc(fCapped);
        TSynLog.Add.Log(sllWarning, 'qwen: no end after % frames, cut "%"',
          [n, copy(fChunks[c], 1, 60)], self);
        break;
      end;
      if fStopped then
        break;
      // the chunk is finished: its last frames no longer wait for more
      fLock.Lock;
      fClosed := c + 1;
      fLock.UnLock;
      fFilled.SetEvent;
    end;
  except
    fFailure := TObject(AcquireExceptionObject); // raised by Finish
  end;
  fLock.Lock;
  fFinished := true;
  fLock.UnLock;
  fFilled.SetEvent;
end;

function TQwenTtsProducer.Push(const aFrame: TQwenTtsQueued): boolean;
var
  room: boolean;
begin
  repeat
    fLock.Lock;
    result := not fStopped;
    room := fTail - fHead < QWEN_TTS_QUEUE;
    if result and room then
    begin
      fQueue[fTail mod QWEN_TTS_QUEUE] := aFrame;
      inc(fTail);
    end;
    fLock.UnLock;
    if not result then
      exit;
    if room then
    begin
      fFilled.SetEvent;
      exit;
    end;
    // the queue is full: the decoder is late, the GPU waits
    fFreed.WaitFor(100);
  until false;
end;

function TQwenTtsProducer.Take(aWant: PtrInt; aCodes: PIntegerArray;
  out aChunk: PtrInt; aWaitMs: cardinal): PtrInt;
var
  ready, chunk, n, i, attempt: PtrInt;
begin
  result := -1;
  aChunk := -1;
  for attempt := 0 to 1 do
  begin
    fLock.Lock;
    ready := fTail - fHead;
    if ready = 0 then
    begin
      if fFinished then
        result := 0;
    end
    else
    begin
      chunk := fQueue[fHead mod QWEN_TTS_QUEUE].Chunk;
      n := 0;
      while (n < aWant) and (n < ready) and
            (fQueue[(fHead + n) mod QWEN_TTS_QUEUE].Chunk = chunk) do
        inc(n);
      // fewer than wanted: only if the chunk will have no others
      if (n = aWant) or (n < ready) or (chunk < fClosed) or fFinished then
      begin
        for i := 0 to n - 1 do
          MoveFast(fQueue[(fHead + i) mod QWEN_TTS_QUEUE].Codes,
            aCodes^[i * QWEN_CODE_GROUPS], QWEN_CODE_GROUPS * SizeOf(integer));
        inc(fHead, n);
        aChunk := chunk;
        result := n;
      end;
    end;
    fLock.UnLock;
    if result >= 0 then
      break;
    if attempt = 0 then
      fFilled.WaitFor(aWaitMs);
  end;
  if result > 0 then
    fFreed.SetEvent;
end;

procedure TQwenTtsProducer.Finish;
var
  failure: TObject;
begin
  fLock.Lock;
  fStopped := true;
  fLock.UnLock;
  fFreed.SetEvent;
  WaitFor;
  failure := fFailure;
  if failure <> nil then
  begin
    fFailure := nil;
    raise failure;
  end;
end;


{ TQwenTts }

constructor TQwenTts.Create(const aSettings: TQwenTtsSettings);
begin
  inherited Create;
  fSettings := aSettings;
  fFramesPerByte := QWEN_TTS_FRAMES_PER_BYTE;
  if fSettings.Voice = '' then
    fSettings.Voice := QWEN_TTS_DEFAULT_VOICE;
  if fSettings.Language = '' then
    fSettings.Language := 'Auto';
end;

destructor TQwenTts.Destroy;
begin
  Close;
  inherited Destroy;
end;

function TQwenTts.Name: RawUtf8;
begin
  // the same before and after Open
  if fSettings.UseGpu and TQwenVulkan.Available then
    result := 'qwen:vulkan'
  else if fSettings.CpuFloat32 then
    result := 'qwen:cpu-f32'
  else
    result := 'qwen:cpu';
end;

procedure TQwenTts.Log(const aLine: RawUtf8);
begin
  TSynLog.Add.Log(sllInfo, aLine, self);
  if Assigned(fOnLog) then
    fOnLog(aLine);
end;

procedure TQwenTts.Open;
var
  folder: TFileName;
  options: TQwenTalkerOptions;
  timer: TPrecisionTimer;
  s: TQwenSampling;
begin
  if fTalker <> nil then
    exit;
  timer.Start;
  folder := fSettings.ModelFolder;
  if folder = '' then
    folder := QwenTtsModelFolder;
  if (folder = '') or not FileExists(IncludeTrailingPathDelimiter(folder) +
     'model.safetensors') then
    raise EQwenTts.CreateUtf8('no % model found (set ModelFolder or HF_HOME)',
      [QWEN_TTS_REPO]);
  folder := IncludeTrailingPathDelimiter(folder);
  if fSettings.UseGpu and TQwenVulkan.Available then
    options := QwenVulkanOptions
  else if fSettings.CpuFloat32 then
    options := QwenCpuOptions(idtF32)
  else
    options := QwenCpuOptions(idtQ8_0);
  options.Threads := fSettings.Threads;
  fTalker := TQwenTalker.Create(folder, options);
  s := fTalker.Sampling; // generation_config.json
  s.Seed := fSettings.Seed;
  if fSettings.Temperature > 0 then
  begin
    s.Temperature := fSettings.Temperature;
    s.SubTemperature := fSettings.Temperature;
  end;
  fTalker.Sampling := s;
  if (fTalker.Vulkan <> nil) and not fSettings.CpuDecoder then
    fCodec := TQwenCodecVulkan.CreateVulkan(folder + 'speech_tokenizer',
      fTalker.Vulkan, fSettings.Threads)
  else
    fCodec := TQwenCodec.Create(folder + 'speech_tokenizer', fSettings.Threads);
  fLoadSeconds := timer.StopInMicroSec / 1e6;
  if fCodec is TQwenCodecVulkan then
    Log(FormatUtf8('% loaded in %s on %, % MB on the GPU (decoder there too)',
      [Name, Fixed2(fLoadSeconds), fTalker.Vulkan.DeviceName,
       (fTalker.Vulkan.DeviceBytes + TQwenCodecVulkan(fCodec).DeviceBytes) shr 20]))
  else if fTalker.Vulkan <> nil then
    Log(FormatUtf8('% loaded in %s on %, % MB on the GPU, % threads', [Name,
      Fixed2(fLoadSeconds), fTalker.Vulkan.DeviceName,
      fTalker.Vulkan.DeviceBytes shr 20, fCodec.Threads]))
  else
    Log(FormatUtf8('% loaded in %s, % threads', [Name, Fixed2(fLoadSeconds),
      fCodec.Threads]));
end;

procedure TQwenTts.Close;
begin
  {$ifdef OSWINDOWS}
  FreeAndNil(fStream);
  {$endif OSWINDOWS}
  FreeAndNil(fCodec);
  FreeAndNil(fTalker);
end;

function TQwenTts.Synthesize(const aText, aVoice: RawUtf8;
  const aSink: TOnQwenPcm; const aAbort: TOnTtsAbort): TQwenTtsStats;
const
  // the frames decoded at once: 1 (the first sound as soon as possible), 2, 4,
  // then 8 (the decoder is faster by packets, and the talker is ahead)
  // - each packet is computed faster than what is already queued plays
  BATCH_MAX = 8;
var
  st: TQwenTtsStats;
  chunks: TRawUtf8DynArray;
  voice: RawUtf8;
  codes: TIntegerDynArray;
  pcm: TSingleDynArray;
  start, now: Int64;
  chunk, c, n, batch: PtrInt;
  talker: TQwenTtsProducer;
begin
  FillCharFast(st, SizeOf(st), 0);
  if fTalker = nil then
    Open;
  voice := aVoice;
  if voice = '' then
    voice := fSettings.Voice;
  chunks := QwenSplitText(aText, QWEN_TTS_FIRST_CHUNK, QWEN_TTS_CHUNK);
  QueryPerformanceMicroSeconds(start);
  SetLength(codes, BATCH_MAX * QWEN_CODE_GROUPS);
  SetLength(pcm, BATCH_MAX * QWEN_FRAME_SAMPLES);
  batch := 1;
  chunk := -1;
  // the talker computes on its thread, the decoder here: each during the other
  talker := TQwenTtsProducer.Create(fTalker, chunks, voice, fSettings.Language,
    fFramesPerByte);
  try
    repeat
      // the abort: at each packet, and every 10 ms of waiting for the talker
      if Assigned(aAbort) and aAbort() then
      begin
        st.Aborted := true;
        break;
      end;
      n := talker.Take(batch, pointer(codes), c, 10);
      if n = 0 then
        break; // all is said
      if n < 0 then
        continue;
      if c <> chunk then
      begin
        fCodec.Reset; // a new text chunk
        chunk := c;
      end;
      fCodec.Decode(pointer(codes), n, pointer(pcm));
      if st.Frames = 0 then
      begin
        QueryPerformanceMicroSeconds(now);
        st.FirstSoundMs := (now - start) / 1000;
      end;
      inc(st.Frames, n);
      if not aSink(pointer(pcm), n * QWEN_FRAME_SAMPLES) then
        st.Aborted := true;
      if batch < BATCH_MAX then
        batch := batch * 2;
    until st.Aborted;
    talker.Finish;
    st.Capped := talker.Capped;
  finally
    talker.Free;
  end;
  QueryPerformanceMicroSeconds(now);
  st.ComputeSeconds := (now - start) / 1e6;
  st.AudioSeconds := st.Frames * QWEN_FRAME_SAMPLES / QWEN_SAMPLE_RATE;
  fLast := st;
  result := st;
end;

{$ifdef OSWINDOWS}

function TQwenTts.Play(aPcm: PSingleArray; aCount: PtrInt): boolean;
begin
  if length(fPcm16) < aCount then
    SetLength(fPcm16, aCount);
  TtsFloatToPcm16(aPcm, aCount, pointer(fPcm16));
  fStream.Write(pointer(fPcm16), aCount * SizeOf(SmallInt));
  result := true;
end;

procedure TQwenTts.Speak(const aText, aVoice: RawUtf8; const aAbort: TOnTtsAbort);
var
  stats: TQwenTtsStats;
begin
  if fTalker = nil then
    Open;
  if fStream = nil then
    fStream := TTtsPcmStream.Create(QWEN_SAMPLE_RATE);
  try
    stats := Synthesize(aText, aVoice, Play, aAbort);
    if not stats.Aborted then
      fStream.Drain(aAbort);
  finally
    fStream.Stop;
  end;
  Log(FormatUtf8('%s of audio in %s (x% real time), first sound % ms%%: %',
    [Fixed2(stats.AudioSeconds), Fixed2(stats.ComputeSeconds),
     Fixed2(stats.AudioSeconds / (stats.ComputeSeconds + 1e-9)),
     round(stats.FirstSoundMs), ABORTED_TEXT[stats.Aborted],
     CAPPED_TEXT[stats.Capped > 0], copy(aText, 1, 60)]));
end;

{$endif OSWINDOWS}

end.
