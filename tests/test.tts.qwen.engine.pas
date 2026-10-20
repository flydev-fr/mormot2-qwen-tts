/// Qwen3-TTS tests - the native engine end to end (text -> streamed sound)
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - no loudspeaker: the sound goes into a collector which notes when each
// piece arrives and replays the clock of an audio output (the gaps: a piece
// arriving after the end of the previous one); measures the first sound and
// the real time factor, writes <ref>/fr-long/native-stream.wav to listen to
// - without model and references (see test.tts.qwen.ref), skipped - but the
// text splitter needs none
unit test.tts.qwen.engine;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.lm,
  mormot.ai.tts.qwen,
  test.tts.qwen.ref;

type
  TTestQwenEngine = class(TSynTestCase)
  protected
    fEngine: TQwenTts;
    fMissing: boolean;
    fPcm: TSingleDynArray;
    fCount: PtrInt;
    fStart: Int64;
    // the simulated playback clock
    fPlayEnd, fGaps: double;
    fCutAfter: PtrInt;
    function Engine: TQwenTts;
    function Collect(aPcm: PSingleArray; aCount: PtrInt): boolean;
    function CutNow: boolean;
    procedure Reset;
  public
    destructor Destroy; override;
  published
    procedure SplitsTextIntoChunks;
    procedure StreamsALongText;
    procedure StopCutsQuickly;
    procedure CapsAChunkWithoutEnd;
  end;


implementation

destructor TTestQwenEngine.Destroy;
begin
  fEngine.Free;
  inherited Destroy;
end;

function TTestQwenEngine.Engine: TQwenTts;
var
  s: TQwenTtsSettings;
begin
  if (fEngine = nil) and not fMissing then
  begin
    s := QwenTtsDefaults;
    s.ModelFolder := QwenTestModel;
    s.Seed := 1;
    s.Threads := QwenTestThreads;
    if s.ModelFolder = '' then
    begin
      fMissing := true;
      AddConsole(QwenNoModelText);
    end
    else
    begin
      fEngine := TQwenTts.Create(s);
      fEngine.Open;
      AddConsole('% loaded in % s, % threads', [fEngine.Name, fEngine.LoadSeconds,
        fEngine.Codec.Threads]);
    end;
  end;
  result := fEngine;
end;

procedure TTestQwenEngine.Reset;
begin
  fPcm := nil;
  fCount := 0;
  fPlayEnd := 0;
  fGaps := 0;
  QueryPerformanceMicroSeconds(fStart);
end;

function TTestQwenEngine.Collect(aPcm: PSingleArray; aCount: PtrInt): boolean;
var
  now: Int64;
  t: double;
begin
  QueryPerformanceMicroSeconds(now);
  t := (now - fStart) / 1e6;
  // an audio output plays without stopping what it has: a piece arriving
  // after the end of the previous one leaves a gap
  if (fPlayEnd > 0) and (t > fPlayEnd) then
    fGaps := fGaps + (t - fPlayEnd);
  if t > fPlayEnd then
    fPlayEnd := t;
  fPlayEnd := fPlayEnd + aCount / QWEN_SAMPLE_RATE;
  if fCount + aCount > length(fPcm) then
    SetLength(fPcm, (fCount + aCount) * 2);
  MoveFast(aPcm^[0], fPcm[fCount], aCount * SizeOf(single));
  inc(fCount, aCount);
  result := true;
end;

function TTestQwenEngine.CutNow: boolean;
begin
  result := fCount >= fCutAfter;
end;

procedure TTestQwenEngine.SplitsTextIntoChunks;
var
  c: TRawUtf8DynArray;
begin
  c := QwenSplitText('Target in sight. Two hostiles!'#10'- warp out now'#10#10 +
    '* ok', 20, 30);
  CheckEqual(length(c), 3, 'chunks');
  if length(c) <> 3 then
    exit;
  // the first chunk is short (the first sound comes sooner), bullets drop, a
  // line without final punctuation gets a period
  CheckEqual(c[0], 'Target in sight.');
  CheckEqual(c[1], 'Two hostiles! warp out now.');
  CheckEqual(c[2], 'ok.');
  CheckEqual(length(QwenSplitText('', 20, 30)), 0, 'empty text');
end;

procedure TTestQwenEngine.StreamsALongText;
var
  ref: TQwenRef;
  st: TQwenTtsStats;
  text: RawUtf8;
begin
  if (Engine = nil) or not QwenLoadRef('fr-long', ref) then
    exit;
  // two more sentences: the text goes in two chunks (QwenSplitText)
  text := ref.Doc.U['text'] + ' Fin de transmission, bonne chasse a tous.';
  Reset;
  st := Engine.Synthesize(text, '', Collect, nil);
  SetLength(fPcm, fCount);
  AddConsole('%: % s of audio (% frames) in % s, x% real time, first sound % ms, ' +
    'gaps % ms, % threads', [Engine.Name, st.AudioSeconds, st.Frames,
    st.ComputeSeconds, st.AudioSeconds / st.ComputeSeconds, round(st.FirstSoundMs),
    round(fGaps * 1000), Engine.Codec.Threads]);
  CheckEqual(fCount, st.Frames * QWEN_FRAME_SAMPLES, 'samples');
  Check(not st.Aborted, 'not cut');
  Check(st.AudioSeconds > 10, 'the whole text');
  QwenSaveWav(ref.Folder + 'native-stream.wav', fPcm);
  AddConsole('listen: %', [ref.Folder + 'native-stream.wav']);
  if Engine.Talker.Vulkan <> nil then
  begin
    // the targets: first sound in less than 300 ms, at least real time
    CheckUtf8(st.FirstSoundMs < 300, 'first sound in % ms', [st.FirstSoundMs]);
    CheckUtf8(st.AudioSeconds > st.ComputeSeconds, 'x% real time',
      [st.AudioSeconds / st.ComputeSeconds]);
    CheckUtf8(fGaps < 0.05, '% ms of gaps', [fGaps * 1000]);
  end;
end;

procedure TTestQwenEngine.StopCutsQuickly;
var
  ref: TQwenRef;
  st: TQwenTtsStats;
  start, now: Int64;
begin
  if (Engine = nil) or not QwenLoadRef('fr-long', ref) then
    exit;
  Reset;
  fCutAfter := 10 * QWEN_FRAME_SAMPLES; // after 800 ms of sound
  QueryPerformanceMicroSeconds(start);
  st := Engine.Synthesize(ref.Doc.U['text'], '', Collect, CutNow);
  QueryPerformanceMicroSeconds(now);
  Check(st.Aborted, 'cut');
  CheckUtf8(st.Frames < 20, '% frames after the cut at 10', [st.Frames]);
  AddConsole('cut after % frames; the whole call (prefill included) took % ms',
    [st.Frames, (now - start) div 1000]);
end;

procedure TTestQwenEngine.CapsAChunkWithoutEnd;
var
  ref: TQwenRef;
  st: TQwenTtsStats;
  chunks: PtrInt;
begin
  if (Engine = nil) or not QwenLoadRef('fr-long', ref) then
    exit;
  // without frames per byte, the cap is 50 per chunk: the long text asks
  // more, as a generation which would not end
  chunks := length(QwenSplitText(ref.Doc.U['text'], QWEN_TTS_FIRST_CHUNK,
    QWEN_TTS_CHUNK));
  Reset;
  Engine.FramesPerByte := 0;
  try
    st := Engine.Synthesize(ref.Doc.U['text'], '', Collect, nil);
  finally
    Engine.FramesPerByte := QWEN_TTS_FRAMES_PER_BYTE;
  end;
  AddConsole('% chunk(s) capped at % frames: % frames', [st.Capped,
    QWEN_TTS_MAX_FRAMES, st.Frames]);
  Check(st.Capped > 0, 'capped');
  CheckUtf8(st.Frames <= chunks * QWEN_TTS_MAX_FRAMES, '% frames for % chunk(s)',
    [st.Frames, chunks]);
  Check(not st.Aborted, 'not cut');
end;

end.
