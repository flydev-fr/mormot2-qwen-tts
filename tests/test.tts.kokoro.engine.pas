/// Kokoro TTS tests - the native engine end to end (text -> sound), no speaker
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the sound goes to a collector replaying the clock of an audio output
// (the gaps: a chunk arriving after the end of the previous one); measures
// the first sound and the real-time factor, writes
// <ref-kokoro>/fr-long/native-stream.wav
// - without model or espeak-ng (see test.tts.kokoro.ref), skipped
unit test.tts.kokoro.engine;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.tts.kokoro.model,
  mormot.ai.tts.kokoro,
  test.tts.qwen.ref,
  test.tts.kokoro.ref;

type
  TTestKokoroEngine = class(TSynTestCase)
  protected
    fEngine: TKokoroTts;
    fMissing: boolean;
    fPcm: TSingleDynArray;
    fCount: PtrInt;
    fStart: Int64;
    fPlayEnd, fGaps: double;
    fCutAfter: PtrInt;
    function Engine: TKokoroTts;
    function Collect(aPcm: PSingleArray; aCount: PtrInt): boolean;
    function CutNow: boolean;
    procedure Reset;
  public
    destructor Destroy; override;
  published
    procedure StreamsALongText;
    procedure StopCutsQuickly;
    procedure ListsTheVoices;
  end;


implementation

const
  LONG_TEXT = 'Alerte, flotte hostile en approche par la porte de Jita. ' +
    'Trente pilotes, dont des intercepteurs et des croiseurs lourds. ' +
    'Restez a quai jusqu''a nouvel ordre, je repete, restez a quai. ' +
    'Fin de transmission, bonne chasse a tous.';

destructor TTestKokoroEngine.Destroy;
begin
  fEngine.Free;
  inherited Destroy;
end;

function TTestKokoroEngine.Engine: TKokoroTts;
var
  s: TKokoroTtsSettings;
begin
  if (fEngine = nil) and not fMissing then
  begin
    s := KokoroTtsDefaults;
    s.ModelFolder := KokoroTestSnapshot;
    s.EspeakFolder := KokoroTestEspeak;
    s.Threads := QwenTestThreads;
    if (s.ModelFolder = '') or (s.EspeakFolder = '') then
    begin
      fMissing := true;
      if s.ModelFolder = '' then
        AddConsole(KokoroNoModelText)
      else
        AddConsole(KokoroNoEspeakText);
    end
    else
    begin
      fEngine := TKokoroTts.Create(s);
      fEngine.Open;
      AddConsole('% loaded in % s', [fEngine.Name, fEngine.LoadSeconds]);
    end;
  end;
  result := fEngine;
end;

procedure TTestKokoroEngine.Reset;
begin
  fPcm := nil;
  fCount := 0;
  fPlayEnd := 0;
  fGaps := 0;
  QueryPerformanceMicroSeconds(fStart);
end;

function TTestKokoroEngine.Collect(aPcm: PSingleArray; aCount: PtrInt): boolean;
var
  now: Int64;
  t: double;
begin
  QueryPerformanceMicroSeconds(now);
  t := (now - fStart) / 1e6;
  if (fPlayEnd > 0) and (t > fPlayEnd) then
    fGaps := fGaps + (t - fPlayEnd);
  if t > fPlayEnd then
    fPlayEnd := t;
  fPlayEnd := fPlayEnd + aCount / KOKORO_SAMPLE_RATE;
  if fCount + aCount > length(fPcm) then
    SetLength(fPcm, (fCount + aCount) * 2);
  MoveFast(aPcm^[0], fPcm[fCount], aCount * SizeOf(single));
  inc(fCount, aCount);
  result := true;
end;

function TTestKokoroEngine.CutNow: boolean;
begin
  result := fCount >= fCutAfter;
end;

procedure TTestKokoroEngine.StreamsALongText;
var
  st: TKokoroTtsStats;
  ref: TKokoroRef;
begin
  if Engine = nil then
    exit;
  Reset;
  st := Engine.Synthesize(LONG_TEXT, '', Collect, nil);
  SetLength(fPcm, fCount);
  AddConsole('%: % s of audio (% chunks, % phonemes) in % s, x% real time, ' +
    'first sound % ms, gaps % ms', [Engine.Name, st.AudioSeconds, st.Chunks,
    st.Phonemes, st.ComputeSeconds, st.AudioSeconds / st.ComputeSeconds,
    round(st.FirstSoundMs), round(fGaps * 1000)]);
  Check(not st.Aborted, 'not cut');
  Check(st.Chunks >= 3, 'the text in chunks');
  Check(st.AudioSeconds > 10, 'the whole text');
  CheckUtf8(st.AudioSeconds > st.ComputeSeconds, 'x% real time',
    [st.AudioSeconds / st.ComputeSeconds]);
  CheckUtf8(fGaps < 0.05, '% ms of gaps', [fGaps * 1000]);
  if KokoroLoadRef('fr-long', ref) then
  begin
    QwenSaveWav(ref.Folder + 'native-stream.wav', fPcm);
    AddConsole('listen: %', [ref.Folder + 'native-stream.wav']);
  end;
end;

procedure TTestKokoroEngine.StopCutsQuickly;
var
  st: TKokoroTtsStats;
begin
  if Engine = nil then
    exit;
  Reset;
  fCutAfter := KOKORO_SAMPLE_RATE div 2; // after the first chunk
  st := Engine.Synthesize(LONG_TEXT, '', Collect, CutNow);
  Check(st.Aborted, 'cut');
  CheckUtf8(st.Chunks <= 2, '% chunks after the cut', [st.Chunks]);
end;

procedure TTestKokoroEngine.ListsTheVoices;
var
  v: TRawUtf8DynArray;
begin
  if Engine = nil then
    exit;
  v := Engine.Voices;
  AddConsole('% voices: %', [length(v), RawUtf8ArrayToCsv(v)], true);
  CheckUtf8(length(v) >= 50, '% voices', [length(v)]);
  Check(FindRawUtf8(v, 'ff_siwis') >= 0, 'ff_siwis');
end;

end.
