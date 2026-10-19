// - kokorotts: say a text with the native Kokoro-82M engine, or write it as WAV
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
//
//   usage: kokorotts "Hostiles en local." [/voice ff_siwis] [/speed 1.1]
//          [/wav out.wav] [/threads 8] [/model <snapshot folder>]
//          [/espeak <espeak-ng folder>] [/voices]
//   - the model: /model, or hexgrad/Kokoro-82M in a Hugging Face cache: .hf
//     next to the executable, HF_HOME or ~/.cache/huggingface
//   - espeak-ng: /espeak, or espeak-ng next to the executable (espeak-ng.dll
//     and espeak-ng-data)
//   - the voice tells the language: ff_ French, ef_/em_ Spanish, if_/im_
//     Italian, pf_/pm_ Brazilian Portuguese, hf_/hm_ Hindi, af_/am_ and
//     bf_/bm_ English
//   - without /wav, the sound plays as it is computed (Windows audio output)
program kokorotts;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.tts.wave,
  mormot.ai.tts.kokoro.model,
  mormot.ai.tts.kokoro;

type
  /// collects the samples for /wav
  TWavSink = class
  public
    Pcm: TSingleDynArray;
    Count: PtrInt;
    function Add(aPcm: PSingleArray; aCount: PtrInt): boolean;
  end;

function TWavSink.Add(aPcm: PSingleArray; aCount: PtrInt): boolean;
begin
  if Count + aCount > length(Pcm) then
    SetLength(Pcm, (Count + aCount) * 2);
  MoveFast(aPcm^[0], Pcm[Count], aCount * SizeOf(single));
  inc(Count, aCount);
  result := true;
end;

procedure ConsoleLog(const aLine: RawUtf8);
begin
  ConsoleWrite(aLine);
end;

procedure Main;
var
  s: TKokoroTtsSettings;
  tts: TKokoroTts;
  text, speed: RawUtf8;
  wav: TFileName;
  voices: boolean;
  sink: TWavSink;
  st: TKokoroTtsStats;
  names: TRawUtf8DynArray;
  i: PtrInt;
begin
  s := KokoroTtsDefaults;
  with Executable.Command do
  begin
    ExeDescription := 'kokorotts: Kokoro-82M in native Pascal';
    text := ArgU(0, 'the #text to say');
    s.Voice := Param('voice', 'one #voice of /voices, e.g. ff_siwis, af_heart',
      s.Voice);
    speed := Param('speed', 'the #factor of the speech rate (default 1)');
    wav := ParamS('wav', 'write the sound into this #file.wav instead of playing it');
    s.ModelFolder := ParamS('model', 'the model snapshot #folder');
    s.EspeakFolder := ParamS('espeak', 'the espeak-ng #folder');
    s.Threads := Param('threads', 0, 'the CPU #threads (0: two thirds of the cores)');
    voices := Option('voices', 'list the voices of the model');
    if ConsoleHelpFailed then
      exit; // /help, or an unknown switch: the help is written
    if (text = '') and not voices then
    begin
      ConsoleWrite(FullDescription);
      ExitCode := 1;
      exit;
    end;
  end;
  if speed <> '' then
    s.Speed := GetExtended(pointer(speed));
  tts := TKokoroTts.Create(s);
  try
    tts.OnLog := ConsoleLog;
    tts.Open;
    if voices then
    begin
      names := tts.Voices;
      for i := 0 to high(names) do
        ConsoleWrite(names[i]);
      if text = '' then
        exit;
    end;
    if wav <> '' then
    begin
      sink := TWavSink.Create;
      try
        st := tts.Synthesize(text, '', sink.Add);
        FileFromString(TtsWav16(pointer(sink.Pcm), sink.Count, KOKORO_SAMPLE_RATE), wav);
        ConsoleWrite('% s of audio in % s, first sound % ms: %', [
          round(st.AudioSeconds * 100) / 100, round(st.ComputeSeconds * 100) / 100,
          round(st.FirstSoundMs), wav]);
      finally
        sink.Free;
      end;
    end
    else
    {$ifdef OSWINDOWS}
      tts.Speak(text);
    {$else}
      ConsoleWrite('no audio output here: use /wav');
    {$endif OSWINDOWS}
  finally
    tts.Free;
  end;
end;

begin
  try
    Main;
  except
    on e: Exception do
    begin
      ConsoleShowFatalException(e, {waitforenterkey=}false);
      ExitCode := 2;
    end;
  end;
end.
