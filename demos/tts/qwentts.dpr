// - qwentts: say a text with the native Qwen3-TTS engine, or write it as WAV
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
//
//   usage: qwentts "Hostiles in local." [/voice ryan] [/language English]
//          [/wav out.wav] [/cpu] [/f32] [/cpudecoder] [/temperature 0.7]
//          [/seed 1] [/threads 8] [/model <snapshot folder>] [/voices]
//   - the model: /model, or Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice in a Hugging
//     Face cache: .hf next to the executable, HF_HOME or ~/.cache/huggingface
//   - without /wav, the sound plays as it is computed (Windows audio output)
program qwentts;

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
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen;

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
  s: TQwenTtsSettings;
  tts: TQwenTts;
  text, voice, temperature: RawUtf8;
  wav: TFileName;
  voices: boolean;
  sink: TWavSink;
  st: TQwenTtsStats;
  names: TRawUtf8DynArray;
  i: PtrInt;
begin
  s := QwenTtsDefaults;
  with Executable.Command do
  begin
    ExeDescription := 'qwentts: Qwen3-TTS-12Hz-0.6B-CustomVoice in native Pascal';
    text := ArgU(0, 'the #text to say');
    voice := Param('voice', 'one #speaker of /voices, e.g. vivian, ryan, aiden',
      s.Voice);
    s.Language := Param('language', 'the #language as Auto, French, English...',
      s.Language);
    wav := ParamS('wav', 'write the sound into this #file.wav instead of playing it');
    s.ModelFolder := ParamS('model', 'the model snapshot #folder');
    s.UseGpu := not Option('cpu', 'compute on the CPU even with a Vulkan GPU');
    s.CpuFloat32 := Option('f32', 'CPU: float32 weights (the codes of PyTorch, slower)');
    s.CpuDecoder := Option('cpudecoder', 'the 12 Hz decoder on the CPU, not the GPU');
    temperature := Param('temperature',
      'the #value of the sampling temperature (default 0.9)');
    s.Seed := Param('seed', 0, 'the #number of a fixed seed (0: random)');
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
  if temperature <> '' then
    s.Temperature := GetExtended(pointer(temperature));
  s.Voice := voice;
  tts := TQwenTts.Create(s);
  try
    tts.OnLog := ConsoleLog;
    tts.Open;
    if voices then
    begin
      names := tts.Talker.Speakers;
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
        FileFromString(TtsWav16(pointer(sink.Pcm), sink.Count, QWEN_SAMPLE_RATE), wav);
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
