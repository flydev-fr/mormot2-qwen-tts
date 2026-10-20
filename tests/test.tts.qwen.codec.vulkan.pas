/// Qwen3-TTS tests - the 12 Hz decoder with its back part on the GPU (Vulkan)
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - compared to the PyTorch reference and to the CPU decoder; without a Vulkan
// GPU, or without model and references (see test.tts.qwen.ref), skipped
unit test.tts.qwen.codec.vulkan;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.codec.vulkan,
  mormot.ai.tts.qwen.lm,
  test.tts.qwen.ref;

type
  TTestQwenCodecVulkan = class(TSynTestCase)
  protected
    fVulkan: TQwenVulkan;
    fCodec: TQwenCodecVulkan;
    fMissing: boolean;
    function Codec: TQwenCodecVulkan;
    function LoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;
    function DecodeAll(const aRef: TQwenRef; aChunk: PtrInt): TSingleDynArray;
    procedure CompareWav(const aRef: TQwenRef; const aPcm: TSingleDynArray);
  public
    destructor Destroy; override;
  published
    procedure GpuMatchesPython;
    procedure GpuChunksGiveTheSameSound;
    procedure GpuSlidingWindow;
    procedure GpuDecodingSpeed;
  end;


implementation

const
  // absolute error allowed on the wav (amplitude -1..1): float32 on the GPU,
  // sums in another order, erf and sine approximated (measured on 2026-10-18:
  // 4.8e-6 at worst, on fr-long)
  GPU_WAV_TOLERANCE = 2e-5;

destructor TTestQwenCodecVulkan.Destroy;
begin
  fCodec.Free;
  fVulkan.Free;
  inherited Destroy;
end;

function TTestQwenCodecVulkan.Codec: TQwenCodecVulkan;
var
  snap: TFileName;
  timer: TPrecisionTimer;
begin
  if (fCodec = nil) and not fMissing then
  begin
    snap := QwenTestModel;
    if (snap = '') or not FileExists(snap + 'speech_tokenizer' + PathDelim +
       'model.safetensors') then
    begin
      fMissing := true;
      AddConsole(QwenNoModelText);
    end
    else if not TQwenVulkan.Available then
    begin
      fMissing := true;
      AddConsole('skipped: no Vulkan device');
    end
    else
    begin
      timer.Start;
      fVulkan := TQwenVulkan.Create(1);
      fCodec := TQwenCodecVulkan.CreateVulkan(snap + 'speech_tokenizer', fVulkan);
      AddConsole('GPU decoder on %: loaded in %, % MB on the GPU',
        [fVulkan.DeviceName, timer.Stop, fCodec.DeviceBytes shr 20]);
    end;
  end;
  result := fCodec;
end;

function TTestQwenCodecVulkan.LoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;
begin
  result := (Codec <> nil) and QwenLoadRef(aName, aRef);
  if (Codec <> nil) and not result then
    AddConsole('skipped: no reference % in % (tools/tts/qwen_ref.py)',
      [aName, QwenTestRefRoot]);
end;

function TTestQwenCodecVulkan.DecodeAll(const aRef: TQwenRef;
  aChunk: PtrInt): TSingleDynArray;
var
  f, n: PtrInt;
begin
  Codec.Reset;
  SetLength(result, aRef.Frames * QWEN_FRAME_SAMPLES);
  f := 0;
  while f < aRef.Frames do
  begin
    n := aRef.Frames - f;
    if n > aChunk then
      n := aChunk;
    Codec.Decode(@aRef.Codes[f * QWEN_CODE_GROUPS], n, @result[f * QWEN_FRAME_SAMPLES]);
    inc(f, n);
  end;
  CheckEqual(Codec.Frames, aRef.Frames, 'frames');
end;

procedure TTestQwenCodecVulkan.CompareWav(const aRef: TQwenRef;
  const aPcm: TSingleDynArray);
var
  ref: TSingleDynArray;
  shape: TInt64DynArray;
  diff, peak: double;
begin
  ref := QwenRefTensor(aRef, 'wav', shape);
  if not CheckUtf8(length(ref) = length(aPcm), 'wav: % samples, % expected',
     [length(aPcm), length(ref)]) then
    exit;
  QwenDiff(pointer(aPcm), pointer(ref), length(ref), diff, peak);
  AddConsole('GPU wav %: % samples, max |diff| %', [aRef.Folder, length(ref), diff]);
  CheckUtf8(diff <= GPU_WAV_TOLERANCE, 'GPU wav error % > %', [diff, GPU_WAV_TOLERANCE]);
end;

procedure TTestQwenCodecVulkan.GpuMatchesPython;
var
  ref: TQwenRef;
begin
  if not LoadRef('fr-vivian', ref) then
    exit;
  CompareWav(ref, DecodeAll(ref, 1));
end;

procedure TTestQwenCodecVulkan.GpuChunksGiveTheSameSound;
var
  ref: TQwenRef;
  one, four, all: TSingleDynArray;
  i: PtrInt;
  same: boolean;
begin
  if not LoadRef('fr-vivian', ref) then
    exit;
  one := DecodeAll(ref, 1);
  four := DecodeAll(ref, 4);
  all := DecodeAll(ref, ref.Frames); // by passes of 16 frames
  // each output sums its terms in the same order, whatever the packet
  same := true;
  for i := 0 to high(one) do
    if (one[i] <> four[i]) or (one[i] <> all[i]) then
    begin
      same := false;
      CheckUtf8(false, 'sample %: % / % / %', [i, one[i], four[i], all[i]]);
      break;
    end;
  Check(same, 'chunks of 1, 4 and all frames give the same samples');
end;

procedure TTestQwenCodecVulkan.GpuSlidingWindow;
var
  ref: TQwenRef;
begin
  if not LoadRef('fr-long', ref) then
    exit;
  CompareWav(ref, DecodeAll(ref, 8));
end;

procedure TTestQwenCodecVulkan.GpuDecodingSpeed;
var
  ref: TQwenRef;
  pcm: TSingleDynArray;
  timer: TPrecisionTimer;
  first, total, audio: double;
  f, n: PtrInt;
begin
  // no speed assertion (the machine may be busy): the log only
  if not LoadRef('fr-long', ref) then
    exit;
  SetLength(pcm, ref.Frames * QWEN_FRAME_SAMPLES);
  Codec.Reset;
  timer.Start;
  Codec.Decode(pointer(ref.Codes), 1, pointer(pcm));
  first := timer.StopInMicroSec / 1000;
  f := 1;
  timer.Start;
  while f < ref.Frames do
  begin
    n := ref.Frames - f;
    if n > 8 then
      n := 8;
    Codec.Decode(@ref.Codes[f * QWEN_CODE_GROUPS], n, @pcm[f * QWEN_FRAME_SAMPLES]);
    inc(f, n);
  end;
  total := timer.StopInMicroSec / 1e6;
  audio := (ref.Frames - 1) * QWEN_FRAME_SAMPLES / QWEN_SAMPLE_RATE;
  AddConsole('GPU decoder: first frame in % ms; then % s of audio in % s by 8 ' +
    'frames (x% real time)', [first, audio, total, audio / total]);
  CompareWav(ref, pcm);
end;

end.
