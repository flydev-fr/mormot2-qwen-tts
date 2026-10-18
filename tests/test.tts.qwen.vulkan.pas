/// Qwen3-TTS tests - the talker and the code predictor on the GPU (Vulkan)
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the same computation as on the CPU (float32 weights), then the sentence
// in Q8_0 against the PyTorch reference, its speed, the GPU memory, and wav
// files to listen to in <ref>/<name>/native-*.wav
// - without a Vulkan GPU, or without model and references (see
// test.tts.qwen.ref), skipped
unit test.tts.qwen.vulkan;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.lm,
  mormot.ai.tts.qwen.lm.vulkan,
  mormot.ai.tts.qwen.talker,
  test.tts.qwen.ref;

type
  TTestQwenVulkan = class(TSynTestCase)
  protected
    fMissing: boolean;
    fSnapshot: TFileName;
    fCodec: TQwenCodec;
    function Ready: boolean;
    function Codec: TQwenCodec;
    /// the sentence of aRef until its end; returns the codes
    function Speak(aTalker: TQwenTalker; const aRef: TQwenRef;
      const aLabel: RawUtf8): TIntegerDynArray;
    /// decode the codes and write <ref>/<name>/aFile
    procedure SaveWav(const aRef: TQwenRef; const aCodes: TIntegerDynArray;
      const aFile: TFileName);
    procedure ComparePredictor(aKvF16: boolean; aTolerance: double);
  public
    destructor Destroy; override;
  published
    procedure PredictorOnGpuMatchesCpu;
    procedure TalkerQ8OnGpu;
    procedure SampledSentenceOnGpu;
    procedure TalkerQ8OnCpu;
  end;


implementation

const
  /// the GPU memory allowed: a game may run beside
  VRAM_BUDGET = Int64(2) shl 30;
  KV_TEXT: array[boolean] of RawUtf8 = ('F32', 'F16');

destructor TTestQwenVulkan.Destroy;
begin
  fCodec.Free;
  inherited Destroy;
end;

function TTestQwenVulkan.Ready: boolean;
begin
  if not fMissing and (fSnapshot = '') then
  begin
    fSnapshot := QwenTestModel;
    if (fSnapshot = '') or not FileExists(fSnapshot + 'model.safetensors') then
    begin
      fMissing := true;
      AddConsole(QwenNoModelText);
    end
    else if not TQwenVulkan.Available then
    begin
      fMissing := true;
      AddConsole('skipped: no Vulkan device');
    end;
  end;
  result := not fMissing;
end;

function TTestQwenVulkan.Codec: TQwenCodec;
begin
  if fCodec = nil then
    fCodec := TQwenCodec.Create(fSnapshot + 'speech_tokenizer');
  result := fCodec;
end;

function TTestQwenVulkan.Speak(aTalker: TQwenTalker; const aRef: TQwenRef;
  const aLabel: RawUtf8): TIntegerDynArray;
var
  n, i, first, same0: PtrInt;
  timer: TPrecisionTimer;
  start, total, audio: double;
begin
  SetLength(result, (aRef.Frames * 2 + 40) * QWEN_CODE_GROUPS);
  timer.Start;
  aTalker.Start(aRef.Doc.U['text'], aRef.Doc.U['speaker'], aRef.Doc.U['language'],
    aRef.Doc.B['non_streaming_mode']);
  start := timer.StopInMicroSec / 1000;
  n := 0;
  timer.Start;
  while (n < aRef.Frames * 2 + 40) and
        aTalker.NextFrame(@result[n * QWEN_CODE_GROUPS]) do
    inc(n);
  total := timer.StopInMicroSec / 1e6;
  SetLength(result, n * QWEN_CODE_GROUPS);
  audio := n * QWEN_FRAME_SAMPLES / QWEN_SAMPLE_RATE;
  // the agreement with PyTorch: the first frame which differs, then all diverges
  first := -1;
  for i := 0 to MinPtrInt(n, aRef.Frames) * QWEN_CODE_GROUPS - 1 do
    if result[i] <> aRef.Codes[i] then
    begin
      first := i div QWEN_CODE_GROUPS;
      break;
    end;
  same0 := 0;
  for i := 0 to MinPtrInt(n, aRef.Frames) - 1 do
    if result[i * QWEN_CODE_GROUPS] = aRef.Codes[i * QWEN_CODE_GROUPS] then
      inc(same0);
  AddConsole('%: prefill % ms, % frames (% s of audio, python %) in % s, ' +
    'x% real time, % ms per frame; first frame differing from python: %, ' +
    'group 0 equal on % frames', [aLabel, start, n, audio, aRef.Frames, total,
    audio / total, total * 1000 / (n + ord(n = 0)), first, same0]);
end;

procedure TTestQwenVulkan.SaveWav(const aRef: TQwenRef;
  const aCodes: TIntegerDynArray; const aFile: TFileName);
var
  pcm: TSingleDynArray;
  n: PtrInt;
begin
  n := length(aCodes) div QWEN_CODE_GROUPS;
  if n = 0 then
    exit;
  SetLength(pcm, n * QWEN_FRAME_SAMPLES);
  Codec.Reset;
  Codec.Decode(pointer(aCodes), n, pointer(pcm));
  QwenSaveWav(aRef.Folder + aFile, pcm);
  AddConsole('listen: %', [aRef.Folder + aFile]);
end;

procedure TTestQwenVulkan.ComparePredictor(aKvF16: boolean; aTolerance: double);
var
  f: TSafeTensorsFile;
  cfg: TDocVariantData;
  w: TQwenLmWeights;
  cpu: TInferCpuFastBackend;
  vk: TQwenVulkan;
  a, b: TQwenLm;
  x, la, lb, ha, hb: TSingleDynArray;
  g, i, rows: PtrInt;
  seed: cardinal;
  diff, peak, worst: double;
begin
  // the float32 weights of the code predictor (5 layers, 15 heads): CPU and
  // GPU must return the same logits, within rounding
  Check(cfg.InitJsonFromFile(fSnapshot + 'config.json', JSON_FAST));
  f := TSafeTensorsFile.Create(fSnapshot + 'model.safetensors');
  cpu := TInferCpuFastBackend.Create;
  vk := TQwenVulkan.Create(1);
  w := nil;
  a := nil;
  b := nil;
  try
    w := TQwenLmWeights.Create(f, 'predictor', 'talker.code_predictor.model.',
      cfg.O['talker_config']^.O['code_predictor_config']^, idtF32);
    for g := 0 to QWEN_CODE_GROUPS - 2 do
      w.AddOutput(f, FormatUtf8('talker.code_predictor.lm_head.%.weight', [g]),
        QWEN_CODEBOOK_SIZE);
    a := TQwenLmCpu.CreateCpu(w, cpu);
    b := TQwenLmVulkan.CreateVulkan(w, vk, aKvF16);
    SetLength(x, 2 * w.Hidden);
    SetLength(la, QWEN_CODEBOOK_SIZE);
    SetLength(lb, QWEN_CODEBOOK_SIZE);
    SetLength(ha, w.Hidden);
    SetLength(hb, w.Hidden);
    seed := 12345;
    worst := 0;
    for g := 0 to QWEN_CODE_GROUPS - 2 do
    begin
      // pseudo-random inputs: 2 rows, then one per group
      rows := 1 + ord(g = 0);
      for i := 0 to rows * w.Hidden - 1 do
      begin
        seed := seed * 1664525 + 1013904223;
        x[i] := (integer(seed shr 8) / (1 shl 24) - 0.5) * 0.2;
      end;
      a.Forward(pointer(x), rows, g, pointer(la), pointer(ha));
      b.Forward(pointer(x), rows, g, pointer(lb), pointer(hb));
      QwenDiff(pointer(lb), pointer(la), QWEN_CODEBOOK_SIZE, diff, peak);
      if diff / peak > worst then
        worst := diff / peak;
      QwenDiff(pointer(hb), pointer(ha), w.Hidden, diff, peak);
      if diff / peak > worst then
        worst := diff / peak;
    end;
    AddConsole('predictor F32, KV %: GPU vs CPU over 15 steps, relative error %',
      [KV_TEXT[aKvF16], worst]);
    CheckUtf8(worst <= aTolerance, 'GPU vs CPU relative error % > %',
      [worst, aTolerance]);
    CheckEqual(b.Length, 16, 'positions');
  finally
    a.Free;
    b.Free;
    w.Free;
    vk.Free;
    cpu.Free;
    f.Free;
  end;
end;

procedure TTestQwenVulkan.PredictorOnGpuMatchesCpu;
begin
  if not Ready then
    exit;
  // measured on 2026-10-18: 2.6e-5 with a float32 cache, 2.3e-2 with half
  // floats (11 significant bits)
  ComparePredictor(false, 1e-4);
  ComparePredictor(true, 5e-2);
end;

procedure TTestQwenVulkan.TalkerQ8OnGpu;
const
  REFS: array[0..1] of RawUtf8 = ('fr-vivian', 'fr-long');
var
  talker: TQwenTalker;
  ref: TQwenRef;
  codes: TIntegerDynArray;
  timer: TPrecisionTimer;
  r, n: PtrInt;
begin
  if not Ready then
    exit;
  timer.Start;
  talker := TQwenTalker.Create(fSnapshot, QwenVulkanOptions);
  try
    AddConsole('Vulkan Q8_0 on %: loaded in %, % MB on the GPU',
      [talker.Vulkan.DeviceName, timer.Stop, talker.Vulkan.DeviceBytes shr 20]);
    talker.Sampling := QwenGreedy;
    for r := 0 to high(REFS) do
    begin
      if not QwenLoadRef(REFS[r], ref) then
        continue;
      codes := Speak(talker, ref, 'Vulkan Q8_0 greedy ' + REFS[r]);
      n := length(codes) div QWEN_CODE_GROUPS;
      Check(n > 0, 'frames');
      if n > 0 then
        CheckUtf8(codes[0] = ref.Codes[0], 'first code % instead of %',
          [codes[0], ref.Codes[0]]);
      // the end of the sentence: the same within a third
      CheckUtf8((n * 3 >= ref.Frames * 2) and (n * 3 <= ref.Frames * 4),
        '% frames, python %', [n, ref.Frames]);
      SaveWav(ref, codes, 'native-vulkan-q8-greedy.wav');
    end;
    AddConsole('GPU memory after the sentences: % MB', [talker.Vulkan.DeviceBytes shr 20]);
    CheckUtf8(talker.Vulkan.DeviceBytes < VRAM_BUDGET, '% MB on the GPU',
      [talker.Vulkan.DeviceBytes shr 20]);
  finally
    talker.Free;
  end;
end;

procedure TTestQwenVulkan.SampledSentenceOnGpu;
var
  talker: TQwenTalker;
  ref: TQwenRef;
  codes: TIntegerDynArray;
  s: TQwenSampling;
  n: PtrInt;
begin
  if not Ready or not QwenLoadRef('fr-accents', ref) then
    exit;
  talker := TQwenTalker.Create(fSnapshot, QwenVulkanOptions);
  try
    // generation_config.json: temperature 0.9, top-k 50, penalty 1.05
    s := talker.Sampling;
    Check(s.DoSample and s.SubDoSample, 'sampling from generation_config.json');
    s.Seed := 1;
    talker.Sampling := s;
    codes := Speak(talker, ref, 'Vulkan Q8_0 sampled fr-accents');
    n := length(codes) div QWEN_CODE_GROUPS;
    CheckUtf8((n * 2 >= ref.Frames) and (n <= ref.Frames * 2),
      '% frames, python greedy %', [n, ref.Frames]);
    SaveWav(ref, codes, 'native-vulkan-q8-sampled.wav');
  finally
    talker.Free;
  end;
end;

procedure TTestQwenVulkan.TalkerQ8OnCpu;
var
  talker: TQwenTalker;
  ref: TQwenRef;
  codes: TIntegerDynArray;
begin
  // the fallback without GPU: Q8_0 on the CPU, for speed
  if fSnapshot = '' then
    fSnapshot := QwenTestModel;
  if (fSnapshot = '') or not FileExists(fSnapshot + 'model.safetensors') or
     not QwenLoadRef('fr-vivian', ref) then
    exit;
  talker := TQwenTalker.Create(fSnapshot, QwenCpuOptions(idtQ8_0));
  try
    talker.Sampling := QwenGreedy;
    codes := Speak(talker, ref, 'CPU Q8_0 greedy fr-vivian');
    Check(length(codes) > 0, 'frames');
    SaveWav(ref, codes, 'native-cpu-q8-greedy.wav');
  finally
    talker.Free;
  end;
end;

end.
