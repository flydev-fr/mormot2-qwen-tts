/// Kokoro TTS tests - native Kokoro-82M against PyTorch (tools/tts/kokoro_ref.py)
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the .pth reader, the ids, the style of the voice, then each stage of the
// network and the sound
// - without model or references (see test.tts.kokoro.ref), skipped
unit test.tts.kokoro.model;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  math,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.tts.torchfile,
  mormot.ai.tts.kokoro.model,
  test.tts.qwen.ref,
  test.tts.kokoro.ref;

type
  TTestKokoroModel = class(TSynTestCase)
  protected
    fModel: TKokoroModel;
    fVoice: TKokoroVoice;
    fMissing: boolean;
    fLog: TQwenProbeLog;
    function Model: TKokoroModel;
    /// compare the collected output aName with the reference; aChannelMajor: the
    // reference is [channels][time] (PyTorch), ours [time][channels]
    procedure Compare(const aRef: TKokoroRef; const aName: RawUtf8;
      aChannelMajor: boolean; aTolerance: double);
    procedure CompareAudio(const aRef: TKokoroRef; const aPcm: TSingleDynArray;
      aTolerance: double; const aLabel: RawUtf8);
  public
    destructor Destroy; override;
  published
    procedure TorchFileReadsTheCheckpoint;
    procedure IdsAndStyleMatchPython;
    procedure DurationsMatchPython;
    procedure EveryStageMatchesPython;
    procedure SoundMatchesPython;
    procedure SynthesisSpeed;
  end;


implementation

const
  // relative error allowed (to the absolute peak of the reference) for a
  // stage: float32, sums in another order, chained LSTM and normalizations
  STAGE_TOLERANCE = 2e-3;
  // the sound, with the source and its STFT forced (those of PyTorch)
  WAV_TOLERANCE = 2e-3;
  // the native source, relative to the peak
  SOURCE_TOLERANCE = 2e-2;

destructor TTestKokoroModel.Destroy;
begin
  fModel.Free;
  fVoice.Free;
  fLog.Free;
  inherited Destroy;
end;

function TTestKokoroModel.Model: TKokoroModel;
var
  snap: TFileName;
  timer: TPrecisionTimer;
begin
  if (fModel = nil) and not fMissing then
  begin
    fLog := TQwenProbeLog.Create;
    snap := KokoroTestSnapshot;
    if (snap = '') or not FileExists(snap + 'kokoro-v1_0.pth') then
    begin
      fMissing := true;
      AddConsole(KokoroNoModelText);
    end
    else
    begin
      timer.Start;
      fModel := TKokoroModel.Create(snap, QwenTestThreads);
      fVoice := TKokoroVoice.Create(snap, 'ff_siwis');
      AddConsole('Kokoro-82M loaded in %, % threads', [timer.Stop, fModel.Cpu.Threads]);
    end;
  end;
  result := fModel;
end;

procedure TTestKokoroModel.Compare(const aRef: TKokoroRef; const aName: RawUtf8;
  aChannelMajor: boolean; aTolerance: double);
var
  ref: TSingleDynArray;
  shape: TInt64DynArray;
  i, rows, cols, t, c: PtrInt;
  mine: PSingleArray;
  d, diff, peak: double;
begin
  ref := KokoroRefTensor(aRef, aName, shape);
  i := fLog.Find(aName);
  if not CheckUtf8((ref <> nil) and (i >= 0), '% missing', [aName]) then
    exit;
  cols := fLog.Cols[i];
  rows := length(fLog.Data[i]) div cols;
  if not CheckUtf8(length(ref) = rows * cols, '% has % values, reference %',
     [aName, rows * cols, length(ref)]) then
    exit;
  mine := pointer(fLog.Data[i]);
  diff := 0;
  peak := 0;
  for t := 0 to rows - 1 do
    for c := 0 to cols - 1 do
    begin
      if aChannelMajor then
        d := ref[c * rows + t]
      else
        d := ref[t * cols + c];
      if abs(d) > peak then
        peak := abs(d);
      d := abs(mine^[t * cols + c] - d);
      if d > diff then
        diff := d;
    end;
  if peak = 0 then
    peak := 1;
  AddConsole('% %x%: max |diff| % (relative %)', [aName, rows, cols, diff,
    diff / peak], true);
  CheckUtf8(diff / peak <= aTolerance, '% relative error % > %',
    [aName, diff / peak, aTolerance]);
end;

procedure TTestKokoroModel.CompareAudio(const aRef: TKokoroRef;
  const aPcm: TSingleDynArray; aTolerance: double; const aLabel: RawUtf8);
var
  ref: TSingleDynArray;
  shape: TInt64DynArray;
  diff, peak, err, sig: double;
  i: PtrInt;
begin
  ref := KokoroRefTensor(aRef, 'audio', shape);
  if not CheckUtf8(length(ref) = length(aPcm), '%: % samples, % expected',
     [aLabel, length(aPcm), length(ref)]) then
    exit;
  QwenDiff(pointer(aPcm), pointer(ref), length(ref), diff, peak);
  err := 0;
  sig := 0;
  for i := 0 to high(ref) do
  begin
    err := err + sqr(aPcm[i] - ref[i]);
    sig := sig + sqr(ref[i]);
  end;
  AddConsole('% %: % samples, max |diff| %, SNR % dB', [aLabel, aRef.Folder,
    length(ref), diff, 10 * Log10((sig + 1e-20) / (err + 1e-20))]);
  CheckUtf8(diff <= aTolerance, '% audio error % > %', [aLabel, diff, aTolerance]);
end;

procedure TTestKokoroModel.TorchFileReadsTheCheckpoint;
var
  f: TTorchFile;
  snap: TFileName;
  t: PTorchTensor;
  total: Int64;
  i: PtrInt;
begin
  snap := KokoroTestSnapshot;
  if (snap = '') or not FileExists(snap + 'kokoro-v1_0.pth') then
    exit;
  f := TTorchFile.Create(snap + 'kokoro-v1_0.pth');
  try
    // 548 float32 tensors, 81.8 M parameters (python: torch.load)
    CheckEqual(f.Count, 548, 'tensors');
    total := 0;
    for i := 0 to f.Count - 1 do
    begin
      Check(f.Item(i)^.DType = tdF32, 'float32');
      inc(total, f.Item(i)^.Count);
    end;
    CheckEqual(total, 81763410, 'parameters');
    t := f.Tensor('decoder.module.decode.0.conv1.weight_v');
    CheckEqual(TorchShapeText(t^.Shape), '[1024,1090,3]');
  finally
    f.Free;
  end;
end;

procedure TTestKokoroModel.IdsAndStyleMatchPython;
var
  ref: TKokoroRef;
  r, i: PtrInt;
  ids: TIntegerDynArray;
  want: PDocVariantData;
  style, refs: TSingleDynArray;
  shape: TInt64DynArray;
  same: boolean;
  u: UnicodeString;
begin
  if Model = nil then
    exit;
  for r := 0 to high(KOKORO_REFS) do
  begin
    if not KokoroLoadRef(KOKORO_REFS[r], ref) then
      continue;
    ids := Model.Ids(ref.Doc.U['phonemes']);
    want := ref.Doc.A['input_ids'];
    same := length(ids) = want^.Count;
    if same then
      for i := 0 to high(ids) do
        if ids[i] <> VariantToIntegerDef(want^.Values[i], -1) then
          same := false;
    CheckUtf8(same, '% ids: % vs %', [KOKORO_REFS[r], IntegerDynArrayToCsv(ids),
      want^.ToJson]);
    // the style: pack[len(phonemes) - 1], len in code points
    u := Utf8ToString(ref.Doc.U['phonemes']);
    refs := KokoroRefTensor(ref, 'ref_s', shape);
    SetLength(style, KOKORO_STYLE);
    MoveFast(fVoice.Style(length(u))^, style[0], KOKORO_STYLE * SizeOf(single));
    Check(CompareMem(pointer(style), pointer(refs), KOKORO_STYLE * SizeOf(single)),
      Utf8ToString('style of ' + KOKORO_REFS[r]));
  end;
end;

procedure TTestKokoroModel.DurationsMatchPython;
var
  ref: TKokoroRef;
  r, i, wrong: PtrInt;
  want: TIntegerDynArray;
  refs: TSingleDynArray;
  shape: TInt64DynArray;
begin
  if Model = nil then
    exit;
  for r := 0 to high(KOKORO_REFS) do
  begin
    if not KokoroLoadRef(KOKORO_REFS[r], ref) then
      continue;
    refs := KokoroRefTensor(ref, 'ref_s', shape);
    want := KokoroRefInts(ref, 'pred.dur');
    Model.Quiet := true;
    Model.ForcedDurations := nil;
    Model.ForcedHar := nil;
    Model.Probe := nil;
    Model.Forward(KokoroRefInts(ref, 'input_ids'), pointer(refs),
      ref.Doc.D['speed']);
    wrong := -1;
    for i := 0 to high(want) do
      if (i > high(Model.LastDurations)) or (Model.LastDurations[i] <> want[i]) then
      begin
        wrong := i;
        break;
      end;
    CheckUtf8(wrong < 0, '%: duration of token % differs (% vs %)',
      [KOKORO_REFS[r], wrong, Model.LastDurations[wrong], want[wrong]]);
  end;
end;

/// the STFT of the reference source, [22][frames] -> [frames][22]
function RefHar(const aRef: TKokoroRef): TSingleDynArray;
var
  har: TSingleDynArray;
  shape: TInt64DynArray;
  t, c, frames: PtrInt;
begin
  har := KokoroRefTensor(aRef, 'gen.har', shape);
  frames := shape[1];
  SetLength(result, length(har));
  for t := 0 to frames - 1 do
    for c := 0 to 21 do
      result[t * 22 + c] := har[c * frames + t];
end;

procedure TTestKokoroModel.EveryStageMatchesPython;
const
  // name, reference [channels][time]
  STAGES: array[0..33] of RawUtf8 = (
    'bert', 'bert_encoder', 'pred.text_encoder', 'pred.lstm',
    'pred.duration_proj', 'pred.en', 'pred.shared', 'pred.F0.0', 'pred.F0.1',
    'pred.F0.2', 'pred.F0', 'pred.N.0', 'pred.N.1', 'pred.N.2', 'pred.N',
    'text_encoder', 'asr', 'dec.F0_conv', 'dec.N_conv', 'dec.encode',
    'dec.asr_res', 'dec.decode.0', 'dec.decode.1', 'dec.decode.2',
    'dec.decode.3', 'gen.noise_conv0', 'gen.noise_res0', 'gen.up0', 'gen.res0',
    'gen.noise_conv1', 'gen.noise_res1', 'gen.up1', 'gen.res1', 'gen.post');
  MAJOR: array[0..33] of boolean = (
    false, true, false, false,
    false, true, false, true, true,
    true, false, true, true, true, false,
    true, true, true, true, true,
    true, true, true, true,
    true, true, true, true, true,
    true, true, true, true, true);
var
  ref: TKokoroRef;
  refs: TSingleDynArray;
  shape: TInt64DynArray;
  i: PtrInt;
begin
  if (Model = nil) or not KokoroLoadRef('fr-siwis', ref) then
    exit;
  refs := KokoroRefTensor(ref, 'ref_s', shape);
  // PyTorch's durations and source STFT: the rest compares as is (the
  // phase of the near-zero frequencies of the STFT is only the rounding
  // noise of PyTorch's FFT)
  fLog.Clear;
  Model.Quiet := true;
  Model.ForcedDurations := KokoroRefInts(ref, 'pred.dur');
  Model.ForcedHar := RefHar(ref);
  Model.Probe := fLog.Collect;
  try
    Model.Forward(KokoroRefInts(ref, 'input_ids'), pointer(refs), ref.Doc.D['speed']);
  finally
    Model.Probe := nil;
    Model.ForcedHar := nil;
  end;
  for i := 0 to high(STAGES) do
    Compare(ref, STAGES[i], MAJOR[i], STAGE_TOLERANCE);
  // then the native source: its phase accumulates over 300 samples per frame
  // the tiny F0 differences, and even more on the 9th harmonic
  fLog.Clear;
  Model.Probe := fLog.Collect;
  try
    Model.Forward(KokoroRefInts(ref, 'input_ids'), pointer(refs), ref.Doc.D['speed']);
  finally
    Model.Probe := nil;
    Model.ForcedDurations := nil;
  end;
  Compare(ref, 'gen.har_source', false, SOURCE_TOLERANCE);
end;

procedure TTestKokoroModel.SoundMatchesPython;
var
  ref: TKokoroRef;
  refs, pcm: TSingleDynArray;
  shape: TInt64DynArray;
  r: PtrInt;
begin
  if Model = nil then
    exit;
  for r := 0 to high(KOKORO_REFS) do
  begin
    if not KokoroLoadRef(KOKORO_REFS[r], ref) then
      continue;
    refs := KokoroRefTensor(ref, 'ref_s', shape);
    Model.Quiet := true;
    Model.ForcedDurations := KokoroRefInts(ref, 'pred.dur');
    Model.ForcedHar := RefHar(ref);
    try
      pcm := Model.Forward(KokoroRefInts(ref, 'input_ids'), pointer(refs),
        ref.Doc.D['speed']);
    finally
      Model.ForcedHar := nil;
    end;
    CompareAudio(ref, pcm, WAV_TOLERANCE, 'forced har');
    // then all native: the STFT of the source, ours
    pcm := Model.Forward(KokoroRefInts(ref, 'input_ids'), pointer(refs),
      ref.Doc.D['speed']);
    Model.ForcedDurations := nil;
    CompareAudio(ref, pcm, 0.5, 'native');
    QwenSaveWav(ref.Folder + 'native-kokoro.wav', pcm);
  end;
end;

procedure TTestKokoroModel.SynthesisSpeed;
var
  ref: TKokoroRef;
  refs, pcm: TSingleDynArray;
  shape: TInt64DynArray;
  timer: TPrecisionTimer;
  seconds, audio: double;
begin
  // no speed assertion: the log
  if (Model = nil) or not KokoroLoadRef('fr-long', ref) then
    exit;
  refs := KokoroRefTensor(ref, 'ref_s', shape);
  Model.Quiet := false;
  Model.Profile := true;
  timer.Start;
  pcm := Model.Forward(KokoroRefInts(ref, 'input_ids'), pointer(refs), 1);
  seconds := timer.StopInMicroSec / 1e6;
  Model.Profile := false;
  AddConsole('stages (ms):%', [Model.LastProfile]);
  audio := length(pcm) / KOKORO_SAMPLE_RATE;
  AddConsole('% s of audio in % s (x% real time), % threads', [audio, seconds,
    audio / seconds, Model.Cpu.Threads]);
  QwenSaveWav(ref.Folder + 'native-kokoro-noise.wav', pcm);
end;

end.
