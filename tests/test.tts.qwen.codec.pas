/// Qwen3-TTS tests - the native 12 Hz decoder against the PyTorch reference
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - reference codes -> intermediate outputs and wav, frame by frame, compared
// to PyTorch (float32, CPU); model and references: see test.tts.qwen.ref
unit test.tts.qwen.codec;

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
  mormot.ai.tts.qwen.codec,
  test.tts.qwen.ref;

type
  TTestQwenCodec = class(TSynTestCase)
  protected
    fCodec: TQwenCodec;
    fMissing: boolean;
    fLog: TQwenProbeLog;
    function Codec: TQwenCodec;
    function LoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;
    /// decode all the frames of aRef by packets of aChunk; returns the PCM
    function DecodeAll(const aRef: TQwenRef; aChunk: PtrInt;
      aProbe: boolean): TSingleDynArray;
    /// compare a collected output to the reference: error relative to the peak
    procedure Compare(const aRef: TQwenRef; const aName: RawUtf8; aTolerance: double);
    procedure CompareWav(const aRef: TQwenRef; const aPcm: TSingleDynArray;
      aTolerance: double);
  public
    destructor Destroy; override;
  published
    procedure FrameByFrameMatchesPython;
    procedure ChunksGiveTheSameSound;
    procedure SlidingWindowOfALongSentence;
    procedure AutoLanguageSentence;
    procedure DecodingSpeed;
  end;


implementation

const
  // relative error allowed (to the absolute peak of the reference): float32,
  // sums in another order than PyTorch, over about twenty layers (measured on
  // 2026-10-18: 2.1e-5 at worst, on dec.dec2)
  STAGE_TOLERANCE = 1e-4;
  // absolute error allowed on the wav, amplitude -1..1 (measured: 2.5e-6)
  WAV_TOLERANCE = 1e-5;

destructor TTestQwenCodec.Destroy;
begin
  fCodec.Free;
  fLog.Free;
  inherited Destroy;
end;

function TTestQwenCodec.Codec: TQwenCodec;
var
  snap: TFileName;
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
    else
      fCodec := TQwenCodec.Create(snap + 'speech_tokenizer', QwenTestThreads);
    fLog := TQwenProbeLog.Create;
  end;
  result := fCodec;
end;

function TTestQwenCodec.LoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;
begin
  result := (Codec <> nil) and QwenLoadRef(aName, aRef);
  if (Codec <> nil) and not result then
    AddConsole('skipped: no reference % in % (tools/tts/qwen_ref.py)',
      [aName, QwenTestRefRoot]);
end;

function TTestQwenCodec.DecodeAll(const aRef: TQwenRef; aChunk: PtrInt;
  aProbe: boolean): TSingleDynArray;
var
  f, n: PtrInt;
begin
  fLog.Clear;
  if aProbe then
    Codec.Probe := fLog.Collect
  else
    Codec.Probe := nil;
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
  Codec.Probe := nil;
  CheckEqual(Codec.Frames, aRef.Frames, 'frames');
end;

procedure TTestQwenCodec.Compare(const aRef: TQwenRef; const aName: RawUtf8;
  aTolerance: double);
var
  ref: TSingleDynArray;
  shape: TInt64DynArray;
  i, rows, cols, t, c: PtrInt;
  mine: PSingleArray;
  d, diff, peak: double;
  timeMajor: boolean;
begin
  ref := QwenRefTensor(aRef, aName, shape);
  i := fLog.Find(aName);
  if not CheckUtf8((ref <> nil) and (i >= 0), '% missing', [aName]) then
    exit;
  cols := fLog.Cols[i];
  rows := length(fLog.Data[i]) div cols;
  // dec.tf.*: [N][512] as here; the convolutions: [C][T] (PyTorch)
  timeMajor := (length(shape) = 2) and (shape[0] = rows) and (shape[1] = cols);
  if not CheckUtf8(timeMajor or ((length(shape) = 2) and (shape[0] = cols) and
     (shape[1] = rows)) or ((length(shape) = 1) and (cols = 1) and (shape[0] = rows)),
     '% shape %x% vs %', [aName, rows, cols, length(ref)]) then
    exit;
  mine := pointer(fLog.Data[i]);
  diff := 0;
  peak := 0;
  for t := 0 to rows - 1 do
    for c := 0 to cols - 1 do
    begin
      if timeMajor or (cols = 1) then
        d := ref[t * cols + c]
      else
        d := ref[c * rows + t];
      if abs(d) > peak then
        peak := abs(d);
      d := abs(mine^[t * cols + c] - d);
      if d > diff then
        diff := d;
    end;
  if peak = 0 then
    peak := 1;
  AddConsole('% %x%: max |diff| % (relative %)',
    [aName, rows, cols, diff, diff / peak], true);
  CheckUtf8(diff / peak <= aTolerance, '% relative error % > %',
    [aName, diff / peak, aTolerance]);
end;

procedure TTestQwenCodec.CompareWav(const aRef: TQwenRef;
  const aPcm: TSingleDynArray; aTolerance: double);
var
  ref: TSingleDynArray;
  shape: TInt64DynArray;
  i, worst: PtrInt;
  d, diff: double;
begin
  ref := QwenRefTensor(aRef, 'wav', shape);
  if not CheckUtf8(length(ref) = length(aPcm), 'wav: % samples, % expected',
     [length(aPcm), length(ref)]) then
    exit;
  diff := 0;
  worst := 0;
  for i := 0 to high(ref) do
  begin
    d := abs(aPcm[i] - ref[i]);
    if d > diff then
    begin
      diff := d;
      worst := i;
    end;
  end;
  AddConsole('wav %: % samples, max |diff| % at %', [aRef.Folder, length(ref),
    diff, worst]);
  CheckUtf8(diff <= aTolerance, 'wav error % > %', [diff, aTolerance]);
end;

procedure TTestQwenCodec.FrameByFrameMatchesPython;
const
  STAGES: array[0..19] of RawUtf8 = (
    'dec.quant', 'dec.preconv', 'dec.tf.in', 'dec.tf.layer0', 'dec.tf.layer1',
    'dec.tf.layer2', 'dec.tf.layer3', 'dec.tf.layer4', 'dec.tf.layer5',
    'dec.tf.layer6', 'dec.tf.layer7', 'dec.tf.out', 'dec.up0', 'dec.up1',
    'dec.dec0', 'dec.dec1', 'dec.dec2', 'dec.dec3', 'dec.dec4', 'dec.dec6');
var
  ref: TQwenRef;
  pcm: TSingleDynArray;
  i: PtrInt;
  timer: TPrecisionTimer;
  seconds: double;
begin
  if not LoadRef('fr-vivian', ref) then
    exit;
  timer.Start;
  pcm := DecodeAll(ref, 1, true);
  seconds := timer.StopInMicroSec / 1e6;
  AddConsole('% frames (% s of audio) decoded one by one in % s, % threads',
    [ref.Frames, ref.Frames * QWEN_FRAME_SAMPLES / QWEN_SAMPLE_RATE, seconds,
     Codec.Threads]);
  for i := 0 to high(STAGES) do
    Compare(ref, STAGES[i], STAGE_TOLERANCE);
  CompareWav(ref, pcm, WAV_TOLERANCE);
end;

procedure TTestQwenCodec.ChunksGiveTheSameSound;
var
  ref: TQwenRef;
  one, four, all: TSingleDynArray;
  i: PtrInt;
  same: boolean;
begin
  if not LoadRef('fr-vivian', ref) then
    exit;
  one := DecodeAll(ref, 1, false);
  four := DecodeAll(ref, 4, false);
  all := DecodeAll(ref, ref.Frames, false);
  // each row goes through the same computations, whatever the packet
  same := true;
  for i := 0 to high(one) do
    if (one[i] <> four[i]) or (one[i] <> all[i]) then
    begin
      same := false;
      CheckUtf8(false, 'sample %: % / % / %', [i, one[i], four[i], all[i]]);
      break;
    end;
  Check(same, 'chunks of 1, 4 and all frames give the same samples');
  CompareWav(ref, four, WAV_TOLERANCE);
end;

procedure TTestQwenCodec.SlidingWindowOfALongSentence;
var
  ref: TQwenRef;
  pcm: TSingleDynArray;
begin
  if not LoadRef('fr-long', ref) then
    exit;
  Check(ref.Frames > 72, 'longer than the attention window');
  pcm := DecodeAll(ref, 4, true);
  Compare(ref, 'dec.tf.layer7', STAGE_TOLERANCE);
  Compare(ref, 'dec.tf.out', STAGE_TOLERANCE);
  CompareWav(ref, pcm, WAV_TOLERANCE);
end;

procedure TTestQwenCodec.AutoLanguageSentence;
var
  ref: TQwenRef;
begin
  if not LoadRef('auto-vivian', ref) then
    exit;
  CompareWav(ref, DecodeAll(ref, 2, false), WAV_TOLERANCE);
end;

procedure TTestQwenCodec.DecodingSpeed;
var
  ref: TQwenRef;
  pcm: TSingleDynArray;
  timer: TPrecisionTimer;
  first, total, audio: double;
  f, n, b: PtrInt;
begin
  // no speed assertion (the machine may be busy): the log only
  if not LoadRef('fr-long', ref) then
    exit;
  SetLength(pcm, ref.Frames * QWEN_FRAME_SAMPLES);
  Codec.Probe := nil;
  Codec.Reset;
  timer.Start;
  Codec.Decode(pointer(ref.Codes), 1, pointer(pcm));
  first := timer.StopInMicroSec / 1000;
  audio := (ref.Frames - 1) * QWEN_FRAME_SAMPLES / QWEN_SAMPLE_RATE;
  // the rest by packets of 4 (what the engine does), then of 8 to compare
  b := 4;
  repeat
    if b > 4 then
    begin
      Codec.Reset;
      Codec.Decode(pointer(ref.Codes), 1, pointer(pcm));
    end;
    f := 1;
    timer.Start;
    while f < ref.Frames do
    begin
      n := ref.Frames - f;
      if n > b then
        n := b;
      Codec.Decode(@ref.Codes[f * QWEN_CODE_GROUPS], n, @pcm[f * QWEN_FRAME_SAMPLES]);
      inc(f, n);
    end;
    total := timer.StopInMicroSec / 1e6;
    if b = 4 then
      AddConsole('first frame (80 ms of audio) in % ms; then % s of audio in % s ' +
        'by 4 frames (x% real time), % threads',
        [first, audio, total, audio / total, Codec.Threads])
    else
      AddConsole('by % frames: % s (x% real time)', [b, total, audio / total]);
    CompareWav(ref, pcm, WAV_TOLERANCE);
    b := b * 2;
  until b > 8;
end;

end.
