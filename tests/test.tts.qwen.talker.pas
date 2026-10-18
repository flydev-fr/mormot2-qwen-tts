/// Qwen3-TTS tests - the native talker and code predictor against PyTorch
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - tools/tts/qwen_ref.py, greedy, float32 on the CPU: prompt ids, inputs,
// layers, logits, then the codes of the whole sentence; model and references:
// see test.tts.qwen.ref
unit test.tts.qwen.talker;

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
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.threads,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.lm,
  mormot.ai.tts.qwen.talker,
  test.tts.qwen.ref;

type
  TTestQwenTalker = class(TSynTestCase)
  protected
    fTalker: TQwenTalker;
    fMissing: boolean;
    fLog: TQwenProbeLog;
    function Talker: TQwenTalker;
    function LoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;
    procedure StartRef(const aRef: TQwenRef);
    /// compare aCount collected values (aName, from row aRow) to those of the
    // reference (aRefName, from aRefOffset)
    procedure Compare(const aRef: TQwenRef; const aName: RawUtf8; aRow: PtrInt;
      const aRefName: RawUtf8; aRefOffset, aCount: PtrInt; aTolerance: double);
  public
    destructor Destroy; override;
  published
    procedure PromptIdsMatchPython;
    procedure PrefillMatchesPython;
    procedure FirstFrameMatchesPython;
    procedure GreedyCodesMatchPython;
    procedure ParallelQ8MatchesSerial;
  end;


implementation

const
  // relative error allowed (to the absolute peak of the reference); measured
  // on 2026-10-18: 2.8e-6 at worst, on the code predictor logits
  LM_TOLERANCE = 3e-5;
  REFS: array[0..3] of RawUtf8 = ('fr-vivian', 'auto-vivian', 'fr-accents', 'fr-long');

destructor TTestQwenTalker.Destroy;
begin
  fTalker.Free;
  fLog.Free;
  inherited Destroy;
end;

function TTestQwenTalker.Talker: TQwenTalker;
var
  snap: TFileName;
  timer: TPrecisionTimer;
begin
  if (fTalker = nil) and not fMissing then
  begin
    fLog := TQwenProbeLog.Create;
    snap := QwenTestModel;
    if (snap = '') or not FileExists(snap + 'model.safetensors') then
    begin
      fMissing := true;
      AddConsole(QwenNoModelText);
    end
    else
    begin
      timer.Start;
      fTalker := TQwenTalker.Create(snap, QwenCpuOptions);
      fTalker.Sampling := QwenGreedy; // as qwen_ref.py
      AddConsole('talker and code predictor loaded in %', [timer.Stop]);
    end;
  end;
  result := fTalker;
end;

function TTestQwenTalker.LoadRef(const aName: RawUtf8; out aRef: TQwenRef): boolean;
begin
  result := (Talker <> nil) and QwenLoadRef(aName, aRef);
  if (Talker <> nil) and not result then
    AddConsole('skipped: no reference % in % (tools/tts/qwen_ref.py)',
      [aName, QwenTestRefRoot]);
end;

procedure TTestQwenTalker.StartRef(const aRef: TQwenRef);
begin
  Talker.Start(aRef.Doc.U['text'], aRef.Doc.U['speaker'], aRef.Doc.U['language'],
    aRef.Doc.B['non_streaming_mode']);
end;

procedure TTestQwenTalker.Compare(const aRef: TQwenRef; const aName: RawUtf8;
  aRow: PtrInt; const aRefName: RawUtf8; aRefOffset, aCount: PtrInt;
  aTolerance: double);
var
  ref: TSingleDynArray;
  shape: TInt64DynArray;
  i: PtrInt;
  diff, peak: double;
begin
  ref := QwenRefTensor(aRef, aRefName, shape);
  i := fLog.Find(aName);
  if not CheckUtf8((ref <> nil) and (i >= 0), '% / % missing', [aName, aRefName]) then
    exit;
  if not CheckUtf8((aRow * fLog.Cols[i] + aCount <= length(fLog.Data[i])) and
     (aRefOffset + aCount <= length(ref)), '% too short', [aName]) then
    exit;
  QwenDiff(@fLog.Data[i][aRow * fLog.Cols[i]], @ref[aRefOffset], aCount, diff, peak);
  if peak = 0 then
    peak := 1;
  AddConsole('% (row %) vs % [%]: max |diff| % (relative %)',
    [aName, aRow, aRefName, aRefOffset, diff, diff / peak], true);
  CheckUtf8(diff / peak <= aTolerance, '% relative error % > %',
    [aName, diff / peak, aTolerance]);
end;

procedure TTestQwenTalker.PromptIdsMatchPython;
var
  ref: TQwenRef;
  r, i: PtrInt;
  mine: TIntegerDynArray;
  expected: PDocVariantData;
  same: boolean;
begin
  for r := 0 to high(REFS) do
  begin
    if not LoadRef(REFS[r], ref) then
      continue;
    mine := Talker.PromptIds(ref.Doc.U['text']);
    expected := ref.Doc.A['prompt_ids'];
    same := length(mine) = expected^.Count;
    if same then
      for i := 0 to high(mine) do
        if mine[i] <> VariantToIntegerDef(expected^.Values[i], -1) then
        begin
          same := false;
          break;
        end;
    CheckUtf8(same, '% ids: % vs %', [REFS[r], IntegerDynArrayToCsv(mine),
      expected^.ToJson]);
  end;
end;

procedure TTestQwenTalker.PrefillMatchesPython;
var
  ref: TQwenRef;
  shape: TInt64DynArray;
  rows, hidden, l, layers: PtrInt;
begin
  if not LoadRef('fr-vivian', ref) then
    exit;
  QwenRefTensor(ref, 'talker.prefill.embeds', shape);
  rows := shape[0];
  hidden := shape[1];
  QwenRefTensor(ref, 'talker.prefill.layers', shape);
  layers := shape[0] - 1;
  fLog.Clear;
  Talker.Probe := fLog.Collect;
  try
    StartRef(ref);
  finally
    Talker.Probe := nil;
  end;
  Compare(ref, 'talker.prefill.embeds', 0, 'talker.prefill.embeds', 0,
    rows * hidden, LM_TOLERANCE);
  Compare(ref, 'talker.trailing', 0, 'talker.tts_pad', 0, hidden, LM_TOLERANCE);
  // hidden_states of transformers: the input of each layer, then the normed
  // output of the last one
  for l := 0 to layers - 2 do
    Compare(ref, FormatUtf8('talker.layer%', [l]), 0, 'talker.prefill.layers',
      (l + 1) * rows * hidden, rows * hidden, LM_TOLERANCE);
  Compare(ref, 'talker.out', 0, 'talker.prefill.layers', layers * rows * hidden,
    rows * hidden, LM_TOLERANCE);
  QwenRefTensor(ref, 'talker.prefill.logits', shape);
  Compare(ref, 'talker.logits', 0, 'talker.prefill.logits', 0, shape[0],
    LM_TOLERANCE);
end;

procedure TTestQwenTalker.FirstFrameMatchesPython;
var
  ref: TQwenRef;
  codes: array[0..QWEN_CODE_GROUPS - 1] of integer;
  g: PtrInt;
begin
  if not LoadRef('fr-vivian', ref) then
    exit;
  StartRef(ref);
  fLog.Clear;
  Talker.Probe := fLog.Collect;
  try
    Check(Talker.NextFrame(@codes), 'first frame');
  finally
    Talker.Probe := nil;
  end;
  for g := 0 to QWEN_CODE_GROUPS - 1 do
    CheckUtf8(codes[g] = ref.Codes[g], 'group %: % instead of %',
      [g, codes[g], ref.Codes[g]]);
  Compare(ref, 'predictor.embeds', 0, 'predictor.f0.embeds', 0, 2 * 1024, LM_TOLERANCE);
  Compare(ref, 'predictor.logits', 0, 'predictor.f0.logits', 0,
    (QWEN_CODE_GROUPS - 1) * QWEN_CODEBOOK_SIZE, LM_TOLERANCE);
  Compare(ref, 'talker.step.embeds', 0, 'talker.step.embeds', 0, 1024, LM_TOLERANCE);
  Compare(ref, 'talker.logits', 0, 'talker.step.logits', 0, 3072, LM_TOLERANCE);
end;

procedure TTestQwenTalker.GreedyCodesMatchPython;
var
  ref: TQwenRef;
  r, n, i, wrong: PtrInt;
  codes: TIntegerDynArray;
  timer: TPrecisionTimer;
  seconds, audio: double;
begin
  for r := 0 to high(REFS) do
  begin
    if not LoadRef(REFS[r], ref) then
      continue;
    SetLength(codes, (ref.Frames + 20) * QWEN_CODE_GROUPS);
    timer.Start;
    StartRef(ref);
    n := 0;
    while (n < ref.Frames + 20) and Talker.NextFrame(@codes[n * QWEN_CODE_GROUPS]) do
      inc(n);
    seconds := timer.StopInMicroSec / 1e6;
    audio := n * QWEN_FRAME_SAMPLES / QWEN_SAMPLE_RATE;
    wrong := -1;
    for i := 0 to MinPtrInt(n, ref.Frames) * QWEN_CODE_GROUPS - 1 do
      if codes[i] <> ref.Codes[i] then
      begin
        wrong := i;
        break;
      end;
    AddConsole('%: % frames (% s of audio) in % s, x% real time; first difference %',
      [REFS[r], n, audio, seconds, audio / seconds, wrong]);
    CheckEqual(n, ref.Frames, REFS[r] + ' frames');
    CheckUtf8(wrong < 0, '%: code % differs (frame %, group %)',
      [REFS[r], wrong, wrong div QWEN_CODE_GROUPS, wrong mod QWEN_CODE_GROUPS]);
  end;
end;

procedure TTestQwenTalker.ParallelQ8MatchesSerial;
var
  snap: TFileName;
  f: TSafeTensorsFile;
  cfg: TDocVariantData;
  pool, p: TInferThreadPool;
  digest: array[boolean] of cardinal;
  ms: array[boolean] of double;
  parallel: boolean;
  w: TQwenLmWeights;
  g: PtrInt;
  timer: TPrecisionTimer;
begin
  // the code predictor (5 layers, 15 heads, the shapes of the talker) in Q8_0,
  // converted on the calling thread then on 8 threads: the same bytes
  if Talker = nil then
    exit;
  snap := QwenTestModel;
  Check(cfg.InitJsonFromFile(snap + 'config.json', JSON_FAST));
  f := TSafeTensorsFile.Create(snap + 'model.safetensors');
  pool := TInferThreadPool.Create(7);
  try
    for parallel := false to true do
    begin
      if parallel then
        p := pool
      else
        p := nil;
      timer.Start;
      w := TQwenLmWeights.Create(f, 'predictor', 'talker.code_predictor.model.',
        cfg.O['talker_config']^.O['code_predictor_config']^, idtQ8_0, p);
      try
        for g := 0 to QWEN_CODE_GROUPS - 2 do
          w.AddOutput(f, FormatUtf8('talker.code_predictor.lm_head.%.weight', [g]),
            QWEN_CODEBOOK_SIZE, p);
        ms[parallel] := timer.StopInMicroSec / 1000;
        digest[parallel] := w.Digest;
      finally
        w.Free;
      end;
    end;
  finally
    pool.Free;
    f.Free;
  end;
  AddConsole('predictor Q8_0: % ms on one thread, % ms on 8, crc32c % / %',
    [round(ms[false]), round(ms[true]), CardinalToHex(digest[false]),
     CardinalToHex(digest[true])]);
  CheckEqual(digest[true], digest[false], 'Q8_0 bytes');
end;

end.
