/// Qwen3-TTS tests - the qwen3-tts-lm dataset: embeddings in, logits out
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - tests/data/infer-ref/qwen3-tts-lm (tools/tts/make_lm_reference.py), read
// by test.infer.ref of mormot2-ai: the talker (28 layers) and the code
// predictor (5 layers, 15 heads) of Qwen3-TTS fed with embedding rows, as
// PyTorch computed them in float32
// - the dataset is checked alone; the stacks need the model (see
// test.tts.qwen.ref), else they are skipped
unit test.tts.qwen.lm;

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
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.threads,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.lm,
  test.infer.ref,
  test.tts.qwen.ref;

type
  /// the tolerances of one run: |a - r| <= Atol + Rtol * |r|, and a cosine
  // of at least MinCosine
  TQwenLmTolerance = record
    StateAtol, StateRtol, StateCosine: double;
    LogitAtol, LogitRtol, LogitCosine: double;
    /// the argmax of each logits row must be the reference code
    SameCodes: boolean;
  end;

  TTestQwenLmReference = class(TSynTestCase)
  protected
    fRef: TInferRefSet;
    fMissing: boolean;
    fLog: TQwenProbeLog;
    function Ref: TInferRefSet;
    function Model: TFileName;
    /// compare aCount values to the reference tensor aName, from aOffset
    function Compare(const aLabel, aName: RawUtf8; aMine: PSingleArray;
      aOffset, aCount: PtrInt; aAtol, aRtol, aCosine: double): TInferCompare;
    /// both stacks with aDType weights, on aVulkan or on the CPU
    procedure Run(const aLabel: RawUtf8; aDType: TInferDType; aVulkan: TQwenVulkan;
      const aTol: TQwenLmTolerance);
  public
    destructor Destroy; override;
  published
    procedure DatasetIsConsistent;
    procedure CpuFloat32MatchesPyTorch;
    procedure CpuQ8MatchesPyTorch;
    procedure VulkanQ8MatchesPyTorch;
  end;


implementation

const
  DATASET = 'qwen3-tts-lm';

destructor TTestQwenLmReference.Destroy;
begin
  fRef.Free;
  fLog.Free;
  inherited Destroy;
end;

function TTestQwenLmReference.Ref: TInferRefSet;
var
  dir: TFileName;
begin
  if (fRef = nil) and not fMissing then
  begin
    dir := FindInferRefDir(DATASET);
    if dir = '' then
    begin
      fMissing := true;
      AddConsole('skipped: no tests/data/infer-ref/% above %', [DATASET,
        Executable.ProgramFilePath]);
    end
    else
      fRef := TInferRefSet.CreateFromFolder(dir);
    fLog := TQwenProbeLog.Create;
  end;
  result := fRef;
end;

function TTestQwenLmReference.Model: TFileName;
begin
  result := QwenTestModel;
  if (result <> '') and not FileExists(result + 'model.safetensors') then
    result := '';
  if result = '' then
    AddConsole(QwenNoModelText);
end;

function ArgMax(p: PSingleArray; n: PtrInt): PtrInt;
var
  i: PtrInt;
begin
  result := 0;
  for i := 1 to n - 1 do
    if p^[i] > p^[result] then
      result := i;
end;

procedure TTestQwenLmReference.DatasetIsConsistent;
var
  bad: RawUtf8;
  n, m, g: PtrInt;
  codes: PIntegerArray;
  logits, embed, norm: PSingleArray;
begin
  if Ref = nil then
    exit;
  Check(Ref.VerifyTensorHashes(bad), Utf8ToString(bad));
  CheckEqual(Ref.Manifest^.U['dataset'], DATASET);
  // the greedy codes of frame 0 are the argmax of each logits row (codes
  // 2048.. of the talker are control tokens, out of the codebook)
  codes := Ref.I32('predictor.codes', n);
  CheckEqual(n, QWEN_CODE_GROUPS, 'codes');
  logits := Ref.F32('talker.logits_last', m);
  CheckEqual(ArgMax(logits, QWEN_CODEBOOK_SIZE), codes^[0], 'talker code');
  logits := Ref.F32('predictor.logits', m);
  CheckEqual(m, (QWEN_CODE_GROUPS - 1) * QWEN_CODEBOOK_SIZE, 'predictor logits');
  for g := 0 to QWEN_CODE_GROUPS - 2 do
    CheckEqual(ArgMax(@logits^[g * QWEN_CODEBOOK_SIZE], QWEN_CODEBOOK_SIZE),
      codes^[g + 1], 'predictor code');
  // the first input row of the predictor is the last normed state of the talker
  embed := Ref.F32('predictor.embed', n);
  norm := Ref.F32('talker.norm', m);
  Check(CompareMem(embed, @norm^[m - 1024], 1024 * SizeOf(single)), 'talker state');
end;

function TTestQwenLmReference.Compare(const aLabel, aName: RawUtf8;
  aMine: PSingleArray; aOffset, aCount: PtrInt;
  aAtol, aRtol, aCosine: double): TInferCompare;
var
  r: PSingleArray;
  n: PtrInt;
begin
  FillCharFast(result, SizeOf(result), 0);
  r := Ref.F32(aName, n);
  if not CheckUtf8(aOffset + aCount <= n, '% too short', [aName]) then
    exit;
  result := InferCompare(aMine, @r^[aOffset], aCount, aAtol, aRtol);
  CheckUtf8(InferCompareOk(result) and (result.Cosine >= aCosine), '% %[%]: %',
    [aLabel, aName, aOffset, InferCompareText(result)]);
end;

procedure TTestQwenLmReference.Run(const aLabel: RawUtf8; aDType: TInferDType;
  aVulkan: TQwenVulkan; const aTol: TQwenLmTolerance);
var
  snap: TFileName;
  f: TSafeTensorsFile;
  cfg: TDocVariantData;
  tc, pc: PDocVariantData;
  pool: TInferThreadPool;
  cpu: TInferCpuFastBackend;
  w: TQwenLmHeads;
  heads: TInferBackend;
  lm: TQwenLm;
  embed: PSingleArray;
  codes: PIntegerArray;
  logits, hidden: TSingleDynArray;
  rows, n, l, g, i, wrong: PtrInt;
  cmp: TInferCompare;
  states, worstAbs, worstCos: double;
  timer: TPrecisionTimer;
begin
  if Ref = nil then
    exit;
  snap := Model;
  if snap = '' then
    exit;
  Check(cfg.InitJsonFromFile(snap + 'config.json', JSON_FAST));
  tc := cfg.O['talker_config'];
  pc := tc^.O['code_predictor_config'];
  f := TSafeTensorsFile.Create(snap + 'model.safetensors');
  pool := TInferThreadPool.Create((SystemInfo.dwNumberOfProcessors * 2 + 2) div 3 - 1);
  cpu := nil;
  if aVulkan = nil then
  begin
    cpu := TInferCpuFastBackend.Create;
    heads := cpu;
  end
  else
    heads := aVulkan.Backend;
  try
    codes := Ref.I32('predictor.codes', n);
    // the talker: 22 rows at once, from position 0
    timer.Start;
    w := TQwenLmHeads.Create('talker', tc^.I['hidden_size'], aDType);
    try
      w.Add(f, 'talker.codec_head.weight', tc^.I['vocab_size'], pool);
      lm := TQwenLm.Create(QwenLmStack(snap, heads, 'talker_config',
        'talker.model.', aDType), w, heads, aVulkan);
      try
        embed := Ref.F32('talker.embed', n);
        rows := n div w.Hidden;
        SetLength(logits, w.Rows(0));
        SetLength(hidden, w.Hidden);
        fLog.Clear;
        lm.Probe := fLog.Collect; // CPU only: the GPU returns the last row
        lm.Forward(embed, rows, 0, pointer(logits), pointer(hidden));
        AddConsole('% talker: % rows in %', [aLabel, rows, timer.Stop]);
        states := 0;
        for l := 0 to lm.Model.Config.Layers - 2 do
        begin
          i := fLog.Find(FormatUtf8('talker.layer%', [l]));
          if i < 0 then
            continue;
          cmp := Compare(aLabel, FormatUtf8('talker.layer%%', [l div 10, l mod 10]),
            pointer(fLog.Data[i]), 0, rows * w.Hidden, aTol.StateAtol,
            aTol.StateRtol, aTol.StateCosine);
          if cmp.MaxAbs > states then
            states := cmp.MaxAbs;
        end;
        if states > 0 then
          AddConsole('% talker.layer00..%: max |diff| %', [aLabel,
            lm.Model.Config.Layers - 2, states]);
        i := fLog.Find('talker.out');
        if i >= 0 then
        begin
          cmp := Compare(aLabel, 'talker.norm', pointer(fLog.Data[i]), 0,
            rows * w.Hidden, aTol.StateAtol, aTol.StateRtol, aTol.StateCosine);
          AddConsole('% talker.norm: %', [aLabel, InferCompareText(cmp)]);
        end;
        cmp := Compare(aLabel, 'talker.norm', pointer(hidden), (rows - 1) * w.Hidden,
          w.Hidden, aTol.StateAtol, aTol.StateRtol, aTol.StateCosine);
        AddConsole('% talker.norm last row: %', [aLabel, InferCompareText(cmp)]);
        cmp := Compare(aLabel, 'talker.logits_last', pointer(logits), 0,
          length(logits), aTol.LogitAtol, aTol.LogitRtol, aTol.LogitCosine);
        AddConsole('% talker.logits_last: %', [aLabel, InferCompareText(cmp)]);
        if aTol.SameCodes then
          CheckEqual(ArgMax(pointer(logits), QWEN_CODEBOOK_SIZE), codes^[0],
            aLabel + ' talker code');
      finally
        lm.Free;
      end;
    finally
      w.Free;
    end;
    // the code predictor: 2 rows, then one per group, each with its head
    timer.Start;
    w := TQwenLmHeads.Create('predictor', pc^.I['hidden_size'], aDType);
    try
      for g := 0 to QWEN_CODE_GROUPS - 2 do
        w.Add(f, FormatUtf8('talker.code_predictor.lm_head.%.weight', [g]),
          pc^.I['vocab_size'], pool);
      lm := TQwenLm.Create(QwenLmStack(snap, heads,
        'talker_config.code_predictor_config', 'talker.code_predictor.model.',
        aDType), w, heads, aVulkan);
      try
        embed := Ref.F32('predictor.embed', n);
        SetLength(logits, w.Rows(0));
        wrong := -1;
        worstAbs := 0;
        worstCos := 1;
        for g := 0 to QWEN_CODE_GROUPS - 2 do
        begin
          if g = 0 then
            lm.Forward(embed, 2, 0, pointer(logits), nil)
          else
            lm.Forward(@embed^[(g + 1) * w.Hidden], 1, g, pointer(logits), nil);
          cmp := Compare(aLabel, 'predictor.logits', pointer(logits),
            g * length(logits), length(logits), aTol.LogitAtol, aTol.LogitRtol,
            aTol.LogitCosine);
          if cmp.MaxAbs > worstAbs then
            worstAbs := cmp.MaxAbs;
          if cmp.Cosine < worstCos then
            worstCos := cmp.Cosine;
          if (wrong < 0) and
             (ArgMax(pointer(logits), QWEN_CODEBOOK_SIZE) <> codes^[g + 1]) then
            wrong := g + 1;
        end;
        AddConsole('% predictor.logits (15 heads in %): max |diff| %, min cosine %, ' +
          'first different code: %', [aLabel, timer.Stop, worstAbs, worstCos, wrong]);
        if aTol.SameCodes then
          CheckUtf8(wrong < 0, '% predictor code of group % differs', [aLabel, wrong]);
        CheckEqual(lm.Length, QWEN_CODE_GROUPS, 'positions');
      finally
        lm.Free;
      end;
    finally
      w.Free;
    end;
  finally
    cpu.Free;
    pool.Free;
    f.Free;
  end;
end;

function Tolerance(aStateAtol, aStateRtol, aStateCosine, aLogitAtol, aLogitRtol,
  aLogitCosine: double; aSameCodes: boolean): TQwenLmTolerance;
begin
  result.StateAtol := aStateAtol;
  result.StateRtol := aStateRtol;
  result.StateCosine := aStateCosine;
  result.LogitAtol := aLogitAtol;
  result.LogitRtol := aLogitRtol;
  result.LogitCosine := aLogitCosine;
  result.SameCodes := aSameCodes;
end;

// measured on 2026-10-18 (Ryzen, RX 9060 XT):
// - F32 on the CPU: layer states within 1.8e-4, final norm within 9.3e-5,
//   logits within 8.3e-5, the same codes
// - Q8_0 on the CPU: states within 3.2 (outlier channels of the first rows),
//   cosine 0.9987; logits within 0.54, cosine 0.9995; the last code differs
// - Q8_0 on Vulkan: last state within 0.38, logits within 0.43, cosine
//   0.9999; the last code differs

procedure TTestQwenLmReference.CpuFloat32MatchesPyTorch;
begin
  Run('CPU F32', idtF32, nil, Tolerance(1e-3, 1e-4, 0.999999, 1e-3, 1e-4,
    0.999999, true));
end;

procedure TTestQwenLmReference.CpuQ8MatchesPyTorch;
begin
  Run('CPU Q8_0', idtQ8_0, nil, Tolerance(8, 0, 0.995, 2, 0, 0.999, false));
end;

procedure TTestQwenLmReference.VulkanQ8MatchesPyTorch;
var
  vk: TQwenVulkan;
begin
  if (Ref = nil) or (Model = '') then
    exit;
  if not TQwenVulkan.Available then
  begin
    AddConsole('skipped: no Vulkan device');
    exit;
  end;
  vk := TQwenVulkan.Create;
  try
    Run('Vulkan Q8_0', idtQ8_0, vk, Tolerance(8, 0, 0.995, 2, 0, 0.999, false));
  finally
    vk.Free;
  end;
end;

end.
