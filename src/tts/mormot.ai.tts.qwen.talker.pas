/// Qwen3-TTS - the language model of Qwen3-TTS-12Hz CustomVoice: text to codes
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the talker (Qwen3, 28 layers) predicts the group 0 code of each frame;
// the code predictor (Qwen3, 5 layers) derives the 15 others, one group after
// the other; their 16 embeddings summed, plus the following text, form the
// input of the next frame
// - reference: qwen_tts 0.1.1, Qwen3TTSForConditionalGeneration.generate in
// core/models/modeling_qwen3_tts.py and the prompt of
// inference/qwen3_tts_model.py
// - both stacks run on the CPU (float32: in greedy mode, the same codes as
// PyTorch; or Q8_0) or on the GPU through Vulkan (Q8_0); the embeddings, the
// text projection and the choice of the codes stay on the CPU
// - the multimodal RoPE of the talker (mrope_section [24, 20, 20], interleaved)
// reduces to the plain RoPE: a single sentence, without padding, gives three
// identical positions
unit mormot.ai.tts.qwen.talker;

{$I mormot.defines.inc}

interface

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.threads,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.sampler,
  mormot.ai.tts.qwen.codec,
  mormot.ai.tts.qwen.lm,
  mormot.ai.tts.qwen.lm.vulkan;

type
  EQwenTalker = class(ESynException);

  /// the Qwen text tokenizer (Qwen2Tokenizer) from vocab.json, merges.txt and
  // the added tokens of tokenizer_config.json: the TTS model has no
  // tokenizer.json, the file TBpeTokenizer.Create reads
  TQwenTokenizer = class(TBpeTokenizer)
  public
    constructor CreateFromFolder(const aFolder: TFileName);
  end;

  /// where the talker and the code predictor run
  TQwenDevice = (
    qdCpu,
    qdVulkan);

  /// the hardware and the weights format
  TQwenTalkerOptions = record
    Device: TQwenDevice;
    /// idtF32 (CPU only) or idtQ8_0
    DType: TInferDType;
    /// the CPU threads, 0 = two thirds of the cores
    Threads: integer;
    /// Vulkan: the K/V cache in half floats
    KvF16: boolean;
  end;

  /// how each code is chosen (generation_config.json)
  TQwenSampling = record
    /// false: greedy (the argmax); otherwise temperature, top-k, top-p
    DoSample: boolean;
    Temperature, TopP: double;
    TopK: integer;
    /// the talker only, on the group 0 codes already returned
    RepetitionPenalty: double;
    /// the code predictor
    SubDoSample: boolean;
    SubTemperature, SubTopP: double;
    SubTopK: integer;
    /// 0: a different seed for each sentence
    Seed: Int64;
  end;

  /// talker and code predictor of Qwen3-TTS-12Hz CustomVoice: a prompt
  // (Start), then one frame of 16 codes per NextFrame, until the end
  // - not to be shared between threads
  TQwenTalker = class
  protected
    fFile: TSafeTensorsFile;
    fOptions: TQwenTalkerOptions;
    fOwnCpu: TInferCpuFastBackend;
    fCpu: TInferCpuFastBackend;
    fVulkan: TQwenVulkan;
    fConfig: TDocVariantData;
    fTokenizer: TQwenTokenizer;
    fImStart, fImEnd: integer;
    fTtsBosId, fTtsEosId, fTtsPadId: integer;
    fCodecBos, fCodecEos, fCodecPad, fThink, fNoThink, fThinkBos, fThinkEos: integer;
    fVocab, fGroups, fPredVocab: PtrInt;
    fHidden, fTextHidden: PtrInt;
    // text: BF16 embeddings [151936][2048] of the file, then the projection
    fTextEmb: PInferTensor;
    fFc1W, fFc1B, fFc2W, fFc2B: TSingleDynArray;
    fTtsBos, fTtsEos, fTtsPad: TSingleDynArray;
    // code embeddings: of the talker [3072][1024], of the code predictor
    // (groups 1..15) [2048][1024]
    fCodecEmb: TSingleDynArray;
    fPredEmb: array of TSingleDynArray;
    // the two stacks
    fTalkerW, fPredW: TQwenLmWeights;
    fTalker, fPredictor: TQwenLm;
    // the choice of the codes
    fSampling: TQwenSampling;
    fSampler, fSubSampler: TInferSampler;
    // the running sentence
    fTrailing: TSingleDynArray;
    fTrailingRows, fStep: PtrInt;
    fPastHidden, fLogits, fWork, fPredLogits, fX: TSingleDynArray;
    fHistory: TIntegerDynArray;
    fProbe: TOnQwenProbe;
    procedure SetProbe(const aProbe: TOnQwenProbe);
    procedure Emit(const aName: RawUtf8; aData: PSingleArray; aRows, aCols: PtrInt);
    function NewLm(aWeights: TQwenLmWeights): TQwenLm;
    procedure TextProject(const aIds: array of integer; y: PSingleArray);
    procedure AddRow(const aEmb: TSingleDynArray; aId: PtrInt; y: PSingleArray);
    function ChooseCode0: integer;
    function ChooseCode(aLogits: PSingleArray): integer;
  public
    /// load the model of aSnapshot (the Hugging Face cache folder with
    // config.json, generation_config.json, model.safetensors, vocab.json,
    // merges.txt); the sampling is the one of generation_config.json
    constructor Create(const aSnapshot: TFileName;
      const aOptions: TQwenTalkerOptions);
    destructor Destroy; override;
    /// the ids of the prompt '<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n'
    function PromptIds(const aText: RawUtf8): TIntegerDynArray;
    /// prepare the sentence and compute the prefill: the first frame is ready
    // - aSpeaker: vivian, ryan... ('' : none); aLanguage: French, English...
    // or Auto; aNonStreaming: the whole text in the prefill (the default of
    // generate_custom_voice), otherwise one text token per frame
    procedure Start(const aText, aSpeaker, aLanguage: RawUtf8;
      aNonStreaming: boolean = true);
    /// the next frame: aCodes[0..15]; false at the end of the sentence
    function NextFrame(aCodes: PIntegerArray): boolean;
    /// the names of the voices (spk_id of config.json)
    function Speakers: TRawUtf8DynArray;
    property Tokenizer: TQwenTokenizer
      read fTokenizer;
    property Options: TQwenTalkerOptions
      read fOptions;
    /// the GPU, nil on the CPU
    property Vulkan: TQwenVulkan
      read fVulkan;
    /// the weights of the two stacks
    property TalkerWeights: TQwenLmWeights
      read fTalkerW;
    property PredictorWeights: TQwenLmWeights
      read fPredW;
    /// to be changed before Start
    property Sampling: TQwenSampling
      read fSampling write fSampling;
    /// the frames already returned since Start
    property Step: PtrInt
      read fStep;
    /// the intermediate outputs (tests); nil by default
    property Probe: TOnQwenProbe
      read fProbe write SetProbe;
  end;

/// CPU, float32: the same codes as PyTorch in greedy mode
function QwenCpuOptions(aDType: TInferDType = idtF32): TQwenTalkerOptions;

/// GPU through Vulkan, Q8_0, K/V cache in float32
function QwenVulkanOptions: TQwenTalkerOptions;

/// greedy everywhere, as tools/tts/qwen_ref.py
function QwenGreedy: TQwenSampling;

/// the values of generation_config.json; those of qwen_tts when the file is
// missing (sampling, temperature 0.9, top-k 50, penalty 1.05)
function QwenSamplingFromConfig(const aFileName: TFileName): TQwenSampling;


implementation

function QwenCpuOptions(aDType: TInferDType): TQwenTalkerOptions;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.Device := qdCpu;
  result.DType := aDType;
end;

function QwenVulkanOptions: TQwenTalkerOptions;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.Device := qdVulkan;
  result.DType := idtQ8_0;
  // K/V cache in float32: 230 KB per talker position, and 100 times less
  // drift than in half floats (measured: 2.6e-5 against 2.3e-2)
  result.KvF16 := false;
end;

function QwenGreedy: TQwenSampling;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.Temperature := 1;
  result.TopP := 1;
  result.RepetitionPenalty := 1.05; // generate() applies it in greedy mode too
  result.SubTemperature := 1;
  result.SubTopP := 1;
end;

function QwenSamplingFromConfig(const aFileName: TFileName): TQwenSampling;
var
  doc: TDocVariantData;
begin
  // the default values of qwen_tts (Qwen3TTSModel._merge_generate_kwargs)
  result.DoSample := true;
  result.Temperature := 0.9;
  result.TopK := 50;
  result.TopP := 1;
  result.RepetitionPenalty := 1.05;
  result.SubDoSample := true;
  result.SubTemperature := 0.9;
  result.SubTopK := 50;
  result.SubTopP := 1;
  result.Seed := 0;
  if not doc.InitJsonFromFile(aFileName, JSON_FAST) then
    exit;
  doc.GetAsBoolean('do_sample', result.DoSample);
  doc.GetAsDouble('temperature', result.Temperature);
  doc.GetAsInteger('top_k', result.TopK);
  doc.GetAsDouble('top_p', result.TopP);
  doc.GetAsDouble('repetition_penalty', result.RepetitionPenalty);
  doc.GetAsBoolean('subtalker_dosample', result.SubDoSample);
  doc.GetAsDouble('subtalker_temperature', result.SubTemperature);
  doc.GetAsInteger('subtalker_top_k', result.SubTopK);
  doc.GetAsDouble('subtalker_top_p', result.SubTopP);
end;


{ TQwenTokenizer }

constructor TQwenTokenizer.CreateFromFolder(const aFolder: TFileName);
var
  vocab, cfg: TDocVariantData;
  added: PDocVariantData;
  texts, a, b, addedText: TRawUtf8DynArray;
  addedId: TIntegerDynArray;
  merges, line: RawUtf8;
  i, n, id, p, q, sp: PtrInt;
begin
  // no inherited Create: the one of TBpeTokenizer reads a tokenizer.json, the
  // one of TInferTokenizer does nothing more than TObject.Create
  if not vocab.InitJsonFromFile(aFolder + 'vocab.json', JSON_FAST) then
    EQwenTalker.RaiseUtf8('%: cannot read %vocab.json', [self, aFolder]);
  SetLength(texts, vocab.Count);
  for i := 0 to vocab.Count - 1 do
  begin
    id := VariantToIntegerDef(vocab.Values[i], -1);
    if (id < 0) or (id >= vocab.Count) or (texts[id] <> '') then
      EQwenTalker.RaiseUtf8('%: vocab id % of "%" invalid or used twice',
        [self, id, vocab.Names[i]]);
    texts[id] := vocab.Names[i];
  end;
  // merges.txt: one "a b" pair per line, after an optional "#version"
  merges := StringFromFile(aFolder + 'merges.txt');
  if merges = '' then
    EQwenTalker.RaiseUtf8('%: cannot read %merges.txt', [self, aFolder]);
  n := 0;
  p := 1;
  while p <= system.length(merges) do
  begin
    q := PosEx(#10, merges, p);
    if q = 0 then
      q := system.length(merges) + 1;
    line := copy(merges, p, q - p);
    p := q + 1;
    if (line <> '') and (line[system.length(line)] = #13) then
      SetLength(line, system.length(line) - 1);
    if (line = '') or IdemPChar(pointer(line), '#VERSION') then
      continue;
    sp := PosEx(' ', line);
    if (sp <= 1) or (sp = system.length(line)) then
      EQwenTalker.RaiseUtf8('%: merge "%" is not "a b"', [self, line]);
    if n = system.length(a) then
    begin
      SetLength(a, n * 2 + 1024);
      SetLength(b, n * 2 + 1024);
    end;
    a[n] := copy(line, 1, sp - 1);
    b[n] := copy(line, sp + 1, maxInt);
    inc(n);
  end;
  SetLength(a, n);
  SetLength(b, n);
  // the special tokens: added_tokens_decoder {"151643": {"content": ...}}
  if not cfg.InitJsonFromFile(aFolder + 'tokenizer_config.json', JSON_FAST) then
    EQwenTalker.RaiseUtf8('%: cannot read %tokenizer_config.json', [self, aFolder]);
  added := cfg.O['added_tokens_decoder'];
  SetLength(addedText, added^.Count);
  SetLength(addedId, added^.Count);
  for i := 0 to added^.Count - 1 do
  begin
    addedId[i] := Utf8ToInteger(added^.Names[i], -1);
    addedText[i] := _Safe(added^.Values[i])^.U['content'];
  end;
  Build(texts, a, b, addedText, addedId);
end;


{ TQwenTalker }

constructor TQwenTalker.Create(const aSnapshot: TFileName;
  const aOptions: TQwenTalkerOptions);
var
  folder: TFileName;
  tc, pc: PDocVariantData;
  g: PtrInt;
  threads: integer;
  pool: TInferThreadPool;
begin
  inherited Create;
  fOptions := aOptions;
  if (aOptions.Device = qdVulkan) and (aOptions.DType <> idtQ8_0) then
    EQwenTalker.RaiseUtf8('%: Vulkan runs Q8_0 weights only', [self]);
  folder := IncludeTrailingPathDelimiter(aSnapshot);
  if not fConfig.InitJsonFromFile(folder + 'config.json', JSON_FAST) then
    EQwenTalker.RaiseUtf8('%: cannot read %config.json', [self, folder]);
  tc := fConfig.O['talker_config'];
  pc := tc^.O['code_predictor_config'];
  fTtsBosId := fConfig.I['tts_bos_token_id'];
  fTtsEosId := fConfig.I['tts_eos_token_id'];
  fTtsPadId := fConfig.I['tts_pad_token_id'];
  fCodecBos := tc^.I['codec_bos_id'];
  fCodecEos := tc^.I['codec_eos_token_id'];
  fCodecPad := tc^.I['codec_pad_id'];
  fThink := tc^.I['codec_think_id'];
  fNoThink := tc^.I['codec_nothink_id'];
  fThinkBos := tc^.I['codec_think_bos_id'];
  fThinkEos := tc^.I['codec_think_eos_id'];
  fVocab := tc^.I['vocab_size'];
  fGroups := tc^.I['num_code_groups'];
  fHidden := tc^.I['hidden_size'];
  fTextHidden := tc^.I['text_hidden_size'];
  fPredVocab := pc^.I['vocab_size'];
  if (fGroups <> QWEN_CODE_GROUPS) or (fPredVocab <> QWEN_CODEBOOK_SIZE) or
     (pc^.I['hidden_size'] <> fHidden) or (fCodecEos <= 0) or (fVocab <= 0) then
    EQwenTalker.RaiseUtf8('%: unexpected configuration in %config.json', [self, folder]);
  fSampling := QwenSamplingFromConfig(folder + 'generation_config.json');
  fTokenizer := TQwenTokenizer.CreateFromFolder(folder);
  fImStart := fTokenizer.TokenId('<|im_start|>');
  fImEnd := fTokenizer.TokenId('<|im_end|>');
  if (fImStart < 0) or (fImEnd < 0) then
    EQwenTalker.RaiseUtf8('%: no <|im_start|> / <|im_end|> token', [self]);
  // the CPU backend: the one of the GPU if any (a single thread pool)
  if aOptions.Device = qdVulkan then
  begin
    fVulkan := TQwenVulkan.Create(aOptions.Threads);
    fCpu := fVulkan.Backend;
  end
  else
  begin
    fOwnCpu := TInferCpuFastBackend.Create(aOptions.Threads);
    fCpu := fOwnCpu;
  end;
  fFile := TSafeTensorsFile.Create(folder + 'model.safetensors');
  fTextEmb := fFile.Tensor('talker.model.text_embedding.weight');
  if not InferShapeIs(fTextEmb^, [tc^.I['text_vocab_size'], fTextHidden]) then
    EQwenTalker.RaiseUtf8('%: text_embedding is %', [self, InferShapeText(fTextEmb^.Shape)]);
  fFc1W := QwenWiden(fFile, 'talker.text_projection.linear_fc1.weight', fTextHidden, fTextHidden);
  fFc1B := QwenWiden(fFile, 'talker.text_projection.linear_fc1.bias', 1, fTextHidden);
  fFc2W := QwenWiden(fFile, 'talker.text_projection.linear_fc2.weight', fHidden, fTextHidden);
  fFc2B := QwenWiden(fFile, 'talker.text_projection.linear_fc2.bias', 1, fHidden);
  fCodecEmb := QwenWiden(fFile, 'talker.model.codec_embedding.weight', fVocab, fHidden);
  SetLength(fPredEmb, fGroups - 1);
  for g := 0 to fGroups - 2 do
    fPredEmb[g] := QwenWiden(fFile, FormatUtf8(
      'talker.code_predictor.model.codec_embedding.%.weight', [g]), fPredVocab, fHidden);
  // the stacks and their heads: codec_head for the talker, lm_head.0..14 for
  // the code predictor; converted (BF16 -> Q8_0 or F32) on a pool during the
  // load
  threads := aOptions.Threads;
  if threads <= 0 then
    threads := (SystemInfo.dwNumberOfProcessors * 2 + 2) div 3;
  pool := nil;
  if threads > 1 then
    pool := TInferThreadPool.Create(threads - 1);
  try
    fTalkerW := TQwenLmWeights.Create(fFile, 'talker', 'talker.model.', tc^,
      aOptions.DType, pool);
    fTalkerW.AddOutput(fFile, 'talker.codec_head.weight', fVocab, pool);
    fPredW := TQwenLmWeights.Create(fFile, 'predictor', 'talker.code_predictor.model.',
      pc^, aOptions.DType, pool);
    for g := 0 to fGroups - 2 do
      fPredW.AddOutput(fFile, FormatUtf8('talker.code_predictor.lm_head.%.weight',
        [g]), fPredVocab, pool);
  finally
    pool.Free;
  end;
  fTalker := NewLm(fTalkerW);
  fPredictor := NewLm(fPredW);
  // tts_bos / tts_eos / tts_pad through the text projection
  SetLength(fTtsBos, fHidden);
  SetLength(fTtsEos, fHidden);
  SetLength(fTtsPad, fHidden);
  TextProject([fTtsBosId], pointer(fTtsBos));
  TextProject([fTtsEosId], pointer(fTtsEos));
  TextProject([fTtsPadId], pointer(fTtsPad));
  SetLength(fPastHidden, fHidden);
  SetLength(fLogits, fVocab);
  SetLength(fWork, fVocab);
  SetLength(fPredLogits, fPredVocab);
  SetLength(fX, 2 * fHidden);
end;

destructor TQwenTalker.Destroy;
begin
  fTalker.Free;
  fPredictor.Free;
  fTalkerW.Free;
  fPredW.Free;
  fSampler.Free;
  fSubSampler.Free;
  fTokenizer.Free;
  fFile.Free;
  fVulkan.Free;
  fOwnCpu.Free;
  inherited Destroy;
end;

function TQwenTalker.NewLm(aWeights: TQwenLmWeights): TQwenLm;
begin
  if fVulkan <> nil then
    result := TQwenLmVulkan.CreateVulkan(aWeights, fVulkan, fOptions.KvF16)
  else
    result := TQwenLmCpu.CreateCpu(aWeights, fCpu);
end;

procedure TQwenTalker.SetProbe(const aProbe: TOnQwenProbe);
begin
  fProbe := aProbe;
  fTalker.Probe := aProbe;
  fPredictor.Probe := aProbe;
end;

procedure TQwenTalker.Emit(const aName: RawUtf8; aData: PSingleArray;
  aRows, aCols: PtrInt);
begin
  if Assigned(fProbe) then
    fProbe(aName, aData, aRows, aCols);
end;

function TQwenTalker.Speakers: TRawUtf8DynArray;
var
  spk: PDocVariantData;
begin
  spk := fConfig.O['talker_config']^.O['spk_id'];
  result := copy(spk^.Names, 0, spk^.Count);
end;

procedure TQwenTalker.TextProject(const aIds: array of integer; y: PSingleArray);
var
  n, r, i: PtrInt;
  e, h: TSingleDynArray;
  v: single;
begin
  // Qwen3TTSTalkerResizeMLP: fc2(silu(fc1(embedding)))
  n := system.length(aIds);
  if n = 0 then
    exit;
  SetLength(e, n * fTextHidden);
  SetLength(h, n * fTextHidden);
  for r := 0 to n - 1 do
  begin
    if (aIds[r] < 0) or (aIds[r] >= fTextEmb^.Shape[0]) then
      EQwenTalker.RaiseUtf8('%: text token % out of range', [self, aIds[r]]);
    QwenRowToF32(fTextEmb^, aIds[r], fTextHidden, @e[r * fTextHidden]);
  end;
  fCpu.Linear(pointer(e), n, fTextHidden, InferWeight(pointer(fFc1W), idtF32),
    fTextHidden, pointer(fFc1B), pointer(h));
  for i := 0 to n * fTextHidden - 1 do
  begin
    v := h[i];
    if v < -80 then
      h[i] := 0 // silu(v) = v / (1 + exp(-v)): exp would overflow
    else
      h[i] := v / (1 + Exp(-v));
  end;
  fCpu.Linear(pointer(h), n, fTextHidden, InferWeight(pointer(fFc2W), idtF32),
    fHidden, pointer(fFc2B), y);
end;

procedure TQwenTalker.AddRow(const aEmb: TSingleDynArray; aId: PtrInt; y: PSingleArray);
var
  c: PtrInt;
begin
  for c := 0 to fHidden - 1 do
    y^[c] := y^[c] + aEmb[aId * fHidden + c];
end;

function TQwenTalker.PromptIds(const aText: RawUtf8): TIntegerDynArray;
var
  n: PtrInt;
begin
  // as the Hugging Face tokenizer: split at the special tokens, encode each
  // piece
  result := nil;
  SetLength(result, 16);
  result[0] := fImStart;
  n := 1;
  fTokenizer.EncodeAppend('assistant'#10 + aText, result, n);
  if n >= system.length(result) then
    SetLength(result, n + 8);
  result[n] := fImEnd;
  inc(n);
  fTokenizer.EncodeAppend(#10, result, n);
  if n >= system.length(result) then
    SetLength(result, n + 8);
  result[n] := fImStart;
  inc(n);
  fTokenizer.EncodeAppend('assistant'#10, result, n);
  SetLength(result, n);
end;

function NewSampler(aDoSample: boolean; aTemperature, aTopP: double;
  aTopK: integer; aSeed: Int64): TInferSampler;
var
  s: TInferSampling;
begin
  result := nil;
  if not aDoSample then
    exit;
  s := InferSamplingGreedy;
  s.Temperature := aTemperature;
  s.TopK := aTopK;
  s.TopP := aTopP;
  s.Seed := aSeed;
  result := TInferSampler.Create(s);
end;

procedure TQwenTalker.Start(const aText, aSpeaker, aLanguage: RawUtf8;
  aNonStreaming: boolean);
var
  ids, codec, text: TIntegerDynArray;
  lang, spk, dialect: RawUtf8;
  tc: PDocVariantData;
  langId, spkId, nCodec, nText, rows, r, i: PtrInt;
  v: integer;
  seed: Int64;
  prefill: TSingleDynArray;
  row: PSingleArray;
begin
  tc := fConfig.O['talker_config'];
  ids := PromptIds(aText);
  // [im_start, assistant, \n] text [im_end, \n, im_start, assistant, \n]
  if system.length(ids) < 9 then
    EQwenTalker.RaiseUtf8('%.Start: empty text', [self]);
  // the voice and the language: codec_think, think_bos, language, think_eos
  // (or nothink, think_bos, think_eos in Auto), the voice, pad, bos
  spk := LowerCase(aSpeaker);
  spkId := -1;
  if spk <> '' then
  begin
    if not tc^.O['spk_id']^.GetAsInteger(spk, v) then
      EQwenTalker.RaiseUtf8('%.Start: unknown speaker "%"', [self, aSpeaker]);
    spkId := v;
  end;
  lang := LowerCase(aLanguage);
  if lang = '' then
    lang := 'auto';
  langId := -1;
  if lang <> 'auto' then
  begin
    if not tc^.O['codec_language_id']^.GetAsInteger(lang, v) then
      EQwenTalker.RaiseUtf8('%.Start: unknown language "%"', [self, aLanguage]);
    langId := v;
  end;
  if (spk <> '') and ((lang = 'chinese') or (lang = 'auto')) then
  begin
    dialect := tc^.O['spk_is_dialect']^.U[spk];
    if (dialect <> '') and (dialect <> 'false') and (dialect <> 'true') and
       tc^.O['codec_language_id']^.GetAsInteger(dialect, v) then
      langId := v;
  end;
  if langId < 0 then
    codec := TIntegerDynArray.Create(fNoThink, fThinkBos, fThinkEos)
  else
    codec := TIntegerDynArray.Create(fThink, fThinkBos, langId, fThinkEos);
  if spkId >= 0 then
  begin
    SetLength(codec, system.length(codec) + 1);
    codec[high(codec)] := spkId;
  end;
  SetLength(codec, system.length(codec) + 2);
  codec[high(codec) - 1] := fCodecPad;
  codec[high(codec)] := fCodecBos;
  nCodec := system.length(codec);
  text := copy(ids, 3, system.length(ids) - 8);
  nText := system.length(text);
  // the prefill: the role (3), tts_pad x (codec - 2) + tts_bos over the codec
  // but its last one, then the text
  if aNonStreaming then
    rows := 3 + (nCodec - 1) + (nText + 1) + 1
  else
    rows := 3 + (nCodec - 1) + 1;
  SetLength(prefill, rows * fHidden);
  TextProject(copy(ids, 0, 3), pointer(prefill));
  r := 3;
  for i := 0 to nCodec - 2 do
  begin
    row := @prefill[r * fHidden];
    if i < nCodec - 2 then
      MoveFast(fTtsPad[0], row^[0], fHidden * SizeOf(single))
    else
      MoveFast(fTtsBos[0], row^[0], fHidden * SizeOf(single));
    AddRow(fCodecEmb, codec[i], row);
    inc(r);
  end;
  if aNonStreaming then
  begin
    // the whole text then tts_eos, each over codec_pad; then tts_pad over
    // codec_bos
    TextProject(text, @prefill[r * fHidden]);
    MoveFast(fTtsEos[0], prefill[(r + nText) * fHidden], fHidden * SizeOf(single));
    for i := 0 to nText do
      AddRow(fCodecEmb, fCodecPad, @prefill[(r + i) * fHidden]);
    inc(r, nText + 1);
    MoveFast(fTtsPad[0], prefill[r * fHidden], fHidden * SizeOf(single));
    AddRow(fCodecEmb, fCodecBos, @prefill[r * fHidden]);
    // what follows: only tts_pad
    fTrailingRows := 1;
    fTrailing := copy(fTtsPad);
  end
  else
  begin
    // the first text token over codec_bos; the next ones and tts_eos, one per
    // frame
    TextProject(copy(text, 0, 1), @prefill[r * fHidden]);
    AddRow(fCodecEmb, fCodecBos, @prefill[r * fHidden]);
    fTrailingRows := nText;
    SetLength(fTrailing, fTrailingRows * fHidden);
    TextProject(copy(text, 1, nText - 1), pointer(fTrailing));
    MoveFast(fTtsEos[0], fTrailing[(nText - 1) * fHidden], fHidden * SizeOf(single));
  end;
  Emit('talker.prefill.embeds', pointer(prefill), rows, fHidden);
  Emit('talker.trailing', pointer(fTrailing), fTrailingRows, fHidden);
  // the choice of the codes: one seed per sentence, or the one of the settings
  FreeAndNil(fSampler);
  FreeAndNil(fSubSampler);
  seed := fSampling.Seed;
  if seed = 0 then
    seed := Random64 shr 1;
  fSampler := NewSampler(fSampling.DoSample, fSampling.Temperature, fSampling.TopP,
    fSampling.TopK, seed);
  fSubSampler := NewSampler(fSampling.SubDoSample, fSampling.SubTemperature,
    fSampling.SubTopP, fSampling.SubTopK, seed xor $5DEECE66D);
  // the prefill of the talker: its last state gives the first frame
  fTalker.Reset;
  fTalker.Forward(pointer(prefill), rows, 0, pointer(fLogits), pointer(fPastHidden));
  Emit('talker.logits', pointer(fLogits), 1, fVocab);
  fHistory := nil;
  fStep := 0;
end;

function TQwenTalker.ChooseCode0: integer;
var
  i, j: PtrInt;
  s, penalty: single;
  ninf: single;
begin
  // the logits processors of generate: repetition_penalty on the group 0 codes
  // already returned, 2048..3071 forbidden but eos, no eos before 2 frames;
  // then the argmax, or temperature, top-k, top-p and the draw
  PCardinal(@ninf)^ := $FF800000; // -Inf
  penalty := fSampling.RepetitionPenalty;
  MoveFast(fLogits[0], fWork[0], fVocab * SizeOf(single));
  for i := fVocab - 1024 to fVocab - 1 do
    if i <> fCodecEos then
      fWork[i] := ninf;
  if system.length(fHistory) < 2 then
    fWork[fCodecEos] := ninf;
  if penalty <> 1 then
    for j := 0 to high(fHistory) do
    begin
      i := fHistory[j];
      s := fLogits[i];
      // only once per code, as the gather / scatter of transformers: when it
      // was already penalized (or forbidden above), it has changed
      if fWork[i] <> s then
        continue;
      if s > 0 then
        fWork[i] := s / penalty
      else
        fWork[i] := s * penalty;
    end;
  if fSampler <> nil then
    result := fSampler.Next(pointer(fWork), fVocab)
  else
    result := InferArgMax(pointer(fWork), fVocab);
end;

function TQwenTalker.ChooseCode(aLogits: PSingleArray): integer;
begin
  if fSubSampler <> nil then
    result := fSubSampler.Next(aLogits, fPredVocab)
  else
    result := InferArgMax(aLogits, fPredVocab);
end;

function TQwenTalker.NextFrame(aCodes: PIntegerArray): boolean;
var
  code, g: PtrInt;
begin
  result := false;
  code := ChooseCode0;
  if code = fCodecEos then
    exit;
  aCodes^[0] := code;
  SetLength(fHistory, system.length(fHistory) + 1);
  fHistory[high(fHistory)] := code;
  // the code predictor: [talker state, embedding of code 0], then each code
  // returned enters in turn
  MoveFast(fPastHidden[0], fX[0], fHidden * SizeOf(single));
  MoveFast(fCodecEmb[code * fHidden], fX[fHidden], fHidden * SizeOf(single));
  Emit('predictor.embeds', pointer(fX), 2, fHidden);
  fPredictor.Reset;
  fPredictor.Forward(pointer(fX), 2, 0, pointer(fPredLogits), nil);
  for g := 0 to fGroups - 2 do
  begin
    if g > 0 then
    begin
      MoveFast(fPredEmb[g - 1][aCodes^[g] * fHidden], fX[0], fHidden * SizeOf(single));
      fPredictor.Forward(pointer(fX), 1, g, pointer(fPredLogits), nil);
    end;
    Emit('predictor.logits', pointer(fPredLogits), 1, fPredVocab);
    aCodes^[g + 1] := ChooseCode(pointer(fPredLogits));
  end;
  // the input of the next frame: the sum of the 16 embeddings, plus the text
  MoveFast(fCodecEmb[code * fHidden], fX[0], fHidden * SizeOf(single));
  for g := 1 to fGroups - 1 do
    AddRow(fPredEmb[g - 1], aCodes^[g], pointer(fX));
  if fStep < fTrailingRows then
    AddRow(fTrailing, fStep, pointer(fX))
  else
    AddRow(fTtsPad, 0, pointer(fX));
  Emit('talker.step.embeds', pointer(fX), 1, fHidden);
  fTalker.Forward(pointer(fX), 1, 0, pointer(fLogits), pointer(fPastHidden));
  Emit('talker.logits', pointer(fLogits), 1, fVocab);
  inc(fStep);
  result := true;
end;

end.
