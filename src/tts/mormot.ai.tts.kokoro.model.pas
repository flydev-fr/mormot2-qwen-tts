/// Text-to-speech - Kokoro-82M (hexgrad, StyleTTS2 + iSTFTNet vocoder) in native Pascal, on the CPU
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - from phonemes (the ids of the config.json vocabulary) and a style
// vector (the voice, 256 values) to 24 kHz PCM sound, in one go: this model
// is not autoregressive
// - reference: kokoro 0.9.4, KModel.forward_with_tokens (model.py),
// modules.py and istftnet.py; weights kokoro-v1_0.pth, read by
// mormot.ai.tts.torchfile
// - ALBERT (PLBERT, 12 shared layers) -> bert_encoder -> the predictor:
// LSTM and AdaLayerNorm (DurationEncoder), durations in 25 ms frames, then
// F0 and energy (N) by AdainResBlk1d blocks
// - the text encoder (convolutions, LSTM) aligned on the durations
// - the decoder: AdainResBlk1d blocks, then the vocoder: a harmonic source
// drawn from F0 (SineGen, 9 harmonics), its STFT, two stages of transposed
// convolutions (x10, x6) and Snake blocks, then an iSTFT (n_fft 20, hop 5)
// - one predictor frame = 600 samples at 24 kHz
unit mormot.ai.tts.kokoro.model;

{$I mormot.defines.inc}

{$ifndef FPC}
  // single operations in single, as PyTorch's float32
  // (Delphi Win64 computes them in double otherwise): the phase of the
  // harmonic source climbs to tens of thousands of radians
  {$EXCESSPRECISION OFF}
{$endif FPC}

interface

uses
  sysutils,
  math,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.ai.infer.kernels,
  mormot.ai.tts.torchfile,
  mormot.ai.tts.kokoro.nn;

const
  KOKORO_SAMPLE_RATE = 24000;
  /// the samples of one duration frame: 2 (decoder) x 300 (vocoder)
  KOKORO_FRAME_SAMPLES = 600;
  /// the phonemes of one chunk at most (ALBERT's context: 512, minus the
  // two bounds)
  KOKORO_MAX_PHONEMES = 510;
  KOKORO_STYLE = 256;

type
  EKokoro = class(ESynException);

  /// an intermediate output, for the tests: aRows time steps of aCols
  // channels
  TOnKokoroProbe = procedure(const aName: RawUtf8; aData: PSingleArray;
    aRows, aCols: PtrInt) of object;

  TKokoroModel = class
  protected
    fFile: TTorchFile;
    fCpu: TKokoroCpu;
    fVocab: array of integer;      // per code point, -1: missing
    fProbe: TOnKokoroProbe;
    fQuiet: boolean;
    fForcedDurations: TIntegerDynArray;
    fForcedHar: TSingleDynArray;
    fLastDurations: TIntegerDynArray;
    fSeed: cardinal;
    fProfile: boolean;
    fMark: Int64;
    fProfileText: RawUtf8;
    // ALBERT
    fWord, fPos, fType, fEmbNormW, fEmbNormB: PSingleArray;
    fMapping: TKLinear;
    fQ, fK, fV, fDense, fFfn, fFfnOut: TKLinear;
    fAttnNormW, fAttnNormB, fFullNormW, fFullNormB: PSingleArray;
    fBertEncoder: TKLinear;
    // predictor
    fDurLstm: array[0..2] of TKLstm;
    fDurNorm: array[0..2] of TKAdaLayerNorm;
    fPredLstm, fShared: TKLstm;
    fDurProj: TKLinear;
    fF0, fN: array[0..2] of TKAdainResBlk;
    fF0Proj, fNProj: TKConv;
    // text encoder
    fTextEmb: PSingleArray;
    fTextConv: array[0..2] of TKConv;
    fTextGamma, fTextBeta: array[0..2] of PSingleArray;
    fTextLstm: TKLstm;
    // decoder
    fF0Conv, fNConv, fAsrRes: TKConv;
    fEncode: TKAdainResBlk;
    fDecode: array[0..3] of TKAdainResBlk;
    // vocoder
    fSourceW: array[0..8] of single;
    fSourceB: single;
    fNoiseConv: array[0..1] of TKConv;
    fNoiseRes: array[0..1] of TKAdaINResBlock1;
    fUps: array[0..1] of TKConvT;
    fRes: array[0..5] of TKAdaINResBlock1;
    fPost: TKConv;
    fWindow: array[0..19] of single;
    fCos, fSin: array[0..19] of double; // cos and sin of 2 pi m / 20
    // the running ALBERT job
    fAq, fAk, fAv, fActx, fAf: PSingleArray;
    fAT: PtrInt;
    // the running vocoder job
    fJPhase: array[0..8] of TSingleDynArray;
    fJF0, fJIn, fJOut: PSingleArray;
    fJFrames: PtrInt;
    fJSeed: cardinal;
    procedure Emit(const aName: RawUtf8; aData: PSingleArray; aRows, aCols: PtrInt);
    procedure AttentionRows(aFirst, aLast: PtrInt);
    procedure GeluRows(aFirst, aLast: PtrInt);
    procedure SourceBlock(aBlock, aFirst, aLast: PtrInt);
    procedure StftRows(aFirst, aLast: PtrInt);
    procedure IstftRows(aFirst, aLast: PtrInt);
    function Albert(const aIds: TIntegerDynArray): TSingleDynArray;
    function Source(const aF0: TSingleDynArray; aFrames: PtrInt): TSingleDynArray;
    function Generator(const x: TSingleDynArray; aLen: PtrInt;
      const aF0: TSingleDynArray): TSingleDynArray;
    function Random01: single;
    function RandomNormal: single;
  public
    /// loads aSnapshot (the folder holding config.json and kokoro-v1_0.pth)
    constructor Create(const aSnapshot: TFileName; aThreads: integer = 0);
    destructor Destroy; override;
    /// the ids of aPhonemes, framed by 0; unknown phonemes are dropped
    function Ids(const aPhonemes: RawUtf8): TIntegerDynArray;
    /// the sound of aIds (framed by 0) with the style aRefS[256]; aSpeed:
    // 1 = normal, higher = faster
    function Forward(const aIds: TIntegerDynArray; aRefS: PSingleArray;
      aSpeed: single = 1): TSingleDynArray;
    property Cpu: TKokoroCpu
      read fCpu;
    /// the intermediate outputs (tests); nil by default
    property Probe: TOnKokoroProbe
      read fProbe write fProbe;
    /// true: the harmonic source without randomness (initial phases and noise
    // at zero), as tools/tts/kokoro_ref.py; false by default
    property Quiet: boolean
      read fQuiet write fQuiet;
    /// tests: forced durations (those of the reference); nil by default
    property ForcedDurations: TIntegerDynArray
      read fForcedDurations write fForcedDurations;
    /// tests: the STFT of the source forced [frames][22]; nil by default
    property ForcedHar: TSingleDynArray
      read fForcedHar write fForcedHar;
    /// the durations of the last Forward, in frames
    property LastDurations: TIntegerDynArray
      read fLastDurations;
    /// the random seed of the source
    property Seed: cardinal
      read fSeed write fSeed;
    /// true: LastProfile tells the time of each stage of the last Forward
    property Profile: boolean
      read fProfile write fProfile;
    /// 'bert=12 pred.lstm=3 ... gemm=185/259/1398' in milliseconds, if Profile
    // - gemm: the matrix products of all the stages, in ms, Gmac and
    // Gmac/s
    property LastProfile: RawUtf8
      read fProfileText;
  end;

  /// the styles of a voice: voices/<name>.pt, [510][1][256]
  TKokoroVoice = class
  protected
    fName: RawUtf8;
    fPack: TSingleDynArray;
  public
    constructor Create(const aSnapshot: TFileName; const aName: RawUtf8);
    /// the style for a sentence of aPhonemes phonemes: pack[aPhonemes - 1]
    function Style(aPhonemes: PtrInt): PSingleArray;
    property Name: RawUtf8
      read fName;
  end;


implementation

const
  HIDDEN = 512;
  STYLE = 128;
  BERT = 768;
  HEADS = 12;
  HEAD_DIM = 64;
  N_FFT = 20;
  HOP = 5;
  BINS = N_FFT div 2 + 1;
  HARMONICS = 9;
  UPSAMPLE = 300;
  HALF: single = 0.5;

function GeluNew(x: single): single;
const
  C = 0.7978845608028654; // sqrt(2 / pi)
var
  t: single;
begin
  t := C * (x + 0.044715 * (x * x * x));
  if t > 20 then
    result := x
  else if t < -20 then
    result := 0
  else
    result := 0.5 * x * (1 + (Exp(2 * double(t)) - 1) / (Exp(2 * double(t)) + 1));
end;


{ TKokoroModel }

constructor TKokoroModel.Create(const aSnapshot: TFileName; aThreads: integer);
var
  folder: TFileName;
  cfg: TDocVariantData;
  vocab: PDocVariantData;
  i, cp: PtrInt;
  u: UnicodeString;
  p: RawUtf8;
const
  KS: array[0..2] of PtrInt = (3, 7, 11);
begin
  inherited Create;
  folder := IncludeTrailingPathDelimiter(aSnapshot);
  if not cfg.InitJsonFromFile(folder + 'config.json', JSON_FAST) then
    raise EKokoro.CreateUtf8('%: cannot read %config.json', [self, folder]);
  if (cfg.I['hidden_dim'] <> HIDDEN) or (cfg.I['style_dim'] <> STYLE) or
     (cfg.O['plbert']^.I['hidden_size'] <> BERT) then
    raise EKokoro.CreateUtf8('%: unexpected configuration in %config.json', [self, folder]);
  vocab := cfg.O['vocab'];
  SetLength(fVocab, 65536);
  for i := 0 to high(fVocab) do
    fVocab[i] := -1;
  for i := 0 to vocab^.Count - 1 do
  begin
    u := Utf8ToString(vocab^.Names[i]);
    if length(u) <> 1 then
      continue;
    cp := ord(u[1]);
    fVocab[cp] := VariantToIntegerDef(vocab^.Values[i], -1);
  end;
  fSeed := Random32;
  fCpu := TKokoroCpu.Create(aThreads);
  fFile := TTorchFile.Create(folder + 'kokoro-v1_0.pth');
  // ALBERT
  p := 'bert.module.';
  fWord := fFile.F32(p + 'embeddings.word_embeddings.weight', [178, 128]);
  fPos := fFile.F32(p + 'embeddings.position_embeddings.weight', [512, 128]);
  fType := fFile.F32(p + 'embeddings.token_type_embeddings.weight', [2, 128]);
  fEmbNormW := fFile.F32(p + 'embeddings.LayerNorm.weight', [128]);
  fEmbNormB := fFile.F32(p + 'embeddings.LayerNorm.bias', [128]);
  fMapping := TKLinear.Create(fCpu, fFile, p + 'encoder.embedding_hidden_mapping_in', 128, BERT);
  p := 'bert.module.encoder.albert_layer_groups.0.albert_layers.0.';
  fQ := TKLinear.Create(fCpu, fFile, p + 'attention.query', BERT, BERT);
  fK := TKLinear.Create(fCpu, fFile, p + 'attention.key', BERT, BERT);
  fV := TKLinear.Create(fCpu, fFile, p + 'attention.value', BERT, BERT);
  fDense := TKLinear.Create(fCpu, fFile, p + 'attention.dense', BERT, BERT);
  fAttnNormW := fFile.F32(p + 'attention.LayerNorm.weight', [BERT]);
  fAttnNormB := fFile.F32(p + 'attention.LayerNorm.bias', [BERT]);
  fFfn := TKLinear.Create(fCpu, fFile, p + 'ffn', BERT, 2048);
  fFfnOut := TKLinear.Create(fCpu, fFile, p + 'ffn_output', 2048, BERT);
  fFullNormW := fFile.F32(p + 'full_layer_layer_norm.weight', [BERT]);
  fFullNormB := fFile.F32(p + 'full_layer_layer_norm.bias', [BERT]);
  fBertEncoder := TKLinear.Create(fCpu, fFile, 'bert_encoder.module', BERT, HIDDEN);
  // predictor
  p := 'predictor.module.';
  for i := 0 to 2 do
  begin
    fDurLstm[i] := TKLstm.Create(fCpu, fFile, FormatUtf8('%text_encoder.lstms.%',
      [p, 2 * i]), HIDDEN + STYLE, HIDDEN div 2);
    fDurNorm[i] := TKAdaLayerNorm.Create(fCpu, fFile, FormatUtf8('%text_encoder.lstms.%',
      [p, 2 * i + 1]), STYLE, HIDDEN);
  end;
  fPredLstm := TKLstm.Create(fCpu, fFile, p + 'lstm', HIDDEN + STYLE, HIDDEN div 2);
  fDurProj := TKLinear.Create(fCpu, fFile, p + 'duration_proj.linear_layer', HIDDEN, 50);
  fShared := TKLstm.Create(fCpu, fFile, p + 'shared', HIDDEN + STYLE, HIDDEN div 2);
  fF0[0] := TKAdainResBlk.Create(fCpu, fFile, p + 'F0.0', HIDDEN, HIDDEN, STYLE);
  fF0[1] := TKAdainResBlk.Create(fCpu, fFile, p + 'F0.1', HIDDEN, HIDDEN div 2, STYLE, true);
  fF0[2] := TKAdainResBlk.Create(fCpu, fFile, p + 'F0.2', HIDDEN div 2, HIDDEN div 2, STYLE);
  fN[0] := TKAdainResBlk.Create(fCpu, fFile, p + 'N.0', HIDDEN, HIDDEN, STYLE);
  fN[1] := TKAdainResBlk.Create(fCpu, fFile, p + 'N.1', HIDDEN, HIDDEN div 2, STYLE, true);
  fN[2] := TKAdainResBlk.Create(fCpu, fFile, p + 'N.2', HIDDEN div 2, HIDDEN div 2, STYLE);
  fF0Proj := TKConv.Create(fCpu, fFile, p + 'F0_proj', HIDDEN div 2, 1, 1, 1, 1, 0, false);
  fNProj := TKConv.Create(fCpu, fFile, p + 'N_proj', HIDDEN div 2, 1, 1, 1, 1, 0, false);
  // text encoder
  p := 'text_encoder.module.';
  fTextEmb := fFile.F32(p + 'embedding.weight', [178, HIDDEN]);
  for i := 0 to 2 do
  begin
    fTextConv[i] := TKConv.Create(fCpu, fFile, FormatUtf8('%cnn.%.0', [p, i]),
      HIDDEN, HIDDEN, 5);
    fTextGamma[i] := fFile.F32(FormatUtf8('%cnn.%.1.gamma', [p, i]), [HIDDEN]);
    fTextBeta[i] := fFile.F32(FormatUtf8('%cnn.%.1.beta', [p, i]), [HIDDEN]);
  end;
  fTextLstm := TKLstm.Create(fCpu, fFile, p + 'lstm', HIDDEN, HIDDEN div 2);
  // decoder
  p := 'decoder.module.';
  fF0Conv := TKConv.Create(fCpu, fFile, p + 'F0_conv', 1, 1, 3, 2, 1, 1);
  fNConv := TKConv.Create(fCpu, fFile, p + 'N_conv', 1, 1, 3, 2, 1, 1);
  fAsrRes := TKConv.Create(fCpu, fFile, p + 'asr_res.0', HIDDEN, 64, 1, 1, 1, 0);
  fEncode := TKAdainResBlk.Create(fCpu, fFile, p + 'encode', HIDDEN + 2, 1024, STYLE);
  for i := 0 to 2 do
    fDecode[i] := TKAdainResBlk.Create(fCpu, fFile, FormatUtf8('%decode.%', [p, i]),
      1024 + 2 + 64, 1024, STYLE);
  fDecode[3] := TKAdainResBlk.Create(fCpu, fFile, p + 'decode.3', 1024 + 2 + 64,
    512, STYLE, true);
  // vocoder
  p := 'decoder.module.generator.';
  MoveFast(fFile.F32(p + 'm_source.l_linear.weight', [1, HARMONICS])^, fSourceW,
    SizeOf(fSourceW));
  fSourceB := fFile.F32(p + 'm_source.l_linear.bias', [1])^[0];
  fNoiseConv[0] := TKConv.Create(fCpu, fFile, p + 'noise_convs.0', N_FFT + 2, 256,
    12, 6, 1, 3, false);
  fNoiseConv[1] := TKConv.Create(fCpu, fFile, p + 'noise_convs.1', N_FFT + 2, 128,
    1, 1, 1, 0, false);
  fNoiseRes[0] := TKAdaINResBlock1.Create(fCpu, fFile, p + 'noise_res.0', 256, 7,
    [1, 3, 5], STYLE);
  fNoiseRes[1] := TKAdaINResBlock1.Create(fCpu, fFile, p + 'noise_res.1', 128, 11,
    [1, 3, 5], STYLE);
  fUps[0] := TKConvT.Create(fCpu, fFile, p + 'ups.0', 512, 256, 20, 10, 5);
  fUps[1] := TKConvT.Create(fCpu, fFile, p + 'ups.1', 256, 128, 12, 6, 3);
  for i := 0 to 5 do
    fRes[i] := TKAdaINResBlock1.Create(fCpu, fFile, FormatUtf8('%resblocks.%', [p, i]),
      256 shr (i div 3), KS[i mod 3], [1, 3, 5], STYLE);
  fPost := TKConv.Create(fCpu, fFile, p + 'conv_post', 128, N_FFT + 2, 7, 1, 1, 3);
  // periodic hann, in float32 as torch.hann_window
  for i := 0 to N_FFT - 1 do
  begin
    fWindow[i] := 0.5 - 0.5 * Cos(single(2 * pi / N_FFT) * i);
    fCos[i] := Cos(2 * pi * i / N_FFT);
    fSin[i] := Sin(2 * pi * i / N_FFT);
  end;
end;

destructor TKokoroModel.Destroy;
var
  i: PtrInt;
begin
  fMapping.Free;
  fQ.Free;
  fK.Free;
  fV.Free;
  fDense.Free;
  fFfn.Free;
  fFfnOut.Free;
  fBertEncoder.Free;
  for i := 0 to 2 do
  begin
    fDurLstm[i].Free;
    fDurNorm[i].Free;
    fF0[i].Free;
    fN[i].Free;
    fTextConv[i].Free;
  end;
  fPredLstm.Free;
  fShared.Free;
  fDurProj.Free;
  fF0Proj.Free;
  fNProj.Free;
  fTextLstm.Free;
  fF0Conv.Free;
  fNConv.Free;
  fAsrRes.Free;
  fEncode.Free;
  for i := 0 to 3 do
    fDecode[i].Free;
  for i := 0 to 1 do
  begin
    fNoiseConv[i].Free;
    fNoiseRes[i].Free;
    fUps[i].Free;
  end;
  for i := 0 to 5 do
    fRes[i].Free;
  fPost.Free;
  fFile.Free;
  fCpu.Free;
  inherited Destroy;
end;

procedure TKokoroModel.Emit(const aName: RawUtf8; aData: PSingleArray;
  aRows, aCols: PtrInt);
var
  now: Int64;
begin
  if fProfile then
  begin
    // the time since the previous stage
    QueryPerformanceMicroSeconds(now);
    fProfileText := FormatUtf8('% %=%', [fProfileText, aName,
      (now - fMark) div 1000]);
    fMark := now;
  end;
  if Assigned(fProbe) then
    fProbe(aName, aData, aRows, aCols);
end;

function TKokoroModel.Ids(const aPhonemes: RawUtf8): TIntegerDynArray;
var
  u: UnicodeString;
  i, n, id: PtrInt;
begin
  u := Utf8ToString(aPhonemes);
  SetLength(result, length(u) + 2);
  result[0] := 0;
  n := 1;
  for i := 1 to length(u) do
  begin
    id := fVocab[ord(u[i])];
    if id >= 0 then
    begin
      result[n] := id;
      inc(n);
    end;
  end;
  result[n] := 0;
  SetLength(result, n + 1);
end;

function TKokoroModel.Random01: single;
begin
  // xorshift32: enough to draw phases and noise, nothing more
  fSeed := fSeed xor (fSeed shl 13);
  fSeed := fSeed xor (fSeed shr 17);
  fSeed := fSeed xor (fSeed shl 5);
  result := (fSeed shr 8) / 16777216;
end;

function TKokoroModel.RandomNormal: single;
var
  u1: single;
begin
  repeat
    u1 := Random01;
  until u1 > 0;
  result := Sqrt(-2 * Ln(u1)) * Cos(2 * pi * Random01);
end;

procedure TKokoroModel.AttentionRows(aFirst, aLast: PtrInt);
var
  r, h, j: PtrInt;
  scores: array[0..511] of single;
  m, sum: single;
  q, ctx: PSingleArray;
begin
  for r := aFirst to aLast do
    for h := 0 to HEADS - 1 do
    begin
      q := @fAq^[r * BERT + h * HEAD_DIM];
      InferDotsF32(@q^[0], @fAk^[h * HEAD_DIM], BERT * SizeOf(single), fAT,
        HEAD_DIM, @scores);
      m := -1e30;
      for j := 0 to fAT - 1 do
      begin
        scores[j] := scores[j] * 0.125; // 1 / sqrt(64)
        if scores[j] > m then
          m := scores[j];
      end;
      sum := 0;
      for j := 0 to fAT - 1 do
      begin
        scores[j] := Exp(scores[j] - m);
        sum := sum + scores[j];
      end;
      for j := 0 to fAT - 1 do
        scores[j] := scores[j] / sum;
      ctx := @fActx^[r * BERT + h * HEAD_DIM];
      FillCharFast(ctx^, HEAD_DIM * SizeOf(single), 0);
      InferAxpyRowsF32(@scores, @fAv^[h * HEAD_DIM], BERT * SizeOf(single), fAT,
        HEAD_DIM, ctx);
    end;
end;

procedure TKokoroModel.GeluRows(aFirst, aLast: PtrInt);
var
  i: PtrInt;
begin
  for i := aFirst * 2048 to (aLast + 1) * 2048 - 1 do
    fAf^[i] := GeluNew(fAf^[i]);
end;

function TKokoroModel.Albert(const aIds: TIntegerDynArray): TSingleDynArray;
var
  t, n, l, c: PtrInt;
  e, h, q, k, v, ctx, a, f, f2, swap: TSingleDynArray;
begin
  n := length(aIds);
  if n > 512 then
    raise EKokoro.CreateUtf8('%: % tokens, 512 at most', [self, n]);
  SetLength(e, n * 128);
  for t := 0 to n - 1 do
    for c := 0 to 127 do
      e[t * 128 + c] := (fWord^[aIds[t] * 128 + c] + fType^[c]) + fPos^[t * 128 + c];
  fCpu.LayerNorm(pointer(e), n, 128, 128, fEmbNormW, fEmbNormB, 1e-12);
  SetLength(h, n * BERT);
  fMapping.Forward(pointer(e), n, pointer(h));
  SetLength(q, n * BERT);
  SetLength(k, n * BERT);
  SetLength(v, n * BERT);
  SetLength(ctx, n * BERT);
  SetLength(a, n * BERT);
  SetLength(f, n * 2048);
  SetLength(f2, n * BERT);
  for l := 0 to 11 do
  begin
    fQ.Forward(pointer(h), n, pointer(q));
    fK.Forward(pointer(h), n, pointer(k));
    fV.Forward(pointer(h), n, pointer(v));
    fAq := pointer(q);
    fAk := pointer(k);
    fAv := pointer(v);
    fActx := pointer(ctx);
    fAT := n;
    fCpu.ParallelRows(n, 4, AttentionRows);
    fDense.Forward(pointer(ctx), n, pointer(a));
    fCpu.Add(pointer(h), pointer(a), n * BERT); // a := h + a
    fCpu.LayerNorm(pointer(a), n, BERT, BERT, fAttnNormW, fAttnNormB, 1e-12);
    fFfn.Forward(pointer(a), n, pointer(f));
    fAf := pointer(f);
    fCpu.ParallelRows(n, 4, GeluRows);
    fFfnOut.Forward(pointer(f), n, pointer(f2));
    // h := f2 + a: a + f2 into f2, then both arrays swapped
    fCpu.Add(pointer(a), pointer(f2), n * BERT);
    swap := h;
    h := f2;
    f2 := swap;
    fCpu.LayerNorm(pointer(h), n, BERT, BERT, fFullNormW, fFullNormB, 1e-12);
  end;
  result := h;
end;

const
  /// the samples of one block of the source: a randomness of its own for
  // each block, the same whatever the number of threads
  SOURCE_BLOCK = 4096;

procedure TKokoroModel.SourceBlock(aBlock, aFirst, aLast: PtrInt);
var
  t, hh, i0, i1: PtrInt;
  scale, x, lambda, v, s, sum, uv, amp, u1, u2: single;
  rnd: cardinal;
  spare: single;
  hasSpare: boolean;

  function Next01: single;
  begin
    rnd := rnd xor (rnd shl 13);
    rnd := rnd xor (rnd shr 17);
    rnd := rnd xor (rnd shl 5);
    result := ((rnd shr 8) + 0.5) / 16777216; // ]0, 1[
  end;

  function Normal: single;
  begin
    // Box-Muller, two values per draw
    if hasSpare then
    begin
      hasSpare := false;
      result := spare;
      exit;
    end;
    u1 := Next01;
    u2 := Next01;
    v := Sqrt(-2 * Ln(u1));
    result := v * Cos(2 * pi * u2);
    spare := v * Sin(2 * pi * u2);
    hasSpare := true;
  end;

begin
  rnd := (fJSeed xor cardinal(aBlock * 2654435761)) or 1;
  hasSpare := false;
  spare := 0;
  scale := 1 / UPSAMPLE;
  for t := aFirst to aLast do
  begin
    // F.interpolate(scale_factor=300, mode=linear, align_corners=False), in
    // float32: scale * (t + 0.5) - 0.5
    x := t;
    x := x + HALF;
    x := scale * x - HALF;
    if x < 0 then
      x := 0;
    i0 := trunc(x);
    lambda := x - i0;
    i1 := i0 + 1;
    if i1 >= fJFrames then
      i1 := fJFrames - 1;
    uv := ord(fJF0^[t div UPSAMPLE] > 10);
    // the noise: 0.003 on voiced parts, 0.1 / 3 elsewhere
    amp := uv * 0.003 + (1 - uv) * (0.1 / 3);
    sum := fSourceB;
    for hh := 0 to HARMONICS - 1 do
    begin
      v := (1 - lambda) * fJPhase[hh][i0] + lambda * fJPhase[hh][i1];
      s := Sin(v) * 0.1;
      if fQuiet then
        s := s * uv
      else
        s := s * uv + amp * Normal;
      sum := sum + fSourceW[hh] * s;
    end;
    // tanh(l_linear(sine_waves))
    if sum > 20 then
      fJOut^[t] := 1
    else if sum < -20 then
      fJOut^[t] := -1
    else
      fJOut^[t] := (Exp(2 * double(sum)) - 1) / (Exp(2 * double(sum)) + 1);
  end;
end;

function TKokoroModel.Source(const aF0: TSingleDynArray; aFrames: PtrInt): TSingleDynArray;
var
  f, hh: PtrInt;
  cum: double;
  v: single;
begin
  // F0 [aFrames] repeated x300: the step of the sample at 24 kHz
  // - SineGen's random initial phase is added to the first sample, which
  // the reduction to 1/300 (samples 149 and 150 of each frame) never
  // reads: it has no effect, and is left out here
  for hh := 0 to HARMONICS - 1 do
  begin
    // rad = (f0 * h / 24000) mod 1, constant over each frame: the reduction
    // by linear interpolation (0.5 a + 0.5 a) returns it as is; then
    // phase = cumsum(rad) * 2 * pi (sum in double, as torch on the CPU),
    // brought back to the sample rate after a * 300
    SetLength(fJPhase[hh], aFrames);
    cum := 0;
    for f := 0 to aFrames - 1 do
    begin
      v := aF0[f] * (hh + 1);
      v := v / KOKORO_SAMPLE_RATE;
      v := v - Floor(v);
      cum := cum + v;
      v := cum;
      v := v * 2;
      v := v * single(pi);
      fJPhase[hh][f] := v * UPSAMPLE;
    end;
  end;
  SetLength(result, aFrames * UPSAMPLE);
  fJF0 := pointer(aF0);
  fJOut := pointer(result);
  fJFrames := aFrames;
  Random01; // a new seed at each call, drawn from Seed
  fJSeed := fSeed;
  fCpu.ParallelBlocks(aFrames * UPSAMPLE, SOURCE_BLOCK, SourceBlock);
end;

procedure TKokoroModel.StftRows(aFirst, aLast: PtrInt);
var
  t, k, j: PtrInt;
  re, im, s: double;
begin
  for t := aFirst to aLast do
    for k := 0 to BINS - 1 do
    begin
      re := 0;
      im := 0;
      for j := 0 to N_FFT - 1 do
      begin
        s := fJIn^[t * HOP + j] * fWindow[j];
        re := re + s * fCos[(k * j) mod N_FFT];
        im := im - s * fSin[(k * j) mod N_FFT];
      end;
      fJOut^[t * (N_FFT + 2) + k] := Sqrt(re * re + im * im);
      fJOut^[t * (N_FFT + 2) + BINS + k] := ArcTan2(im, re);
    end;
end;

procedure TKokoroModel.IstftRows(aFirst, aLast: PtrInt);
var
  t, k, j: PtrInt;
  re, im: array[0..BINS - 1] of double;
  frame: PSingleArray;
  mag, ph, s: double;
begin
  for t := aFirst to aLast do
  begin
    frame := @fJIn^[t * (N_FFT + 2)];
    // amplitude exp(x), phase sin(x): mag * exp(i phase)
    for k := 0 to BINS - 1 do
    begin
      mag := Exp(frame^[k]);
      ph := Sin(frame^[BINS + k]);
      re[k] := mag * Cos(ph);
      im[k] := mag * Sin(ph);
    end;
    // irfft (C2R: the imaginary parts of 0 and N/2 ignored), window
    for j := 0 to N_FFT - 1 do
    begin
      s := re[0] + re[BINS - 1] * fCos[((BINS - 1) * j) mod N_FFT];
      for k := 1 to BINS - 2 do
        s := s + 2 * (re[k] * fCos[(k * j) mod N_FFT] - im[k] * fSin[(k * j) mod N_FFT]);
      fJOut^[t * N_FFT + j] := (s / N_FFT) * fWindow[j];
    end;
  end;
end;

function TKokoroModel.Generator(const x: TSingleDynArray; aLen: PtrInt;
  const aF0: TSingleDynArray): TSingleDynArray;
var
  src, har, y, xs, r, acc, post: TSingleDynArray;
  n, frames, t, k, j, i, len, ch, nl, p: PtrInt;
  padded: TSingleDynArray;
  envelope, signal: array of double;
  third: single;
begin
  // the harmonic source and its STFT (n_fft 20, hop 5, centered, reflect)
  frames := aLen * UPSAMPLE div HOP + 1;
  if fForcedHar <> nil then
  begin
    if length(fForcedHar) <> frames * (N_FFT + 2) then
      raise EKokoro.CreateUtf8('%: forced har of % values, % expected',
        [self, length(fForcedHar), frames * (N_FFT + 2)]);
    har := fForcedHar;
  end
  else
  begin
    src := Source(aF0, aLen);
    n := length(src);
    Emit('gen.har_source', pointer(src), n, 1);
    SetLength(padded, n + N_FFT);
    for t := 0 to N_FFT div 2 - 1 do
    begin
      padded[t] := src[N_FFT div 2 - t];
      padded[n + N_FFT div 2 + t] := src[n - 2 - t];
    end;
    MoveFast(src[0], padded[N_FFT div 2], n * SizeOf(single));
    SetLength(har, frames * (N_FFT + 2));
    fJIn := pointer(padded);
    fJOut := pointer(har);
    fCpu.ParallelRows(frames, 64, StftRows);
  end;
  Emit('gen.har', pointer(har), frames, N_FFT + 2);
  // two stages: LeakyReLU 0.1, the source through its convolution and its
  // block, the transposed convolution, then the mean of the three Snake blocks
  y := copy(x);
  len := aLen;
  third := 3;
  for i := 0 to 1 do
  begin
    ch := 256 shr i;
    fCpu.Leaky(pointer(y), len * (512 shr i), 0.1);
    nl := fNoiseConv[i].OutLen(frames);
    SetLength(xs, nl * ch);
    fNoiseConv[i].Forward(pointer(har), frames, pointer(xs));
    Emit(FormatUtf8('gen.noise_conv%', [i]), pointer(xs), nl, ch);
    SetLength(r, nl * ch);
    fNoiseRes[i].Forward(pointer(xs), nl, pointer(r)); // style: see Forward
    Emit(FormatUtf8('gen.noise_res%', [i]), pointer(r), nl, ch);
    p := fUps[i].OutLen(len);
    if i = 1 then
    begin
      // ReflectionPad1d((1, 0)): row 1 in front; the transposed convolution
      // writes after it
      SetLength(xs, (p + 1) * ch);
      fUps[i].Forward(pointer(y), len, @xs[ch]);
      MoveFast(xs[2 * ch], xs[0], ch * SizeOf(single));
      inc(p);
    end
    else
    begin
      SetLength(xs, p * ch);
      fUps[i].Forward(pointer(y), len, pointer(xs));
    end;
    Emit(FormatUtf8('gen.up%', [i]), pointer(xs), p, ch);
    if p <> nl then
      raise EKokoro.CreateUtf8('%: stage % has % rows, the source %', [self, i, p, nl]);
    fCpu.Add(pointer(r), pointer(xs), p * ch);
    SetLength(acc, p * ch);
    SetLength(r, p * ch);
    fRes[i * 3].Forward(pointer(xs), p, pointer(acc));
    for j := 1 to 2 do
    begin
      fRes[i * 3 + j].Forward(pointer(xs), p, pointer(r));
      fCpu.Add(pointer(r), pointer(acc), p * ch);
    end;
    fCpu.Divide(pointer(acc), p * ch, third);
    Emit(FormatUtf8('gen.res%', [i]), pointer(acc), p, ch);
    y := acc;
    acc := nil;
    len := p;
  end;
  // LeakyReLU 0.01, conv_post -> 11 amplitudes (exp) and 11 phases (sin)
  fCpu.Leaky(pointer(y), len * 128, 0.01);
  SetLength(post, len * (N_FFT + 2));
  fPost.Forward(pointer(y), len, pointer(post));
  Emit('gen.post', pointer(post), len, N_FFT + 2);
  // iSTFT: irfft of each frame and window (in parallel), then addition
  // and division by the sum of the windows^2
  SetLength(xs, len * N_FFT);
  fJIn := pointer(post);
  fJOut := pointer(xs);
  fCpu.ParallelRows(len, 64, IstftRows);
  SetLength(signal, (len - 1) * HOP + N_FFT);
  SetLength(envelope, length(signal));
  for t := 0 to len - 1 do
    for j := 0 to N_FFT - 1 do
    begin
      signal[t * HOP + j] := signal[t * HOP + j] + xs[t * N_FFT + j];
      envelope[t * HOP + j] := envelope[t * HOP + j] + sqr(double(fWindow[j]));
    end;
  n := (len - 1) * HOP;
  SetLength(result, n);
  for t := 0 to n - 1 do
    if envelope[t + N_FFT div 2] > 1e-11 then
      result[t] := signal[t + N_FFT div 2] / envelope[t + N_FFT div 2]
    else
      result[t] := 0;
end;

function TKokoroModel.Forward(const aIds: TIntegerDynArray; aRefS: PSingleArray;
  aSpeed: single): TSingleDynArray;
var
  n, t, i, f, frames, c: PtrInt;
  s, sd: PSingleArray;
  hb, den, d, x, lstm, dur, en, sh, a, b, f0, nn, ten, asr, cat, ar,
    f0c, nc: TSingleDynArray;
  durations: TIntegerDynArray;
  sum, v: single;
begin
  n := length(aIds);
  if n < 3 then
    raise EKokoro.CreateUtf8('%.Forward: no phoneme', [self]);
  fProfileText := '';
  fCpu.GemmMicroSec := 0;
  fCpu.GemmMac := 0;
  QueryPerformanceMicroSeconds(fMark);
  sd := @aRefS^[STYLE]; // the predictor's style: the last 128 values
  s := aRefS;           // the decoder's: the first 128
  // ALBERT and its projection
  hb := Albert(aIds);
  Emit('bert', pointer(hb), n, BERT);
  SetLength(den, n * HIDDEN);
  fBertEncoder.Forward(pointer(hb), n, pointer(den));
  Emit('bert_encoder', pointer(den), n, HIDDEN);
  // DurationEncoder: [x, s] -> 3 x (LSTM, AdaLayerNorm, [., s])
  SetLength(d, n * (HIDDEN + STYLE));
  for t := 0 to n - 1 do
  begin
    MoveFast(den[t * HIDDEN], d[t * (HIDDEN + STYLE)], HIDDEN * SizeOf(single));
    MoveFast(sd^[0], d[t * (HIDDEN + STYLE) + HIDDEN], STYLE * SizeOf(single));
  end;
  SetLength(x, n * HIDDEN);
  for i := 0 to 2 do
  begin
    fDurLstm[i].Forward(pointer(d), n, pointer(x));
    fDurNorm[i].Prepare(sd);
    fDurNorm[i].Forward(pointer(x), n, HIDDEN);
    for t := 0 to n - 1 do
      MoveFast(x[t * HIDDEN], d[t * (HIDDEN + STYLE)], HIDDEN * SizeOf(single));
  end;
  Emit('pred.text_encoder', pointer(d), n, HIDDEN + STYLE);
  // the durations: LSTM, projection on 50, sum of the sigmoids / speed
  SetLength(lstm, n * HIDDEN);
  fPredLstm.Forward(pointer(d), n, pointer(lstm));
  Emit('pred.lstm', pointer(lstm), n, HIDDEN);
  SetLength(dur, n * 50);
  fDurProj.Forward(pointer(lstm), n, pointer(dur));
  Emit('pred.duration_proj', pointer(dur), n, 50);
  SetLength(durations, n);
  frames := 0;
  for t := 0 to n - 1 do
  begin
    sum := 0;
    for i := 0 to 49 do
      sum := sum + 1 / (1 + Exp(-dur[t * 50 + i]));
    v := sum / aSpeed;
    durations[t] := Round(v); // to the nearest even, as torch.round
    if durations[t] < 1 then
      durations[t] := 1;
  end;
  if fForcedDurations <> nil then
  begin
    if length(fForcedDurations) <> n then
      raise EKokoro.CreateUtf8('%: % forced durations for % tokens',
        [self, length(fForcedDurations), n]);
    durations := copy(fForcedDurations);
  end;
  for t := 0 to n - 1 do
    inc(frames, durations[t]);
  fLastDurations := durations;
  // each phoneme repeated over its frames
  SetLength(en, frames * (HIDDEN + STYLE));
  f := 0;
  for t := 0 to n - 1 do
    for i := 1 to durations[t] do
    begin
      MoveFast(d[t * (HIDDEN + STYLE)], en[f * (HIDDEN + STYLE)],
        (HIDDEN + STYLE) * SizeOf(single));
      inc(f);
    end;
  Emit('pred.en', pointer(en), frames, HIDDEN + STYLE);
  // F0 and N: shared LSTM, then three AdainResBlk1d each (x2 in the middle)
  SetLength(sh, frames * HIDDEN);
  fShared.Forward(pointer(en), frames, pointer(sh));
  Emit('pred.shared', pointer(sh), frames, HIDDEN);
  for c := 0 to 1 do
  begin
    a := copy(sh);
    f := frames;
    for i := 0 to 2 do
    begin
      if c = 0 then
      begin
        fF0[i].Prepare(sd);
        SetLength(b, fF0[i].OutLen(f) * fF0[i].OutCount);
        fF0[i].Forward(pointer(a), f, pointer(b));
        f := fF0[i].OutLen(f);
        Emit(FormatUtf8('pred.F0.%', [i]), pointer(b), f, fF0[i].OutCount);
      end
      else
      begin
        fN[i].Prepare(sd);
        SetLength(b, fN[i].OutLen(f) * fN[i].OutCount);
        fN[i].Forward(pointer(a), f, pointer(b));
        f := fN[i].OutLen(f);
        Emit(FormatUtf8('pred.N.%', [i]), pointer(b), f, fN[i].OutCount);
      end;
      a := b;
      b := nil;
    end;
    if c = 0 then
    begin
      SetLength(f0, f);
      fF0Proj.Forward(pointer(a), f, pointer(f0));
      Emit('pred.F0', pointer(f0), f, 1);
    end
    else
    begin
      SetLength(nn, f);
      fNProj.Forward(pointer(a), f, pointer(nn));
      Emit('pred.N', pointer(nn), f, 1);
    end;
  end;
  // the text encoder: embedding, 3 x (conv 5, LayerNorm, LeakyReLU), LSTM
  SetLength(a, n * HIDDEN);
  SetLength(b, n * HIDDEN);
  for t := 0 to n - 1 do
    MoveFast(fTextEmb^[aIds[t] * HIDDEN], a[t * HIDDEN], HIDDEN * SizeOf(single));
  for i := 0 to 2 do
  begin
    fTextConv[i].Forward(pointer(a), n, pointer(b));
    fCpu.LayerNorm(pointer(b), n, HIDDEN, HIDDEN, fTextGamma[i], fTextBeta[i], 1e-5);
    fCpu.Leaky(pointer(b), n * HIDDEN, 0.2);
    a := copy(b);
  end;
  SetLength(ten, n * HIDDEN);
  fTextLstm.Forward(pointer(a), n, pointer(ten));
  Emit('text_encoder', pointer(ten), n, HIDDEN);
  SetLength(asr, frames * HIDDEN);
  f := 0;
  for t := 0 to n - 1 do
    for i := 1 to durations[t] do
    begin
      MoveFast(ten[t * HIDDEN], asr[f * HIDDEN], HIDDEN * SizeOf(single));
      inc(f);
    end;
  Emit('asr', pointer(asr), frames, HIDDEN);
  // the decoder
  SetLength(f0c, frames);
  SetLength(nc, frames);
  fF0Conv.Forward(pointer(f0), 2 * frames, pointer(f0c));
  fNConv.Forward(pointer(nn), 2 * frames, pointer(nc));
  Emit('dec.F0_conv', pointer(f0c), frames, 1);
  Emit('dec.N_conv', pointer(nc), frames, 1);
  SetLength(cat, frames * (HIDDEN + 2));
  for t := 0 to frames - 1 do
  begin
    MoveFast(asr[t * HIDDEN], cat[t * (HIDDEN + 2)], HIDDEN * SizeOf(single));
    cat[t * (HIDDEN + 2) + HIDDEN] := f0c[t];
    cat[t * (HIDDEN + 2) + HIDDEN + 1] := nc[t];
  end;
  fEncode.Prepare(s);
  SetLength(x, frames * 1024);
  fEncode.Forward(pointer(cat), frames, pointer(x));
  Emit('dec.encode', pointer(x), frames, 1024);
  SetLength(ar, frames * 64);
  fAsrRes.Forward(pointer(asr), frames, pointer(ar));
  Emit('dec.asr_res', pointer(ar), frames, 64);
  SetLength(cat, frames * 1090);
  for i := 0 to 3 do
  begin
    for t := 0 to frames - 1 do
    begin
      MoveFast(x[t * 1024], cat[t * 1090], 1024 * SizeOf(single));
      MoveFast(ar[t * 64], cat[t * 1090 + 1024], 64 * SizeOf(single));
      cat[t * 1090 + 1088] := f0c[t];
      cat[t * 1090 + 1089] := nc[t];
    end;
    fDecode[i].Prepare(s);
    SetLength(b, fDecode[i].OutLen(frames) * fDecode[i].OutCount);
    fDecode[i].Forward(pointer(cat), frames, pointer(b));
    Emit(FormatUtf8('dec.decode.%', [i]), pointer(b), fDecode[i].OutLen(frames),
      fDecode[i].OutCount);
    x := b;
    b := nil;
  end;
  // the vocoder, over 2 x frames, with the decoder's style
  for i := 0 to 1 do
    fNoiseRes[i].Prepare(s);
  for i := 0 to 5 do
    fRes[i].Prepare(s);
  result := Generator(x, 2 * frames, f0);
  Emit('istft', pointer(result), length(result), 1);
  if fProfile and (fCpu.GemmMicroSec > 0) then
    // including the products: ms, Gmac, Gmac/s
    fProfileText := FormatUtf8('% gemm=%/%/%', [fProfileText,
      fCpu.GemmMicroSec div 1000, fCpu.GemmMac div 1000000000,
      fCpu.GemmMac div (fCpu.GemmMicroSec * 1000)]);
end;


{ TKokoroVoice }

constructor TKokoroVoice.Create(const aSnapshot: TFileName; const aName: RawUtf8);
var
  f: TTorchFile;
  t: PTorchTensor;
begin
  inherited Create;
  fName := aName;
  f := TTorchFile.Create(IncludeTrailingPathDelimiter(aSnapshot) + 'voices' +
    PathDelim + Utf8ToString(aName) + '.pt');
  try
    t := f.Tensor('');
    if (t^.DType <> tdF32) or (t^.Count <> KOKORO_MAX_PHONEMES * KOKORO_STYLE) then
      raise EKokoro.CreateUtf8('%: voice % is %', [self, aName, TorchShapeText(t^.Shape)]);
    SetLength(fPack, t^.Count);
    MoveFast(t^.Data^, fPack[0], t^.Count * SizeOf(single));
  finally
    f.Free;
  end;
end;

function TKokoroVoice.Style(aPhonemes: PtrInt): PSingleArray;
begin
  if aPhonemes < 1 then
    aPhonemes := 1
  else if aPhonemes > KOKORO_MAX_PHONEMES then
    aPhonemes := KOKORO_MAX_PHONEMES;
  result := @fPack[(aPhonemes - 1) * KOKORO_STYLE];
end;

end.
