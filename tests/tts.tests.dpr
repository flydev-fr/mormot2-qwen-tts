// - console test runner of the native Qwen3-TTS and Kokoro engines (src/tts)
// - part of mormot2-qwen-tts, built on mormot2-ai
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the qwen3-tts-lm dataset is committed under tests/data/infer-ref; the
//   model comes from QWEN_TTS_MODEL or the Hugging Face cache, the sentence
//   references of tools/tts/qwen_ref.py from QWEN_TTS_REF; Kokoro from
//   KOKORO_TTS_MODEL, KOKORO_ESPEAK and KOKORO_TTS_REF - the tests which
//   need what is missing are skipped
// - /noenter: do not wait for ENTER at the end
program tts.tests;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.log,
  mormot.core.test,
  test.tts.qwen.ref,
  test.tts.qwen.lm,
  test.tts.qwen.codec,
  test.tts.qwen.codec.vulkan,
  test.tts.qwen.talker,
  test.tts.qwen.vulkan,
  test.tts.qwen.engine,
  test.tts.kokoro.ref,
  test.tts.kokoro.g2p,
  test.tts.kokoro.kernels,
  test.tts.kokoro.model,
  test.tts.kokoro.engine;

type
  TTtsTests = class(TSynTestsLogged)
  published
    /// the two Qwen3 stacks against the embeddings -> logits dataset
    procedure Lm;
    /// the 12 Hz audio decoder
    procedure Codec;
    /// the talker and the code predictor: from text to codes
    procedure Talker;
    /// the same on the GPU (Vulkan, Q8_0)
    procedure Vulkan;
    /// the engine end to end: text -> streamed sound
    procedure Engine;
    /// Kokoro-82M: espeak-ng phonemes, the CPU kernels, the model against
    // PyTorch, the engine end to end
    procedure Kokoro;
  end;

{ TTtsTests }

procedure TTtsTests.Lm;
begin
  AddCase([
    TTestQwenLmReference]);
end;

procedure TTtsTests.Codec;
begin
  AddCase([
    TTestQwenCodec,
    TTestQwenCodecVulkan]);
end;

procedure TTtsTests.Talker;
begin
  AddCase([
    TTestQwenTalker]);
end;

procedure TTtsTests.Vulkan;
begin
  AddCase([
    TTestQwenVulkan]);
end;

procedure TTtsTests.Engine;
begin
  AddCase([
    TTestQwenEngine]);
end;

procedure TTtsTests.Kokoro;
begin
  AddCase([
    TTestKokoroG2P,
    TTestKokoroKernels,
    TTestKokoroModel,
    TTestKokoroEngine]);
end;

begin
  TTtsTests.RunAsConsole('mORMot2 TTS Regression Tests', LOG_VERBOSE);
end.
