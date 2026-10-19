# mormot2-qwen-tts

A native Pascal text-to-speech engine for
[Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice),
built on [mORMot 2](https://github.com/synopse/mORMot2) and the inference engine
of [mormot2-ai](https://github.com/landrix/mormot2-ai). No Python and no
external runtime: the talker, the code predictor and the 12 Hz audio decoder
run in Delphi code on the CPU (AVX2) or on any Vulkan GPU, and the sound
streams as 24 kHz PCM while the rest of the sentence is computed.

Every stage was checked against the PyTorch reference of the official
`qwen_tts` package (greedy, CPU, float32).

## Measured

On a Ryzen 9 9950X3D (22 threads) with a Radeon RX 9060 XT, Debug build, from
the test suite and the demo (2026-10-18):

| | |
|---|---|
| Engine, end to end on the GPU | x4.4 real time, first sound in 102-108 ms, ~0.9 GB of VRAM |
| Talker + code predictor on the GPU (Vulkan, Q8_0) | x5.5, ~14.5 ms per frame of 80 ms |
| 12 Hz decoder on the GPU | x19.5, first frame in 5.7 ms, 348 MB |
| 12 Hz decoder on the CPU | x4.2-4.6 (4 or 8 frames at once) |
| Talker + code predictor on the CPU, Q8_0 / float32 | x3.2 / x0.8 |
| Load (model already in the OS cache) | 1.6-2.3 s |

Against PyTorch:
- decoder: wav within 2.5e-6 (CPU) and 4.8e-6 (GPU) of the reference, and
  bit-identical whatever the number of frames decoded at once;
- talker and code predictor on the CPU in float32: the same codes as PyTorch,
  frame by frame, on four sentences (French, English, `Auto` language, 156
  frames); layer states within 1.8e-4, logits within 8.3e-5;
- on the GPU in Q8_0: logits within 0.43 (cosine 0.9999), so the greedy codes
  drift after a few frames, as any quantized run does - the sound stays right.

## Requirements

- Delphi with mORMot 2 in the library path, as mORMot 2 recommends. Built
  and tested with Delphi 13 (Win64) only.
- mormot2-ai: the git submodule `external/mormot2-ai`, pinned at `5b30ded`
  (`git submodule update --init`).
- Optional: a Vulkan GPU (1.0 or later) with ~1 GB of free memory; without
  one, everything runs on the CPU.
- The model, about 2.4 GB, downloaded once from Hugging Face, e.g. with
  `hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (revision
  `85e237c12c027371202489a0ec509ded67b5e4b5` was tested). The engine finds it
  in a Hugging Face cache: `.hf` next to the executable, `HF_HOME`, then
  `~/.cache/huggingface`; or give its folder (the one with `config.json` and
  `speech_tokenizer`) explicitly.

The sound output (`TTtsPcmStream`, waveOut) is Windows only; `Synthesize`
hands the samples to any callback on every platform mormot2-ai runs on.

## Demo

`demos/tts/qwentts.dproj` builds `bin/Win64-Debug/qwentts.exe`:

```
qwentts "Target in sight, two hostiles in local."
qwentts "Hello from mORMot 2." /voice ryan /language English /wav hello.wav
qwentts /voices
qwentts /help
```

Voices: serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan.
`/temperature` (default 0.9): lower gives a calmer, steadier delivery, higher
a more expressive one. `/seed` fixes the voice of a text. `/cpu`, `/f32` and
`/cpudecoder` choose where it runs.

## In code

```pascal
uses
  mormot.ai.tts.qwen;

var
  s: TQwenTtsSettings;
  tts: TQwenTts;
begin
  s := QwenTtsDefaults;    // GPU when there is one, vivian, Auto language
  s.Voice := 'ryan';
  tts := TQwenTts.Create(s);
  try
    tts.Speak('Hello from mORMot 2.');  // plays while it computes
    // or tts.Synthesize(text, voice, OnPcm, OnAbort) to get the samples
  finally
    tts.Free;
  end;
end;
```

A long text is split into sentences (`QwenSplitText`) said in one stream.
`Speak` and `Synthesize` take an abort callback, checked at least every
100 ms.

## Layout

The folders follow mormot2-ai, so that this code can also live as a branch of
a mormot2-ai fork: the project files search both `external/mormot2-ai/src/infer`
and `src/infer`.

| Unit (`src/tts`) | Role |
|---|---|
| `mormot.ai.tts.qwen` | `TQwenTts`: settings, text chunks, the talker on its own thread feeding the decoder, streaming, measures |
| `mormot.ai.tts.qwen.talker` | the prompt (as `qwen_tts`), the talker (Qwen3, 28 layers) which predicts the first code of each frame, the code predictor (Qwen3, 5 layers) which derives the 15 others, the sampling of `generation_config.json` |
| `mormot.ai.tts.qwen.lm` | a Qwen3 stack fed with embeddings: weights in float32 or Q8_0 (the bytes ggml writes), CPU computation |
| `mormot.ai.tts.qwen.lm.vulkan` | the same stack on the GPU, after `TQwen3Vulkan.Chunk` of mormot2-ai: one recorded pass per frame |
| `mormot.ai.tts.qwen.codec` | the 12 Hz decoder, frame by frame and causal: 16 codes -> 1920 samples at 24 kHz |
| `mormot.ai.tts.qwen.codec.vulkan` | its heavy back part (upsampling, ConvNeXt, dilated convolutions, SnakeBeta) on the GPU |
| `mormot.ai.tts.wave` | float to PCM 16, WAV files, the streamed Windows output |

`tools/tts/vk` holds the four GLSL shaders of the decoder (a convolution as a
tiled matrix product with fused epilogues, SnakeBeta with an accurate sine,
the depthwise convolution, LayerNorm); `build-shaders.ps1` compiles them with
[glslang](https://github.com/KhronosGroup/glslang/releases) into
`src/tts/mormot.ai.tts.qwen.spv.inc`, which is committed: building needs no
shader compiler.

## Tests

`tests/tts.tests.dproj` builds `bin/Win64-Debug/tts.tests.exe`; run it with
`/noenter` (without it, it waits for ENTER), or `--test <class>` for one
class. What a test needs and does not find, it skips:

- the model: `QWEN_TTS_MODEL` (the snapshot folder), or the Hugging Face cache;
- the sentence references: `QWEN_TTS_REF`, the output folder of
  `tools/tts/qwen_ref.py` (about 120 MB); the tests write `native-*.wav`
  files there to listen to;
- `QWEN_THREADS` sets the threads of the decoder and engine tests.

The four references the tests read, made with `pip install qwen-tts==0.1.1
soundfile` (torch 2.9.1 was used):

```
python tools/tts/qwen_ref.py --out fr-vivian
python tools/tts/qwen_ref.py --out auto-vivian --language Auto --no-blocks --text "Hostile fleet in local, ten pilots."
python tools/tts/qwen_ref.py --out fr-accents --no-blocks --text "Arrivée imminente : évacuez la zone, la flotte ennemie est à deux sauts."
python tools/tts/qwen_ref.py --out fr-long --no-blocks --text "Alerte, flotte hostile en approche par la porte de Jita. Trente pilotes, dont des intercepteurs et des croiseurs lourds. Restez a quai jusqu'a nouvel ordre, je repete, restez a quai."
```

## A reference for mormot2-ai: embeddings in, logits out

`tests/data/infer-ref/qwen3-tts-lm` is a small dataset (2.7 MB) in the format
of mormot2-ai's `tests/data/infer-ref` (`manifest.json` + `tensors.bin`,
sha256 per tensor, read by its `test.infer.ref`); mormot2-ai adopted it as
is on 2026-10-19 (commit 7c1824f). Both Qwen3 stacks of the model take
embedding rows instead of token ids:

- `talker.embed` [22, 1024]: the prefill of one sentence; `talker.layer00`
  .. `layer26`, `talker.norm` [22, 1024]; `talker.logits_last` [3072]
  (`talker.codec_head.weight` on the last row);
- `predictor.embed` [16, 1024] and `predictor.logits` [15, 2048]: row g of
  the logits is `talker.code_predictor.lm_head.{g}.weight` on position
  g + 1, so one forward of the 16 rows gives them all; `predictor.codes`
  [16] are the greedy codes.

The weights come from the model (`talker.model.*`,
`talker.code_predictor.model.*`; plain Qwen3 decoders with q_norm / k_norm,
rope theta 1e6, eps 1e-6). The multimodal RoPE of the talker reduces to the
plain one here, its three positions being equal. The values were computed by
PyTorch in float32 (`tools/tts/qwen_ref.py`), the dataset was assembled by
`tools/tts/make_lm_reference.py` (standard library only).
`test.tts.qwen.lm` checks it, then runs both stacks on it: on the CPU in
float32 (within 2e-4), in Q8_0, and on Vulkan in Q8_0.

## Notes from the port

What mormot2-ai did not cover yet, and how this code works around it:
- `InferRowToF32` has no BF16: `QwenRowToF32` widens BF16 and F16 rows.
- `TBpeTokenizer` reads `tokenizer.json` or GGUF only; the model ships
  `vocab.json` + `merges.txt`: `TQwenTokenizer.CreateFromFolder` feeds the
  same `Build`.
- The forward passes take token ids: `TQwenLm` takes embedding rows, on the
  CPU and on Vulkan, repeating the push constant records of mormot2-ai's
  shaders.
- A half-float K/V cache drifts in autoregressive loops (2.3e-2 against
  2.6e-5 in float32 over 15 code predictor steps): the K/V cache is float32.
- A `TInferThreadPool` work item must not allocate on the heap (3.5 s
  instead of 0.6 s with 22 threads queued on the memory manager): the Q8_0
  conversion works in a stack buffer.
- The Vulkan backend keeps device weights keyed by their host address:
  weights freed and reallocated at the same address would be served stale,
  so free them only with the backend, or after `ClearWeights`.

## License

The mORMot three-license, MPL 1.1 / GPL 2.0 / LGPL 2.1: see [LICENSE](LICENSE)
and, for the model and other references, [NOTICE](NOTICE). The Qwen3-TTS
model and weights are Apache-2.0 and are not part of this repository.
