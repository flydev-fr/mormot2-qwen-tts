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

A second engine, for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
is much lighter (82 M parameters, 330 MB) and runs on the CPU only, about 30
times faster than real time: see [Kokoro-82M](#kokoro-82m) below.

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
- mormot2-ai: the git submodule `external/mormot2-ai`, pinned at `21aa4bb`
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

What the port needed from mormot2-ai, reported as its backlog item B14 and
taken up on 2026-10-19:
- BF16 and F16 rows: `InferRowToF32` widens them.
- The `vocab.json` + `merges.txt` of the model (no `tokenizer.json`):
  `TBpeTokenizer.CreateFromVocabMerges`, the same ids as transformers.
- Embedding rows instead of token ids, and one stack of a larger checkpoint:
  `TDecoderModel.ForwardEmbeds` and `CreateStack` (see below why `TQwenLm`
  does not use them yet).
- A half-float K/V cache drifts in loops that feed floats back (2.3e-2
  against 2.6e-5 in float32 over 15 code predictor steps): the K/V cache is
  float32, as the `KvF16` docs of mormot2-ai now say.
- Device weights are keyed by their host address: weights freed and
  reallocated at the same address were served stale. `ForgetWeights`: a GPU
  stack forgets its weights when it is freed.
- A `TInferThreadPool` work item that allocates on the heap queues the
  threads on Delphi's memory manager (3.5 s instead of 0.6 s with 22
  threads): the Q8_0 conversion works in a stack buffer. `RunWorkers` now
  gives a worker index for a buffer per thread; the 32 floats of a block do
  not need it.

Still ours, both passes of `TQwenLm`:
- On the CPU, `CreateStack` + `ForwardEmbeds` (the final norm read through
  `aOnState`, our heads applied to it) gave the same states, within 1.8e-4
  of PyTorch, but `CreateStack` converts the weights on one thread: both
  stacks loaded in 6.1 s in Q8_0 against 0.56 s with the pool of
  `TQwenLmWeights` (0.61 s against 0.26 s in float32), and a Q8_0 frame
  took about 10% longer. `TQwenLmCpu` stays until the conversion is
  parallel.
- On the GPU (`mormot.ai.tts.qwen.lm.vulkan`), a device pass of mormot2-ai
  returns the logits of one head; the talker needs the normed state of its
  last row (the input of the code predictor) and the code predictor a
  different head at each step. The pass repeats the push constant records
  of mormot2-ai's shaders, which are private to its units: they grew in
  21aa4bb (`Window`, `Norm`, `Rope`), and the pass computed wrong values
  without any error until its records followed.

## Kokoro-82M

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (hexgrad, Apache-2.0)
is a StyleTTS 2 model with an iSTFTNet vocoder: ALBERT reads the phonemes,
a predictor gives each one a duration, F0 and energy, then the decoder and
the vocoder output the whole chunk at once - it is not autoregressive. Its
54 voices cover English, French, Spanish, Italian, Brazilian Portuguese,
Hindi, Japanese and Chinese; the French one, `ff_siwis`, is the default.

The phonemes come from [espeak-ng](https://github.com/espeak-ng/espeak-ng),
loaded at runtime, through the same steps as the Python package (misaki's
`EspeakG2P` over phonemizer): the native engine gets the same phonemes,
hence the same ids and durations.

### Measured

Same machine, Debug build, 22 threads, from the test suite (2026-10-19):

| | |
|---|---|
| Model alone (9.45 s of audio) | x31 real time, 0.30 s |
| Engine, end to end (14.6 s of audio, 3 chunks) | x28, first sound in 56 ms |
| Load | 0.37 s |

The matrix products run on a register-blocked kernel (6 x 16 outputs in
AVX2/FMA, 6 x 32 in AVX-512, detected at runtime; Pascal loops otherwise).

Against PyTorch (kokoro 0.9.4, CPU, float32), on four French sentences:
- phonemes, ids and durations identical;
- every stage within 2e-3 of the peak of the reference;
- with PyTorch's STFT of the harmonic source forced in, the wav within 4e-6
  (SNR 111 dB);
- fully native, an SNR of 24-26 dB: the phase of the harmonic source
  accumulates the tiny F0 differences over 300 samples per frame, which
  changes the waveform but not what one hears.

### Requirements

- The model, about 340 MB, e.g. `hf download hexgrad/Kokoro-82M --include
  config.json kokoro-v1_0.pth "voices/*"` (revision
  `f3ff3571791e39611d31c381e3a41a3af07b4987` was tested), found in a Hugging
  Face cache as above, or given as a folder.
- espeak-ng: `espeak-ng.dll` and its `espeak-ng-data` folder, side by side,
  e.g. the ones of the `espeakng-loader` wheel (0.2.4 was tested: unzip
  `espeakng_loader/` from the win_amd64 wheel). By default the engine looks
  for an `espeak-ng` folder next to the executable. Other systems would load
  `libespeak-ng.so.1` or `libespeak-ng.dylib` from that folder: not tested.
- Nothing else: no GPU, no Python.

### Demo

`demos/tts/kokorotts.dproj` builds `bin/Win64-Debug/kokorotts.exe`:

```
kokorotts "Cible en vue, deux hostiles en local."
kokorotts "Hello from mORMot 2." /voice af_heart /wav hello.wav
kokorotts "Rapido, por favor." /voice ef_dora /speed 1.2
kokorotts /voices
```

The first letter of a voice gives its language: `a` American English, `b`
British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `p`
Brazilian Portuguese; English goes through espeak-ng too, not through
misaki's dictionary. Japanese and Chinese voices are not supported here.

### In code

```pascal
uses
  mormot.ai.tts.kokoro;

var
  s: TKokoroTtsSettings;
  tts: TKokoroTts;
begin
  s := KokoroTtsDefaults;  // ff_siwis, speed 1
  tts := TKokoroTts.Create(s);
  try
    tts.Speak('Cible en vue, deux hostiles en local.');
    // or tts.Synthesize(text, voice, OnPcm, OnAbort) to get the samples
  finally
    tts.Free;
  end;
end;
```

The text is split as for Qwen (`TtsSplitText`), with a short first chunk
cut at a comma, so that the first sound comes after the first few words.

### Units

| Unit (`src/tts`) | Role |
|---|---|
| `mormot.ai.tts.kokoro` | `TKokoroTts`: settings, text chunks, espeak-ng per language, voices, streaming, measures |
| `mormot.ai.tts.kokoro.model` | the network: ALBERT, the predictor, the text encoder, the decoder, the vocoder (harmonic source, STFT, iSTFT); the voices |
| `mormot.ai.tts.kokoro.nn` | its building blocks on a thread pool: linear, 1D and transposed convolutions, bidirectional LSTM, AdaIN, AdaLayerNorm, the residual blocks |
| `mormot.ai.tts.kokoro.kernels` | the CPU kernels: matrix product, AdaIN + Snake, LSTM cell, sine |
| `mormot.ai.tts.espeak` | text to Kokoro phonemes through espeak-ng |
| `mormot.ai.tts.torchfile` | reads PyTorch `.pth` / `.pt` files (`torch.save`) without Python: the zip and a small pickle machine |
| `mormot.ai.tts.common` | what both engines share: text chunks, model folders |

### Tests

The same runner, `tests/tts.tests.dproj`, holds the Kokoro tests:

- the model: `KOKORO_TTS_MODEL` (the snapshot folder), or the Hugging Face
  cache;
- espeak-ng: `KOKORO_ESPEAK` (the folder of `espeak-ng.dll`), or `espeak-ng`
  next to the executable;
- the references: `KOKORO_TTS_REF`, the output folder of
  `tools/tts/kokoro_ref.py` (about 340 MB), or `ref-kokoro` next to the
  executable; the docstring of the script gives the four commands (kokoro
  0.9.4 and torch 2.14.1 were used). The script sets the random parts of the
  harmonic source to zero, which the native model can do as well, so that
  the sound compares sample by sample.

The kernel tests need nothing: they compare the assembler with the Pascal
loops, bit for bit.

### Notes from the port

- Single-precision code needs `{$EXCESSPRECISION OFF}`: otherwise Win64
  computes singles in double, and the phase of the harmonic source, tens of
  thousands of radians, drifts away from PyTorch.
- PyTorch's `cumsum` on the CPU sums float32 in double: so does the port.
- Each output of the matrix product sums in blocks of 16 in single, then an
  intermediate total in single, then the total in double: a plain float32
  sum drifted to 9e-6 from PyTorch (SNR 97 dB), this order gives the SNR of
  an all-double product (111 dB). The order does not depend on the tiling or
  the number of threads.
- The random initial phase of `SineGen` only touches the first sample of
  each frame, which the 1/300 interpolation never reads: it has no effect.
- The weights use `weight_norm` (g, v): w = v * (g / |v|) is computed once
  at load time, in float32 as `torch._weight_norm`.

## License

The mORMot three-license, MPL 1.1 / GPL 2.0 / LGPL 2.1: see [LICENSE](LICENSE)
and, for the model and other references, [NOTICE](NOTICE). The Qwen3-TTS
and Kokoro-82M models and weights are Apache-2.0 and are not part of this
repository. espeak-ng is GPL-3.0-or-later: it is neither shipped nor linked
at build time, only loaded at runtime from a folder the user provides.
`mormot.ai.tts.espeak` follows the punctuation handling of phonemizer
(GPL-3.0-or-later) step by step, so that the phonemes match: see NOTICE.
