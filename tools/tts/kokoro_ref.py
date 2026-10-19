"""Kokoro-82M reference for the native Pascal engine (mormot.ai.tts.kokoro).

Says one sentence with the official kokoro package, on the CPU in float32,
and saves everything needed to compare the port step by step: phonemes
(before and after misaki's replacements), ids, style vector, outputs of
ALBERT, of the predictor (durations, F0, N), of the text encoder, of each
stage of the iSTFTNet decoder, and the wav.

The harmonic source of the vocoder (SineGen) draws random initial phases and
noise: here they are set to zero, for a deterministic wav the port can
replay exactly (it has the same switch).

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install kokoro==0.9.4
    python tools/tts/kokoro_ref.py
    python tools/tts/kokoro_ref.py --text "Hostile fleet in local." --voice af_heart --out en-hostile

The test references (tests/test.tts.kokoro.ref):

    python tools/tts/kokoro_ref.py --out fr-siwis
    python tools/tts/kokoro_ref.py --out fr-long --text "Alerte, flotte hostile en approche par la porte de Jita. Trente pilotes, dont des intercepteurs et des croiseurs lourds. Restez à quai jusqu'à nouvel ordre."
    python tools/tts/kokoro_ref.py --out fr-accents --text "Arrivée imminente : évacuez la zone, la flotte ennemie est à deux sauts (Amarr) !"
    python tools/tts/kokoro_ref.py --out fr-numbers --speed 1.2 --text "12 pilotes en local, 3 Loki et 450 millions d'ISK; « attention » au portail."

Output: <root>/<out>/ (root: --root, or the KOKORO_TTS_REF environment
variable, or ./ref-kokoro): manifest.json and one raw file per tensor,
float32 (.f32) or int32 (.i32), little-endian, C order (the last dimension
varies fastest).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from kokoro import KModel, KPipeline
from kokoro import istftnet

REPO = "hexgrad/Kokoro-82M"


class Dump:
    """The saved tensors and their description for the manifest."""

    def __init__(self, folder):
        self.folder = folder
        self.tensors = {}
        os.makedirs(folder, exist_ok=True)

    def add(self, name, value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            value = value.float() if value.is_floating_point() else value.long()
            value = value.numpy()
        a = np.ascontiguousarray(value)
        if a.dtype.kind == "f":
            a, kind = a.astype("<f4"), "f32"
        else:
            a, kind = a.astype("<i4"), "i32"
        file = f"{name}.{kind}"
        a.tofile(os.path.join(self.folder, file))
        self.tensors[name] = {"file": file, "dtype": kind, "shape": list(a.shape)}


def quiet_sine(self, f0):
    """SineGen.forward without randomness: initial phases and noise at zero."""
    fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
    rad = (fn / self.sampling_rate) % 1
    rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.upsample_scale,
                        mode="linear").transpose(1, 2)
    phase = torch.cumsum(rad, dim=1) * 2 * torch.pi
    phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale,
                          scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
    sine_waves = torch.sin(phase) * self.sine_amp
    uv = self._f02uv(f0)
    noise = torch.zeros_like(sine_waves)
    return sine_waves * uv + noise, uv, noise


def adain_block(dump, name, block, x, s):
    out = block(x, s)
    dump.add(name, out[0])
    return out


def generator(dump, g, x, s, f0):
    """Generator.forward step by step."""
    f0u = g.f0_upsamp(f0[:, None]).transpose(1, 2)
    har_source, _, _ = g.m_source(f0u)
    har_source = har_source.transpose(1, 2).squeeze(1)
    dump.add("gen.har_source", har_source[0])
    spec, phase = g.stft.transform(har_source)
    har = torch.cat([spec, phase], dim=1)
    dump.add("gen.har", har[0])
    for i in range(g.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = g.noise_convs[i](har)
        dump.add(f"gen.noise_conv{i}", x_source[0])
        x_source = g.noise_res[i](x_source, s)
        dump.add(f"gen.noise_res{i}", x_source[0])
        x = g.ups[i](x)
        if i == g.num_upsamples - 1:
            x = g.reflection_pad(x)
        dump.add(f"gen.up{i}", x[0])
        x = x + x_source
        xs = None
        for j in range(g.num_kernels):
            r = g.resblocks[i * g.num_kernels + j](x, s)
            xs = r if xs is None else xs + r
        x = xs / g.num_kernels
        dump.add(f"gen.res{i}", x[0])
    x = F.leaky_relu(x)
    x = g.conv_post(x)
    dump.add("gen.post", x[0])
    spec = torch.exp(x[:, :g.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, g.post_n_fft // 2 + 1:, :])
    return g.stft.inverse(spec, phase)


def forward(dump, model, input_ids, ref_s, speed):
    """KModel.forward_with_tokens and Decoder.forward step by step."""
    n = input_ids.shape[1]
    lengths = torch.full((1,), n, dtype=torch.long)
    mask = torch.zeros((1, n), dtype=torch.bool)
    bert = model.bert(input_ids, attention_mask=(~mask).int())
    dump.add("bert", bert[0])
    d_en = model.bert_encoder(bert).transpose(-1, -2)
    dump.add("bert_encoder", d_en[0])
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, lengths, mask)
    dump.add("pred.text_encoder", d[0])
    x, _ = model.predictor.lstm(d)
    dump.add("pred.lstm", x[0])
    duration = model.predictor.duration_proj(x)
    dump.add("pred.duration_proj", duration[0])
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    dump.add("pred.duration", duration[0])
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    dump.add("pred.dur", pred_dur)
    indices = torch.repeat_interleave(torch.arange(n), pred_dur)
    aln = torch.zeros((n, indices.shape[0]))
    aln[indices, torch.arange(indices.shape[0])] = 1
    aln = aln.unsqueeze(0)
    en = d.transpose(-1, -2) @ aln
    dump.add("pred.en", en[0])
    p = model.predictor
    sh, _ = p.shared(en.transpose(-1, -2))
    dump.add("pred.shared", sh[0])
    f0 = sh.transpose(-1, -2)
    for i, block in enumerate(p.F0):
        f0 = adain_block(dump, f"pred.F0.{i}", block, f0, s)
    f0 = p.F0_proj(f0).squeeze(1)
    dump.add("pred.F0", f0[0])
    nn_ = sh.transpose(-1, -2)
    for i, block in enumerate(p.N):
        nn_ = adain_block(dump, f"pred.N.{i}", block, nn_, s)
    nn_ = p.N_proj(nn_).squeeze(1)
    dump.add("pred.N", nn_[0])
    t_en = model.text_encoder(input_ids, lengths, mask)
    dump.add("text_encoder", t_en[0])
    asr = t_en @ aln
    dump.add("asr", asr[0])
    dec = model.decoder
    style = ref_s[:, :128]
    f0c = dec.F0_conv(f0.unsqueeze(1))
    nc = dec.N_conv(nn_.unsqueeze(1))
    dump.add("dec.F0_conv", f0c[0])
    dump.add("dec.N_conv", nc[0])
    x = torch.cat([asr, f0c, nc], axis=1)
    x = adain_block(dump, "dec.encode", dec.encode, x, style)
    asr_res = dec.asr_res(asr)
    dump.add("dec.asr_res", asr_res[0])
    res = True
    for i, block in enumerate(dec.decode):
        if res:
            x = torch.cat([x, asr_res, f0c, nc], axis=1)
        x = adain_block(dump, f"dec.decode.{i}", block, x, style)
        if block.upsample_type != "none":
            res = False
    audio = generator(dump, dec.generator, x, style, f0)
    return audio.squeeze(), pred_dur


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--text", default="Cible en vue, deux hostiles en local.")
    p.add_argument("--voice", default="ff_siwis")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", default="fr-siwis")
    p.add_argument("--root", default=os.environ.get("KOKORO_TTS_REF", "ref-kokoro"))
    args = p.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)  # the order of the sums, from one run to the next
    istftnet.SineGen.forward = quiet_sine
    folder = os.path.join(args.root, args.out)
    dump = Dump(folder)
    snap = snapshot_download(REPO, allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/*.pt"])
    model = KModel(repo_id=REPO, config=os.path.join(snap, "config.json"),
                   model=os.path.join(snap, "kokoro-v1_0.pth")).eval()
    pipe = KPipeline(lang_code=args.voice[0], repo_id=REPO, model=False)
    raw = pipe.g2p.backend.phonemize([args.text.replace("«", chr(8220)).replace("»", chr(8221))
                                      .replace("(", "«").replace(")", "»")])
    ps, _ = pipe.g2p(args.text)
    ids = [i for i in (model.vocab.get(c) for c in ps) if i is not None]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    dump.add("input_ids", input_ids[0])
    pack = torch.load(os.path.join(snap, "voices", args.voice + ".pt"), weights_only=True)
    ref_s = pack[len(ps) - 1]
    dump.add("ref_s", ref_s[0])

    t0 = time.perf_counter()
    with torch.inference_mode():
        audio, pred_dur = forward(dump, model, input_ids, ref_s, args.speed)
        check, _ = model.forward_with_tokens(input_ids, ref_s, args.speed)
    dt = time.perf_counter() - t0
    gap = float((audio - check).abs().max())
    dump.add("audio", audio)

    dur = len(audio) / 24000
    manifest = {
        "model": REPO,
        "snapshot": snap,
        "text": args.text,
        "voice": args.voice,
        "speed": args.speed,
        "espeak_raw": raw[0] if raw else "",
        "phonemes": ps,
        "input_ids": [0, *ids, 0],
        "frames": int(pred_dur.sum()),
        "samples": len(audio),
        "audio_seconds": dur,
        "seconds": dt,
        "replay_max_abs": gap,
        "torch": torch.__version__,
        "note": "SineGen without randomness (initial phases and noise set to 0), one thread",
        "tensors": dump.tensors,
    }
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    import wave
    with wave.open(os.path.join(folder, "out.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((np.clip(audio.numpy(), -1, 1) * 32767).round().astype("<i2").tobytes())
    print(f"{ps!r}: {dur:.2f}s of audio ({int(pred_dur.sum())} frames) in {dt:.2f}s, "
          f"replay max |diff| {gap:.2e}; written to {folder}", flush=True)


if __name__ == "__main__":
    main()
