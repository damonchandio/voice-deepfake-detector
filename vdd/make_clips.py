"""Build the test set: real LibriSpeech clips, and the same sentences spoken by two free TTS engines.

    python -m vdd.make_clips      -> clips/real/*.wav, clips/kokoro-*/*.wav, clips/piper-*/*.wav, clips/index.csv

Real: hf-internal-testing/librispeech_asr_dummy (73 clips of LibriSpeech validation-clean, CC BY 4.0).
Fake: Kokoro-82M (Apache-2.0) and Piper (MIT) voices, run locally on a CPU. Same text, so the only thing that
differs between a real and a fake clip is who (or what) is speaking.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA, CLIPS = ROOT / "data", ROOT / "clips"
KOKORO_VOICES = ["af_heart", "bm_george"]   # one US, one British
PIPER_VOICES = ["en_GB-alan-medium", "en_US-lessac-medium"]


def real_clips() -> list[tuple[str, np.ndarray, int, str]]:
    import pyarrow.parquet as pq
    out = []
    for r in pq.read_table(DATA / "ls.parquet").to_pylist():
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        out.append((r["id"], wav, sr, r["text"].capitalize()))
    return out


def main() -> None:
    rows = []
    real = real_clips()
    (CLIPS / "real").mkdir(parents=True, exist_ok=True)
    for cid, wav, sr, text in real:
        p = CLIPS / "real" / f"{cid}.wav"
        sf.write(p, wav, sr)
        rows.append({"path": str(p.relative_to(ROOT)), "label": "real", "source": "librispeech", "text": text})

    from kokoro_onnx import Kokoro
    k = Kokoro(str(DATA / "kokoro.onnx"), str(DATA / "voices.bin"))
    for v in KOKORO_VOICES:
        d = CLIPS / f"kokoro-{v}"
        d.mkdir(exist_ok=True)
        for cid, _, _, text in real:
            wav, sr = k.create(text, voice=v, speed=1.0, lang="en-gb" if v.startswith("b") else "en-us")
            sf.write(d / f"{cid}.wav", wav, sr)
            rows.append({"path": str((d / f"{cid}.wav").relative_to(ROOT)), "label": "fake", "source": f"kokoro-{v}",
                         "text": text})

    from piper import PiperVoice
    for v in PIPER_VOICES:
        voice = PiperVoice.load(str(DATA / f"{v}.onnx"))
        d = CLIPS / f"piper-{v}"
        d.mkdir(exist_ok=True)
        for cid, _, _, text in real:
            chunks = list(voice.synthesize(text))
            wav = np.concatenate([c.audio_float_array for c in chunks])
            sf.write(d / f"{cid}.wav", wav, chunks[0].sample_rate)
            rows.append({"path": str((d / f"{cid}.wav").relative_to(ROOT)), "label": "fake", "source": f"piper-{v}",
                         "text": text})

    with (CLIPS / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips: {len(real)} real, {len(rows) - len(real)} fake")


if __name__ == "__main__":
    main()
