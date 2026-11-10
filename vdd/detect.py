"""Run free Hugging Face voice-deepfake detectors on a clip and return P(fake).

Each detector names its classes differently ("fake"/"real", in either order), so the fake class is looked up
from the model's own id2label instead of assumed.
"""
from __future__ import annotations

import numpy as np

DETECTORS = [   # the five most-downloaded audio deepfake classifiers on Hugging Face with clear labels (Nov 2026)
    "mo-thecreator/Deepfake-audio-detection",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
    "garystafford/wav2vec2-deepfake-voice-detector",
    "MelodyMachine/Deepfake-audio-detection-V2",
    "Vansh180/deepfake-audio-wav2vec2",
]
SR = 16_000
MAX_SECONDS = 12   # long clips are cut: these models were trained on short clips and memory grows with length


def load_audio(path: str) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav[: SR * MAX_SECONDS]


class Detector:
    def __init__(self, name: str):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.name, self.torch = name, torch
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModelForAudioClassification.from_pretrained(name).eval()
        labels = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        fake = [i for i, l in labels.items() if l in ("fake", "spoof", "deepfake")]
        if len(fake) != 1:
            raise ValueError(f"{name}: can't tell which class is fake from {labels}")
        self.fake = fake[0]

    def p_fake(self, wav: np.ndarray) -> float:
        x = self.fe(wav, sampling_rate=SR, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**x).logits[0]
        return float(self.torch.softmax(logits, -1)[self.fake])
