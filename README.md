# voice-deepfake-detector

![licence](https://img.shields.io/badge/licence-MIT-16A6F0) ![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)

**Do the free voice-deepfake detectors on Hugging Face catch voices made with free text-to-speech? I tested the five most downloaded. Mostly, no.**

Scam calls and voice notes made with cloned or synthetic voices are getting cheaper every month. When people ask "is this voice real?", the free answer is usually one of a handful of Hugging Face models. I wanted to know how much to trust them, so I made a small, fair test: the same 73 sentences, read once by real people and four times by free AI voices that anyone can run on a laptop.

A fake voice (Kokoro `af_heart`, made on a laptop in a few seconds):

```
$ python -m vdd.check voicemail.wav
  0.0% fake  mo-thecreator/Deepfake-audio-detection
 16.0% fake  Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
 94.5% fake  garystafford/wav2vec2-deepfake-voice-detector
  0.0% fake  MelodyMachine/Deepfake-audio-detection-V2
  7.5% fake  Vansh180/deepfake-audio-wav2vec2

1 of 5 detectors say fake. See the README before trusting either answer.
```

## What I found

At each model's own threshold (50%). "Real voices flagged" is the share of genuine human clips it called fake. The other columns are the share of each AI voice it caught. AUC is how well it ranks fakes above real clips overall: 1.0 is perfect, 0.5 is a coin toss.

| Detector | Real voices flagged fake | Kokoro af_heart caught | Kokoro bm_george caught | Piper alan (GB) caught | Piper lessac (US) caught | AUC |
|---|---|---|---|---|---|---|
| mo-thecreator/Deepfake-audio-detection | 3% | **14%** | 96% | 99% | 62% | 0.81 |
| Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | 14% | 30% | 75% | 77% | 34% | 0.82 |
| garystafford/wav2vec2-deepfake-voice-detector | **67%** | 90% | 100% | 100% | 99% | 0.92 |
| MelodyMachine/Deepfake-audio-detection-V2 | 3% | 0% | 0% | 0% | 0% | **0.05** |
| Vansh180/deepfake-audio-wav2vec2 | 0% | 0% | 0% | 0% | 0% | 0.41 |

Three things stood out:

1. **One free voice slips past the most popular detector.** The most downloaded model caught 96 to 99% of three AI voices, then only 14% of Kokoro's `af_heart`, a free voice you can run on a CPU. Whether you're caught depends on which voice you picked, not on whether the voice is fake.
2. **The model with the best score is the least usable.** garystafford's detector separates real from fake best (AUC 0.92), but at its default setting it called two thirds of the real human recordings fake. A detector that says "fake" to almost everything catches everything.
3. **Two of the five catch nothing at all.** MelodyMachine's V2 scored every clip as real. Its AUC of 0.05 means it ranks the clips almost perfectly, but backwards: real voices look more fake to it than AI voices do, which points to swapped labels or a model that learned the opposite of what it says. Vansh180's model also scored everything as real, and ranked no better than chance.

**If you need to check a voice, don't trust any single free detector.** Ask a question only the real person could answer, or call them back on a number you already have.

## How it works

- **Real speech:** 73 clips of LibriSpeech validation-clean (via `hf-internal-testing/librispeech_asr_dummy`, CC BY 4.0), people reading audiobooks.
- **Fake speech:** the same 73 sentences from four free voices run locally: Kokoro-82M (`af_heart`, `bm_george`, Apache-2.0) and Piper (`en_GB-alan-medium`, `en_US-lessac-medium`, MIT). Same words, so the only difference between a real and a fake clip is the speaker.
- **Detectors:** the five most downloaded audio deepfake classifiers on Hugging Face (November 2026) with clear real/fake labels. The fake class is read from each model's own `id2label`, not assumed. Audio is resampled to 16 kHz, and anything over 12 seconds is cut.
- **Scores:** catch rate, false alarm rate and accuracy at 0.5, plus AUC, overall and per voice ([`vdd/metrics.py`](vdd/metrics.py), with a test).

```
pip install -r requirements.txt
python -m vdd.make_clips   # needs the model files in data/ (see make_clips.py), about 10 minutes on a CPU
python -m vdd.bench        # about 25 minutes on a 2-core CPU; writes results/scores.csv and results/summary.json
python -m vdd.check your_clip.wav
```

## Limits

- 365 clips is small. The ordering of the detectors is clear, but treat each percentage as roughly plus or minus 10 points.
- The real clips are clean audiobook reading. Phone calls, voice notes and noisy rooms would push every false alarm rate up.
- Only two free TTS engines. ElevenLabs and other commercial voices weren't tested.
- One threshold (50%). Some detectors would do better with a tuned threshold, but a person using them doesn't get one.

## Licence

MIT for the code. The audio isn't redistributed: `make_clips` rebuilds it from the sources above under their own licences.
