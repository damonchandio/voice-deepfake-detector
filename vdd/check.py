"""Ask every detector about one clip.

    python -m vdd.check voicemail.wav
"""
from __future__ import annotations

import sys

from .detect import DETECTORS, Detector, load_audio


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m vdd.check <audio file>")
    wav = load_audio(sys.argv[1])
    votes = 0
    for name in DETECTORS:
        p = Detector(name).p_fake(wav)
        votes += p >= 0.5
        print(f"{p:6.1%} fake  {name}")
    print(f"\n{votes} of {len(DETECTORS)} detectors say fake. See the README before trusting either answer.")


if __name__ == "__main__":
    main()
