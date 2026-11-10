"""Run every detector on every clip in clips/index.csv and write results/.

    python -m vdd.bench            -> results/scores.csv, results/summary.json, prints a markdown table
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from .detect import DETECTORS, Detector, load_audio
from .metrics import auc, rates

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    clips = list(csv.DictReader((ROOT / "clips" / "index.csv").open(encoding="utf-8")))
    wavs = [load_audio(str(ROOT / c["path"])) for c in clips]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows, summary = [], {}
    for name in DETECTORS:
        t0 = time.time()
        det = Detector(name)
        scores = [det.p_fake(w) for w in wavs]
        labels = [c["label"] for c in clips]
        s = {"overall": {**rates(scores, labels), "auc": round(auc(scores, labels), 3)},
             "seconds": round(time.time() - t0)}
        for src in sorted({c["source"] for c in clips if c["label"] == "fake"}):
            keep = [i for i, c in enumerate(clips) if c["source"] in (src, "librispeech")]
            s[src] = {"caught": rates([scores[i] for i in keep], [labels[i] for i in keep])["caught"],
                      "auc": round(auc([scores[i] for i in keep], [labels[i] for i in keep]), 3)}
        summary[name] = s
        rows += [{"detector": name, "path": c["path"], "label": c["label"], "source": c["source"], "p_fake": round(p, 4)}
                 for c, p in zip(clips, scores)]
        print(name, json.dumps(s), flush=True)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    srcs = [k for k in next(iter(summary.values())) if k not in ("overall", "seconds")]
    print("\n| Detector | real voices flagged fake | " + " | ".join(f"{s} caught" for s in srcs) + " | AUC |")
    print("|---|---|" + "---|" * len(srcs) + "---|")
    for name, s in summary.items():
        cells = [f"{s[src]['caught']:.0%}" for src in srcs]
        print(f"| {name} | {s['overall']['false_alarm']:.0%} | " + " | ".join(cells) + f" | {s['overall']['auc']} |")


if __name__ == "__main__":
    main()
