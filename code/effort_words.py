# Copyright (c) 2026 the authors of Effort. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Effort-Limited-Review-Verification-1.0
# Licensed under the Limited Review and Verification License; see ../LICENSE.md.

"""effort_words.py — assign one Laban Effort word to each training phase.

A pure function of phase_log.csv, implementing the Weight / Time / Space rule
documented in the supplementary appendix (Section "Translating the training
trace into Laban Effort words"). For phase p, with mean gradient norm mu, its
gradient-norm standard deviation sigma, and its gradient-norm temporal trend
tau (grad_norm_trend):

    Weight = strong if mu    >= median(mu)      else light
    Time   = sudden if sigma >= mu              else sustained
    Space  = direct if |tau| >= median(|tau|)   else indirect

The three binary factors index the eight Efforts of the Laban cube.

Usage:
  python effort_words.py --input data/phase_log.csv [--output data/effort_words.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# eight Efforts, keyed by (weight, time, space)
EFFORT_CUBE = {
    ("light",  "sustained", "indirect"): "Float",
    ("light",  "sustained", "direct"):   "Glide",
    ("light",  "sudden",    "direct"):   "Dab",
    ("light",  "sudden",    "indirect"): "Flick",
    ("strong", "sustained", "direct"):   "Press",
    ("strong", "sustained", "indirect"): "Wring",
    ("strong", "sudden",    "direct"):   "Punch",
    ("strong", "sudden",    "indirect"): "Slash",
}


def effort_words(ph: pd.DataFrame) -> pd.DataFrame:
    """Return phase_id, weight, time, space, effort for each phase in phase_log."""
    mu = ph["mean_grad_norm"].to_numpy(dtype=float)
    sigma = ph["std_grad_norm"].to_numpy(dtype=float)
    tau = ph["grad_norm_trend"].to_numpy(dtype=float)
    m_mu = float(np.median(mu))
    m_tau = float(np.median(np.abs(tau)))

    rows = []
    for i in range(len(ph)):
        weight = "strong" if mu[i] >= m_mu else "light"
        time = "sudden" if sigma[i] >= mu[i] else "sustained"
        space = "direct" if abs(tau[i]) >= m_tau else "indirect"
        rows.append({
            "phase_id": int(ph["phase_id"].iloc[i]),
            "weight": weight,
            "time": time,
            "space": space,
            "effort": EFFORT_CUBE[(weight, time, space)],
        })
    return pd.DataFrame(rows)


def main(args):
    ph = pd.read_csv(args.input)
    out = effort_words(ph)
    print(out.to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\n[effort_words] wrote {len(out)} phases -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Laban Effort word per training phase")
    p.add_argument("--input", default="data/phase_log.csv")
    p.add_argument("--output", default="")
    main(p.parse_args())
