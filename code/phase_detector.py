# Copyright (c) 2026 the authors of Efforts. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Efforts-Limited-Review-Verification-1.0
# Licensed under the Limited Review and Verification License; see ../LICENSE.md.

"""phase_detector.py — Phase segmentation on training_log.csv.

Algorithm:
  0. Downsample the per-batch log by STRIDE=15 before segmentation. RBF PELT on the
     full ~155k-row log is infeasible (the Gram matrix is ~n^2); the released
     phase_log was computed on this downsample, so all row-based quantities below
     are in downsampled rows.
  1. PELT change-point detection on loss_smoothed (ruptures, model=rbf, penalty=10)
  2. PELT on grad_norm series independently
  3. Merge breakpoints (union; collapse points within 200 downsampled rows)
  4. Enforce minimum phase length of 500 downsampled rows
  5. Compute per-phase summary statistics
  6. Write phase_log.csv

Usage:
  python phase_detector.py \\
      --input  data/training_log.csv \\
      --output data/phase_log.csv
  or import and call detect_phases(df) → pd.DataFrame
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PELT_MODEL = "rbf"
PELT_PENALTY = 10
MERGE_WINDOW = 200       # collapse breakpoints closer than this (downsampled rows)
MIN_PHASE_STEPS = 500    # minimum phase length (downsampled rows)
STRIDE = 15              # downsample the per-batch log by this factor before PELT


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------
def _pelt_breakpoints(series: np.ndarray, penalty: float = PELT_PENALTY) -> list[int]:
    """Run PELT on a 1-D signal. Returns breakpoint indices (exclusive end, 1-indexed)."""
    series_2d = series.reshape(-1, 1)
    algo = rpt.Pelt(model=PELT_MODEL, min_size=MIN_PHASE_STEPS).fit(series_2d)
    breakpoints = algo.predict(pen=penalty)
    # ruptures returns [bp1, bp2, ..., N] — last entry is len(series)
    return breakpoints[:-1]  # drop the trailing N


def _merge_breakpoints(bp1: list[int], bp2: list[int], n: int) -> list[int]:
    """Union of two breakpoint lists; collapse points within MERGE_WINDOW steps.

    Uses first-wins: when breakpoints cluster together, keep the earliest one.
    This avoids systematic drift from repeated midpoint replacement.
    """
    combined = sorted(set(bp1 + bp2))
    if not combined:
        return []
    merged = [combined[0]]
    for pt in combined[1:]:
        if pt - merged[-1] >= MERGE_WINDOW:
            merged.append(pt)
    return merged


def _enforce_min_duration(breakpoints: list[int], n: int, min_steps: int = MIN_PHASE_STEPS) -> list[int]:
    """Remove breakpoints that would create phases shorter than min_steps.

    Forward pass: skip interior breakpoints that would create a short phase.
    Then trim the final phase: if the tail is too short, remove the last
    interior breakpoint (merging it into the penultimate phase).
    """
    filtered = [0]
    for bp in breakpoints:
        if bp - filtered[-1] >= min_steps:
            filtered.append(bp)
    # Trim trailing short phase: if tail < min_steps, drop last interior bp
    if n - filtered[-1] < min_steps and len(filtered) > 1:
        filtered.pop()
    filtered.append(n)
    return filtered[1:-1]  # return only interior breakpoints


def _phase_stats(df: pd.DataFrame, start: int, end: int) -> dict:
    """Compute summary statistics for steps in [start, end)."""
    chunk = df.iloc[start:end]

    loss_vals = chunk["loss_smoothed"].dropna().values
    grad_vals = chunk["grad_norm"].dropna().values
    wu_vals = chunk["weight_update_magnitude"].dropna().values
    ae_vals = chunk["attention_entropy_mean"].dropna().values

    def safe_slope(y: np.ndarray) -> float:
        if len(y) < 2:
            return float("nan")
        x = np.arange(len(y), dtype=float)
        slope, *_ = stats.linregress(x, y)
        return float(slope)

    return {
        "mean_loss": float(np.mean(loss_vals)) if len(loss_vals) else float("nan"),
        "std_loss": float(np.std(loss_vals)) if len(loss_vals) else float("nan"),
        "mean_grad_norm": float(np.mean(grad_vals)) if len(grad_vals) else float("nan"),
        "std_grad_norm": float(np.std(grad_vals)) if len(grad_vals) else float("nan"),
        "mean_weight_update": float(np.mean(wu_vals)) if len(wu_vals) else float("nan"),
        "mean_attention_entropy": float(np.mean(ae_vals)) if len(ae_vals) else float("nan"),
        "loss_slope": safe_slope(loss_vals),
        "grad_norm_trend": safe_slope(grad_vals),
        "flag_contains_spike": bool(chunk["flag_loss_spike"].any()) if "flag_loss_spike" in chunk.columns else False,
        "flag_contains_plateau": bool(chunk["flag_plateau"].any()) if "flag_plateau" in chunk.columns else False,
        "flag_contains_grad_explosion": bool(chunk["flag_grad_explode"].any()) if "flag_grad_explode" in chunk.columns else False,
        "flag_contains_grad_vanish": bool(chunk["flag_grad_vanish"].any()) if "flag_grad_vanish" in chunk.columns else False,
    }


def detect_phases(df: pd.DataFrame, stride: int = STRIDE) -> pd.DataFrame:
    """
    Run phase detection on a training_log DataFrame.

    Args:
        df: DataFrame with columns matching training_log.csv schema.
            Must contain at least: step, epoch, loss_smoothed, grad_norm,
            weight_update_magnitude, attention_entropy_mean, flag_* columns.

    Returns:
        phase_df: DataFrame matching phase_log.csv schema.
    """
    # Downsample in ACQUISITION ORDER (do not sort first). The raw log concatenates
    # requeued runs and so contains step/epoch decreases; sorting by step would select
    # a different set of rows under df.iloc[::stride] and change the phase statistics.
    df = df.reset_index(drop=True)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    n = len(df)

    if n < MIN_PHASE_STEPS:
        # Too few rows for any meaningful phase detection; return single phase.
        print(f"[phase_detector] Only {n} steps — too few for PELT (min={MIN_PHASE_STEPS}). Returning single phase.")
        stats_dict = _phase_stats(df, 0, n)
        row = {
            "phase_id": 1,
            "step_start": int(df["step"].iloc[0]) if n > 0 else 0,
            "step_end": int(df["step"].iloc[-1]) if n > 0 else 0,
            "epoch_start": int(df["epoch"].iloc[0]) if n > 0 else 1,
            "epoch_end": int(df["epoch"].iloc[-1]) if n > 0 else 1,
            "mean_loss": stats_dict["mean_loss"],
            "std_loss": stats_dict["std_loss"],
            "mean_grad_norm": stats_dict["mean_grad_norm"],
            "std_grad_norm": stats_dict["std_grad_norm"],
            "mean_weight_update": stats_dict["mean_weight_update"],
            "mean_attention_entropy": stats_dict["mean_attention_entropy"],
            "loss_slope": stats_dict["loss_slope"],
            "grad_norm_trend": stats_dict["grad_norm_trend"],
            "phase_duration_steps": n,  # row count == step count for single phase
            "flag_contains_spike": stats_dict["flag_contains_spike"],
            "flag_contains_plateau": stats_dict["flag_contains_plateau"],
            "flag_contains_grad_explosion": stats_dict["flag_contains_grad_explosion"],
            "flag_contains_grad_vanish": stats_dict["flag_contains_grad_vanish"],
        }
        return pd.DataFrame([row])

    # Fill NaN in signal columns with forward fill for change-point detection
    loss_signal = df["loss_smoothed"].ffill().fillna(0.0).values.astype(float)
    grad_signal = df["grad_norm"].ffill().fillna(0.0).values.astype(float)

    # PELT on each signal
    print(f"[phase_detector] Running PELT on loss_smoothed ({n} steps) …")
    bp_loss = _pelt_breakpoints(loss_signal)
    print(f"[phase_detector] Running PELT on grad_norm ({n} steps) …")
    bp_grad = _pelt_breakpoints(grad_signal)
    print(f"[phase_detector] loss breakpoints: {len(bp_loss)}, grad breakpoints: {len(bp_grad)}")

    # Merge and enforce minimum duration
    merged = _merge_breakpoints(bp_loss, bp_grad, n)
    merged = _enforce_min_duration(merged, n)
    print(f"[phase_detector] Final breakpoints after merge+filter: {len(merged)}")

    # Build phase boundaries (row indices, inclusive start, exclusive end)
    boundaries = [0] + merged + [n]
    phases = list(zip(boundaries[:-1], boundaries[1:]))
    print(f"[phase_detector] {len(phases)} phases detected")

    # Build phase_log rows
    rows = []
    for phase_id, (start_row, end_row) in enumerate(phases, start=1):
        step_start = int(df["step"].iloc[start_row])
        step_end = int(df["step"].iloc[end_row - 1])
        epoch_start = int(df["epoch"].iloc[start_row])
        epoch_end = int(df["epoch"].iloc[end_row - 1])

        stats_dict = _phase_stats(df, start_row, end_row)
        rows.append({
            "phase_id": phase_id,
            "step_start": step_start,
            "step_end": step_end,
            "epoch_start": epoch_start,
            "epoch_end": epoch_end,
            "mean_loss": stats_dict["mean_loss"],
            "std_loss": stats_dict["std_loss"],
            "mean_grad_norm": stats_dict["mean_grad_norm"],
            "std_grad_norm": stats_dict["std_grad_norm"],
            "mean_weight_update": stats_dict["mean_weight_update"],
            "mean_attention_entropy": stats_dict["mean_attention_entropy"],
            "loss_slope": stats_dict["loss_slope"],
            "grad_norm_trend": stats_dict["grad_norm_trend"],
            "phase_duration_steps": end_row - start_row,
            "flag_contains_spike": stats_dict["flag_contains_spike"],
            "flag_contains_plateau": stats_dict["flag_contains_plateau"],
            "flag_contains_grad_explosion": stats_dict["flag_contains_grad_explosion"],
            "flag_contains_grad_vanish": stats_dict["flag_contains_grad_vanish"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[phase_detector] Loading {input_path} …")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[phase_detector] Loaded {len(df)} rows, {len(df.columns)} columns")

    phase_df = detect_phases(df, stride=args.stride)
    phase_df.to_csv(output_path, index=False)
    print(f"[phase_detector] Wrote {len(phase_df)} phases → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase detection on training_log.csv")
    parser.add_argument("--input",  default="data/training_log.csv")
    parser.add_argument("--output", default="data/phase_log.csv")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help="downsample the per-batch log by this factor before PELT")
    args = parser.parse_args()
    main(args)
