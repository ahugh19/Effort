#!/usr/bin/env python3
# Copyright (c) 2026 the authors of Efforts. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Efforts-Limited-Review-Verification-1.0
# Licensed under the Limited Review and Verification License; see ../LICENSE.md.

"""
build_stage_timeline.py — data for the stage companion animation.

Reads the phase-proportional sonification MIDI + its bar breakdown and emits
stage_timeline.json:

  duration_frac base is 1.0 (all times are fractions of the piece); the panel
  multiplies by the actual audio.duration at runtime, so a Suno cover of any
  length fits automatically.

  notes:  per-note {t, dur, pitch, vel, inst, phase} — drives the spark field.
          When --wav is given, note times are snapped to detected audio onsets
          (precise to the Suno performance), each carrying the pitch/vel/phase
          of the nearest MIDI note; otherwise MIDI times are used.
  bars:   per-bar  {t, phase, gn, vol, dens}  — drives the two data lines.
  phases: {phase, t0, t1}                      — the reserved word slots.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pretty_midi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from effort_words import effort_words           # noqa: E402  (shared Laban rule)

# Release-relative defaults: this file lives in <release>/code/, so the package
# root is one level up. Every path below is overridable on the command line.
ROOT = Path(__file__).resolve().parents[1]
MID = ROOT / "audio" / "sonification.mid"
NOTES = ROOT / "audio" / "sonification_notes.md"
PHASELOG = ROOT / "data" / "phase_log.csv"
OUT = ROOT / "timeline" / "stage_timeline.json"
WAV = ROOT / "audio" / "cover.mp3"

BAR_RE = re.compile(
    r"bar\s+(\d+)\s+\[\w+\s*\]\s+P\s*(\d+).*?gn=([\d.]+)\s+vol=([\d.]+)\s+dens=\s*(\d+)")


def analyze_audio(pm: "pretty_midi.PrettyMIDI", wav_path: Path):
    """Load the Suno cover once and return three things:

      onsets : detected attack times (s) — the precise accent layer.
      warp   : a monotone function midi_seconds -> suno_seconds, from a
               chroma DTW between the MIDI and the cover. Suno re-performs the
               piece (non-uniform tempo), so a linear midi/total mapping drifts;
               DTW aligns by melody/harmony and puts every MIDI event on the
               cover's clock.
      adur   : cover duration (s).
    """
    import librosa
    y, sr = librosa.load(str(wav_path), mono=True)
    adur = len(y) / sr

    # accent layer: onset detection
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    of = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=512,
                                    backtrack=True, delta=0.06, wait=3)
    onsets = librosa.frames_to_time(of, sr=sr, hop_length=512)

    # chroma DTW warp (coarser hop=1024 is plenty for a time-warp envelope)
    hop = 1024
    fs = sr / hop
    # +1e-6 floor so silent (all-zero) frames don't become NaN under L2 norm
    Cs = librosa.util.normalize(
        librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop) + 1e-6, norm=2, axis=0)
    Cm = librosa.util.normalize(pm.get_chroma(fs=fs) + 1e-6, norm=2, axis=0)
    _, wp = librosa.sequence.dtw(X=Cm, Y=Cs, metric="cosine",
                                 global_constraints=True, band_rad=0.25)
    wp = wp[::-1]                                    # ascending along the path
    mi = wp[:, 0] / fs                               # midi seconds
    si = librosa.frames_to_time(wp[:, 1], sr=sr, hop_length=hop)  # suno seconds
    uniq = np.unique(mi)                             # collapse vertical moves
    smap = np.maximum.accumulate(np.array([si[mi == u].mean() for u in uniq]))

    def warp(sec: float) -> float:
        return float(np.interp(sec, uniq, smap))

    return onsets, warp, adur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", default=str(MID), help="sonification MIDI score")
    ap.add_argument("--notes", default=str(NOTES), help="sonification bar breakdown (.md)")
    ap.add_argument("--phase-log", default=str(PHASELOG), help="phase_log.csv")
    ap.add_argument("--wav", default=str(WAV),
                    help="cover audio; note times snap to its detected onsets.")
    ap.add_argument("--output", default=str(OUT), help="output timeline JSON")
    ap.add_argument("--no-wav", action="store_true", help="Use MIDI times only.")
    args = ap.parse_args()
    notes_text = Path(args.notes).read_text()

    pm = pretty_midi.PrettyMIDI(str(args.midi))
    total = pm.get_end_time()
    downbeats = list(pm.get_downbeats()) + [total]      # bar start times + end

    # bar→phase map (from the breakdown), so any time can be tagged with a phase
    bar_phase = {}
    for line in notes_text.splitlines():
        m = BAR_RE.search(line)
        if m:
            bar_phase[int(m.group(1))] = int(m.group(2))
    def phase_at_frac(f: float) -> int:
        for b in range(len(downbeats) - 1, 0, -1):
            if f * total >= downbeats[b - 1]:
                return bar_phase.get(b, bar_phase.get(max(bar_phase), 1))
        return 1

    midi_notes = []
    for inst in pm.instruments:
        name = inst.name or f"prog{inst.program}"
        for n in inst.notes:
            midi_notes.append((n.start, n.end, int(n.pitch), int(n.velocity), name))
    midi_notes.sort()

    # audio analysis: onsets + a chroma-DTW warp midi_seconds -> suno_seconds.
    # to_frac() puts ANY midi time onto the cover's clock as a [0,1] fraction of
    # the cover duration, so notes/bars/phases all share Suno's (warped) timeline.
    use_wav = (not args.no_wav) and Path(args.wav).exists()
    warp = None
    if use_wav:
        onsets, warp, adur = analyze_audio(pm, Path(args.wav))

    def to_frac(sec: float) -> float:
        return warp(sec) / adur if warp else sec / total

    # fine layer: every MIDI note, warped onto the cover clock (kind="midi")
    fine = [{"t": round(to_frac(s), 5), "dur": round(max(0.0, to_frac(e) - to_frac(s)), 5),
             "pitch": p, "vel": v, "inst": nm, "phase": phase_at_frac(s / total),
             "kind": "midi"}
            for s, e, p, v, nm in midi_notes]

    # accent layer: real audio onsets (already on the cover clock); pitch/vel
    # come from the MIDI note whose WARPED time is nearest the onset (kind="onset")
    accents = []
    if use_wav:
        wstarts = np.array([warp(m[0]) for m in midi_notes])   # midi starts in suno-seconds
        for ot in onsets:
            j = int(np.argmin(np.abs(wstarts - ot)))
            s0, e0, pit, vel, nm = midi_notes[j]
            accents.append({"t": round(float(ot / adur), 5),
                            "dur": round(max(0.0, to_frac(e0) - to_frac(s0)), 5),
                            "pitch": pit, "vel": vel, "inst": nm,
                            "phase": phase_at_frac(s0 / total), "kind": "onset"})
        print(f"  [onset] {len(accents)} audio accents (cover {adur:.1f}s vs "
              f"midi {total:.1f}s) over {len(fine)} MIDI notes, chroma-DTW warped")

    notes = sorted(fine + accents, key=lambda x: (x["t"], x["kind"]))

    bars = []
    for line in notes_text.splitlines():
        m = BAR_RE.search(line)
        if not m:
            continue
        b = int(m.group(1))
        t0 = downbeats[b - 1] if b - 1 < len(downbeats) else total
        bars.append({
            "bar": b,
            "t": round(to_frac(t0), 5),                 # bar downbeat on the cover clock
            "phase": int(m.group(2)),
            "gn": float(m.group(3)),
            "vol": float(m.group(4)),
            "dens": int(m.group(5)),
        })

    phases = []
    for b in bars:
        if not phases or phases[-1]["phase"] != b["phase"]:
            phases.append({"phase": b["phase"], "t0": b["t"], "t1": 1.0})
        phases[-1]["t1"] = b["t"]
    for i, p in enumerate(phases):                       # extend each to the next start
        p["t1"] = phases[i + 1]["t0"] if i + 1 < len(phases) else 1.0

    # per-phase loss + the full Laban verdict (Weight/Time/Space/Effort) from the
    # shared effort_words() function, so the displayed word matches the audited rule.
    # From phase_log.csv (the same source as the words).
    import pandas as pd
    ph_df = pd.read_csv(args.phase_log)
    lo, hi = ph_df["mean_loss"].min(), ph_df["mean_loss"].max()
    loss_by_id = {int(r["phase_id"]): round(float((r["mean_loss"] - lo) / max(hi - lo, 1e-9)), 4)
                  for _, r in ph_df.iterrows()}
    by_id = {}
    for _, e in effort_words(ph_df).iterrows():
        pid = int(e["phase_id"])
        by_id[pid] = {
            "loss": loss_by_id.get(pid, 0.0),
            "weight": e["weight"], "time": e["time"],
            "space": e["space"], "effort": e["effort"],
        }
    for p in phases:
        p.update(by_id.get(p["phase"],
                 {"loss": 0.0, "weight": "light", "time": "sustained",
                  "space": "indirect", "effort": "Float"}))

    Path(args.output).write_text(json.dumps({
        "source": "sonification.mid",
        "onset_snapped": use_wav,
        "n_notes": len(notes), "n_fine": len(fine), "n_accents": len(accents),
        "n_bars": len(bars), "n_phases": len(phases),
        "pitch_lo": min(n["pitch"] for n in notes),
        "pitch_hi": max(n["pitch"] for n in notes),
        "notes": notes, "bars": bars, "phases": phases,
    }))
    print(f"wrote {args.output}: {len(fine)} midi + {len(accents)} "
          f"onset accents, {len(bars)} bars, {len(phases)} phases")


if __name__ == "__main__":
    main()
