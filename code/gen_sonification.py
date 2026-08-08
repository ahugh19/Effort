#!/usr/bin/env python3
# Copyright (c) 2026 the authors of Efforts. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Efforts-Limited-Review-Verification-1.0
# Licensed under the Limited Review and Verification License; see ../LICENSE.md.

"""
gen_sonification.py — cinematic player-piano score from training signals.

A rule-based reading of the Lodge ``training_log.csv`` in a cinematic
player-piano idiom: a mechanical player piano that carries an actual motivic
*melody* (not just a broken-chord bed), a string quartet that enters one
instrument at a time with moving inner lines, and a low synth drone whose
brightness tracks the training trend.

Mapping (artist-designed rules, bound to this run's three usable signals):

  pitch        grad_norm (log, self-normalized) → register of both the melody
               and the accompaniment, constrained to 2nds/3rds inside D
               natural minor. Higher grad → higher register.
  rhythm       grad_sign_change_rate → density. The sign-change rate *is* a
               volatility measure (how often the gradient flips); stable bars
               get sparse, sustained notes, volatile bars get tighter
               subdivisions and syncopation.
  anomaly      |loss_delta| spikes → string tremolo + an unresolved chromatic
               passing tone (the only place chromaticism enters the lines).
  trend        per-bar grad level → synth-pad brightness (CC 74). Training
               converges, so the pad darkens across the piece.

Form follows the data's own arc, not a redemption arc, and each section has
its *own* progression + ostinato figure so the loop never simply repeats:

  INTRO   solo player piano, head motif derived from the window's first data
          points, over a rocking  i – VII – VI – VII  modal figure.
  BUILD   melody develops; strings enter one at a time (cello → viola →
          violin II → violin I) over the classic  i – VI – III – VII.
  CAESURA the last build bar drops to a single suspended note + drone — a
          breath — so the climax lands by contrast.
  CLIMAX  the run's single largest grad_norm (epoch 2, batch 29): driving
          16ths, tremolo strings, an unresolved  iv – i – V – i  (the A-major
          V brings the chromatic C#).
  DECAY   reverse dropout (violins out, then viola, then cello), the piano
          thinning over  i – VI – iv – i , sampled from the long convergence
          tail (epochs 3→end, log-compressed). Ends on a bare open fifth
          (D–A), no third, no resolution.

Texture stays transparent despite occasional enrichment layers, and silence is
used as a structural element. MIDI only; final mix and instrument choice happen
in Logic Pro. The file carries GM Bank Select + Volume + Pan +
a GM-System-On SysEx so Logic's "open in new project" path resolves each
Program Change to a GM voice.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi


# ─── instruments (GM programs) ───────────────────────────────────────────────

PLAYER_PIANO = 0      # Acoustic Grand — treated as a mechanical player piano
VIOLIN1 = 40
VIOLIN2 = 40
VIOLA = 41
CELLO = 42
SYNTH_DRONE = 89      # Pad 2 (warm) — low evolving texture
# enrichment layers (added around the melody/rhythm spine, not replacing it)
CONTRABASS = 43       # low foundation under the cello
STRING_PAD = 49       # slow strings — sustained harmonic bed
CELESTA = 8           # music-box doubling of the melody, an octave up
HARP = 46             # rolled-chord shimmer at phrase boundaries


# ─── form (scalable; every length-dependent constant is derived) ─────────────

BEATS_PER_BAR = 4
BEND_RANGE_SEMITONES = 2

# Populated by configure_length(); declared here so the rest of the module
# can `from gen_sonification import INTRO_BARS` etc. without import-time errors.
INTRO_BARS = BUILD_BARS = CLIMAX_BARS = DECAY_BARS = TOTAL_BARS = 0
INTRO = BUILD = CLIMAX = DECAY = range(0)
CAESURA_BAR = CLIMAX_PEAK_BAR = CLIMAX_PEAK_BEAT = PRE_DECAY_BEATS = 0
MELODY_BARS: set[int] = set()
PHRASE_STARTS: set[int] = set()
_BUILD_QUARTERS: tuple[int, int, int] = (0, 0, 0)
_DECAY_THIRDS: tuple[int, int] = (0, 0)
_MEMORY_BARS: set[int] = set()
_STRING_PAD_LO = _STRING_PAD_HI = _CB_DECAY_END = 0


def configure_length(scale: float = 1.0) -> None:
    """Rescale every form constant by ``scale``. scale=1.0 reproduces the
    original 36-bar / ~2:07 sonification exactly; scale=1.3 is ~2:43, etc.

    Each additional bar samples a *fresh* batch from the training-signals
    window, so the longer piece has new musical content per bar — it is
    regenerated against more data, not a time-stretched copy of the
    shorter version."""
    global INTRO_BARS, BUILD_BARS, CLIMAX_BARS, DECAY_BARS, TOTAL_BARS
    global INTRO, BUILD, CLIMAX, DECAY
    global CAESURA_BAR, CLIMAX_PEAK_BAR, CLIMAX_PEAK_BEAT, PRE_DECAY_BEATS
    global MELODY_BARS, PHRASE_STARTS
    global _BUILD_QUARTERS, _DECAY_THIRDS, _MEMORY_BARS
    global _STRING_PAD_LO, _STRING_PAD_HI, _CB_DECAY_END

    INTRO_BARS = max(4, round(6 * scale))
    BUILD_BARS = max(8, round(12 * scale))
    CLIMAX_BARS = max(4, round(5 * scale))
    DECAY_BARS = max(8, round(13 * scale))
    TOTAL_BARS = INTRO_BARS + BUILD_BARS + CLIMAX_BARS + DECAY_BARS

    INTRO = range(0, INTRO_BARS)
    BUILD = range(INTRO_BARS, INTRO_BARS + BUILD_BARS)
    CLIMAX = range(INTRO_BARS + BUILD_BARS, INTRO_BARS + BUILD_BARS + CLIMAX_BARS)
    DECAY = range(INTRO_BARS + BUILD_BARS + CLIMAX_BARS, TOTAL_BARS)

    CAESURA_BAR = BUILD.stop - 1
    CLIMAX_PEAK_BAR = INTRO_BARS + BUILD_BARS + (CLIMAX_BARS - 1) // 2
    CLIMAX_PEAK_BEAT = CLIMAX_PEAK_BAR * BEATS_PER_BAR
    PRE_DECAY_BEATS = (INTRO_BARS + BUILD_BARS + CLIMAX_BARS) * BEATS_PER_BAR

    _BUILD_QUARTERS = (BUILD_BARS // 4, BUILD_BARS // 2, 3 * BUILD_BARS // 4)
    _DECAY_THIRDS = (max(1, DECAY_BARS // 6), max(2, DECAY_BARS // 2))
    _STRING_PAD_LO = max(1, INTRO_BARS * 2 // 3)
    _STRING_PAD_HI = DECAY.start + round(0.55 * DECAY_BARS)
    _CB_DECAY_END = DECAY.start + round(0.46 * DECAY_BARS)
    _MEMORY_BARS = {
        DECAY.start + round(0.62 * DECAY_BARS),
        DECAY.start + round(0.77 * DECAY_BARS),
    }

    # MELODY_BARS: intro middle, build "play 3 rest 1" minus the last two
    # build bars (caesura prep + caesura), all of climax, then a first decay
    # cluster + gap + a fragmenting second cluster.
    melody: set[int] = set()
    for b in range(2, max(3, INTRO_BARS - 1)):
        melody.add(b)
    for i, b in enumerate(BUILD):
        if b >= CAESURA_BAR - 1:
            continue
        if (i % 4) != 3:
            melody.add(b)
    melody.update(CLIMAX)
    decay_list = list(DECAY)
    n_first = max(1, round(2 * scale))
    n_gap = max(1, round(2 * scale))
    n_second = max(2, round(3 * scale))
    for b in decay_list[:n_first]:
        melody.add(b)
    sec_start = n_first + n_gap
    for b in decay_list[sec_start : sec_start + n_second]:
        melody.add(b)
    MELODY_BARS = melody

    ps: set[int] = set()
    if 2 in MELODY_BARS:
        ps.add(2)
    for b in range(BUILD.start, BUILD.stop, 4):
        if b in MELODY_BARS:
            ps.add(b)
    ps.add(CLIMAX.start)
    if CLIMAX_BARS >= 4:
        ps.add(CLIMAX.start + (CLIMAX_BARS + 1) // 2)
    if decay_list[:1]:
        ps.add(decay_list[0])
    if sec_start < len(decay_list):
        ps.add(decay_list[sec_start])
    PHRASE_STARTS = ps


# Initialize at module import with the original 36-bar form.
configure_length(1.0)


# ─── D natural minor material ────────────────────────────────────────────────

_DEG = (0, 2, 3, 5, 7, 8, 10)


def build_scale(low: int = 38, high: int = 88) -> list[int]:
    """Every D-natural-minor pitch in [low, high], ascending. _DEG is relative
    to D (pc 2), so the octave grid must be anchored on a D, not on C."""
    out = []
    p = low - ((low - 2) % 12)          # the D at or just below `low`
    while p <= high:
        for d in _DEG:
            n = p + d
            if low <= n <= high:
                out.append(n)
        p += 12
    return sorted(set(out))


SCALE = build_scale()

# Chord vocabulary as (bass MIDI pitch, triad pitch-classes). D = pc 2.
CHORDS: dict[str, tuple[int, tuple[int, int, int]]] = {
    "i":   (38, (2, 5, 9)),     # Dm
    "III": (41, (5, 9, 0)),     # F
    "iv":  (43, (7, 10, 2)),    # Gm
    "v":   (45, (9, 0, 4)),     # Am
    "V":   (45, (9, 1, 4)),     # A major (C# — the chromatic dominant)
    "VI":  (34, (10, 2, 5)),    # B♭
    "VII": (36, (0, 4, 7)),     # C
    "bII": (39, (3, 7, 10)),    # E♭ (Neapolitan colour)
}

# Per-section progressions (4-chord cells, cycled to fill the section).
PROGRESSIONS = {
    "intro":  ("i", "VII", "VI", "VII"),
    "build":  ("i", "VI", "III", "VII"),
    "climax": ("iv", "i", "V", "i"),
    "decay":  ("i", "VI", "iv", "i"),
}

# Per-section ostinato figures (indices into the realized 4-tone arpeggio).
ARP_FIGURES = {
    "intro":  (0, 2, 1, 2),
    "build":  (0, 1, 2, 3, 2, 1),
    "climax": (0, 1, 2, 3, 3, 2, 1, 0),
    "decay":  (0, 2, 1),
}


def section_of(bar: int) -> str:
    if bar in INTRO:
        return "intro"
    if bar in BUILD:
        return "build"
    if bar in CLIMAX:
        return "climax"
    return "decay"


def chord_name_for(bar: int) -> str:
    sec = section_of(bar)
    cell = PROGRESSIONS[sec]
    # index within the section so each section starts its own cell at bar 0
    start = {"intro": INTRO.start, "build": BUILD.start,
             "climax": CLIMAX.start, "decay": DECAY.start}[sec]
    return cell[(bar - start) % len(cell)]


def realize(name: str, floor: int = 48) -> tuple[int, tuple[int, ...], tuple[int, int, int]]:
    """Return (bass pitch, 4-tone ascending arpeggio, triad pitch-classes)."""
    bass, pcs = CHORDS[name]
    tones: list[int] = []
    prev = floor
    for pc in (pcs[0], pcs[1], pcs[2], pcs[0]):     # root, third, fifth, root+oct
        n = prev + ((pc - prev) % 12)
        if tones and n <= tones[-1]:
            n += 12
        tones.append(n)
        prev = n
    return bass, tuple(tones), pcs


def nearest_scale_index(pitch: int) -> int:
    return int(np.argmin([abs(s - pitch) for s in SCALE]))


def nearest_chord_index(pitch: int, pcs: tuple[int, ...]) -> int:
    """Index of the nearest SCALE pitch whose pitch-class is in the chord."""
    cands = [i for i, s in enumerate(SCALE) if s % 12 in pcs]
    return min(cands, key=lambda i: abs(SCALE[i] - pitch))


# ─── helpers ─────────────────────────────────────────────────────────────────


def add_note(inst: pretty_midi.Instrument, pitch: int, vel: int,
             start: float, end: float) -> None:
    p = int(np.clip(pitch, 21, 108))
    v = int(np.clip(vel, 1, 127))
    if start < end:
        inst.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=start, end=end))


def pct_normalize(arr: np.ndarray, ref: np.ndarray, lo=1.0, hi=99.0,
                  use_log: bool = False) -> np.ndarray:
    """Normalize ``arr`` against ``ref``'s [lo, hi] percentile window → [0, 1]."""
    clean = ref[~np.isnan(ref)]
    if use_log:
        clean = np.abs(clean)
        arr_a = np.abs(arr)
        eps = max(1e-6, np.percentile(clean[clean > 0], 1) * 0.1) if (clean > 0).any() else 1e-6
        ref_t = np.log(clean + eps)
        arr_t = np.log(arr_a + eps)
    else:
        ref_t = clean
        arr_t = arr
    p_lo, p_hi = np.percentile(ref_t, [lo, hi])
    if p_hi - p_lo < 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr_t - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def gm_init(inst: pretty_midi.Instrument, volume: int = 100) -> None:
    inst.control_changes.append(pretty_midi.ControlChange(0, 0, 0.0))
    inst.control_changes.append(pretty_midi.ControlChange(32, 0, 0.0))
    inst.control_changes.append(pretty_midi.ControlChange(7, int(np.clip(volume, 0, 127)), 0.0))


def write_pan(inst: pretty_midi.Instrument, pan_0_127: int) -> None:
    inst.control_changes.append(
        pretty_midi.ControlChange(10, int(np.clip(pan_0_127, 0, 127)), 0.0))


def write_rpn(inst: pretty_midi.Instrument, semitones: int) -> None:
    for n, v in [(101, 0), (100, 0), (6, semitones), (38, 0), (101, 127), (100, 127)]:
        inst.control_changes.append(pretty_midi.ControlChange(n, v, 0.0))
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=0.0))


def set_brightness(inst: pretty_midi.Instrument, value: int, time: float) -> None:
    inst.control_changes.append(
        pretty_midi.ControlChange(74, int(np.clip(value, 0, 127)), max(0.0, time)))


def set_expression(inst: pretty_midi.Instrument, value: int, time: float) -> None:
    """CC 11 (Expression) — the dynamic volume orchestral patches use for
    swells. Distinct from CC 7 (static channel volume); fluidsynth and Logic
    both honor it, so a CC 11 ramp reads as a real crescendo/decrescendo."""
    inst.control_changes.append(
        pretty_midi.ControlChange(11, int(np.clip(value, 0, 127)), max(0.0, time)))


def set_sustain(inst: pretty_midi.Instrument, value: int, time: float) -> None:
    """CC 64 (Sustain Pedal). Press at the bar/chord change, lift just before
    the next one — gives the short staccato-ish arp notes a wet cinematic
    bloom without smearing across chord changes."""
    inst.control_changes.append(
        pretty_midi.ControlChange(64, int(np.clip(value, 0, 127)), max(0.0, time)))


def expr_ramp(inst: pretty_midi.Instrument, lo: int, hi: int,
              start: float, dur: float, n: int = 8) -> None:
    for f in np.linspace(0.0, 1.0, n):
        set_expression(inst, int(lo + (hi - lo) * f), start + f * dur)


def apply_transpose(mid: pretty_midi.PrettyMIDI, semitones: int) -> None:
    """Shift every note up/down by ``semitones`` and update the key signature.
    A clean-room way to produce a fingerprint-distinct variant without
    touching the musical content: 100% of the structure is preserved."""
    if semitones == 0:
        return
    for inst in mid.instruments:
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + semitones, 21, 108))
    for ks in mid.key_signature_changes:
        if 12 <= ks.key_number <= 23:           # pretty_midi: minor keys 12..23
            ks.key_number = 12 + (ks.key_number - 12 + semitones) % 12
        else:                                   # major keys 0..11
            ks.key_number = (ks.key_number + semitones) % 12


def inject_gm_mode_on(midi_path: str | Path) -> None:
    """Insert a GM-System-On SysEx (F0 7E 7F 09 01 F7) at the file start."""
    import mido
    mf = mido.MidiFile(str(midi_path))
    mf.tracks[0].insert(0, mido.Message('sysex', data=[0x7E, 0x7F, 0x09, 0x01], time=0))
    mf.save(str(midi_path))


def add_ritardando(midi_path: str | Path, base_bpm: int,
                   ritard_bars: int = 4, depth: float = 0.46) -> None:
    """Write a real tempo ritardando over the final ``ritard_bars``.

    pretty_midi writes one tempo; the note ticks are already fixed, so we only
    need to *slow the clock* under the final bars. We add a per-beat ``set_tempo``
    curve to the meta track that eases from base_bpm to base_bpm·(1-depth)
    (≈38 BPM by the closing bare fifth), with an ease-in so it sags more as it
    ends. Done in absolute ticks, then re-deltified."""
    import mido
    mf = mido.MidiFile(str(midi_path))
    bar_ticks = mf.ticks_per_beat * BEATS_PER_BAR
    start_bar = TOTAL_BARS - ritard_bars

    points = []
    for k in range(ritard_bars * BEATS_PER_BAR + 1):
        frac = k / (ritard_bars * BEATS_PER_BAR)
        bpm = base_bpm * (1.0 - depth * (frac ** 1.6))
        tick = start_bar * bar_ticks + int(round(k * mf.ticks_per_beat))
        points.append((tick, mido.bpm2tempo(bpm)))

    track = mf.tracks[0]
    abs_msgs, t = [], 0
    for msg in track:
        t += msg.time
        abs_msgs.append([t, msg])
    abs_msgs.extend([tick, mido.MetaMessage('set_tempo', tempo=tempo, time=0)]
                    for tick, tempo in points)

    eot_tick = max(t for t, _ in abs_msgs)
    body = [m for m in abs_msgs if m[1].type != 'end_of_track']
    body.sort(key=lambda x: x[0])              # stable: ties keep original order

    out = mido.MidiTrack()
    prev = 0
    for tick, msg in body:
        msg.time = tick - prev
        prev = tick
        out.append(msg)
    out.append(mido.MetaMessage('end_of_track', time=max(0, eot_tick - prev)))
    mf.tracks[0] = out
    mf.save(str(midi_path))


# ─── data window ─────────────────────────────────────────────────────────────


def sample_bars(df: pd.DataFrame) -> list[dict]:
    """Per-bar data for the whole piece. INTRO+BUILD+CLIMAX read one contiguous
    chronological window ending so the global grad_norm peak lands on
    CLIMAX_PEAK_BEAT; DECAY log-compresses the convergence tail."""
    s = df.sort_values(["epoch", "batch_index"]).reset_index(drop=True)
    n = len(s)

    ref_gn = s["grad_norm"].values
    ref_gscr = s["grad_sign_change_rate"].values
    ref_ld = s["loss_delta"].values

    peak_pos = int(np.nanargmax(ref_gn))
    win_start = int(np.clip(peak_pos - CLIMAX_PEAK_BEAT, 0, n - PRE_DECAY_BEATS - 1))

    pre_rows = [win_start + b for b in range(PRE_DECAY_BEATS)]
    decay_beats = DECAY_BARS * BEATS_PER_BAR
    tail_lo, tail_hi = win_start + PRE_DECAY_BEATS, n - 1
    frac = np.linspace(0.0, 1.0, decay_beats)
    decay_rows = [int(round(tail_lo + (tail_hi - tail_lo) * (np.expm1(3 * f) / np.expm1(3))))
                  for f in frac]
    rows = pre_rows + decay_rows

    # Self-normalize grad_norm / |loss_delta| against the piece's own sampled
    # series (log) — this run is violent from batch 0, so whole-run norm pins
    # epochs 0-2 to the ceiling and erases the arc.
    gn = pct_normalize(ref_gn[rows], ref_gn[rows], use_log=True)
    gscr = pct_normalize(ref_gscr[rows], ref_gscr)
    ld = pct_normalize(np.abs(ref_ld[rows]), np.abs(ref_ld[rows]), use_log=True)

    bars = []
    for bar in range(TOTAL_BARS):
        sl = slice(bar * BEATS_PER_BAR, (bar + 1) * BEATS_PER_BAR)
        gn_b, gscr_b, ld_b = gn[sl], gscr[sl], ld[sl]
        r0 = rows[bar * BEATS_PER_BAR]
        bars.append({
            "bar": bar,
            "gn": np.nan_to_num(gn_b),
            "gscr": np.nan_to_num(gscr_b),
            "ld": np.nan_to_num(ld_b),
            "gn_mean": float(np.nan_to_num(gn_b.mean())),
            "vol": float(np.nan_to_num(gscr_b.mean())),
            "ld_max": float(np.nan_to_num(ld_b.max())),
            "epoch": int(s["epoch"].iloc[r0]),
            "batch": int(s["batch_index"].iloc[r0]),
        })

    cut = float(np.percentile([b["ld_max"] for b in bars], 82))
    for b in bars:
        b["anomaly"] = b["ld_max"] >= cut and b["ld_max"] > 0.0
    return bars


def allocate_phase_bars(durations: np.ndarray, total_bars: int,
                        min_bars: int = 2) -> list[int]:
    """Bars per phase ∝ phase duration (largest-remainder rounding, floor of
    ``min_bars`` so every phase stays performable)."""
    raw = durations / durations.sum() * total_bars
    bars = np.maximum(min_bars, np.floor(raw)).astype(int)
    while bars.sum() > total_bars:                    # trim the largest
        bars[int(np.argmax(bars))] -= 1
    order = np.argsort(-(raw - np.floor(raw)))        # top up by remainder
    j = 0
    while bars.sum() < total_bars:
        bars[order[j % len(order)]] += 1
        j += 1
    return [int(b) for b in bars]


def sample_bars_phase(df: pd.DataFrame, run_dir: str) -> list[dict]:
    """Phase-proportional sampling (mode 'phase').

    Replaces the peak-anchored window + log-compressed decay with a single
    linear clock: each PELT phase from ``phase_log.csv`` receives bars in
    proportion to its ``phase_duration_steps`` (min 2), and its beats sample
    that phase's own batches evenly. Music time is therefore proportional to
    training time — every phase is audible and danceable, and downstream
    stage-shares equal the bar allocation (one clock, no contradiction).
    Normalization and the anomaly cutoff are unchanged from ``sample_bars``.
    """
    ph = pd.read_csv(Path(run_dir) / "phase_log.csv")
    s = df.sort_values(["epoch", "batch_index"]).reset_index(drop=True)
    steps = s["step"].values
    ref_gn = s["grad_norm"].values
    ref_gscr = s["grad_sign_change_rate"].values
    ref_ld = s["loss_delta"].values

    per_phase = allocate_phase_bars(
        ph["phase_duration_steps"].values.astype(float), TOTAL_BARS)
    print("  [phase mode] bars per phase:",
          {int(p): b for p, b in zip(ph["phase_id"], per_phase)})

    rows: list[int] = []
    bar_phase: list[int] = []
    for (_, p), nbars in zip(ph.iterrows(), per_phase):
        idxs = np.where((steps >= p.step_start) & (steps <= p.step_end))[0]
        pick = np.linspace(0, len(idxs) - 1, nbars * BEATS_PER_BAR)
        rows += [int(idxs[i]) for i in pick.round().astype(int)]
        bar_phase += [int(p.phase_id)] * nbars

    gn = pct_normalize(ref_gn[rows], ref_gn[rows], use_log=True)
    gscr = pct_normalize(ref_gscr[rows], ref_gscr)
    ld = pct_normalize(np.abs(ref_ld[rows]), np.abs(ref_ld[rows]), use_log=True)

    bars = []
    for bar in range(TOTAL_BARS):
        sl = slice(bar * BEATS_PER_BAR, (bar + 1) * BEATS_PER_BAR)
        gn_b, gscr_b, ld_b = gn[sl], gscr[sl], ld[sl]
        r0 = rows[bar * BEATS_PER_BAR]
        bars.append({
            "bar": bar,
            "gn": np.nan_to_num(gn_b),
            "gscr": np.nan_to_num(gscr_b),
            "ld": np.nan_to_num(ld_b),
            "gn_mean": float(np.nan_to_num(gn_b.mean())),
            "vol": float(np.nan_to_num(gscr_b.mean())),
            "ld_max": float(np.nan_to_num(ld_b.max())),
            "epoch": int(s["epoch"].iloc[r0]),
            "batch": int(s["batch_index"].iloc[r0]),
            "phase": bar_phase[bar],
        })

    cut = float(np.percentile([b["ld_max"] for b in bars], 82))
    for b in bars:
        b["anomaly"] = b["ld_max"] >= cut and b["ld_max"] > 0.0

    return bars


def step_timeline(run_dir: str, bpm: int = 70,
                  ritard_bars: int = 4, depth: float = 0.46) -> dict:
    """Audio-time → training-step map for the *current* length config.

    Mirrors ``sample_bars``' row selection (contiguous pre-decay window anchored
    on the global grad_norm peak, log-spaced decay tail) and reproduces
    ``add_ritardando``'s per-beat tempo curve, so the returned ``points`` give
    exactly where in training each musical moment was sampled from. The viewer
    uses this to drive the chart playhead along the *real* arc (the playhead
    sweeps the contiguous window, then accelerates through the log-compressed
    convergence tail) instead of a generic phase schedule.

    Returns a dict with:

    * ``points``  — ``[{"t", "step"}]`` per beat: audio fraction → the training
      ``step`` that beat was sampled from (drives the HUD step readout).
    * ``t``       — audio fraction per beat (the music-order chart x domain).
    * ``series``  — the per-beat signal values the score actually reads,
      normalized exactly as the generator does (grad_norm / |loss_delta| in log
      space self-referenced to the window, grad_sign_change_rate linear vs the
      whole run, loss_smoothed linear over the window). Plotting these gives a
      chart in *music order*, so the playhead moves uniformly instead of being
      glued to the left edge while the music dwells in the first few epochs.
    * ``section_bands`` — intro/build/climax/decay as beat-fraction spans, for
      shading the music-order chart (PELT step bands don't apply in this domain).
    """
    s = pd.read_csv(Path(run_dir) / "training_log.csv.gz")
    s = s.sort_values(["epoch", "batch_index"]).reset_index(drop=True)
    n = len(s)
    ref_gn = s["grad_norm"].values
    ref_gscr = s["grad_sign_change_rate"].values
    ref_ld = s["loss_delta"].values
    ref_ls = (s["loss_smoothed"].values if "loss_smoothed" in s.columns
              else np.zeros(n))
    steps = s["step"].values

    peak_pos = int(np.nanargmax(ref_gn))
    win_start = int(np.clip(peak_pos - CLIMAX_PEAK_BEAT, 0, n - PRE_DECAY_BEATS - 1))
    pre_rows = [win_start + b for b in range(PRE_DECAY_BEATS)]
    decay_beats = DECAY_BARS * BEATS_PER_BAR
    tail_lo, tail_hi = win_start + PRE_DECAY_BEATS, n - 1
    frac = np.linspace(0.0, 1.0, decay_beats)
    decay_rows = [int(round(tail_lo + (tail_hi - tail_lo) * (np.expm1(3 * f) / np.expm1(3))))
                  for f in frac]
    rows = pre_rows + decay_rows                          # one row per beat

    # Per-beat normalized signals — same transforms as sample_bars(), so the
    # chart curve is exactly what the music reads, in beat order.
    gn = pct_normalize(ref_gn[rows], ref_gn[rows], use_log=True)
    gscr = pct_normalize(ref_gscr[rows], ref_gscr)
    ld = pct_normalize(np.abs(ref_ld[rows]), np.abs(ref_ld[rows]), use_log=True)
    ls = pct_normalize(ref_ls[rows], ref_ls[rows])
    series = {
        "grad_norm":             [round(float(x), 5) for x in np.nan_to_num(gn)],
        "grad_sign_change_rate": [round(float(x), 5) for x in np.nan_to_num(gscr)],
        "loss_delta":            [round(float(x), 5) for x in np.nan_to_num(ld)],
        "loss_smoothed":         [round(float(x), 5) for x in np.nan_to_num(ls)],
    }

    # Per-beat duration: constant bpm, ritardando over the final ritard_bars.
    total_beats = TOTAL_BARS * BEATS_PER_BAR
    ritard_start_beat = (TOTAL_BARS - ritard_bars) * BEATS_PER_BAR
    base_dt = 60.0 / bpm
    beat_dt = []
    for b in range(total_beats):
        if b < ritard_start_beat:
            beat_dt.append(base_dt)
        else:
            f = (b - ritard_start_beat) / (ritard_bars * BEATS_PER_BAR)
            cur_bpm = bpm * (1.0 - depth * (f ** 1.6))
            beat_dt.append(60.0 / cur_bpm)

    t_cum = 0.0
    times = []
    for b in range(total_beats):
        times.append(t_cum)
        t_cum += beat_dt[b]
    duration = t_cum

    t_frac = [round(times[b] / duration, 5) for b in range(total_beats)]
    points = [{"t": t_frac[b], "step": int(steps[rows[b]])} for b in range(total_beats)]
    points.append({"t": 1.0, "step": int(steps[rows[-1]])})

    def band(lo_bar: int, hi_bar: int, label: str) -> dict:
        return {"label": label,
                "xStart": round(lo_bar * BEATS_PER_BAR / total_beats, 5),
                "xEnd": round(hi_bar * BEATS_PER_BAR / total_beats, 5)}
    section_bands = [
        band(INTRO.start, INTRO.stop, "intro"),
        band(BUILD.start, BUILD.stop, "build"),
        band(CLIMAX.start, CLIMAX.stop, "climax"),
        band(DECAY.start, DECAY.stop, "decay"),
    ]

    return {"duration_sec": round(duration, 2),
            "total_steps": int(np.nanmax(steps)),
            "total_beats": total_beats,
            "points": points,
            "t": t_frac,
            "series": series,
            "section_bands": section_bands}


# ─── arrangement-level dynamic arc ───────────────────────────────────────────


def section_dynamic(bar: int) -> float:
    """Macro loudness arc in [0, 1] (the data is loud-flat then quiet, so the
    crescendo/decrescendo is supplied by the form)."""
    if bar == CAESURA_BAR:
        return 0.42                           # the breath
    if bar in INTRO:
        return 0.62
    if bar in BUILD:
        return float(np.interp(bar - BUILD.start, [0, BUILD_BARS - 1], [0.68, 0.96]))
    if bar in CLIMAX:
        # arrive *into* the peak rather than bursting in at full after the breath
        d = bar - CLIMAX.start
        peak_d = CLIMAX_PEAK_BAR - CLIMAX.start
        if d <= peak_d:
            return float(np.interp(d, [0, peak_d], [0.84, 1.0]))
        return float(np.interp(d, [peak_d, CLIMAX_BARS - 1], [1.0, 0.92]))
    d = bar - DECAY.start
    return float(np.interp(d, [0, DECAY_BARS - 1], [0.9, 0.4]))


def active_strings(bar: int) -> set[str]:
    """Select at most two active quartet voices for the current bar."""
    if bar == CAESURA_BAR:
        return set()                          # caesura instrumentation handled explicitly
    if bar in BUILD:
        b = bar - INTRO_BARS
        q1, q2, q3 = _BUILD_QUARTERS
        if b < q1:
            return {"vc"}
        if b < q2:
            return {"vc", "va"}
        if b < q3:
            return {"vc", "v2"}
        return {"vc", "v1"}
    if bar in CLIMAX:
        return {"vc", "v1"}
    if bar in DECAY:
        d = bar - DECAY.start
        early, mid = _DECAY_THIRDS
        if d < early:
            return {"vc", "v1"}
        if d < mid:
            return {"vc"}
        return set()
    return set()


# ─── melodic walker ──────────────────────────────────────────────────────────


class Walker:
    """Steps through SCALE by ±1/±2 indices (2nds/3rds), held inside a register
    window whose center is set by the data each call."""

    def __init__(self, start_pitch: int):
        self.idx = nearest_scale_index(start_pitch)

    def step(self, center: int, rising: bool, span: int = 4) -> int:
        lo = max(0, nearest_scale_index(center) - span)
        hi = min(len(SCALE) - 1, nearest_scale_index(center) + span)
        delta = (1 if rising else -1) * (1 if (self.idx % 2 == 0) else 2)
        self.idx = int(np.clip(self.idx + delta, lo, hi))
        return SCALE[self.idx]

    def to(self, pitch: int) -> int:
        self.idx = nearest_scale_index(pitch)
        return SCALE[self.idx]


# ─── melody ──────────────────────────────────────────────────────────────────


def melody_rhythm(vol: float) -> list[tuple[float, float]]:
    """(onset, duration) pairs in beats. Volatility tightens subdivisions and
    introduces syncopation; stable bars sing in longer values with space."""
    if vol < 0.30:
        return [(0.0, 2.0), (2.0, 1.7)]                       # two long notes, breath
    if vol < 0.55:
        return [(0.0, 1.0), (1.0, 1.0), (2.0, 1.8)]
    if vol < 0.72:
        return [(0.0, 1.0), (1.0, 0.5), (1.5, 0.5), (2.0, 1.0), (3.0, 0.9)]
    return [(0.0, 0.75), (0.75, 0.75), (1.5, 0.5),            # syncopated
            (2.0, 0.5), (2.5, 0.5), (3.0, 0.9)]


def emit_melody(piano, walker, motif, bar_info, chord_pcs, phrase_start,
                dyn, bs, beat, anomaly, rng) -> list[tuple[int, int, float, float]]:
    gn_m, vol = bar_info["gn_mean"], bar_info["vol"]
    center = int(np.interp(gn_m, [0, 1], [64, 81]))
    rhythm = melody_rhythm(vol)
    rising = gn_m >= 0.5
    notes = []
    for k, (onset, dur) in enumerate(rhythm):
        strong = abs(onset - round(onset)) < 1e-6 and int(round(onset)) % 2 == 0
        if phrase_start and k == 0:
            pitch = SCALE[nearest_chord_index(center, chord_pcs)]
            walker.to(pitch)
        elif phrase_start and k <= len(motif):
            walker.idx = int(np.clip(walker.idx + motif[k - 1], 0, len(SCALE) - 1))
            pitch = SCALE[walker.idx]
        else:
            pitch = walker.step(center, rising, span=4)
            if strong:                                   # consonant landing
                walker.idx = nearest_chord_index(pitch, chord_pcs)
                pitch = SCALE[walker.idx]
        if anomaly and not strong:                       # unresolved chromatic
            pitch += int(rng.choice([-1, 1]))
        vel = int((76 + (8 if strong else 0)) * dyn)     # mechanical: no random
        t0 = bs + onset * beat
        t1 = t0 + dur * beat * 0.95
        add_note(piano, pitch, vel, t0, t1)
        notes.append((pitch, vel, t0, t1))
    return notes


# ─── core generator ──────────────────────────────────────────────────────────


def generate(run_dir: str, bpm: int = 70, seed: int = 7,
             motif_override: list[int] | None = None,
             mode: str = "window") -> tuple[pretty_midi.PrettyMIDI, list[str]]:
    df = pd.read_csv(Path(run_dir) / "training_log.csv.gz")
    bars = sample_bars_phase(df, run_dir) if mode == "phase" else sample_bars(df)
    rng = np.random.default_rng(seed)

    beat = 60.0 / bpm
    bar_sec = BEATS_PER_BAR * beat
    duration = TOTAL_BARS * bar_sec
    print(f"[gen_sonification] bpm={bpm}  {TOTAL_BARS} bars  ({duration:.1f}s)")
    print(f"  intro {len(INTRO)} | build {len(BUILD)} | climax {len(CLIMAX)} "
          f"(peak @ bar {CLIMAX_PEAK_BAR + 1}) | decay {len(DECAY)}    [bar #s are Logic 1-based]")

    mid = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    mid.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    mid.key_signature_changes.append(pretty_midi.KeySignature(14, 0.0))   # D minor

    piano = pretty_midi.Instrument(program=PLAYER_PIANO, name="Player Piano")
    v1 = pretty_midi.Instrument(program=VIOLIN1, name="Violin I")
    v2 = pretty_midi.Instrument(program=VIOLIN2, name="Violin II")
    va = pretty_midi.Instrument(program=VIOLA, name="Viola")
    vc = pretty_midi.Instrument(program=CELLO, name="Cello")
    pad = pretty_midi.Instrument(program=SYNTH_DRONE, name="Synth Pad / Drone")
    cb = pretty_midi.Instrument(program=CONTRABASS, name="Contrabass")
    spad = pretty_midi.Instrument(program=STRING_PAD, name="String Pad")
    cel = pretty_midi.Instrument(program=CELESTA, name="Celesta")
    hp = pretty_midi.Instrument(program=HARP, name="Harp")

    for inst, vol in [(piano, 96), (v1, 88), (v2, 84), (va, 86), (vc, 92),
                      (pad, 68), (cb, 84), (spad, 74), (cel, 70), (hp, 78)]:
        gm_init(inst, vol)
    write_pan(piano, 64); write_pan(v1, 44); write_pan(v2, 52)
    write_pan(va, 74); write_pan(vc, 82); write_pan(pad, 64)
    write_pan(cb, 70); write_pan(spad, 60); write_pan(cel, 58); write_pan(hp, 48)
    write_rpn(v1, BEND_RANGE_SEMITONES)
    write_rpn(pad, BEND_RANGE_SEMITONES)
    set_expression(v1, 0, 0.0)      # Violin I starts silent; it swells in (see V1_ENTER)

    # First bar where Violin I takes the lead — it emerges from niente over the
    # two bars before this so the entrance isn't a cold high tremolo.
    V1_ENTER = next(b for b in BUILD if "v1" in active_strings(b))

    # Head motif: 3 scale-step intervals derived from the window's first beats
    # (sign of the slope). Re-stated at every phrase start → recognizable.
    if motif_override is not None:
        motif = list(motif_override)[:3]
        print(f"  head motif (overridden):  {motif}")
    else:
        d = np.diff(bars[0]["gn"])
        motif = [int(np.clip(round(x * 4), -2, 2)) or (1 if x >= 0 else -1) for x in d][:3]
        if len(set(motif)) <= 1:
            motif = [1, 1, -2]                   # default sighing head
        print(f"  head motif (scale steps): {motif}")

    PIANO_VEL = 66
    ARP_PATTERN = ARP_FIGURES
    mel_walker = Walker(74)
    v1_walker = Walker(74)
    v2_walker = Walker(67)
    va_walker = Walker(60)
    log: list[str] = []
    last_bar = TOTAL_BARS - 1
    counts = {"melody": 0, "arp": 0, "strings": 0,
              "celesta": 0, "string_pad": 0, "contrabass": 0, "harp": 0}

    for bd in bars:
        bar = bd["bar"]
        bs = bar * bar_sec
        gn_m, vol, anomaly = bd["gn_mean"], bd["vol"], bd["anomaly"]
        sec = section_of(bar)
        dyn = section_dynamic(bar)
        cname = chord_name_for(bar)
        bass, tones, pcs = realize(cname)

        # Piano sustain pedal — press at the chord change, lift just before
        # the next bar so the short arp notes ring out into the harmony
        # without smearing across the chord boundary. The final bar holds
        # the pedal through the ritardando into the bare fifth.
        set_sustain(piano, 127, bs)
        if bar != last_bar:
            set_sustain(piano, 0, bs + bar_sec * 0.95)

        density = int(round(np.interp(vol, [0, 1], [4, 16])))
        if bar in INTRO:
            density = min(density, 8)
        if bar in DECAY:
            dd = bar - DECAY.start
            density = max(3, int(round(density * (1 - 0.6 * dd / DECAY_BARS))))
        octave = 12 if gn_m > 0.88 else (-12 if gn_m < 0.30 else 0)

        arp_on = bar not in (CAESURA_BAR, last_bar)
        melody_on = bar in MELODY_BARS and bar != last_bar

        # ── player piano: melody (foreground) ──────────────────────────────
        if melody_on:
            mnotes = emit_melody(
                piano, mel_walker, motif, bd, pcs, bar in PHRASE_STARTS,
                dyn, bs, beat, anomaly, rng)
            counts["melody"] += len(mnotes)
            if sec in ("build", "climax"):       # celesta music-box double, 8ve up
                for (p, v, t0, t1) in mnotes:
                    add_note(cel, p + 12, int(v * 0.5), t0, t1)
                    counts["celesta"] += 1

        # ── player piano: accompaniment ostinato (background, breathes) ─────
        if arp_on:
            fig = ARP_PATTERN[sec]
            step = bar_sec / density
            acc_dyn = dyn * (0.72 if melody_on else 0.92)    # duck under melody
            start = bar % 2
            for i in range(density):
                cell = tones[fig[(i + start) % len(fig)]] + octave
                if anomaly and i % 4 == 3:                   # chromatic lower neighbor
                    cell -= 1
                t0 = bs + i * step
                if vol > 0.6 and i % 2 == 1:
                    t0 += step * 0.18
                gate = 0.9 if vol > 0.5 else 0.62
                add_note(piano, cell, int(PIANO_VEL * acc_dyn), t0, t0 + step * gate)
                counts["arp"] += 1
            add_note(piano, bass + (octave if octave > 0 else 0),
                     int((PIANO_VEL - 6) * dyn), bs, bs + beat * 1.6)
            counts["arp"] += 1

        if bar == CAESURA_BAR:
            # The breath: a soft suspended piano note + soft cello, held for ~3
            # beats; then a wind-up on beat 4 (rising piano anacrusis + harp
            # roll into the next downbeat) so the climax is *arrived at*, not a
            # cold burst after silence. The drone keeps sounding underneath, so
            # this is a suspended hush, not dead air.
            sus = SCALE[nearest_chord_index(71, pcs) + 1]    # the 4th, unresolved
            add_note(piano, sus, 46, bs, bs + bar_sec * 0.70)
            add_note(vc, bass + 12, 40, bs, bs + bar_sec * 0.72)
            counts["melody"] += 1; counts["strings"] += 1
            nb, nt, _ = realize(chord_name_for(bar + 1))     # the climax chord
            for j, p in enumerate([nt[0] - 12, nt[1] - 12, nt[2] - 12, nt[0]]):
                t0 = bs + (3.0 + j * 0.25) * beat            # rising pickup, crescendoing
                add_note(piano, p, 38 + 9 * j, t0, t0 + 0.30 * beat)
            for j, p in enumerate([t + 12 for t in nt] + [nt[0] + 24]):
                t0 = bs + (3.0 + j * 0.11) * beat            # harp roll into the downbeat
                add_note(hp, p, 30 + 4 * j, t0, bs + bar_sec * 0.99)
                counts["harp"] += 1

        if bar == last_bar:                                  # bare open fifth, no third
            add_note(piano, 50, 54, bs, bs + bar_sec * 0.98)   # D3
            add_note(piano, 57, 50, bs, bs + bar_sec * 0.98)   # A3
            add_note(piano, 38, 46, bs, bs + bar_sec * 0.98)   # D2

        # ── synth drone: tonic D pedal, brightness tracks the trend ─────────
        # Enters at the late intro (bar 4) so the build doesn't drop in cold.
        if 4 <= bar < last_bar:
            set_brightness(pad, int(np.interp(gn_m, [0, 1], [26, 120])), bs)
            dv = int(np.clip((40 + 30 * gn_m) * dyn, 26, 80))
            add_note(pad, 26, dv, bs, bs + bar_sec * 0.99)             # D1 pedal
            if bar in CLIMAX:
                add_note(pad, 33, dv - 8, bs, bs + bar_sec * 0.99)    # add the 5th (A1)

        # ── string quartet — moving inner lines, not held whole notes ───────
        act = active_strings(bar) if bar != last_bar else set()

        if "vc" in act:                          # cello: root then fifth (moving bass)
            cv = int(np.clip((52 + 34 * gn_m) * dyn, 36, 102))
            if anomaly:                          # low same-pitch tremolo over a held root
                add_note(vc, bass, cv - 8, bs, bs + bar_sec * 0.98)       # held root (continuity)
                _tremolo(vc, bass + 12, int(cv * 0.78), bs, bar_sec, dense=False)
            else:
                add_note(vc, bass + 12, cv, bs, bs + bar_sec * 0.5 - 0.02)
                add_note(vc, bass + 19, cv - 6, bs + bar_sec * 0.5, bs + bar_sec * 0.98)
            counts["strings"] += 1

        if "va" in act:                          # viola: inner line, two steps
            base = nearest_chord_index(58, pcs)
            p1 = SCALE[base]
            p2 = SCALE[max(0, base - 1)]
            vv = int(np.clip((46 + 28 * gn_m) * dyn, 32, 90))
            add_note(va, p1, vv, bs, bs + bar_sec * 0.5 - 0.02)
            add_note(va, p2, vv - 4, bs + bar_sec * 0.5, bs + bar_sec * 0.92)
            counts["strings"] += 2

        if "v2" in act:                          # violin II: countermelody by 2nds
            vv = int(np.clip((46 + 30 * gn_m) * dyn, 32, 94))
            for off in (0.0, 2.0):
                p = v2_walker.step(67, rising=gn_m > 0.5, span=3)
                add_note(v2, p, vv, bs + off * beat, bs + (off + 1.9) * beat)
            counts["strings"] += 2

        # Violin I emergence — a near-silent sustained note swelling in (CC 11)
        # over the two bars before it takes the lead, so it isn't a cold open.
        if bar in (V1_ENTER - 2, V1_ENTER - 1):
            ped = SCALE[nearest_chord_index(72, pcs)]
            add_note(v1, ped, 26, bs, bs + bar_sec * 0.98)
            lo, hi = (4, 28) if bar == V1_ENTER - 2 else (28, 56)
            expr_ramp(v1, lo, hi, bs, bar_sec)
            counts["strings"] += 1

        if "v1" in act:                          # violin I: one continuous lyrical line
            if bar == V1_ENTER:                  # lyrical, swelling arrival
                vp = v1_walker.to(SCALE[nearest_chord_index(73, pcs)])
                vv = int(np.clip((46 + 26 * gn_m) * dyn, 36, 86))
                expr_ramp(v1, 56, 98, bs, bar_sec)   # finish the crescendo into the line
                add_note(v1, vp, vv, bs, bs + bar_sec * 0.97)
            else:
                target = int(np.interp(gn_m, [0, 1], [70, 85]))
                vp = v1_walker.step(target, rising=gn_m > 0.5, span=4)
                vv = int(np.clip((54 + 40 * gn_m) * dyn, 40, 112))
                if anomaly:
                    # anomaly = a chromatically-inflected sustained note that
                    # swells then eases — a sforzando hairpin, not a high
                    # tremolo siren. Keeps the line continuous.
                    vp += int(rng.choice([-1, 1]))      # unresolved chromatic neighbor
                    expr_ramp(v1, 85, 120, bs, bar_sec * 0.55)
                    expr_ramp(v1, 120, 92, bs + bar_sec * 0.55, bar_sec * 0.45)
                    add_note(v1, vp, vv, bs, bs + bar_sec * 0.97)
                else:
                    dur = 0.96 if bar in CLIMAX else 0.9
                    add_note(v1, vp, vv, bs, bs + bar_sec * dur)
            counts["strings"] += 1

        # ── enrichment layers: warmth around the melody/rhythm spine ────────
        rich = bar not in (CAESURA_BAR, last_bar)
        # string pad — sustained triad bed; swells in late intro (bar 4),
        # thins back out by the late decay so the lonely ending survives.
        if rich and _STRING_PAD_LO <= bar <= _STRING_PAD_HI:
            pv = int(np.clip((30 + 26 * gn_m) * dyn, 22, 76))
            for tone in tones[:3]:
                add_note(spad, tone, pv, bs, bs + bar_sec * 0.99)
                counts["string_pad"] += 1
        # contrabass — low foundation under the cello, drops when the cello does
        if rich and (bar in BUILD or bar in CLIMAX or (bar in DECAY and bar < _CB_DECAY_END)):
            cbv = int(np.clip((46 + 26 * gn_m) * dyn, 30, 88))
            add_note(cb, bass - 12, cbv, bs, bs + bar_sec * 0.98)
            counts["contrabass"] += 1
        # harp — rolled-chord shimmer at phrase boundaries and climax downbeats
        if rich and (bar in PHRASE_STARTS or bar in CLIMAX):
            hv = int(np.clip((26 + 22 * gn_m) * dyn, 20, 72))
            roll = [t + 12 for t in tones] + [tones[0] + 24]
            for j, p in enumerate(roll):
                t0 = bs + j * 0.035
                add_note(hp, p, hv, t0, bs + bar_sec * 0.6)
                counts["harp"] += 1

        # celesta "memory" echo — the head motif restated very high and soft
        # in the late decay, like a distant music-box recalling the opening.
        # Two glimpses (positions derived from DECAY_BARS), then silence.
        if bar in _MEMORY_BARS:
            wi = nearest_chord_index(72, pcs)
            walk = [wi]
            for step in motif:
                wi = int(np.clip(wi + step, 0, len(SCALE) - 1))
                walk.append(wi)
            t0 = bs + 1.5 * beat
            for k, si in enumerate(walk):
                p = SCALE[si] + 12              # one more octave up — distant
                add_note(cel, p, int(38 + 18 * dyn),
                         t0 + k * 0.55 * beat, t0 + (k + 1) * 0.55 * beat)
                counts["celesta"] += 1

        if bar == last_bar:
            voices = ["bare5th"]
        else:
            voices = (["mel"] if melody_on else []) + (["arp"] if arp_on else []) \
                + sorted(act) + (["pad"] if bar in (set(BUILD) | set(CLIMAX) | set(DECAY)) else [])
            if bar in (V1_ENTER - 2, V1_ENTER - 1):
                voices.append("v1~")            # Violin I swelling in from niente
        ph_tag = f" P{bd['phase']:>2d}" if "phase" in bd else ""
        log.append(
            f"bar {bar + 1:2d} [{sec:6s}]{ph_tag} ep{bd['epoch']:>4d}/b{bd['batch']:02d}  "
            f"{cname:3s} gn={gn_m:.2f} vol={vol:.2f} dens={density:2d} oct={octave:+d}"
            f"{' ANOMALY' if anomaly else ''}{' CAESURA' if bar == CAESURA_BAR else ''}"
            f"  → {'+'.join(voices) or 'silence'}")

    for inst in (piano, cel, hp, cb, vc, va, v2, v1, spad, pad):
        mid.instruments.append(inst)
    print("  notes: " + "  ".join(f"{k} {v}" for k, v in counts.items() if v))
    return mid, log


def _tremolo(inst: pretty_midi.Instrument, pitch: int, vel: int,
             bs: float, bar_sec: float, dense: bool, alt: int | None = None) -> None:
    """Fast bowed agitation across the bar. With ``alt`` it is a *fingered*
    tremolo — alternating two pitches — which reads as intentional string
    tension rather than a stuttered single note on a sustained GM patch."""
    n = 14 if dense else 8
    step = bar_sec / n
    for i in range(n):
        p = alt if (alt is not None and i % 2 == 1) else pitch
        t0 = bs + i * step
        add_note(inst, p, int(vel * (0.82 + 0.18 * (i % 2))), t0, t0 + step * 0.9)


DEFAULT_SOUNDFONT = "soundfonts/MuseScore_General.sf3"


def render_mp3(mid_path: Path, mp3_path: Path, soundfont: str,
               master: str = "standard") -> None:
    """MIDI -> fluidsynth (dry WAV) -> ffmpeg (loudness-normalized MP3).

    A *preview* of the GM rendering; final timbres are chosen in Logic Pro.
    This dry render carries no reverb or compression. The retained
    audio/sonification_render.mp3 was mastered with a hall reverb and gentle
    compression, which the GPLv3 ``pedalboard`` library provided; that step was
    removed here to keep the package free of a copyleft dependency, so this
    preview sounds drier and closer than the retained mastered file.
    fluidsynth and ffmpeg are invoked as separate processes, not linked.
    """
    import subprocess
    import tempfile
    sf = Path(soundfont)
    if not sf.exists():
        raise FileNotFoundError(
            f"soundfont not found: {sf}\n  download once with:\n"
            f"  curl -L -o {sf} https://ftp.osuosl.org/pub/musescore/"
            f"soundfont/MuseScore_General/MuseScore_General.sf3")

    with tempfile.TemporaryDirectory() as td:
        dry = Path(td) / "dry.wav"
        subprocess.run(
            ["fluidsynth", "-ni", "-g", "1.0", "-r", "44100",
             f"--fast-render={dry}", str(sf), str(mid_path)],
            check=True, capture_output=True, text=True)

        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(dry),
             "-af", ("loudnorm=I=-16:TP=-1:LRA=20" if master == "wide"
                     else "loudnorm=I=-14:TP=-1:LRA=11"),
             "-codec:a", "libmp3lame", "-b:a", "320k", "-q:a", "0", str(mp3_path)],
            check=True, capture_output=True, text=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Cinematic player-piano + quartet + drone score from training "
                    "signals (cinematic-style). Writes MIDI; optionally an MP3 preview.")
    p.add_argument("--run_dir", default="lodge_training/outputs/global/signal_logs")
    p.add_argument("--output", required=True, help="Output .mid path.")
    p.add_argument("--bpm", type=int, default=70)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--transpose", type=int, default=0,
                   help="Shift every pitch by N semitones (preserves all structure).")
    p.add_argument("--motif", default="",
                   help='Override the head motif as "a,b,c" scale steps, e.g. "2,-1,1".')
    p.add_argument("--length", type=float, default=1.0,
                   help="Scale all section bar counts. 1.0 = original ~2:07; "
                        "1.3 ≈ 2:43; 1.45 ≈ 3:00. Each extra bar samples a new "
                        "batch from the data (regeneration, not stretching).")
    p.add_argument("--mode", choices=["window", "phase"], default="window",
                   help="Sampling schedule. 'window' = peak-anchored window + "
                        "log-compressed decay (the original warped clock). "
                        "'phase' = bars per PELT phase ∝ phase duration — one "
                        "linear clock, every phase audible and danceable.")
    p.add_argument("--notes", help="Optional path to write the bar-by-bar breakdown.")
    p.add_argument("--mp3", help="Also render an MP3 preview to this path.")
    p.add_argument("--soundfont", default=DEFAULT_SOUNDFONT)
    p.add_argument("--master", choices=["standard", "wide"], default="standard",
                   help="MP3 master. 'wide' skips the compressor and relaxes "
                        "loudnorm so per-phase level differences survive.")
    args = p.parse_args(argv)

    motif_override = None
    if args.motif:
        motif_override = [int(x.strip()) for x in args.motif.split(",")]

    if args.length != 1.0:
        configure_length(args.length)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mid, log = generate(args.run_dir, bpm=args.bpm, seed=args.seed,
                        motif_override=motif_override, mode=args.mode)
    apply_transpose(mid, args.transpose)
    mid.write(str(out))
    inject_gm_mode_on(out)
    add_ritardando(out, base_bpm=args.bpm)

    print("\n  ── bar-by-bar breakdown (bar numbers are Logic 1-based) ──")
    for line in log:
        print("   " + line)
    if args.notes:
        header = "# bar numbers are Logic Pro 1-based (the first bar is bar 1)\n"
        Path(args.notes).write_text(header + "\n".join(log) + "\n")
        print(f"\n  breakdown → {args.notes}")
    print(f"\nWritten: {out}  (+ GM Mode On SysEx + Bank Select + Volume + Pan + ritardando)")

    if args.mp3:
        print(f"Rendering MP3 preview → {args.mp3} …")
        render_mp3(out, Path(args.mp3), args.soundfont, master=args.master)
        print(f"Written: {args.mp3}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
