# Effort: public release (minimal reproducible subset)

This package is the minimal set of files that accompanies the supplementary
appendix of the paper (`appendix/Effort_appendix.pdf`) and matches its
reproducibility section. It lets a reader regenerate the deterministic parts of the pipeline and
inspect every retained artifact, without redistributing FineDance. The re-performed
cover is included as a fixed generative output, not
as something to be regenerated.

## Contents

```
code/
  phase_detector.py            PELT change-point segmentation: training log -> 14 phases
  gen_sonification.py          rule-based sonification: log + phases -> symbolic score + dry preview
  build_stage_timeline.py      note/bar/phase/word timeline + chroma-DTW alignment to the cover
  effort_words.py              Weight/Time/Space -> one Laban Effort word per phase
  config/
    lodge_coarse_finedance.yaml  Lodge Global-module training config (Adan, LR 4e-4, seed 1234, ...)
data/
  training_log.csv.gz          per-batch training trace (155,232 rows, 39 columns)
  phase_log.csv                14-phase summary (means, std, trend, flags)
audio/
  sonification.mid             symbolic sonification score
  sonification_render.mp3      fixed mastered render of the deterministic score
  cover.mp3                     the re-performed cover heard in the video, a fixed generative output
  cover_kun.mp3                 the Kunqu-style cover that conditions the dance, a fixed generative output
timeline/
  stage_timeline.json          aligned note/bar/phase/word timeline used by the video
appendix/
  Effort_appendix.pdf          the supplementary appendix of the paper, compiled from the files below
  Effort_appendix.tex          standalone wrapper (title, authors, bibliography)
  appendix.tex                 the appendix text, identical to tex/appendix.tex in the paper source
  reference.bib, neurips_2026.sty   bibliography and NeurIPS 2026 style needed to rebuild the PDF
```

The Weight/Time/Space to Effort-word rule and all thresholds are specified in the
supplementary appendix (`appendix/Effort_appendix.pdf`, Section C) and are a pure
function of `data/phase_log.csv`. To rebuild the appendix PDF, run
`latexmk -pdf Effort_appendix.tex` inside `appendix/`.

## Not included (obtain from the source)

| asset | why | how to get it |
|---|---|---|
| FineDance dataset | Not ours to redistribute. | From the dataset authors. |
| Trained checkpoint | Large, and not needed for the trace-based results. | Not reproducible from this minimal subset, available on request. |

## Reproduce the deterministic pipeline

Training itself is not runnable from this package because the Lodge training stack is
not included. Start from step 2 with the provided `data/training_log.csv.gz`.

```bash
# 1. training is NOT included here (no Lodge training stack). The config records the
#    settings, and the resulting per-batch log is provided as data/training_log.csv.gz

# 2. segment the log into the 14 phases  (pandas reads the .gz directly)
#    (downsamples by 15 in acquisition order. RBF PELT on ~10.3k rows needs ~1 GB
#     of RAM and takes roughly 40 minutes. Its cost is quadratic in row count.
#     This is the provided path for recomputing the phase boundaries.)
python code/phase_detector.py --input data/training_log.csv.gz --output data/phase_log.csv

# 3. sonify: log + phases -> symbolic MIDI score (exact flags in the script header)
python code/gen_sonification.py --run_dir data --mode phase --length 1.3 \
  --output audio/sonification.mid --notes audio/sonification_notes.md
# to also render the MP3 preview, add --mp3 and --soundfont (needs FluidSynth + FFmpeg):
#   python code/gen_sonification.py --run_dir data --mode phase --length 1.3 \
#     --output audio/sonification.mid --notes audio/sonification_notes.md \
#     --mp3 audio/sonification_render.mp3 --soundfont MuseScore_General.sf3
# SoundFont: MuseScore General (MuseScore_General.sf3), from
#   https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3
# This renders a dry preview. The retained audio/sonification_render.mp3 was
# mastered with a hall reverb and gentle compression, so it sounds fuller than
# the dry render. That mastering step was dropped to keep the package free of a
# GPLv3 dependency.

# 4. Effort words: one Laban Effort word per phase (Weight/Time/Space rule)
python code/effort_words.py --input data/phase_log.csv --output data/effort_words.csv

# 5. the cover (audio/cover.mp3) is provided as a fixed artifact, not regenerated

# 6. align the render to the cover -> timeline/stage_timeline.json
python code/build_stage_timeline.py --wav audio/cover.mp3
```

## License

Copyright (c) 2026 the authors of Effort. All rights reserved.

The code and supplementary materials are provided under the
[Limited Review and Verification License](LICENSE.md). They may be inspected
and executed in unmodified form for peer review, private verification, and
private non-commercial scholarly study, but they may not be redistributed,
modified, commercially reused, incorporated into other works or datasets, or
used for machine-learning training without prior written permission.

FineDance is not included and remains subject to its own terms. Third-party
software and externally generated materials remain subject to their respective
licenses, service terms, and underlying rights.
