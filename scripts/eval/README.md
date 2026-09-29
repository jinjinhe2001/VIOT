# Evaluation

Evaluators for the paper's held-out ("Generalization") table and the momentum-residual
diagnostic. They reproduce the original evaluation code exactly (same pair sampling, rollout and
metrics; checked per pair against the original scripts).

| script | what it does |
|---|---|
| `eval_2d_heldout.py` | 2D operator on a pool `.pt`: 100 pairs drawn with `torch.Generator().manual_seed(42)`, unit mass, 50-step MacCormack rollout with clamp + mass renormalisation |
| `eval_mnist_split.py` | 2D MNIST operator on the MNIST train or test split through the training sampler (`torch.manual_seed(42)`, batches of 10) |
| `eval_3d_heldout.py` | 3D operator on a source / target pool: training sampler (raw + peak normalisation + domain check), `torch.manual_seed(42)`, batches of 4, 50-step first-order semi-Lagrangian rollout with clamp + mass renormalisation |
| `eval_momentum_residual.py` | Leray-projected momentum residual r = \|\|P m\|\| / \|\|m\|\| of 2D rollouts (20 pairs, 50 and 100 steps) |

Each script prints mean / std / median of the per-pair terminal L2 (\|\|rho_T - rho_1\|\|_2), relative
L2, final mass %, mean \|div v\|, mean enstrophy and mean kinetic energy, and writes all per-pair
values (and, in 2D, the sampled pool indices) to the `--out` JSON.

**Model selection.** `--model <name>` loads a released model (`viot.pretrained.load_pretrained`:
from the Hugging Face Hub, or from a local copy of the checkpoint repository given with
`--checkpoint-dir` or `$VIOT_CHECKPOINT_DIR`). `--ckpt <file.pt|.safetensors>` loads any state_dict,
with the architecture from `--resolution --k-max --fno-width --fno-modes --fno-layers` (defaults:
the 2D 256^2 / 64 / 32 / 8 / k_max 0.25 models, or the 3D 128^3 / 32 / 16 / 6 / k_max 0.25 models;
`latin_font_2d` needs `--k-max 0.0625 --fno-modes 16`). This is how the held-out retrains from
`scripts/train/*_heldout_split.sh` are evaluated.

**Advection.** The paper evaluated every 2D model with MacCormack advection, including
`latin_font_2d`, which was trained with first-order semi-Lagrangian advection (`--advection` changes
it). The 3D models are evaluated with first-order semi-Lagrangian advection, as trained.

**Data.** Build the pools with `scripts/data/` (see its README). The seeded pair draws index pool
rows, so the pools must have the paper's row order: the 2D glyph pools and the HuMMan pools are
deterministic, MPEG-7 needs `--keep-list`, the 3D glyph pools need `--match-ids`, and the airplane
split files follow the shipped id lists.

## Held-out table

All rows: 100 pairs, seed 42, 50 inference steps; terminal L2 mean +- sample std, mean relative L2 in
parentheses. "Re-run" is this code with the released weights on an RTX 5080 (torch 2.11) with pools
rebuilt by `scripts/data/`; blank = not re-run (data not available locally or retrain needed).

| row (paper) | command | paper | re-run |
|---|---|---|---|
| MNIST 2D, in-pool (train split) | `python scripts/eval/eval_mnist_split.py --model mnist_2d --split train --out runs/eval/mnist_2d_train.json` | 0.00112 +- 0.00038 (0.127) | 0.00111 +- 0.00038 (0.127) |
| MNIST 2D, official test split | `python scripts/eval/eval_mnist_split.py --model mnist_2d --split test --out runs/eval/mnist_2d_test.json` | 0.00108 +- 0.00036 (0.123) | 0.00108 +- 0.00036 (0.122) |
| CJK 2D, in-pool | `python scripts/eval/eval_2d_heldout.py --model cjk_2d --data-path data/chinese_chars_256.pt --out runs/eval/cjk_2d_inpool.json` | 0.00152 +- 0.00045 (0.173) | |
| CJK 2D, 300 unseen characters | `python scripts/eval/eval_2d_heldout.py --model cjk_2d --data-path data/cjk_heldout_chars_256.pt --out runs/eval/cjk_2d_heldout_chars.json` | 0.00123 +- 0.00021 (0.140) | 0.00123 +- 0.00021 (0.140) |
| CJK 2D, 2 unseen typefaces | `python scripts/eval/eval_2d_heldout.py --model cjk_2d --data-path data/cjk_heldout_fonts_256.pt --out runs/eval/cjk_2d_heldout_fonts.json` | 0.00165 +- 0.00090 (0.187) | 0.00165 +- 0.00090 (0.187) |
| Font 2D, in-pool | `python scripts/eval/eval_2d_heldout.py --model latin_font_2d --data-path data/font_letters_256.pt --out runs/eval/latin_font_2d_inpool.json` | 0.00197 +- 0.00068 (0.224) | (pool not reproducible, see `scripts/data/README.md`) |
| Font 2D, 8 unseen font families | `python scripts/eval/eval_2d_heldout.py --model latin_font_2d --data-path data/font_heldout_fonts_256.pt --out runs/eval/latin_font_2d_heldout_fonts.json` | 0.00185 +- 0.00047 (0.210) | 0.00184 +- 0.00047 (0.210) |
| MPEG-7 2D, full-pool operator on the 129 split-out shapes (seen) | `python scripts/eval/eval_2d_heldout.py --model mpeg7_2d --data-path data/mpeg7_256_test_s0.pt --out runs/eval/mpeg7_2d_test129.json` | 0.00102 +- 0.00043 (0.129) | 0.00101 +- 0.00043 (0.127) |
| MPEG-7 2D, retrain on 1165, in-pool | `python scripts/eval/eval_2d_heldout.py --ckpt runs/mpeg7_2d_heldout_split/spectral_bb_visc_model_best.pt --data-path data/mpeg7_256_train_s0.pt --out runs/eval/mpeg7_split_train1165.json` | 0.00102 +- 0.00046 (0.128) | |
| MPEG-7 2D, retrain on 1165, 129 split-out shapes | `python scripts/eval/eval_2d_heldout.py --ckpt runs/mpeg7_2d_heldout_split/spectral_bb_visc_model_best.pt --data-path data/mpeg7_256_test_s0.pt --out runs/eval/mpeg7_split_test129.json` | 0.00110 +- 0.00049 (0.139) | |
| Sphere->airplane 3D, full-pool operator on the 808 PC15k test airplanes (seen) | `python scripts/eval/eval_3d_heldout.py --model sphere2airplane_3d --src-path data/sphere_match_airplanes_wtt_s02_128.pt --tgt-path data/airplanes_wtt_test_128.pt --out runs/eval/sphere2airplane_3d_test808.json` | 18.71 +- 3.21 (0.239) | |
| Sphere->airplane 3D, retrain on 3237, in-pool | `python scripts/eval/eval_3d_heldout.py --ckpt runs/sphere2airplane_3d_heldout_split/spectral_bb_visc_3d_model_best.pt --src-path data/sphere_match_airplanes_wtt_s02_128.pt --tgt-path data/airplanes_wtt_train_128.pt --out runs/eval/sphere2airplane_split_train3237.json` | 18.47 +- 5.00 (0.241) | |
| Sphere->airplane 3D, retrain on 3237, 808 test airplanes | `python scripts/eval/eval_3d_heldout.py --ckpt runs/sphere2airplane_3d_heldout_split/spectral_bb_visc_3d_model_best.pt --src-path data/sphere_match_airplanes_wtt_s02_128.pt --tgt-path data/airplanes_wtt_test_128.pt --out runs/eval/sphere2airplane_split_test808.json` | 18.66 +- 3.41 (0.238) | |
| HuMMan 3D, in-pool | `python scripts/eval/eval_3d_heldout.py --model humman_3d --src-path data/humman_volresize_n14000_128.pt --tgt-path data/humman_volresize_n14000_128.pt --out runs/eval/humman_3d_inpool.json` | 29.19 +- 4.58 (0.269) | |
| HuMMan 3D, 1181 midpoint poses (interpolation) | `python scripts/eval/eval_3d_heldout.py --model humman_3d --src-path data/humman_interp_heldout_n14000_128.pt --tgt-path data/humman_interp_heldout_n14000_128.pt --out runs/eval/humman_3d_interp.json` | 26.64 +- 3.48 (0.245) | |
| Font 3D, in-pool | `python scripts/eval/eval_3d_heldout.py --model font_3d --src-path data/font_3d_volresize_n8000_128.pt --tgt-path data/font_3d_volresize_n8000_128.pt --out runs/eval/font_3d_inpool.json` | 25.76 +- 3.32 (0.305) | |
| Font 3D, 8 unseen faces | `python scripts/eval/eval_3d_heldout.py --model font_3d --src-path data/font3d_heldout_fonts_n8000_128.pt --tgt-path data/font3d_heldout_fonts_n8000_128.pt --out runs/eval/font_3d_heldout_fonts.json` | 25.89 +- 3.52 (0.307) | |

Notes:
* The paper's sphere->airplane rows (full-pool operator and retrain) all use the sphere matched
  to the PC15k training split, `sphere_match_airplanes_wtt_s02_128.pt`, as the source.
* The retrain rows need the models of `scripts/train/mpeg7_2d_heldout_split.sh` and
  `scripts/train/sphere2airplane_3d_heldout_split.sh`; these two retrains are not among the released
  checkpoints.
* The 3D evaluator loads whole pools into host RAM (HuMMan 42 GB, airplane test split 6.8 GB).
  Lowering `--batch` reduces GPU memory but draws different pairs than the paper protocol.
* Terminal L2 reproduces to about 1% across GPUs and torch versions (PyTorch uses TF32 convolutions
  on recent NVIDIA GPUs; see the main README). The finite-difference `mean_abs_div` column is much
  more sensitive to the backend (it differs by up to ~2x between CPU and GPU runs of the same
  model and pairs) and is not a paper-table quantity.

## Momentum residual

`eval_momentum_residual.py` computes r = \|\|P m\|\| / \|\|m\|\|, the fraction of the Navier-Stokes
momentum imbalance m = dv/dt + (v.grad)v - mu lap v (mu = 0) that a pressure gradient cannot absorb
(P is the Leray projector), averaged over 20 pairs (seed 42), for 50- and 100-step rollouts. The
values of the original runs:

| model | command | r (50 steps) | r (100 steps) | spectral mean \|div v\| |
|---|---|---|---|---|
| `mnist_2d` | `python scripts/eval/eval_momentum_residual.py --model mnist_2d --data-path data/mnist_test_pool_256.pt --out runs/eval/mom_mnist_2d.json` | 0.819 | 0.819 | 6.3e-6 |
| `cjk_2d` | `python scripts/eval/eval_momentum_residual.py --model cjk_2d --data-path data/cjk_heldout_chars_256.pt --out runs/eval/mom_cjk_2d.json` | 0.923 | 0.924 | 2.4e-6 |
| `latin_font_2d` | `python scripts/eval/eval_momentum_residual.py --model latin_font_2d --data-path data/font_letters_256.pt --out runs/eval/mom_latin_font_2d.json` | 0.824 | 0.825 | 6.7e-6 |
| `mpeg7_2d` | `python scripts/eval/eval_momentum_residual.py --model mpeg7_2d --data-path data/mpeg7_256_datamean_clean.pt --out runs/eval/mom_mpeg7_2d.json` | 0.863 | 0.865 | 7.0e-6 |

Re-running the `mpeg7_2d` row with this code on the rebuilt pool gives 0.864 / 0.870 (6.9e-6).

The arXiv momentum / numerical-transport table (columns delta_M, delta_H, F_RMS, q_50, q_100 and
the spectral divergence, 100 pairs per setting, including the 3D rows) was produced by a later
diagnostic suite that is not part of this release; `eval_momentum_residual.py` is the r diagnostic
that preceded it.
