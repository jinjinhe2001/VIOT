#!/usr/bin/env bash
# Recipe of the released `latin_font_2d` model (paper run exp31).
#
# Data   : data/font_letters_256.pt  [496, 1, 256, 256] = 62 glyphs (A-Z a-z 0-9) x 8 font faces,
#          face-major order, rendered with the glyph-pool pipeline (PR fraction 0.197).
#          KNOWN GAP: the build script and the 8 font faces of the paper file were not archived
#          (see scripts/data/README.md). A comparable pool can be rendered with
#          `scripts/data/make_glyph_pool.py --latin --fonts <8 font files>`, but it is not the paper data.
# GPUs   : 1 GPU (the paper run used one H200 141 GB at batch size 64; the smaller model,
#          16 modes / k_max 0.0625, needs far less memory per sample than the 32-mode models).
# Time   : 28,722 s (8.0 h) for 40k steps on one H200.
# Output : runs/latin_font_2d/spectral_bb_visc_model_best.pt (released weights; step 35000 in the paper run)
#          runs/latin_font_2d/spectral_bb_visc_model.pt      (final weights, step 40000)
#
# This model was trained (and its end-of-training metric computed) with FIRST-ORDER semi-Lagrangian
# advection, hence --advection semi_lagrangian. Note k_max 0.0625, 16 modes, batch 64, 40k steps,
# lambda_mu 0.05 (all different from the other 2D recipes). Training is not seeded.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m viot.train_2d \
  --resolution 256 --sampler pool --data-path data/font_letters_256.pt \
  --batch-size 64 --lr 1e-4 --n-steps 40000 \
  --n-rollout 10 --n-test 20 --n-infer-steps 50 \
  --k-max 0.0625 \
  --arch fno --fno-width 64 --fno-modes 16 --fno-layers 8 \
  --bb-visc-only --lambda-ke 1.0 --lambda-mu 0.05 \
  --advection semi_lagrangian \
  --save-dir runs/latin_font_2d
