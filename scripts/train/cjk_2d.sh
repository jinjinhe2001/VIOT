#!/usr/bin/env bash
# Recipe of the released `cjk_2d` model (paper run exp51).
#
# Data   : data/chinese_chars_256.pt  [9000, 1, 256, 256] = the first 3000 CJK code points
#          (U+4E00..U+59B7) x 3 fonts (SimHei, Microsoft YaHei, SimSun). Build it with
#          scripts/data/generate_chinese_chars.py (see scripts/data/README.md; the fonts are not
#          redistributable and must be supplied by the user).
# GPUs   : 1 x 80 GB (~70 GB used at batch size 4 on an A100-SXM4-80GB).
# Time   : 49,855 s (13.8 h) for 80k steps on one A100-80GB.
# Output : runs/cjk_2d/spectral_bb_visc_model_best.pt (released weights; step 71000 in the paper run)
#          runs/cjk_2d/spectral_bb_visc_model.pt      (final weights, step 80000)
#
# Training is not seeded (as in the paper run). Select GPUs with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m viot.train_2d \
  --resolution 256 --sampler pool --data-path data/chinese_chars_256.pt \
  --batch-size 4 --lr 1e-4 --n-steps 80000 \
  --n-rollout 10 --n-test 20 --n-infer-steps 50 \
  --k-max 0.25 \
  --arch fno --fno-width 64 --fno-modes 32 --fno-layers 8 \
  --bb-visc-only --lambda-ke 1.0 --lambda-mu 0.01 \
  --advection maccormack \
  --save-dir runs/cjk_2d
