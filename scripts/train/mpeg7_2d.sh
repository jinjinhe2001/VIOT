#!/usr/bin/env bash
# Recipe of the released `mpeg7_2d` model (paper run exp70).
#
# Data   : data/mpeg7_256_datamean_clean.pt  [1294, 1, 256, 256], built from the MPEG-7 CE-Shape-1
#          silhouettes with scripts/data/prepare_mpeg7.py (see scripts/data/README.md).
# GPUs   : 1 x 80 GB (~70 GB used at batch size 4 on an A100-SXM4-80GB).
# Time   : 68,232 s (19.0 h) for 80k steps on one A100-80GB.
# Output : runs/mpeg7_2d/spectral_bb_visc_model_best.pt (released weights; step 79000 in the paper run)
#          runs/mpeg7_2d/spectral_bb_visc_model.pt      (final weights, step 80000)
#
# The batch size is 4 (an internal note that says 12 is wrong). Training is not seeded (as in the
# paper run). Select GPUs with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m viot.train_2d \
  --resolution 256 --sampler pool --data-path data/mpeg7_256_datamean_clean.pt \
  --batch-size 4 --lr 1e-4 --n-steps 80000 \
  --n-rollout 10 --n-test 20 --n-infer-steps 50 \
  --k-max 0.25 \
  --arch fno --fno-width 64 --fno-modes 32 --fno-layers 8 \
  --bb-visc-only --lambda-ke 1.0 --lambda-mu 0.01 \
  --advection maccormack \
  --save-dir runs/mpeg7_2d
