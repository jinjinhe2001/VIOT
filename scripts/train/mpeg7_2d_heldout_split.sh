#!/usr/bin/env bash
# Held-out protocol retrain for MPEG-7 (paper run exp90; the "MPEG-7 2D, retrain on 1,165" row of the
# held-out table). Exactly the mpeg7_2d recipe, trained on a 90/10 split of the 1294 silhouettes.
#
# Data   : data/mpeg7_256_datamean_clean.pt (see mpeg7_2d.sh). This script first writes
#          data/mpeg7_256_train_s0.pt (1165) and data/mpeg7_256_test_s0.pt (129) with
#          scripts/data/split_mpeg7.py (torch.randperm with generator seed 0, first 10% = test) and
#          checks the indices against scripts/data/splits/mpeg7_256_split_s0.json.
# GPUs   : 1 x 80 GB (batch 4).
# Time   : 32,037 s (8.9 h) for 80k steps on one H200.
# Output : runs/mpeg7_2d_heldout_split/spectral_bb_visc_model_best.pt (the held-out row uses this
#          file; last best at step 68000 in the paper run)
# Evaluate with scripts/eval/eval_2d_heldout.py (see scripts/eval/README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/data/split_mpeg7.py \
  --data data/mpeg7_256_datamean_clean.pt --out-dir data \
  --check scripts/data/splits/mpeg7_256_split_s0.json

python -u -m viot.train_2d \
  --resolution 256 --sampler pool --data-path data/mpeg7_256_train_s0.pt \
  --batch-size 4 --lr 1e-4 --n-steps 80000 \
  --n-rollout 10 --n-test 20 --n-infer-steps 50 \
  --k-max 0.25 \
  --arch fno --fno-width 64 --fno-modes 32 --fno-layers 8 \
  --bb-visc-only --lambda-ke 1.0 --lambda-mu 0.01 \
  --advection maccormack \
  --save-dir runs/mpeg7_2d_heldout_split
