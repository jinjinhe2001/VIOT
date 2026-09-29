#!/usr/bin/env bash
# Recipe of the optional `humman_3d_figures` model (paper run exp63): a continuation of the
# `humman_3d` best weights with a larger viscosity weight (lambda_mu 0.007 instead of 0.005).
# This model produced the paper's HuMMan appendix figure, the HuMMan teaser row and the HuMMan
# timing row; the HuMMan metrics and held-out numbers use `humman_3d`.
#
# Data   : data/humman_volresize_n14000_128.pt (as for humman_3d.sh, ~42 GB in host RAM).
# Needs  : runs/humman_3d/spectral_bb_visc_3d_model_best.pt from scripts/train/humman_3d.sh, or set
#          INIT=<path> to another .pt state_dict. To start from the released humman_3d weights:
#            python -c "import torch; from viot.pretrained import load_state_dict; \
#              torch.save(load_state_dict('<ckpt repo>/humman_3d/model.safetensors'), 'humman_3d.pt')"
#            INIT=humman_3d.pt bash scripts/train/humman_3d_figures.sh
# GPUs   : 1 x 80 GB (batch 6).
# Time   : 103,037 s (28.6 h) for 5000 steps on one A100-SXM4-80GB.
# Output : runs/humman_3d_figures/spectral_bb_visc_3d_model_best.pt (released weights; = step 3000)
#          runs/humman_3d_figures/spectral_bb_visc_3d_model.pt      (final weights, step 5000)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/humman_volresize_n14000_128.pt
INIT=${INIT:-runs/humman_3d/spectral_bb_visc_3d_model_best.pt}

python -u -m viot.train_3d \
  --resolution 128 --sampler shapenet \
  --data-path-src "$DATA" --data-path-tgt "$DATA" \
  --batch-size 6 --lr 5e-5 --n-steps 5000 \
  --n-rollout 10 --n-test 10 --n-infer-steps 50 \
  --k-max 0.25 --fno-width 32 --fno-modes 16 --fno-layers 6 \
  --bb-visc-only --lambda-ke 0 --lambda-mu 0.007 --lambda-mu-warmup 0 --lambda-terminal 10 \
  --amp --checkpoint --log-every 100 --no-scheduler --grad-clip 1.0 \
  --raw-sampler --peak-norm --advection semi_lagrangian \
  --init-from "$INIT" \
  --save-dir runs/humman_3d_figures
