#!/usr/bin/env bash
# Recipe of the released `mnist_2d` model (paper run exp50).
#
# Data   : torchvision MNIST (60k training split), downloaded to data/MNIST on first use.
#          Pairs are generated on the fly (upsample 28->256, random blur, PR/area normalisation);
#          no pre-built file is needed.
# GPUs   : 1 x 80 GB (the original run used ~70 GB on an A100-SXM4-80GB at batch size 4).
# Time   : 47,190 s (13.1 h) for 80k steps on one A100-80GB.
# Output : runs/mnist_2d/spectral_bb_visc_model_best.pt  (the released weights are the `best`
#          file: lowest training-batch terminal loss at a 1000-step save; step 64000 in the paper run)
#          runs/mnist_2d/spectral_bb_visc_model.pt       (final weights, step 80000)
#
# Training is not seeded (as in the paper run), so a rerun gives a statistically equivalent,
# not bit-identical, model. Select GPUs with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m viot.train_2d \
  --resolution 256 --sampler mnist --mnist-root data \
  --batch-size 4 --lr 1e-4 --n-steps 80000 \
  --n-rollout 10 --n-test 20 --n-infer-steps 50 \
  --k-max 0.25 \
  --arch fno --fno-width 64 --fno-modes 32 --fno-layers 8 \
  --bb-visc-only --lambda-ke 1.0 --lambda-mu 0.01 \
  --advection maccormack \
  --save-dir runs/mnist_2d
