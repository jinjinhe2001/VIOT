#!/usr/bin/env bash
# Recipe of the released `sphere2airplane_3d` model (paper run exp49).
#
# Data   : data/sphere_match_airplanes_s02_128.pt            [8, 1, 128, 128, 128] source: one soft
#            sphere, 8 identical copies (scripts/data/prepare_sphere_volume_match_128.py)
#          data/shapenet_airplanes_wtt_volresize_n8000_128.pt [4045, 1, 128, 128, 128] targets
#            (scripts/data/voxelize_airplanes_wtt_volresize.py)
#          The target pool is ~34 GB and is loaded into host RAM once per rank (~70 GB for 2 ranks).
# GPUs   : 2 x 80 GB (torchrun, 6 pairs per rank = global batch 12).
# Time   : 58,903 s (16.4 h) for 5000 steps on two A100-SXM4-80GB.
# Output : runs/sphere2airplane_3d/spectral_bb_visc_3d_model_best.pt (released weights; = step 4000)
#          runs/sphere2airplane_3d/spectral_bb_visc_3d_model.pt      (final weights, step 5000)
#
# Advection is first-order semi-Lagrangian (the trainer default); the learning rate is constant
# (--no-scheduler). Select GPUs with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m viot.train_3d \
  --resolution 128 --sampler shapenet \
  --data-path-src data/sphere_match_airplanes_s02_128.pt \
  --data-path-tgt data/shapenet_airplanes_wtt_volresize_n8000_128.pt \
  --batch-size 6 --lr 1e-4 --n-steps 5000 \
  --n-rollout 10 --n-test 10 --n-infer-steps 50 \
  --k-max 0.25 --fno-width 32 --fno-modes 16 --fno-layers 6 \
  --bb-visc-only --lambda-ke 0 --lambda-mu 0.003 --lambda-mu-warmup 1000 \
  --lambda-terminal 10 \
  --amp --checkpoint --log-every 100 --no-scheduler \
  --grad-clip 1.0 --raw-sampler --peak-norm --init-scale 10.0 \
  --advection semi_lagrangian \
  --save-dir runs/sphere2airplane_3d
