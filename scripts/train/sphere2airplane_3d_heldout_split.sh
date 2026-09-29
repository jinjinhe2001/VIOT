#!/usr/bin/env bash
# Held-out protocol retrain for sphere -> airplane (paper run exp91; the "Sphere->airplane 3D, retrain
# on 3,237" row of the held-out table). Exactly the sphere2airplane_3d recipe, trained on the official
# ShapeNetCore.v2.PC15k split (train + val = 3237 airplanes; the 808 test airplanes are held out).
#
# Data   : data/shapenet_airplanes_wtt_volresize_n8000_128.pt (+ its _ids.txt; see
#          sphere2airplane_3d.sh). This script first runs scripts/data/split_airplanes_wtt.py with the
#          shipped PC15k id lists (scripts/data/splits/airplanes_wtt_{train,test}_128_ids.txt). It writes
#          data/airplanes_wtt_{train,test}_128.pt and a sphere matched to the train-set mean mass,
#          data/sphere_match_airplanes_wtt_s02_128.pt (softness 0.2, 8 copies).
#          Needs ~34 GB of host RAM for the split and ~27 GB per rank for training.
# GPUs   : 2 x 80 GB (global batch 12).
# Time   : 39,603 s (11.0 h) for 5000 steps on two H200.
# Output : runs/sphere2airplane_3d_heldout_split/spectral_bb_visc_3d_model_best.pt (the held-out row
#          uses this file; = step 4000 in the paper run)
# Evaluate with scripts/eval/eval_3d_heldout.py (see scripts/eval/README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/data/split_airplanes_wtt.py \
  --data data/shapenet_airplanes_wtt_volresize_n8000_128.pt --out-dir data \
  --train-ids scripts/data/splits/airplanes_wtt_train_128_ids.txt \
  --test-ids scripts/data/splits/airplanes_wtt_test_128_ids.txt

python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m viot.train_3d \
  --resolution 128 --sampler shapenet \
  --data-path-src data/sphere_match_airplanes_wtt_s02_128.pt \
  --data-path-tgt data/airplanes_wtt_train_128.pt \
  --batch-size 6 --lr 1e-4 --n-steps 5000 \
  --n-rollout 10 --n-test 10 --n-infer-steps 50 \
  --k-max 0.25 --fno-width 32 --fno-modes 16 --fno-layers 6 \
  --bb-visc-only --lambda-ke 0 --lambda-mu 0.003 --lambda-mu-warmup 1000 \
  --lambda-terminal 10 \
  --amp --checkpoint --log-every 100 --no-scheduler \
  --grad-clip 1.0 --raw-sampler --peak-norm --init-scale 10.0 \
  --advection semi_lagrangian \
  --save-dir runs/sphere2airplane_3d_heldout_split
