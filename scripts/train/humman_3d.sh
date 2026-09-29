#!/usr/bin/env bash
# Recipe of the released `humman_3d` model: paper run exp55 (fresh, 2 GPUs) followed by exp59
# (continuation from the exp55 best weights, 1 GPU). The released weights are the stage-2 best file.
#
# Data   : data/humman_volresize_n14000_128.pt  [5000, 1, 128, 128, 128], source = target pool
#          ("n14000" is the target mass per pose, not a count). Build it with
#          scripts/data/voxelize_humman_volresize.py from humman_sub_all179_5k.npz
#          (see scripts/data/README.md: the raw-HuMMan -> npz step is not included).
#          The pool is ~42 GB and is loaded into host RAM once per rank.
# GPUs   : stage 1: 2 x 80 GB (global batch 12); stage 2: 1 x 80 GB (batch 6).
# Time   : stage 1: 121,299 s (33.7 h) for 5000 steps on two A100-SXM4-80GB;
#          stage 2: 111,079 s (30.9 h) for 5000 steps on one A100-SXM4-80GB.
# Output : runs/humman_3d_stage1/  (exp55; best == final == step 5000 in the paper run)
#          runs/humman_3d/spectral_bb_visc_3d_model_best.pt (released weights; = stage-2 step 3000)
#          runs/humman_3d/spectral_bb_visc_3d_model.pt      (stage-2 final weights, step 5000)
#
# Set STAGE=2 to skip stage 1 (continue from an existing runs/humman_3d_stage1).
# Advection: first-order semi-Lagrangian; constant learning rate. Select GPUs with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/humman_volresize_n14000_128.pt
STAGE1=runs/humman_3d_stage1
STAGE2=runs/humman_3d
COMMON=(--resolution 128 --sampler shapenet
        --data-path-src "$DATA" --data-path-tgt "$DATA"
        --batch-size 6 --n-rollout 10 --n-test 10 --n-infer-steps 50
        --k-max 0.25 --fno-width 32 --fno-modes 16 --fno-layers 6
        --bb-visc-only --lambda-ke 0 --lambda-terminal 10
        --amp --checkpoint --log-every 100 --no-scheduler
        --grad-clip 1.0 --raw-sampler --peak-norm --advection semi_lagrangian)

if [ "${STAGE:-1}" -le 1 ]; then
  # Stage 1 (exp55): fresh, 2 GPUs, lr 1e-4, lambda_mu 0.005 with a 1000-step warm-up.
  python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m viot.train_3d "${COMMON[@]}" \
    --lr 1e-4 --n-steps 5000 \
    --lambda-mu 0.005 --lambda-mu-warmup 1000 --init-scale 10.0 \
    --save-dir "$STAGE1"
fi

# Stage 2 (exp59): continue from the stage-1 best weights, 1 GPU, lr 5e-5, no warm-up.
python -u -m viot.train_3d "${COMMON[@]}" \
  --lr 5e-5 --n-steps 5000 \
  --lambda-mu 0.005 --lambda-mu-warmup 0 \
  --init-from "$STAGE1/spectral_bb_visc_3d_model_best.pt" \
  --save-dir "$STAGE2"
