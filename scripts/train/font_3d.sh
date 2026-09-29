#!/usr/bin/env bash
# Recipe of the released `font_3d` model: three stages, paper runs exp52 -> exp53 -> exp68.
# The released weights are the stage-3 best file.
#
# Data   : data/font_3d_volresize_n8000_128.pt  [1364, 1, 128, 128, 128] = 62 glyphs (0-9 A-Z a-z)
#          x 22 DejaVu faces, extruded and voxelized with scripts/data/voxelize_font_3d.py
#          (source = target pool, ~11.4 GB in host RAM per rank).
# GPUs   : stage 1: 2 x 80 GB (global batch 12); stages 2 and 3: 1 x 80 GB (batch 6).
# Time   : stage 1: ~44,400 s (12.3 h) to step 3000 on two A100-SXM4-80GB;
#          stage 2: 70,823 s (19.7 h) for 3000 steps on one A100-SXM4-80GB;
#          stage 3: 34,113 s (9.5 h) for 1500 steps on one A100-SXM4-80GB.
# Output : runs/font_3d_stage1/, runs/font_3d_stage2/,
#          runs/font_3d/spectral_bb_visc_3d_model_best.pt (released weights; = stage-3 step 1000)
#          runs/font_3d/spectral_bb_visc_3d_model.pt      (stage-3 final weights, step 1500)
#
# Lineage (from the original launch scripts and logs):
#   * Stage 1 (exp52) is a fresh run with lambda_ke 0.1. It was launched with --n-steps 5000 but
#     stopped at step ~3400; its best file, which stage 2 loaded, is the step-3000 checkpoint.
#     With a constant learning rate the first 3000 steps do not depend on --n-steps, so the default
#     here is 3000 steps (STAGE1_STEPS=5000 runs the full original launch command instead, but then
#     the best file may come from a later step than in the paper lineage).
#   * Stage 2 (exp53) continues from the stage-1 best with lambda_ke 0 (best == final == step 3000).
#   * Stage 3 (exp68) continues from the stage-2 best with lambda_mu 0.0025 and lr 5e-5.
#   The older exp48 font model is not an ancestor of this model.
# Set STAGE=2 or STAGE=3 to start from a later stage. Advection: first-order semi-Lagrangian;
# constant learning rate. Select GPUs with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/font_3d_volresize_n8000_128.pt
STAGE1=runs/font_3d_stage1
STAGE2=runs/font_3d_stage2
STAGE3=runs/font_3d
COMMON=(--resolution 128 --sampler shapenet
        --data-path-src "$DATA" --data-path-tgt "$DATA"
        --batch-size 6 --n-rollout 10 --n-test 10 --n-infer-steps 50
        --k-max 0.25 --fno-width 32 --fno-modes 16 --fno-layers 6
        --bb-visc-only --lambda-terminal 10
        --amp --checkpoint --log-every 100 --no-scheduler
        --grad-clip 1.0 --raw-sampler --peak-norm --advection semi_lagrangian)

if [ "${STAGE:-1}" -le 1 ]; then
  # Stage 1 (exp52): fresh, 2 GPUs, lr 1e-4, lambda_ke 0.1, lambda_mu 0.003 with a 1000-step warm-up.
  python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m viot.train_3d "${COMMON[@]}" \
    --lr 1e-4 --n-steps "${STAGE1_STEPS:-3000}" \
    --lambda-ke 0.1 --lambda-mu 0.003 --lambda-mu-warmup 1000 --init-scale 10.0 \
    --save-dir "$STAGE1"
fi

if [ "${STAGE:-1}" -le 2 ]; then
  # Stage 2 (exp53): from the stage-1 best, 1 GPU, lr 1e-4, lambda_ke 0, lambda_mu 0.003, no warm-up.
  python -u -m viot.train_3d "${COMMON[@]}" \
    --lr 1e-4 --n-steps 3000 \
    --lambda-ke 0 --lambda-mu 0.003 --lambda-mu-warmup 0 \
    --init-from "$STAGE1/spectral_bb_visc_3d_model_best.pt" \
    --save-dir "$STAGE2"
fi

# Stage 3 (exp68): from the stage-2 best, 1 GPU, lr 5e-5, lambda_ke 0, lambda_mu 0.0025, no warm-up.
python -u -m viot.train_3d "${COMMON[@]}" \
  --lr 5e-5 --n-steps 1500 \
  --lambda-ke 0 --lambda-mu 0.0025 --lambda-mu-warmup 0 \
  --init-from "$STAGE2/spectral_bb_visc_3d_model_best.pt" \
  --save-dir "$STAGE3"
