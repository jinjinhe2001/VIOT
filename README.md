# VIOT: A Variational Optimal Transport Operator on Incompressible Flow

**Jinjin He, Shenyifan Lu, Sinan Wang, Zhiqi Li, Duowen Chen, Bo Zhu** · Georgia Institute of Technology

[Paper (arXiv 2609.13729)](https://arxiv.org/abs/2609.13729) ·
[Project page](https://jinjinhe2001.github.io/viot/) ·
[Live demo](https://jinjinhe2001.github.io/viot/#demo) ·
[Checkpoints](https://huggingface.co/jinjinhe2001/VIOT)

![VIOT transports](https://jinjinhe2001.github.io/viot/static/images/results.webp)

VIOT is a generative neural operator for incompressible density transport. Given
a source and a target density, it predicts a stream function (2D) or a vector
potential (3D) at every step. Its curl is a divergence-free velocity, and
advecting the density along it produces the whole transport trajectory in one
feed-forward rollout, in seconds, with no per-pair optimization.

This repository contains the operator, training and evaluation code, the
dataset builders, the released checkpoints of the paper's models, two desktop
paint-chain applications, and a WebGPU version that runs in the browser.

## Installation

```bash
git clone https://github.com/jinjinhe2001/VIOT && cd VIOT
pip install -e .            # torch>=2.0, torchvision, numpy, safetensors, huggingface_hub
pip install -e ".[viz]"     # optional: matplotlib, Pillow, imageio (examples, GUIs)
```

## Quick start

```python
import torch
from viot import load_pretrained
from viot.data_2d import sample_mnist_pairs
from viot.ops_2d import rollout_2d

model, cfg = load_pretrained("mnist_2d", device="cuda")      # downloads from the Hub
rho_0, rho_1 = sample_mnist_pairs(1, 256, 256, device="cuda", train=False)
out = rollout_2d(model, rho_0, rho_1, n_steps=50, scheme=cfg["rollout"]["advection"],
                 return_frames=True)
frames, final = out["frames"], out["final"]                   # 51 densities, [1, 1, 256, 256] each
```

`load_pretrained(name, local_dir=...)` reads a local copy of the checkpoint
repository instead of downloading it.

`python examples/quickstart_2d.py` writes a GIF of a random MNIST test pair, and
`python examples/text_chain_2d.py --model cjk_2d --text 春江潮水连海平 --font <a CJK .ttf>`
chains a sequence of characters. `examples/quickstart_3d.py` does the same in 3D.

## Released models

All checkpoints are the `model_best` weights used by the paper's figures and
held-out evaluation. Each comes with a `config.json` holding the architecture,
rollout settings and exact training recipe.

| name | task | grid | FNO (width / modes / layers) | k_max | advection |
|---|---|---|---|---|---|
| `mnist_2d` | MNIST digits | 256² | 64 / 32 / 8 | 0.25 | MacCormack |
| `cjk_2d` | Chinese characters | 256² | 64 / 32 / 8 | 0.25 | MacCormack |
| `mpeg7_2d` | MPEG-7 silhouettes | 256² | 64 / 32 / 8 | 0.25 | MacCormack |
| `latin_font_2d` | Latin glyphs | 256² | 64 / 16 / 8 | 0.0625 | semi-Lagrangian |
| `sphere2airplane_3d` | sphere to ShapeNet airplanes | 128³ | 32 / 16 / 6 | 0.25 | semi-Lagrangian |
| `humman_3d` | HuMMan human poses | 128³ | 32 / 16 / 6 | 0.25 | semi-Lagrangian |
| `font_3d` | extruded glyphs | 128³ | 32 / 16 / 6 | 0.25 | semi-Lagrangian |
| `humman_3d_figures` | HuMMan (continuation used for the HuMMan figures) | 128³ | 32 / 16 / 6 | 0.25 | semi-Lagrangian |

The "advection" column is the scheme each model was trained with, and the
default of `rollout_2d` / `rollout_3d` for it (`cfg["rollout"]["advection"]`).

## Interactive applications

* **Browser:** [`web/`](web/) runs the `mnist_2d` operator with WebGPU on the
  visitor's GPU, with no server. It is the demo on the project page.
* **Desktop:** `python apps/paint_chain_gui.py` (2D, any 2D model) and
  `python apps/paint_chain_3d_gui.py` (3D glyphs, `font_3d`). Paint a source and
  a target and press Enter. *Continue* chains the result into a new target.

## Training

1. Build the datasets with the scripts in [`scripts/data/`](scripts/data/)
   (its README lists sources, licences and commands). MNIST is downloaded
   automatically.
2. Run the recipe of a model from [`scripts/train/`](scripts/train/), e.g.

```bash
bash scripts/train/mnist_2d.sh              # 1 GPU, ~13 h on an A100
bash scripts/train/sphere2airplane_3d.sh    # 2 GPUs (torch.distributed.run), ~16 h on 2 A100s
```

Each released model's `config.json` records the exact command it was trained
with. The 3D recipes load whole voxel datasets into host memory (34-42 GB).

## Evaluation

[`scripts/eval/`](scripts/eval/) contains the held-out evaluators
(`eval_2d_heldout.py`, `eval_3d_heldout.py`, `eval_mnist_split.py`) and the
momentum-residual diagnostic, with the command for every row of the paper's
held-out table. Re-running them with the released weights reproduces the paper
(terminal L2, 100 pairs, seed 42):

| setting | paper | this repository |
|---|---|---|
| MNIST 2D, test split | 0.00108 ± 0.00036 | 0.00108 ± 0.00036 |
| CJK 2D, 300 unseen characters | 0.00123 ± 0.00021 | 0.00123 ± 0.00021 |
| CJK 2D, 2 unseen typefaces | 0.00165 ± 0.00090 | 0.00165 ± 0.00090 |
| Latin 2D, 8 unseen fonts | 0.00185 ± 0.00047 | 0.00184 ± 0.00047 |
| MPEG-7 2D, 129 split-out shapes | 0.00102 ± 0.00043 | 0.00101 ± 0.00043 |

## Reproducibility notes

* **Checkpoint selection.** `model_best` is the step-checkpoint with the lowest
  training-batch terminal loss, which is what the arXiv version's figures and
  tables use.
* **Advection.** The 3D models and `latin_font_2d` were trained with
  first-order semi-Lagrangian advection. The other 2D models used MacCormack.
  Both schemes are in `viot.ops_2d` / `viot.ops_3d`.
* **Precision.** PyTorch runs convolutions in TF32 on recent NVIDIA GPUs by
  default. That changes a 50-step 2D rollout by about 3 % relative to strict
  fp32, without affecting the metrics.
* **Sketches and glyphs.** The area normalization of the training data can
  enlarge thin shapes (a 1, an I, a straight stroke) past the border. The 2D
  paint GUI, the web demo and `examples/text_chain_2d.py` therefore keep shapes
  within 75% of the frame, which leaves room for the transport, and draw thin
  shapes bolder until they have the training area
  (`viot.data_2d.fit_area_in_frame`, `thicken_to_area`; `--max-extent 0` turns
  this off). Training and evaluation use the uncapped normalization.
* **Tests.** `tests/test_parity_*.py` check that this package reproduces the
  original research code bit-for-bit (they need the original code, so they are
  skipped otherwise).

## Citation

```bibtex
@misc{he2026variational,
  title={A Variational Optimal Transport Operator on Incompressible Flow},
  author={Jinjin He and Shenyifan Lu and Sinan Wang and Zhiqi Li and Duowen Chen and Bo Zhu},
  year={2026},
  eprint={2609.13729},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.13729}
}
```

## License

The code is released under the MIT license (see `LICENSE`). The checkpoints are
released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/);
the terms of their training data apply in addition (see the model card).
