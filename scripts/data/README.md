# Datasets

Scripts that rebuild every training and evaluation pool used for the released models.
Run them from the repository root (after `pip install -e .`; without it the scripts put the
repository on `sys.path` themselves). They write to `data/`, which is git-ignored. Every script
has `--help`.

Only code and small index files are shipped. The data tensors are large (up to 42 GB) and several
sources do not allow redistribution (ShapeNet, HuMMan, the Windows fonts).

| pool (`data/...`) | used by | shape | size | builder |
|---|---|---|---|---|
| MNIST (torchvision) | `mnist_2d` | 60k train images, pairs made on the fly | ~60 MB | automatic download |
| `chinese_chars_256.pt` | `cjk_2d` | [9000, 1, 256, 256] | 2.4 GB | `generate_chinese_chars.py` |
| `font_letters_256.pt` | `latin_font_2d` | [496, 1, 256, 256] | 130 MB | **not available** (see gaps) |
| `mpeg7_256_datamean_clean.pt` | `mpeg7_2d` | [1294, 1, 256, 256] | 339 MB | `prepare_mpeg7.py` |
| `sphere_match_airplanes_s02_128.pt` | `sphere2airplane_3d` (source) | [8, 1, 128, 128, 128] | 67 MB | `prepare_sphere_volume_match_128.py` |
| `shapenet_airplanes_wtt_volresize_n8000_128.pt` | `sphere2airplane_3d` (targets) | [4045, 1, 128, 128, 128] | 34 GB | `voxelize_airplanes_wtt_volresize.py` |
| `humman_volresize_n14000_128.pt` | `humman_3d`, `humman_3d_figures` | [5000, 1, 128, 128, 128] | 42 GB | `voxelize_humman_volresize.py` |
| `font_3d_volresize_n8000_128.pt` | `font_3d` | [1364, 1, 128, 128, 128] | 11.4 GB | `voxelize_font_3d.py` |

All 2D pools are float32 with unit mass per sample. All 3D pools are float32 with values in
[0, 1] (peak ~1); the trainer and the evaluators peak-normalise every sample.

Held-out / evaluation pools (paper table "Generalization", see `scripts/eval/README.md`):

| pool | shape | builder |
|---|---|---|
| MNIST official test split | 10k images | automatic (`scripts/eval/eval_mnist_split.py`) |
| `mnist_test_pool_256.pt` (momentum diagnostic) | [300, 1, 256, 256] | `dump_mnist_pool.py` |
| `cjk_heldout_chars_256.pt` | [900, 1, 256, 256] | `make_glyph_pool.py` |
| `cjk_heldout_fonts_256.pt` | [600, 1, 256, 256] | `make_glyph_pool.py` |
| `font_heldout_fonts_256.pt` | [496, 1, 256, 256] | `make_glyph_pool.py` |
| `mpeg7_256_{train,test}_s0.pt` | [1165 / 129, 1, 256, 256] | `split_mpeg7.py` |
| `airplanes_wtt_{train,test}_128.pt`, `sphere_match_airplanes_wtt_s02_128.pt` | [3237 / 808 / 8, 1, 128^3] | `split_airplanes_wtt.py` |
| `humman_interp_heldout_n14000_128.pt` | [1181, 1, 128^3] | `make_humman_heldout_interp.py` + `voxelize_humman_volresize.py` |
| `font3d_heldout_fonts_n8000_128.pt` | [496, 1, 128^3] | `voxelize_font_3d.py` |

Dependencies: the 2D builders need torch, torchvision, numpy and Pillow. The 3D mesh voxelizers
also need `trimesh` and `scipy` (`pip install -e ".[data3d]"`). `pool_stats.py` prints shape, mass,
peak and participation-ratio statistics of any pool, which is a quick check against the tables here.

## `splits/`: index files of the paper data

| file | content |
|---|---|
| `mpeg7_256_datamean_clean_files.txt` | the 1294 MPEG-7 GIF names of `mpeg7_256_datamean_clean.pt`, in row order |
| `mpeg7_256_split_s0.json` | the 1165 / 129 row indices of the MPEG-7 held-out split (seed 0) |
| `cjk_heldout_chars_256_meta.txt`, `cjk_heldout_fonts_256_meta.txt`, `font_heldout_fonts_256_meta.txt` | `font<TAB>char` of every row of the 2D held-out glyph pools |
| `shapenet_airplanes_wtt_volresize_n8000_128_ids.txt` | ShapeNet model id of every row of the airplane pool (paper row order) |
| `shapenet_airplanes_wtt_volresize_n8000_128_clipped.txt` | the 79 airplanes whose mass is below 8000 because they hit the grid margin |
| `airplanes_wtt_train_128_ids.txt`, `airplanes_wtt_test_128_ids.txt`, `airplanes_wtt_split_meta.json` | the PC15k split of the airplane pool: 3237 (2832 train + 405 val) / 808 test |
| `humman_sub_all179_5k_index.tsv` | HuMMan subject, action and frame of each of the 5000 poses (row = npz row = pool row) |
| `humman_volresize_n14000_128_ids.txt`, `humman_interp_heldout_n14000_128_ids.txt` | npz row of every pool row (0 ... N-1) |
| `font_3d_volresize_n8000_128_ids.txt`, `font3d_heldout_fonts_n8000_128_ids.txt` | `<font file stem>_<char>` of every row of the 3D glyph pools |

The mesh and glyph voxelizers run in a process pool and emit rows in completion order. Pass
`--match-ids splits/<pool>_ids.txt` to get the paper row order (this matters for anything that
indexes rows, e.g. the evaluators' seeded pair draws).

---

## MNIST (`mnist_2d`)

* **Source.** torchvision `MNIST` (Y. LeCun, C. Cortes, C. Burges), downloaded to `data/MNIST`
  on first use by the trainer (`--mnist-root data`) and the evaluators. Commonly distributed under
  CC BY-SA 3.0.
* **Training data.** Pairs are generated on the fly from all 60k training images
  (`viot.data_2d.sample_mnist_pairs`): 28 -> 256 bilinear, Gaussian blur sigma ~ U[0.5, 2.0],
  1e-4 floor, mass normalisation, participation-ratio normalisation to 0.197 x 256^2, mass
  normalisation. There is no file to build.
* **Held-out.** The official 10k test split (`scripts/eval/eval_mnist_split.py --split test`).
  The momentum diagnostic uses a fixed pool from the test split:
  ```bash
  python scripts/data/dump_mnist_pool.py --split test --n 300 --seed 0 --out data/mnist_test_pool_256.pt
  ```

## CJK characters (`cjk_2d`)

* **Source.** Rendered from fonts: the first 3000 code points of the CJK Unified Ideographs block,
  U+4E00 ... U+59B7 (a code-point range, not a frequency list), in SimHei (`simhei.ttf`),
  Microsoft YaHei (`msyh.ttc`, first face) and SimSun (`simsun.ttc`, first face).
* **Licence.** These are commercial fonts bundled with Windows. They are not redistributable, and
  neither are the rendered pools; supply your own licensed copies. Open fonts such as Noto Sans /
  Serif CJK (SIL OFL) work with the same scripts but give a different pool than the paper's.
* **Build** (`[9000, 1, 256, 256]`, font-major order):
  ```bash
  python scripts/data/generate_chinese_chars.py --resolution 256 --n-chars 3000 \
      --font-dir <dir with the fonts> --fonts simhei.ttf msyh.ttc simsun.ttc \
      --output data/chinese_chars_256.pt
  ```
  Pipeline: glyph centred at 80% of the canvas, Gaussian blur sigma 8 (kernel 49), 1e-5 floor,
  mass normalisation, participation ratio 0.197 x H x W, mass normalisation.
* **Held-out pools** (same pipeline):
  ```bash
  # 300 unseen characters U+59B8 ... U+5AE3, the three training fonts -> [900, 1, 256, 256]
  python scripts/data/make_glyph_pool.py --cjk-start 3000 --cjk-n 300 \
      --font-dir <fonts> --fonts simhei.ttf msyh.ttc simsun.ttc --output data/cjk_heldout_chars_256.pt
  # the first 300 training characters in two unseen typefaces, KaiTi and FangSong -> [600, 1, 256, 256]
  python scripts/data/make_glyph_pool.py --cjk-start 0 --cjk-n 300 \
      --font-dir <fonts> --fonts simkai.ttf simfang.ttf --output data/cjk_heldout_fonts_256.pt
  ```
  With the Windows 11 fonts the rebuilt held-out pools match the paper's row for row (same glyph
  order as `splits/*_meta.txt`, participation ratios within 0.01 pixel), and `cjk_2d` evaluated on
  them reproduces the paper's numbers.

## Latin glyphs, 2D (`latin_font_2d`)

* **Paper data.** `font_letters_256.pt`, [496, 1, 256, 256] = the 62 glyphs `A-Z a-z 0-9` in
  8 font faces, face-major, rendered with the glyph-pool pipeline (PR fraction 0.197).
* **Known gap.** The script and the 8 faces used for this file were not archived. A
  nearest-neighbour search against 14 DejaVu and 13 Windows faces found no match, so the
  description of this pool as "DejaVu glyphs" is unverified. A comparable pool can be rendered with
  any 8 fonts, but it is not the paper data, and a model trained on it is not the released one:
  ```bash
  python scripts/data/make_glyph_pool.py --latin --fonts <8 font files> --output data/font_letters_256.pt
  ```
* **Held-out pool.** The same 62 glyphs in 8 unseen Windows font families (commercial fonts, not
  redistributable) -> [496, 1, 256, 256]:
  ```bash
  python scripts/data/make_glyph_pool.py --latin --font-dir <fonts> \
      --fonts arial.ttf times.ttf georgia.ttf calibri.ttf consola.ttf segoeui.ttf tahoma.ttf cour.ttf \
      --output data/font_heldout_fonts_256.pt
  ```

## MPEG-7 silhouettes (`mpeg7_2d`)

* **Source.** MPEG-7 CE-Shape-1 (L. J. Latecki, R. Lakamper, U. Eckhardt, CVPR 2000), 1400 binary
  GIFs in 70 classes, <https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip> (3.4 MB,
  reachable in September 2026).
* **Licence.** Distributed for academic benchmarking by Temple University; no explicit licence.
* **Build** (`[1294, 1, 256, 256]`, PR fraction 0.244):
  ```bash
  mkdir -p data/raw/mpeg7 && (cd data/raw/mpeg7 && wget https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip && unzip -q MPEG7dataset.zip)
  python scripts/data/prepare_mpeg7.py --data-dir data/raw/mpeg7/original --resolution 256 \
      --keep-list scripts/data/splits/mpeg7_256_datamean_clean_files.txt \
      --output data/mpeg7_256_datamean_clean.pt
  ```
  The classes `pencil` and `watch` are skipped (1360 shapes remain). Each shape is flipped
  vertically, cropped, padded to a square with a 10% margin, resized, blurred (sigma 2.05) and
  normalised to the data-mean participation ratio in three passes (details in the script).
* **About the keep list.** The paper file keeps 1294 of the 1360 shapes; the 66 removed ones are
  thin shapes (16 Bone, 14 spring, 12 fork, 12 hammer, 8 spoon, 2 guitar, 2 sea_snake, 1 fish).
  The command that removed them was not archived, and the script's own PR filter
  (`--pr-min-frac 0.8`) removes none of them. We recovered the kept list by matching the paper
  tensor row by row against a rebuild. With `--keep-list` the rebuilt rows correspond one to one
  to the paper file (so the split indices below apply) and agree to float32 rounding (relative L2
  ~1e-6 per sample with torch 2.11), not bit for bit.
* **Held-out split** (random 90/10, seed 0; identical to `splits/mpeg7_256_split_s0.json`):
  ```bash
  python scripts/data/split_mpeg7.py --data data/mpeg7_256_datamean_clean.pt --out-dir data \
      --check scripts/data/splits/mpeg7_256_split_s0.json
  ```

## Sphere source (`sphere2airplane_3d`)

A sigmoid soft ball, peak 1, whose mass matches the mean mass of the airplane pool (7967),
stored as 8 identical copies. This command reproduces the paper file bit for bit (radius 12.3803,
sum 7966.98); the script defaults (`--softness 1.5 --target-sum 8000`) do not:
```bash
python scripts/data/prepare_sphere_volume_match_128.py --softness 0.2 --target-sum 7967 \
    --n-copies 8 --output data/sphere_match_airplanes_s02_128.pt
```

## ShapeNet airplanes (`sphere2airplane_3d`)

* **Source.** Watertight, normalised and decimated meshes of the ShapeNetCore.v2 airplanes
  (synset 02691156), distributed as one archive `model_wtt_normed_dcmed_outwardN.zip`
  (534,464,704 bytes; 4045 models as `<model id>/models/model_decimated.obj`).
* **Known gap.** The watertight conversion was not done by this project's code and its provenance
  is not documented; we obtained the prepared archive. The 4045 model ids are in
  `splits/shapenet_airplanes_wtt_volresize_n8000_128_ids.txt`; the original meshes are in
  ShapeNetCore.v2.
* **Licence.** ShapeNet terms of use: non-commercial research and education only, no
  redistribution of the data or derivatives without permission. We ship only the id lists.
* **Build** (`[4045, 1, 128, 128, 128]`, 0 failed, 79 clipped, mass min/mean/max 2217/7967/8221;
  about 2 minutes with 24 workers):
  ```bash
  unzip -q model_wtt_normed_dcmed_outwardN.zip -d data/raw
  python scripts/data/voxelize_airplanes_wtt_volresize.py \
      --mesh-root data/raw/model_wtt_normed_dcmed_outwardN \
      --resolution 128 --sigma 0.7 --n-target 8000 --margin 0.9 --init-fill 0.7 --workers 24 \
      --output data/shapenet_airplanes_wtt_volresize_n8000_128.pt \
      --match-ids scripts/data/splits/shapenet_airplanes_wtt_volresize_n8000_128_ids.txt
  ```
  Per mesh: centre, scale to 70% of the grid, voxelize at pitch 1/128 with `trimesh` and fill,
  Gaussian blur sigma 0.7, isotropic resize to mass 8000 (at most 90% of the grid), clip to [0, 1].
* **Held-out split.** The official split of ShapeNetCore.v2.PC15k (the point-cloud release of
  PointFlow, Yang et al. 2019): train + val -> 3237 training airplanes, test -> 808 held-out
  airplanes; plus a sphere matched to the training-set mean mass (radius 12.3789):
  ```bash
  python scripts/data/split_airplanes_wtt.py --data data/shapenet_airplanes_wtt_volresize_n8000_128.pt \
      --train-ids scripts/data/splits/airplanes_wtt_train_128_ids.txt \
      --test-ids scripts/data/splits/airplanes_wtt_test_128_ids.txt --out-dir data
  ```
  (`--pc15k-root <ShapeNetCore.v2.PC15k>/02691156` derives the same lists from a copy of the
  PC15k release.)

## HuMMan bodies (`humman_3d`, `humman_3d_figures`)

* **Source.** HuMMan (Cai et al., ECCV 2022) SMPL fits: 5000 poses from 179 subjects and 2657
  action clips, stored as `humman_sub_all179_5k.npz` with keys `betas [5000, 10]`,
  `body_pose [5000, 69]`, `global_orient [5000, 3]`, `transl [5000, 3]`,
  `verts [5000, 6890, 3]`, `faces [13776, 3]`, `pid`, `seq`, `frame`.
* **Known gap.** The script that selected these 5000 frames from the HuMMan release and posed the
  SMPL body model (`verts`) was not archived (which SMPL model variant was used is also not
  recorded). `splits/humman_sub_all179_5k_index.tsv` lists the subject, action and frame of every
  pose, so the subset can be reassembled from a licensed copy of HuMMan and SMPL.
* **Licence.** HuMMan: the dataset's S-Lab / OpenXDLab terms (non-commercial research, no
  redistribution). SMPL: the MPI SMPL licence (non-commercial). We ship only the index.
* **Build** (`[5000, 1, 128, 128, 128]`, 0 clipped; "n14000" is the target mass per pose):
  ```bash
  python scripts/data/voxelize_humman_volresize.py --input data/raw/humman_sub_all179_5k.npz \
      --n-target 14000 --sigma 0.7 --init-fill 0.7 --margin 0.9 \
      --output data/humman_volresize_n14000_128.pt
  ```
  Per pose: undo `global_orient` and `transl`, then the airplane pipeline with mass 14000. Rows
  follow the npz order.
* **Held-out interpolation pool** (1181 new poses: vertex midpoints of consecutive sampled frames of
  the same clip, RMS displacement in [0.03, 0.25]):
  ```bash
  python scripts/data/make_humman_heldout_interp.py --input data/raw/humman_sub_all179_5k.npz \
      --output data/humman_interp_heldout.npz
  python scripts/data/voxelize_humman_volresize.py --input data/humman_interp_heldout.npz \
      --n-target 14000 --sigma 0.7 --init-fill 0.7 --margin 0.9 \
      --output data/humman_interp_heldout_n14000_128.pt
  ```

## Extruded glyphs, 3D (`font_3d`)

* **Source.** The 22 TrueType faces of DejaVu 2.37 (Debian/Ubuntu packages `fonts-dejavu-core`,
  `fonts-dejavu-extra`, `fonts-dejavu-mono`, in `/usr/share/fonts/truetype/dejavu`) x the 62 glyphs
  `0-9 A-Z a-z`.
* **Licence.** DejaVu fonts licence (free, redistributable).
* **Build** (`[1364, 1, 128, 128, 128]`, 0 clipped, mass 7728 ... 8277):
  ```bash
  python scripts/data/voxelize_font_3d.py --font-paths /usr/share/fonts/truetype/dejavu \
      --output data/font_3d_volresize_n8000_128.pt \
      --match-ids scripts/data/splits/font_3d_volresize_n8000_128_ids.txt
  ```
  Per glyph: binary mask at 256 px, extrusion depth 0.3 x the glyph size, scale to 70% of the
  grid, blur sigma 0.7, isotropic resize to mass 8000 (at most 90% of the grid), clip to [0, 1].
* **Held-out pool** (the same glyphs in 8 unseen faces -> [496, 1, 128^3]): FreeMono,
  FreeMonoBold, FreeSans, FreeSansBold, FreeSerif, FreeSerifBold (GNU FreeFont 20211204, GPLv3 with
  font exception; package `fonts-freefont-ttf`) and RobotoSlab-Regular, RobotoSlab-Bold (Apache 2.0;
  package `fonts-roboto-slab`, `.otf`). The exact command was not archived; the defaults and this
  face list reproduce the shipped id list:
  ```bash
  F=/usr/share/fonts/truetype/freefont; R=/usr/share/fonts/opentype/roboto/slab
  python scripts/data/voxelize_font_3d.py --font-paths $F/FreeMono.ttf $F/FreeMonoBold.ttf \
      $F/FreeSans.ttf $F/FreeSansBold.ttf $F/FreeSerif.ttf $F/FreeSerifBold.ttf \
      $R/RobotoSlab-Regular.otf $R/RobotoSlab-Bold.otf \
      --output data/font3d_heldout_fonts_n8000_128.pt \
      --match-ids scripts/data/splits/font3d_heldout_fonts_n8000_128_ids.txt
  ```

## Summary of known gaps

1. **Latin glyphs 2D**: no build script, and the 8 font faces of `font_letters_256.pt` are unknown.
2. **HuMMan**: the raw HuMMan -> `humman_sub_all179_5k.npz` step (frame selection and SMPL posing)
   is missing; only the (subject, action, frame) index is shipped.
3. **ShapeNet airplanes**: the provenance of the watertight archive
   `model_wtt_normed_dcmed_outwardN.zip` is undocumented.
4. **MPEG-7**: the raw archive is only available from the Temple University URL above; the
   paper's 1294-shape subset is reproduced with the recovered keep list, not with the original
   (unarchived) command.
