# -*- coding: utf-8 -*-
"""Interactive paint-chain GUI for the released 3D glyph operator (``font_3d``).

Paint a 2D front-view source and target; the GUI extrudes each into a 128^3
density with the recipe of the ``font_3d`` training set
(``viot.glyphs.mask_to_volume``: extrusion depth 0.3, blur sigma 0.7, resize to
total mass 8000, then peak-normalized as in training), runs a 50-step rollout
through the trained 3D operator, and animates it in a 45-degree MIP view.

Workflow
--------
1. Paint a glyph on the LEFT card (front XY view) -> rho_0.
2. Paint a target glyph on the MIDDLE card -> rho_1.
3. Click "Run transition" -> 50-step 3D rollout, animated at 45 degrees in
   the RIGHT card.
4. Click "Continue" -> final volume becomes next rho_0; clear MIDDLE; paint
   the next target. Thumbnail strip tracks every keyframe.
5. "Save chain GIF" dumps the entire concatenated 45 degrees rollout.

Advection defaults to the scheme the model was trained with (first-order
semi-Lagrangian for all released 3D models). A CUDA GPU with >= 8 GB is
recommended; on CPU a transition takes minutes.

    python apps/paint_chain_3d_gui.py                    # font_3d from the Hub
    python apps/paint_chain_3d_gui.py --local-dir path/to/VIOT-checkpoints
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # duplicate OpenMP runtimes on Windows/conda

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageTk

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from viot.glyphs import mask_to_volume
from viot.ops_3d import advect_3d
from viot.pretrained import build_model, load_config, load_pretrained, load_state_dict

_inferno = matplotlib.colormaps["inferno"]


# ---------------------------------------------------------------------------
# Model settings (overwritten from config.json in main()).
# ---------------------------------------------------------------------------
RES = 128
N_INFER_STEPS = 50
ADVECTION = "semi_lagrangian"
MODEL_NAME = "font_3d"

# Paint canvas: 256x256 grayscale, displayed at 1x.
PAINT_RES = 256
PAINT_CANVAS_PX = 256
ANIM_PX = 480       # right card render size

# Extrusion parameters of the font_3d training set (scripts/data/voxelize_font_3d.py).
DEPTH_FRAC = 0.30
INIT_FILL_FRAC = 0.70
BLUR_SIGMA_3D = 0.7
N_TARGET = 8000
MARGIN = 0.9

# 45 degrees viewing direction (rotate around Y, then tilt slightly).
VIEW_YAW_DEG = 45.0
VIEW_PITCH_DEG = 0.0

HEADER_TITLE = "VIOT  ·  Interactive 3D Paint-Chain"
HEADER_SUBTITLE = ""   # filled in main() from the model config

HISTORY_THUMB_PX = 96
HISTORY_MAX_KEYS = 10

PALETTE = {
    "app_bg":      "#fafaf7",
    "panel_bg":    "#ffffff",
    "header_bg":   "#1a1f24",
    "header_fg":   "#ffffff",
    "header_sub":  "#d0d7de",
    "ink":         "#1a1a1a",
    "muted":       "#6b6b6b",
    "border":      "#cfd2cc",
    "canvas_brd":  "#1a1a1a",
    "accent":      "#b6543a",
    "accent_bg":   "#f5e8e4",
}


_RESAMP = getattr(Image, "Resampling", Image)
NEAREST = _RESAMP.NEAREST
BILINEAR = _RESAMP.BILINEAR


_INFERNO_LUT = (_inferno(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Styling.
# ---------------------------------------------------------------------------
def apply_style(root: tk.Tk) -> tuple[str, str]:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    serif = "Libertinus Serif" if _font_available("Libertinus Serif") else \
            "Times New Roman" if _font_available("Times New Roman") else \
            "DejaVu Serif"
    sans = "Inter" if _font_available("Inter") else \
           "Segoe UI" if _font_available("Segoe UI") else "DejaVu Sans"

    root.configure(bg=PALETTE["app_bg"])
    style.configure(".", background=PALETTE["app_bg"],
                    foreground=PALETTE["ink"])
    style.configure("CardTitle.TLabel", background=PALETTE["panel_bg"],
                    foreground=PALETTE["ink"], font=(serif, 13, "bold"))
    style.configure("CardSubtitle.TLabel", background=PALETTE["panel_bg"],
                    foreground=PALETTE["muted"], font=(sans, 9))
    style.configure("Accent.TButton", padding=(10, 6),
                    font=(sans, 10, "bold"))
    style.configure("TButton", padding=(10, 6), font=(sans, 10))
    return serif, sans


def _font_available(name: str) -> bool:
    try:
        return name in tkfont.families()
    except tk.TclError:
        return False


# ---------------------------------------------------------------------------
# Paint -> 3D extruded density.
# ---------------------------------------------------------------------------
def strokes_to_3d_density(stroke_img: Image.Image,
                          device: torch.device) -> torch.Tensor | None:
    """Convert a [PAINT_RES, PAINT_RES] grayscale paint to a [1, 1, R, R, R]
    density distributed like the font_3d training volumes.

    The painted strokes are binarized and tightly cropped (like a rendered
    glyph mask), extruded with ``viot.glyphs.mask_to_volume`` (total mass
    8000, peak ~1) and peak-normalized as the training sampler does.
    """
    arr = np.asarray(stroke_img, dtype=np.float32) / 255.0
    if arr.sum() < 1.0:
        return None
    nz = np.argwhere(arr > 32 / 255.0)
    if len(nz) == 0:
        return None
    (y0, x0), (y1, x1) = nz.min(axis=0), nz.max(axis=0)
    mask = (arr[y0:y1 + 1, x0:x1 + 1] > 32 / 255.0).astype(np.float32)
    vol, _ = mask_to_volume(mask, res=RES, n_target=N_TARGET, sigma=BLUR_SIGMA_3D,
                            margin=MARGIN, init_fill=INIT_FILL_FRAC, depth_frac=DEPTH_FRAC)
    if vol is None:
        return None
    gt = torch.from_numpy(vol).clamp(min=0)[None, None].to(device)
    return gt / (gt.amax() + 1e-12)


# ---------------------------------------------------------------------------
# 3D density -> 45 degrees MIP RGB image (for animation panel + thumbs).
# ---------------------------------------------------------------------------
def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> torch.Tensor:
    """Return 3x3 rotation: pitch around X, then yaw around Y."""
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    Ry = np.array([
        [np.cos(y), 0.0, np.sin(y)],
        [0.0,       1.0, 0.0],
        [-np.sin(y), 0.0, np.cos(y)],
    ])
    Rx = np.array([
        [1.0, 0.0,       0.0],
        [0.0, np.cos(p), -np.sin(p)],
        [0.0, np.sin(p), np.cos(p)],
    ])
    return torch.from_numpy(Ry @ Rx).float()


def render_density_45(rho: torch.Tensor, view_size: int = ANIM_PX,
                       yaw_deg: float = VIEW_YAW_DEG,
                       pitch_deg: float = VIEW_PITCH_DEG) -> np.ndarray:
    """Render a [1,1,D,H,W] (or [D,H,W]) volume as a single 2D MIP image
    seen from the given yaw/pitch direction. Returns uint8 RGB [H,H,3].

    Uses torch.grid_sample on a rotated coordinate grid for speed (GPU
    if rho is on cuda, ~5 ms / 128^3 frame).
    """
    if rho.ndim == 3:
        rho = rho.unsqueeze(0).unsqueeze(0)
    elif rho.ndim == 4:
        rho = rho.unsqueeze(0)
    device = rho.device
    D = rho.shape[-1]

    # Rotated sampling grid: each grid point (x, y, z) in [-1, 1]^3 of the
    # output volume maps to R @ (x, y, z) in the input volume.
    Rmat = _rotation_matrix(yaw_deg, pitch_deg).to(device)
    Rmat_inv = Rmat.T  # rotation inverse is transpose
    lin = torch.linspace(-1.0, 1.0, D, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1)  # [D,D,D,3]
    pts_rot = pts @ Rmat_inv.T              # [D,D,D,3]
    grid = pts_rot.unsqueeze(0)             # [1,D,D,D,3]

    rho_rot = F.grid_sample(
        rho, grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )  # [1,1,D,D,D]
    rho_rot = rho_rot.squeeze().detach().cpu().numpy()

    # MIP along the new Z axis (axis 0 in rho_rot since grid_sample
    # interprets the last dim of grid as (x,y,z) and depth-first ordering
    # gives Z=axis 0 in the output).
    mip = rho_rot.max(axis=0)
    mip = mip / (mip.max() + 1e-12)

    rgb = (_inferno(mip)[..., :3] * 255).astype(np.uint8)
    pil = Image.fromarray(rgb).resize((view_size, view_size), BILINEAR)
    return np.asarray(pil)


# ---------------------------------------------------------------------------
# Density -> 2D RGB for paint preview (when ρ_0 is locked to a previous final).
# ---------------------------------------------------------------------------
def density3d_to_front_preview(rho: torch.Tensor) -> np.ndarray:
    """Project [1,1,D,H,W] onto front view (sum along Z), inferno-colored."""
    if rho.ndim == 5:
        rho = rho[0, 0]
    mip = rho.detach().cpu().numpy().max(axis=0)
    mip = mip / (mip.max() + 1e-12)
    return (_inferno(mip)[..., :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Card factory.
# ---------------------------------------------------------------------------
def make_card(parent, title, subtitle=""):
    outer = tk.Frame(parent, bg=PALETTE["panel_bg"], bd=0,
                     highlightthickness=1,
                     highlightbackground=PALETTE["border"])
    inner = tk.Frame(outer, bg=PALETTE["panel_bg"])
    inner.pack(padx=12, pady=10)
    ttk.Label(inner, text=title, style="CardTitle.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(inner, text=subtitle, style="CardSubtitle.TLabel").pack(
            anchor="w", pady=(0, 6))
    body = tk.Frame(inner, bg=PALETTE["panel_bg"])
    body.pack()
    return outer, body


# ---------------------------------------------------------------------------
# Paint canvas (front XY view) — same as 2D version but smaller.
# ---------------------------------------------------------------------------
class PaintCanvas:
    def __init__(self, parent, brush_var):
        self.brush_var = brush_var
        self.image = Image.new("L", (PAINT_RES, PAINT_RES), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.tk_image: ImageTk.PhotoImage | None = None
        wrap = tk.Frame(parent, bg=PALETTE["canvas_brd"])
        wrap.pack()
        self.canvas = tk.Canvas(wrap, width=PAINT_CANVAS_PX, height=PAINT_CANVAS_PX,
                                bg="black", highlightthickness=0,
                                cursor="pencil", bd=0)
        self.canvas.pack(padx=1, pady=1)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.last_xy = None
        self.locked = False
        self._render()

    def _press(self, e):
        if self.locked:
            return
        self.last_xy = (e.x, e.y)
        self._stroke(e.x, e.y, e.x, e.y)

    def _drag(self, e):
        if self.locked:
            return
        if self.last_xy is None:
            self.last_xy = (e.x, e.y)
        self._stroke(*self.last_xy, e.x, e.y)
        self.last_xy = (e.x, e.y)

    def _release(self, e):
        self.last_xy = None

    def _stroke(self, x0, y0, x1, y1):
        r = max(1, self.brush_var.get() // 2)
        # Tk canvas px = paint px since PAINT_CANVAS_PX == PAINT_RES.
        self.draw.line([(x0, y0), (x1, y1)], fill=255, width=r * 2)
        # Round endcaps.
        for (x, y) in [(x0, y0), (x1, y1)]:
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self._render()

    def _render(self):
        rgb = np.stack([np.asarray(self.image)] * 3, axis=-1)
        pil = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

    def clear(self):
        self.image = Image.new("L", (PAINT_RES, PAINT_RES), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.locked = False
        self.canvas.configure(cursor="pencil")
        self._render()

    def show_density_preview(self, rho: torch.Tensor):
        """Lock the canvas to a static 3D-density front-view preview."""
        rgb = density3d_to_front_preview(rho)
        pil = Image.fromarray(rgb).resize((PAINT_RES, PAINT_RES), BILINEAR)
        self.image = pil.convert("L")
        self.draw = ImageDraw.Draw(self.image)
        # paint over a tinted color in display:
        color_pil = pil.convert("RGB")
        self.tk_image = ImageTk.PhotoImage(color_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.locked = True
        self.canvas.configure(cursor="arrow")


# ---------------------------------------------------------------------------
# Animation canvas (right card).
# ---------------------------------------------------------------------------
class AnimCanvas:
    def __init__(self, parent):
        wrap = tk.Frame(parent, bg=PALETTE["canvas_brd"])
        wrap.pack()
        self.canvas = tk.Canvas(wrap, width=ANIM_PX, height=ANIM_PX,
                                bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(padx=1, pady=1)
        self.tk_image: ImageTk.PhotoImage | None = None

    def show(self, rgb: np.ndarray):
        pil = Image.fromarray(rgb)
        if pil.size != (ANIM_PX, ANIM_PX):
            pil = pil.resize((ANIM_PX, ANIM_PX), BILINEAR)
        self.tk_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

    def clear(self):
        self.canvas.delete("all")


# ---------------------------------------------------------------------------
# History thumbnail strip.
# ---------------------------------------------------------------------------
class HistoryStrip:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=PALETTE["app_bg"])
        self.frame.pack(fill="x", padx=8, pady=(6, 0))
        self.tk_imgs: list[ImageTk.PhotoImage] = []
        self.labels: list[tk.Label] = []

    def update_keyframes(self, keyframes: list[torch.Tensor]):
        for lbl in self.labels:
            lbl.destroy()
        self.tk_imgs.clear()
        self.labels.clear()
        keep = keyframes[-HISTORY_MAX_KEYS:]
        for rho in keep:
            rgb = render_density_45(rho.to(rho.device), view_size=HISTORY_THUMB_PX)
            pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(pil)
            lbl = tk.Label(self.frame, image=tk_img, bg=PALETTE["app_bg"])
            lbl.pack(side="left", padx=2)
            self.tk_imgs.append(tk_img)
            self.labels.append(lbl)


# ---------------------------------------------------------------------------
# Main app.
# ---------------------------------------------------------------------------
class PaintChain3DApp:
    def __init__(self, root: tk.Tk, load_fn, device_pref: str | None):
        self.root = root
        self.device = torch.device(device_pref or
                                    ("cuda" if torch.cuda.is_available() else "cpu"))
        self.serif, self.sans = apply_style(root)

        self.model: torch.nn.Module | None = None
        self.load_fn = load_fn

        self.frames: list[torch.Tensor] = []      # all frames across the chain
        self.keyframes: list[torch.Tensor] = []   # segment boundary states
        self.last_density: torch.Tensor | None = None
        self.q: queue.Queue = queue.Queue()
        self._anim_after_id: str | None = None
        self._anim_segment: list[torch.Tensor] = []
        self._anim_idx: int = 0

        root.title(f"VIOT  -  3D Paint-Chain  ({MODEL_NAME})")
        root.resizable(False, False)

        header = tk.Frame(root, bg=PALETTE["header_bg"])
        header.pack(fill="x")
        tk.Label(header, text=HEADER_TITLE, bg=PALETTE["header_bg"],
                 fg=PALETTE["header_fg"],
                 font=(self.serif, 16, "bold")).pack(anchor="w",
                                                     padx=18, pady=(12, 0))
        tk.Label(header, text=HEADER_SUBTITLE, bg=PALETTE["header_bg"],
                 fg=PALETTE["header_sub"],
                 font=(self.sans, 9)).pack(anchor="w",
                                           padx=18, pady=(2, 12))

        # Top brush control.
        ctrl = tk.Frame(root, bg=PALETTE["app_bg"])
        ctrl.pack(fill="x", padx=16, pady=(10, 4))
        tk.Label(ctrl, text="Brush size", bg=PALETTE["app_bg"],
                 fg=PALETTE["ink"], font=(self.sans, 10, "bold")).pack(side="left")
        self.brush_var = tk.IntVar(value=22)
        self.brush_label = tk.Label(ctrl, text="22 px", bg=PALETTE["app_bg"],
                                     fg=PALETTE["muted"], font=(self.sans, 9))
        self.brush_label.pack(side="left", padx=(8, 8))
        brush_scale = ttk.Scale(ctrl, from_=4, to=60, orient="horizontal",
                                variable=self.brush_var,
                                command=lambda v: self.brush_label.configure(
                                    text=f"{int(float(v))} px"))
        brush_scale.pack(side="left", fill="x", expand=True, padx=(0, 16))

        # Three cards.
        body = tk.Frame(root, bg=PALETTE["app_bg"])
        body.pack(fill="x", padx=16, pady=(4, 8))
        left_outer, left_body = make_card(body, "ρ₀ source",
                                          "front view · paint here")
        mid_outer, mid_body = make_card(body, "ρ₁ target",
                                        "front view · paint here")
        right_outer, right_body = make_card(body, "rollout @ 45°",
                                            "trained 3D operator")
        left_outer.grid(row=0, column=0, padx=(0, 8))
        mid_outer.grid(row=0, column=1, padx=(0, 8))
        right_outer.grid(row=0, column=2)

        self.left = PaintCanvas(left_body, self.brush_var)
        self.right = PaintCanvas(mid_body, self.brush_var)
        self.anim_canvas = AnimCanvas(right_body)

        # Buttons.
        btnbar = tk.Frame(root, bg=PALETTE["app_bg"])
        btnbar.pack(fill="x", padx=16, pady=(2, 8))
        self.run_btn = ttk.Button(btnbar, text="Run transition",
                                   style="Accent.TButton",
                                   command=self.on_run)
        self.run_btn.pack(side="left")
        self.cont_btn = ttk.Button(btnbar, text="Continue",
                                    command=self.on_continue,
                                    state="disabled")
        self.cont_btn.pack(side="left", padx=(6, 0))
        ttk.Button(btnbar, text="Clear ρ₀",
                   command=self.left.clear).pack(side="left", padx=(6, 0))
        ttk.Button(btnbar, text="Clear ρ₁",
                   command=self.right.clear).pack(side="left", padx=(6, 0))
        ttk.Button(btnbar, text="Reset chain",
                   command=self.on_reset).pack(side="left", padx=(6, 0))
        ttk.Button(btnbar, text="Save chain GIF...",
                   command=self.on_save_gif).pack(side="right")

        # Frame counter + status.
        self.frame_label_var = tk.StringVar(value="step 0 / 0")
        tk.Label(btnbar, textvariable=self.frame_label_var,
                 bg=PALETTE["app_bg"], fg=PALETTE["muted"],
                 font=(self.sans, 9)).pack(side="right", padx=(0, 16))

        self.status_var = tk.StringVar(value="Loading model ...")
        tk.Label(root, textvariable=self.status_var,
                 bg=PALETTE["app_bg"], fg=PALETTE["muted"],
                 font=(self.sans, 9), anchor="w").pack(fill="x", padx=18,
                                                       pady=(0, 4))

        # History thumbs.
        self.history = HistoryStrip(root)

        root.bind("<Return>", lambda e: self.on_run())
        root.bind("<Escape>", lambda e: self.on_reset())

        threading.Thread(target=self._load_model_thread, daemon=True).start()
        root.after(60, self._poll_queue)

    # -----------------------------------------------------------------
    def _load_model_thread(self):
        try:
            model = self.load_fn().to(self.device).eval()
            n_params = sum(p.numel() for p in model.parameters())
            self.q.put(("model_ready", model, n_params))
        except Exception as e:
            self.q.put(("error", f"Model load failed: {e}"))

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.root.after(60, self._poll_queue)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "model_ready":
            _, self.model, n_params = msg
            self.status_var.set(
                f"Ready  —  3D model loaded "
                f"({n_params/1e6:.1f}M params, device={self.device}).  "
                f"Paint ρ₀ on the left, ρ₁ in the middle, "
                f"then press ⏎ or 'Run transition'.")
        elif kind == "rollout_done":
            _, frames = msg
            if not self.keyframes:
                self.keyframes.append(frames[0].clone())
            self.keyframes.append(frames[-1].clone())
            if self.frames:
                self.frames.extend(frames[1:])
            else:
                self.frames.extend(frames)
            self.last_density = frames[-1].clone()
            self.history.update_keyframes(self.keyframes)
            self._start_animation(frames)
            self.cont_btn.configure(state="normal")
            self.run_btn.configure(state="normal")
            self.status_var.set(
                f"Rollout done.  Chain: {len(self.keyframes) - 1} segment(s), "
                f"{len(self.frames)} frames total.  "
                f"Click 'Continue' to chain another step.")
        elif kind == "error":
            self.run_btn.configure(state="normal")
            self.status_var.set(f"ERROR: {msg[1]}")
            messagebox.showerror("Error", msg[1])

    # -----------------------------------------------------------------
    # Animation playback (Tk after() loop).
    # -----------------------------------------------------------------
    def _cancel_animation(self):
        if self._anim_after_id is not None:
            try:
                self.root.after_cancel(self._anim_after_id)
            except tk.TclError:
                pass
            self._anim_after_id = None

    def _start_animation(self, frames: list[torch.Tensor]):
        self._cancel_animation()
        self._anim_segment = frames
        self._anim_idx = 0
        self._tick_animation()

    def _tick_animation(self):
        if self._anim_idx >= len(self._anim_segment):
            return
        rho = self._anim_segment[self._anim_idx]
        rgb = render_density_45(rho.to(self.device), view_size=ANIM_PX)
        self.anim_canvas.show(rgb)
        self.frame_label_var.set(
            f"step {self._anim_idx + 1} / {len(self._anim_segment)}")
        self._anim_idx += 1
        self._anim_after_id = self.root.after(60, self._tick_animation)

    # -----------------------------------------------------------------
    # Button handlers.
    # -----------------------------------------------------------------
    def on_run(self):
        if self.model is None:
            messagebox.showinfo("Wait", "Model is still loading.")
            return

        if self.last_density is None:
            rho0 = strokes_to_3d_density(self.left.image, self.device)
            if rho0 is None:
                messagebox.showinfo("Empty ρ₀",
                                    "Paint a glyph on the left canvas first.")
                return
        else:
            rho0 = self.last_density.to(self.device)

        rho1 = strokes_to_3d_density(self.right.image, self.device)
        if rho1 is None:
            messagebox.showinfo("Empty ρ₁",
                                "Paint a target glyph on the middle canvas.")
            return

        self.run_btn.configure(state="disabled")
        self.cont_btn.configure(state="disabled")
        self.status_var.set(
            "Running 50-step 3D rollout ... this takes a few seconds on GPU.")
        threading.Thread(target=self._rollout_thread,
                         args=(rho0, rho1), daemon=True).start()

    def _rollout_thread(self, rho_0: torch.Tensor, rho_1: torch.Tensor):
        try:
            frames = self._rollout(rho_0, rho_1, N_INFER_STEPS)
            self.q.put(("rollout_done", frames))
        except Exception as e:
            self.q.put(("error", f"Rollout failed: {e}"))

    def _rollout(self, rho_0: torch.Tensor, rho_1: torch.Tensor,
                 n_steps: int) -> list[torch.Tensor]:
        dt = 1.0 / n_steps
        rho = rho_0.clone()
        initial_mass = rho.sum(dim=(-3, -2, -1), keepdim=True)
        frames = [rho.detach().cpu()]
        with torch.no_grad():
            for step in range(n_steps):
                t = torch.full((rho.shape[0],), step / n_steps,
                                device=rho.device, dtype=rho.dtype)
                v = self.model.forward_velocity_only(rho, t, rho_1)
                rho = advect_3d(rho, v.float(), dt, scheme=ADVECTION)
                rho = rho.clamp(min=0)
                rho = rho * initial_mass / (rho.sum(dim=(-3, -2, -1), keepdim=True) + 1e-12)
                frames.append(rho.detach().cpu())
        return frames

    def on_continue(self):
        if self.last_density is None:
            return
        self.left.show_density_preview(self.last_density)
        self.right.clear()
        self.cont_btn.configure(state="disabled")
        self.status_var.set(
            "Final density carried into ρ₀.  Paint next target on the middle "
            "canvas, then press ⏎.")

    def on_reset(self):
        self._cancel_animation()
        self.frames.clear()
        self.keyframes.clear()
        self.last_density = None
        self.left.clear()
        self.right.clear()
        self.cont_btn.configure(state="disabled")
        self.anim_canvas.clear()
        self.frame_label_var.set("step 0 / 0")
        self.history.update_keyframes([])
        self.status_var.set(
            "Chain reset.  Paint ρ₀ left, ρ₁ middle, then press ⏎.")

    def on_save_gif(self):
        if not self.frames:
            messagebox.showinfo("Empty",
                                 "Run at least one transition before saving.")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF animation", "*.gif")],
            initialfile="chain3d.gif")
        if not out:
            return
        try:
            pil_frames = []
            for rho in self.frames:
                rgb = render_density_45(rho.to(self.device), view_size=ANIM_PX)
                pil_frames.append(Image.fromarray(rgb))
            pil_frames[0].save(out, save_all=True,
                               append_images=pil_frames[1:],
                               duration=80, loop=0)
            self.status_var.set(
                f"Saved {len(pil_frames)}-frame chain GIF to {out}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main():
    global RES, N_INFER_STEPS, ADVECTION, MODEL_NAME, HEADER_SUBTITLE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="font_3d",
                    help="released 3D model name or checkpoint folder (default: font_3d)")
    ap.add_argument("--local-dir", default=None, help="local copy of the checkpoint repository")
    ap.add_argument("--ckpt", default=None,
                    help="optional weights file (.safetensors or original .pt); "
                         "the architecture is taken from --model's config")
    ap.add_argument("--steps", type=int, default=None, help="rollout steps (default: config)")
    ap.add_argument("--advection", default=None, choices=["semi_lagrangian", "maccormack"],
                    help="advection scheme (default: the one the model was trained with)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    args = ap.parse_args()

    cfg = load_config(args.model, args.local_dir)
    if cfg["dim"] != 3:
        sys.exit(f"{cfg['name']} is a 2D model; use apps/paint_chain_gui.py")
    MODEL_NAME = cfg["name"]
    RES = cfg["arch"]["max_res"]
    N_INFER_STEPS = args.steps or cfg["rollout"]["n_steps"]
    ADVECTION = args.advection or cfg["rollout"]["advection"]
    HEADER_SUBTITLE = (f"{MODEL_NAME}  ·  {RES}³  ·  k_max={cfg['arch']['k_max']}  ·  "
                       f"{ADVECTION}  ·  {N_INFER_STEPS} steps  ·  45° MIP view")

    def load_fn():
        if args.ckpt:
            model = build_model(cfg)
            model.load_state_dict(load_state_dict(args.ckpt))
            return model
        return load_pretrained(args.model, local_dir=args.local_dir)[0]

    root = tk.Tk()
    PaintChain3DApp(root, load_fn, args.device)
    root.mainloop()


if __name__ == "__main__":
    main()
