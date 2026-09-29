# -*- coding: utf-8 -*-
"""Interactive paint-chain GUI for the released 2D VIOT operators.

Workflow
--------
1. Paint a digit on the LEFT card  (rho_0 source).
2. Paint a target digit on the MIDDLE card  (rho_1 target).
3. Click "Run transition" -> 50-step rollout is animated in the RIGHT card.
4. Click "Continue" -> the final density of the previous segment is moved
   into the LEFT card (locked, rendered as a heatmap), the MIDDLE card
   clears, and you paint the next target. A thumbnail strip at the bottom
   keeps a record of every keyframe in the chain. Repeat indefinitely.
5. "Save chain GIF..." dumps the entire concatenated rollout as a single GIF.

Hand-drawn strokes are normalized to match the MNIST sampler used at training
time so size and position do not matter:
    bbox crop -> pad to square -> resize to 70% fill -> Gaussian blur ->
    floor -> mass-normalize -> participation-ratio area-normalize -> mass-normalize.
The area normalization is capped so that every sketch stays within 75% of the
canvas, which leaves room for the transport; thin sketches (a 1, an I, a
straight stroke) are drawn bolder until they have the training area instead of
being enlarged past the border (``--max-extent``; 0 restores the uncapped
normalization of the paper's GUI).

The model (default ``mnist_2d``, the one shown in the paper) is fetched from
the Hugging Face Hub, read from a local copy of the checkpoint repository
(``--local-dir``), or given as a checkpoint file (``--ckpt``). Architecture,
advection scheme and step count come from the model's ``config.json``.

    python apps/paint_chain_gui.py                      # mnist_2d
    python apps/paint_chain_gui.py --model cjk_2d
    python apps/paint_chain_gui.py --local-dir path/to/VIOT-checkpoints
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # duplicate OpenMP runtimes on Windows/conda

import argparse
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from viot.data_2d import FRAME_MAX_EXTENT, MNIST_TARGET_PR_FRAC, normalize_area_on_grid
from viot.data_2d import strokes_to_density as _strokes_to_density
from viot.ops_2d import advect
from viot.pretrained import build_model, load_config, load_pretrained, load_state_dict


# ---------------------------------------------------------------------------
# Model settings (overwritten from config.json in main()).
# ---------------------------------------------------------------------------
RES = 256
N_INFER_STEPS = 50
ADVECTION = "maccormack"
MODEL_NAME = "mnist_2d"

# Participation ratio (effective area / H*W) of each training set; sketches are
# rescaled to it like the training densities (MNIST, CJK and Latin use 0.197).
PR_FRAC = {"mpeg7_2d": 0.244}
TARGET_PR_FRAC = MNIST_TARGET_PR_FRAC
# Largest size of a sketch after the area normalization (fraction of the canvas);
# None: uncapped, thin sketches can then be enlarged past the border.
MAX_EXTENT = FRAME_MAX_EXTENT

DISPLAY_SCALE = 2          # 256-px image rendered at 512 px on screen
CANVAS_PX = RES * DISPLAY_SCALE
ANIM_DELAY_MS = 50         # per-frame delay while playing back a rollout
PAINT_FILL_FRAC = 0.70     # bbox of strokes is rescaled to 70% of canvas
PAINT_BLUR_SIGMA = 2.0     # Gaussian blur applied before density-norm

# Chain history strip.
HISTORY_THUMB_PX = 96
HISTORY_MAX_KEYS = 10

HEADER_TITLE = "VIOT  ·  Interactive Paint-Chain"
HEADER_SUBTITLE = ""   # filled in main() from the model config

# PIL deprecation: prefer the new resampling enum if present.
_RESAMP = getattr(Image, "Resampling", Image)
NEAREST = _RESAMP.NEAREST
BILINEAR = _RESAMP.BILINEAR

_INFERNO_LUT = (matplotlib.colormaps["inferno"](np.linspace(0.0, 1.0, 256))[:, :3]
                * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visual style: paper-grade light palette + serif/sans typography.
# ---------------------------------------------------------------------------
PALETTE = {
    "app_bg":       "#f4f5f7",
    "panel_bg":     "#ffffff",
    "header_bg":    "#1f2937",
    "header_fg":    "#ffffff",
    "header_sub":   "#9ca3af",
    "ink":          "#111827",
    "muted":        "#6b7280",
    "accent":       "#2563eb",
    "accent_dark":  "#1d4ed8",
    "border":       "#e5e7eb",
    "canvas_brd":   "#9ca3af",
    "status_bg":    "#eef0f3",
}


def _pick_font(candidates: list[str], default: str) -> str:
    """Return the first installed font in ``candidates``, else ``default``."""
    available = set(tkfont.families())
    for name in candidates:
        if name in available:
            return name
    return default


def apply_paper_style(root: tk.Tk) -> tuple[str, str]:
    """Configure ttk theme + color/typography palette. Returns (serif, sans)."""
    serif = _pick_font(
        ["Cambria", "Georgia", "Libertinus Serif", "Linux Libertine O",
         "Times New Roman", "DejaVu Serif"], default="Times New Roman")
    sans = _pick_font(
        ["Inter", "Segoe UI Variable", "Segoe UI", "Helvetica Neue",
         "Arial", "DejaVu Sans"], default="Segoe UI")

    root.configure(bg=PALETTE["app_bg"])
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=PALETTE["app_bg"],
                    foreground=PALETTE["ink"], font=(sans, 10))
    style.configure("TFrame", background=PALETTE["app_bg"])
    style.configure("TLabel", background=PALETTE["app_bg"],
                    foreground=PALETTE["ink"])
    style.configure("Card.TFrame", background=PALETTE["panel_bg"])
    style.configure("Card.TLabel", background=PALETTE["panel_bg"],
                    foreground=PALETTE["ink"])
    style.configure("CardTitle.TLabel", background=PALETTE["panel_bg"],
                    foreground=PALETTE["ink"], font=(serif, 13, "bold"))
    style.configure("CardSubtitle.TLabel", background=PALETTE["panel_bg"],
                    foreground=PALETTE["muted"], font=(serif, 10, "italic"))
    style.configure("CardCaption.TLabel", background=PALETTE["panel_bg"],
                    foreground=PALETTE["muted"], font=(sans, 9))

    style.configure("Status.TLabel", background=PALETTE["status_bg"],
                    foreground=PALETTE["muted"], font=(sans, 9), padding=(10, 6))
    style.configure("Status.TFrame", background=PALETTE["status_bg"])
    style.configure("History.TFrame", background=PALETTE["app_bg"])
    style.configure("History.TLabel", background=PALETTE["app_bg"],
                    foreground=PALETTE["muted"], font=(sans, 9))
    style.configure("HistoryArrow.TLabel", background=PALETTE["app_bg"],
                    foreground=PALETTE["muted"], font=(sans, 14))

    # Buttons.
    style.configure("TButton", padding=(10, 5), font=(sans, 10))
    style.configure("Primary.TButton", padding=(14, 6),
                    font=(sans, 10, "bold"),
                    foreground="white", background=PALETTE["accent"],
                    borderwidth=0)
    style.map("Primary.TButton",
              background=[("active", PALETTE["accent_dark"]),
                          ("disabled", "#9ca3af")])
    style.configure("Secondary.TButton", padding=(10, 5), font=(sans, 10))

    # Slider.
    style.configure("Horizontal.TScale", troughcolor=PALETTE["border"],
                    background=PALETTE["app_bg"])
    style.configure("Horizontal.TSeparator",
                    background=PALETTE["border"])

    return serif, sans


# ---------------------------------------------------------------------------
# Hand-drawing -> training-distribution density.
# ---------------------------------------------------------------------------
def strokes_to_density(stroke_img: Image.Image,
                       device: torch.device) -> torch.Tensor | None:
    """Convert a [H, H] grayscale (mode 'L') stroke image to a [1, 1, H, H]
    density matching the training distribution (see
    ``viot.data_2d.strokes_to_density``). Returns ``None`` for an empty canvas.
    """
    if MAX_EXTENT is not None:
        # frame-capped area normalization straight to the model's target area
        canvas = _strokes_to_density(stroke_img, resolution=RES,
                                     fill_frac=PAINT_FILL_FRAC, blur_sigma=PAINT_BLUR_SIGMA,
                                     max_extent=MAX_EXTENT, pr_frac=TARGET_PR_FRAC)
        return None if canvas is None else canvas.to(device)
    canvas = _strokes_to_density(stroke_img, resolution=RES,
                                 fill_frac=PAINT_FILL_FRAC, blur_sigma=PAINT_BLUR_SIGMA)
    if canvas is None:
        return None
    if TARGET_PR_FRAC != MNIST_TARGET_PR_FRAC:
        target_pr = TARGET_PR_FRAC * RES * RES
        canvas[0] = normalize_area_on_grid(canvas[0], target_pr, RES, RES)
        canvas = canvas.clamp(min=0)
        canvas = canvas / (canvas.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    return canvas.to(device)


# ---------------------------------------------------------------------------
# Density -> RGB for the Tk display.
# ---------------------------------------------------------------------------
def density_to_rgb(rho: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Map a [H, W] non-negative array to an [H, W, 3] uint8 inferno LUT."""
    rho = np.asarray(rho, dtype=np.float32)
    if vmax is None:
        vmax = float(rho.max()) + 1e-12
    n = np.clip(rho / vmax, 0.0, 1.0)
    idx = (n * 255).astype(np.uint8)
    return _INFERNO_LUT[idx]


# ---------------------------------------------------------------------------
# Card-style panel factory: titled frame with a thin border + drop shadow feel.
# ---------------------------------------------------------------------------
def make_card(parent: tk.Widget, title: str, subtitle: str = "") -> tuple[tk.Frame, tk.Frame]:
    """Returns (outer, body). Outer is a parent frame with title/subtitle,
    body is where to pack the actual widget (usually a Canvas)."""
    outer = tk.Frame(parent, bg=PALETTE["panel_bg"], bd=0,
                     highlightthickness=1,
                     highlightbackground=PALETTE["border"],
                     highlightcolor=PALETTE["border"])
    inner = tk.Frame(outer, bg=PALETTE["panel_bg"])
    inner.pack(padx=12, pady=10)
    title_lbl = ttk.Label(inner, text=title, style="CardTitle.TLabel")
    title_lbl.pack(anchor="w")
    if subtitle:
        sub_lbl = ttk.Label(inner, text=subtitle, style="CardSubtitle.TLabel")
        sub_lbl.pack(anchor="w", pady=(0, 6))
    body = tk.Frame(inner, bg=PALETTE["panel_bg"])
    body.pack()
    return outer, body


# ---------------------------------------------------------------------------
# Paint canvas widget (one panel: rho_0 or rho_1).
# ---------------------------------------------------------------------------
class PaintCanvas:
    """A 256x256 PIL grayscale image displayed at 2x via a Tk Canvas.

    Paintable in mouse-drag mode; can also be locked into a "density preview"
    mode (used for showing the final rollout frame as the next rho_0).
    """

    def __init__(self, parent: tk.Widget, brush_var: tk.IntVar):
        self.brush_var = brush_var
        self.image = Image.new("L", (RES, RES), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.tk_image: ImageTk.PhotoImage | None = None
        # Wrap canvas in a thin gray border frame so it reads as a print-style
        # panel rather than a raw widget.
        wrap = tk.Frame(parent, bg=PALETTE["canvas_brd"])
        wrap.pack()
        self.canvas = tk.Canvas(wrap, width=CANVAS_PX, height=CANVAS_PX,
                                bg="black", highlightthickness=0,
                                cursor="pencil", bd=0)
        self.canvas.pack(padx=1, pady=1)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.last_xy: tuple[int, int] | None = None
        self.locked = False
        self._render_strokes()

    # --- public API ------------------------------------------------------
    def lock(self) -> None:
        self.locked = True
        self.canvas.configure(cursor="arrow")

    def unlock(self) -> None:
        self.locked = False
        self.canvas.configure(cursor="pencil")

    def clear(self) -> None:
        self.image = Image.new("L", (RES, RES), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.unlock()
        self._render_strokes()

    def set_density_preview(self, rho: np.ndarray,
                            vmax: float | None = None) -> None:
        """Lock the canvas and show ``rho`` as an inferno-mapped heatmap."""
        self.lock()
        rgb = density_to_rgb(rho, vmax=vmax)
        img = Image.fromarray(rgb).resize((CANVAS_PX, CANVAS_PX), BILINEAR)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    # --- mouse handlers --------------------------------------------------
    def _to_img(self, x: int, y: int) -> tuple[int, int]:
        return (max(0, min(RES - 1, x // DISPLAY_SCALE)),
                max(0, min(RES - 1, y // DISPLAY_SCALE)))

    def _press(self, ev: tk.Event) -> None:
        if self.locked:
            return
        ix, iy = self._to_img(ev.x, ev.y)
        self.last_xy = (ix, iy)
        self._stamp(ix, iy)
        self._render_strokes()

    def _drag(self, ev: tk.Event) -> None:
        if self.locked:
            return
        ix, iy = self._to_img(ev.x, ev.y)
        if self.last_xy is not None:
            r = max(int(self.brush_var.get()) // 2, 1)
            self.draw.line([self.last_xy, (ix, iy)], fill=255, width=2 * r)
        self._stamp(ix, iy)
        self.last_xy = (ix, iy)
        self._render_strokes()

    def _release(self, _ev: tk.Event) -> None:
        self.last_xy = None

    def _stamp(self, x: int, y: int) -> None:
        r = max(int(self.brush_var.get()) // 2, 1)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _render_strokes(self) -> None:
        arr = np.asarray(self.image, dtype=np.float32)
        rgb = density_to_rgb(arr, vmax=255.0)
        img = Image.fromarray(rgb).resize((CANVAS_PX, CANVAS_PX), NEAREST)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)


# ---------------------------------------------------------------------------
# Chain history strip: thumbnails of the chain's keyframes.
# ---------------------------------------------------------------------------
class ChainHistoryStrip(tk.Frame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent, bg=PALETTE["app_bg"])
        self.thumbs: list[ImageTk.PhotoImage] = []  # GC anchors
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        self.thumbs.clear()
        ttk.Label(self, text="(no keyframes yet — paint and run a transition)",
                  style="History.TLabel").pack(side="left", padx=4)

    def update_keyframes(self, keyframes: list[torch.Tensor]) -> None:
        for w in self.winfo_children():
            w.destroy()
        self.thumbs.clear()
        if not keyframes:
            self._show_placeholder()
            return
        kfs = keyframes[-HISTORY_MAX_KEYS:]
        # Per-keyframe normalization gives crisp thumbnails regardless of mass.
        for i, f in enumerate(kfs):
            if i > 0:
                ttk.Label(self, text="→",
                          style="HistoryArrow.TLabel").pack(side="left",
                                                            padx=4)
            wrap = tk.Frame(self, bg=PALETTE["canvas_brd"])
            wrap.pack(side="left")
            cv = tk.Canvas(wrap, width=HISTORY_THUMB_PX,
                           height=HISTORY_THUMB_PX, bg="black",
                           highlightthickness=0, bd=0)
            cv.pack(padx=1, pady=1)
            arr = f[0, 0].numpy()
            vmax = float(arr.max()) + 1e-12
            rgb = density_to_rgb(arr, vmax=vmax)
            img = Image.fromarray(rgb).resize(
                (HISTORY_THUMB_PX, HISTORY_THUMB_PX), BILINEAR)
            tk_img = ImageTk.PhotoImage(img)
            self.thumbs.append(tk_img)
            cv.create_image(0, 0, anchor="nw", image=tk_img)
            label = ttk.Label(self, text=f"k{i}", style="History.TLabel")
            label.place(in_=wrap, relx=0.0, rely=0.0, x=4, y=2)


# ---------------------------------------------------------------------------
# Main application.
# ---------------------------------------------------------------------------
class PaintChainApp:
    def __init__(self, root: tk.Tk, load_fn, device: torch.device):
        self.root = root
        self.device = device
        self.load_fn = load_fn

        self.serif, self.sans = apply_paper_style(root)

        # Chain state.
        self.model: torch.nn.Module | None = None
        self.frames: list[torch.Tensor] = []
        self.keyframes: list[torch.Tensor] = []
        self.last_density: torch.Tensor | None = None
        self.q: queue.Queue = queue.Queue()

        # Animation state.
        self._anim_after_id: str | None = None
        self._anim_segment: list[torch.Tensor] = []
        self._anim_vmax: float = 1.0
        self._anim_idx: int = 0

        root.title(f"VIOT  -  Paint-Chain  ({MODEL_NAME})")
        root.resizable(False, False)
        try:
            root.option_add("*tearOff", False)
        except tk.TclError:
            pass

        # ----- header ---------------------------------------------------
        header = tk.Frame(root, bg=PALETTE["header_bg"])
        header.pack(fill="x")
        tk.Label(header, text=HEADER_TITLE, bg=PALETTE["header_bg"],
                 fg=PALETTE["header_fg"],
                 font=(self.serif, 17, "bold")).pack(anchor="w",
                                                     padx=18, pady=(12, 0))
        tk.Label(header, text=HEADER_SUBTITLE, bg=PALETTE["header_bg"],
                 fg=PALETTE["header_sub"],
                 font=(self.sans, 9)).pack(anchor="w",
                                           padx=18, pady=(2, 12))

        # ----- top control bar (brush) ---------------------------------
        ctrl = tk.Frame(root, bg=PALETTE["app_bg"])
        ctrl.pack(fill="x", padx=16, pady=(10, 4))
        tk.Label(ctrl, text="Brush size", bg=PALETTE["app_bg"],
                 fg=PALETTE["ink"],
                 font=(self.sans, 10, "bold")).pack(side="left")
        self.brush_var = tk.IntVar(value=22)
        self.brush_label = tk.Label(ctrl, text="22 px",
                                    bg=PALETTE["app_bg"],
                                    fg=PALETTE["muted"],
                                    font=(self.sans, 10), width=6,
                                    anchor="w")
        slider = ttk.Scale(ctrl, from_=4, to=60, length=220,
                           variable=self.brush_var,
                           command=self._on_brush_change)
        slider.pack(side="left", padx=(8, 6))
        self.brush_label.pack(side="left")

        # ----- three canvas cards ---------------------------------------
        body = tk.Frame(root, bg=PALETTE["app_bg"])
        body.pack(padx=16, pady=8)

        card_l, body_l = make_card(
            body, title="ρ₀  Source",
            subtitle="paint a digit with the mouse")
        card_l.grid(row=0, column=0, padx=8, pady=4, sticky="n")
        self.left = PaintCanvas(body_l, self.brush_var)

        card_m, body_m = make_card(
            body, title="ρ₁  Target",
            subtitle="paint the next digit to morph into")
        card_m.grid(row=0, column=1, padx=8, pady=4, sticky="n")
        self.right = PaintCanvas(body_m, self.brush_var)

        card_r, body_r = make_card(
            body, title="Generated chain",
            subtitle=f"learned div-free flow rolled out for {N_INFER_STEPS} steps")
        card_r.grid(row=0, column=2, padx=8, pady=4, sticky="n")
        wrap_a = tk.Frame(body_r, bg=PALETTE["canvas_brd"])
        wrap_a.pack()
        self.anim_canvas = tk.Canvas(wrap_a, width=CANVAS_PX,
                                     height=CANVAS_PX, bg="black",
                                     highlightthickness=0, bd=0)
        self.anim_canvas.pack(padx=1, pady=1)
        self._anim_tk: ImageTk.PhotoImage | None = None

        self.frame_label = ttk.Label(body_r, text="step 0 / 0",
                                     style="CardCaption.TLabel")
        self.frame_label.pack(anchor="w", pady=(6, 0))

        # ----- chain history strip --------------------------------------
        hist_outer = tk.Frame(root, bg=PALETTE["app_bg"])
        hist_outer.pack(fill="x", padx=16, pady=(8, 4))
        tk.Label(hist_outer, text="Chain history",
                 bg=PALETTE["app_bg"], fg=PALETTE["ink"],
                 font=(self.serif, 11, "bold")).pack(anchor="w")
        tk.Label(hist_outer,
                 text="keyframes from each transition (most recent on the right)",
                 bg=PALETTE["app_bg"], fg=PALETTE["muted"],
                 font=(self.serif, 9, "italic")).pack(anchor="w",
                                                      pady=(0, 6))
        self.history = ChainHistoryStrip(hist_outer)
        self.history.pack(anchor="w", pady=(0, 4))

        # ----- button row -----------------------------------------------
        btns = tk.Frame(root, bg=PALETTE["app_bg"])
        btns.pack(fill="x", padx=16, pady=(8, 4))
        self.run_btn = ttk.Button(btns, text="▶  Run transition",
                                  style="Primary.TButton",
                                  command=self.on_run)
        self.run_btn.pack(side="left", padx=(0, 6))
        self.cont_btn = ttk.Button(btns, text="↻  Continue",
                                   style="Secondary.TButton",
                                   command=self.on_continue,
                                   state="disabled")
        self.cont_btn.pack(side="left", padx=6)
        ttk.Separator(btns, orient="vertical").pack(side="left", fill="y",
                                                    padx=10)
        ttk.Button(btns, text="Clear ρ₀",
                   command=self.on_clear_left).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear ρ₁",
                   command=self.right.clear).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset chain",
                   command=self.on_reset).pack(side="left", padx=4)
        ttk.Separator(btns, orient="vertical").pack(side="left", fill="y",
                                                    padx=10)
        ttk.Button(btns, text="Save chain GIF...",
                   command=self.on_save_gif).pack(side="left", padx=4)

        # ----- status bar ------------------------------------------------
        status_wrap = tk.Frame(root, bg=PALETTE["status_bg"])
        status_wrap.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(
            value=f"Loading model on {device}... "
                  f"(window may be unresponsive for a few seconds)")
        ttk.Label(status_wrap, textvariable=self.status_var,
                  style="Status.TLabel").pack(anchor="w", padx=4)

        # Keyboard shortcuts.
        root.bind("<Return>", lambda _e: self.on_run())
        root.bind("<space>", lambda _e: self.on_run())
        root.bind("<Control-z>", lambda _e: self.on_clear_left())
        root.bind("<Control-x>", lambda _e: self.right.clear())
        root.bind("<Control-r>", lambda _e: self.on_reset())

        # Load model on a background thread so the window appears immediately.
        threading.Thread(target=self._load_model_thread,
                         daemon=True).start()
        self.root.after(50, self._poll_queue)

    # ---------------- helpers --------------------------------------------
    def _on_brush_change(self, _val: str) -> None:
        self.brush_label.configure(text=f"{int(self.brush_var.get())} px")

    # ---------------- model loading ---------------------------------------
    def _load_model_thread(self) -> None:
        try:
            model = self.load_fn().to(self.device).eval()
            n_params = sum(p.numel() for p in model.parameters())
            self.q.put(("model_ready", model, n_params))
        except Exception as exc:
            self.q.put(("error", f"Failed to load model: {exc}"))

    # ---------------- queue plumbing --------------------------------------
    def _poll_queue(self) -> None:
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _handle(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "model_ready":
            _, self.model, n_params = msg
            self.status_var.set(
                f"Ready  —  model loaded "
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

    # ---------------- button handlers ------------------------------------
    def on_clear_left(self) -> None:
        self.last_density = None
        self.left.clear()

    def on_reset(self) -> None:
        self._cancel_animation()
        self.frames.clear()
        self.keyframes.clear()
        self.last_density = None
        self.left.clear()
        self.right.clear()
        self.cont_btn.configure(state="disabled")
        self.anim_canvas.delete("all")
        self.frame_label.configure(text="step 0 / 0")
        self.history.update_keyframes([])
        self.status_var.set(
            "Chain reset.  Paint ρ₀ left, ρ₁ middle, "
            "then press ⏎.")

    def on_run(self) -> None:
        if self.model is None:
            messagebox.showinfo("Wait", "Model is still loading.")
            return

        if self.last_density is None:
            rho0 = strokes_to_density(self.left.image, self.device)
            if rho0 is None:
                messagebox.showinfo(
                    "Empty ρ₀",
                    "Paint a digit on the left canvas first.")
                return
        else:
            rho0 = self.last_density.to(self.device)

        rho1 = strokes_to_density(self.right.image, self.device)
        if rho1 is None:
            messagebox.showinfo(
                "Empty ρ₁",
                "Paint a target digit on the middle canvas.")
            return

        self.run_btn.configure(state="disabled")
        self.cont_btn.configure(state="disabled")
        self.status_var.set(
            f"Running rollout ({N_INFER_STEPS} steps on {self.device})...")
        threading.Thread(target=self._rollout_thread,
                         args=(rho0, rho1), daemon=True).start()

    def on_continue(self) -> None:
        if self.last_density is None:
            return
        rho_np = self.last_density[0, 0].cpu().numpy()
        self.left.set_density_preview(rho_np)
        self.right.clear()
        self.cont_btn.configure(state="disabled")
        self.status_var.set(
            "ρ₀ is now the previous final frame.  Paint the next "
            "ρ₁ and press ⏎.")

    def on_save_gif(self) -> None:
        if not self.frames:
            messagebox.showinfo("Empty",
                                "No frames yet — run a transition first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".gif", filetypes=[("GIF", "*.gif")],
            initialfile="paint_chain.gif")
        if not path:
            return
        vmax = max(float(f.max()) for f in self.frames) + 1e-12
        images = []
        for f in self.frames:
            rgb = density_to_rgb(f[0, 0].numpy(), vmax=vmax)
            images.append(Image.fromarray(rgb).resize(
                (CANVAS_PX, CANVAS_PX), BILINEAR))
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=ANIM_DELAY_MS, loop=0)
        self.status_var.set(f"Saved chain GIF: {path}")

    # ---------------- inference ------------------------------------------
    def _rollout_thread(self, rho0: torch.Tensor,
                        rho1: torch.Tensor) -> None:
        try:
            with torch.no_grad():
                rho = rho0.clone()
                init_mass = rho.sum(dim=(-2, -1), keepdim=True)
                dt = 1.0 / N_INFER_STEPS
                frames = [rho.detach().cpu().clone()]
                for s in range(N_INFER_STEPS):
                    t = torch.full((1,), s / N_INFER_STEPS,
                                   device=self.device, dtype=rho.dtype)
                    v = self.model.forward_velocity_only(rho, t, rho1)
                    rho = advect(rho, v, dt, scheme=ADVECTION)
                    rho = rho.clamp(min=0)
                    rho = rho * init_mass / (
                        rho.sum(dim=(-2, -1), keepdim=True) + 1e-12)
                    frames.append(rho.detach().cpu().clone())
            self.q.put(("rollout_done", frames))
        except Exception as exc:
            self.q.put(("error", f"Rollout failed: {exc}"))

    # ---------------- animation playback ---------------------------------
    def _start_animation(self, frames: list[torch.Tensor]) -> None:
        self._cancel_animation()
        self._anim_segment = frames
        self._anim_vmax = max(float(f.max()) for f in frames) + 1e-12
        self._anim_idx = 0
        self._step_animation()

    def _step_animation(self) -> None:
        if self._anim_idx >= len(self._anim_segment):
            self._anim_after_id = None
            return
        f = self._anim_segment[self._anim_idx][0, 0].numpy()
        rgb = density_to_rgb(f, vmax=self._anim_vmax)
        img = Image.fromarray(rgb).resize((CANVAS_PX, CANVAS_PX), BILINEAR)
        self._anim_tk = ImageTk.PhotoImage(img)
        self.anim_canvas.delete("all")
        self.anim_canvas.create_image(0, 0, anchor="nw",
                                      image=self._anim_tk)
        self.frame_label.configure(
            text=f"step {self._anim_idx} / {len(self._anim_segment) - 1}")
        self._anim_idx += 1
        self._anim_after_id = self.root.after(ANIM_DELAY_MS,
                                              self._step_animation)

    def _cancel_animation(self) -> None:
        if self._anim_after_id is not None:
            try:
                self.root.after_cancel(self._anim_after_id)
            except Exception:
                pass
            self._anim_after_id = None


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    global RES, N_INFER_STEPS, ADVECTION, MODEL_NAME, TARGET_PR_FRAC, MAX_EXTENT, CANVAS_PX, HEADER_SUBTITLE
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mnist_2d",
                   help="released 2D model name or checkpoint folder (default: mnist_2d)")
    p.add_argument("--local-dir", default=None, help="local copy of the checkpoint repository")
    p.add_argument("--ckpt", default=None,
                   help="optional weights file (.safetensors or original .pt); "
                        "the architecture is taken from --model's config")
    p.add_argument("--steps", type=int, default=None, help="rollout steps (default: config)")
    p.add_argument("--advection", default=None, choices=["maccormack", "semi_lagrangian", "weno"],
                   help="advection scheme (default: the one the model was trained with)")
    p.add_argument("--pr-frac", type=float, default=None,
                   help="participation-ratio fraction that sketches are rescaled to")
    p.add_argument("--max-extent", type=float, default=MAX_EXTENT,
                   help="largest sketch size as a fraction of the canvas (default 0.75; "
                        "0 or negative: uncapped area normalization, as in the paper's GUI)")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                   help="Override torch device (default: cuda if available).")
    args = p.parse_args()

    cfg = load_config(args.model, args.local_dir)
    if cfg["dim"] != 2:
        sys.exit(f"{cfg['name']} is a 3D model; use apps/paint_chain_3d_gui.py")
    MODEL_NAME = cfg["name"]
    RES = cfg["arch"]["max_res"]
    CANVAS_PX = RES * DISPLAY_SCALE
    N_INFER_STEPS = args.steps or cfg["rollout"]["n_steps"]
    ADVECTION = args.advection or cfg["rollout"]["advection"]
    TARGET_PR_FRAC = args.pr_frac or PR_FRAC.get(MODEL_NAME, MNIST_TARGET_PR_FRAC)
    MAX_EXTENT = args.max_extent if args.max_extent > 0 else None
    HEADER_SUBTITLE = (f"{MODEL_NAME}  ·  {RES}²  ·  k_max={cfg['arch']['k_max']}  ·  "
                       f"{ADVECTION}  ·  {N_INFER_STEPS} steps")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_fn():
        if args.ckpt:
            model = build_model(cfg)
            model.load_state_dict(load_state_dict(args.ckpt))
            return model
        return load_pretrained(args.model, local_dir=args.local_dir)[0]

    root = tk.Tk()
    PaintChainApp(root, load_fn, device)
    root.mainloop()


if __name__ == "__main__":
    main()
