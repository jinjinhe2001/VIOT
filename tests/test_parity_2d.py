"""Parity tests: ``viot`` 2D code vs. the original research code.

These tests prove that the released 2D modules (model, advection, rollout,
samplers, stroke preprocessing, trainer) reproduce the code that trained the
paper's models bit for bit. They need a copy of the original code, which is
not part of the public repository; without it every test is skipped.

Locations (environment variables):

``VIOT_REF_SNAPSHOT``  original code snapshot (``fluidot/``, ``fluidot_improved/``,
                       ``paint_chain_gui.py``). Default: ``../_work/gt_snapshot``
                       next to the repository root.
``VIOT_REF_EXTRA``     originals that are not in the snapshot: ``fluidot_ops_git_HEAD.py``
                       (first-order advection, git HEAD), ``mnist_grid_chain.py`` (WENO),
                       ``eval_2d_heldout.py`` (circular divergence).
                       Default: ``<snapshot>/../ref_extra``.
``VIOT_REF_CKPT_2D``   an original 2D checkpoint (exp50, MNIST 256^2).
                       Default: ``<snapshot>/../ckpt/exp50_mnist_best.pt``.
``VIOT_MNIST_ROOT``    torchvision MNIST root. Default: ``<snapshot>/../.data``.
``VIOT_PARITY_LONG=1`` also run a 2000-step training parity check that covers the
                       1000-step checkpoints and the ``model_best`` rule (slow, CPU).

Run with ``pytest tests/test_parity_2d.py -v`` or ``python tests/test_parity_2d.py``.
"""

import ast
import contextlib
import copy
import hashlib
import importlib.util
import os
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

# Some Windows/conda installs ship two OpenMP runtimes; the original modules set this too.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from viot import data_2d, ops_2d, train_2d  # noqa: E402
from viot.model_2d import FNO2D, FNOHybridSpectralFlowMatcher2D  # noqa: E402

REF = Path(os.environ.get("VIOT_REF_SNAPSHOT", REPO_ROOT.parent / "_work" / "gt_snapshot"))
REF_EXTRA = Path(os.environ.get("VIOT_REF_EXTRA", REF.parent / "ref_extra"))
REF_CKPT = Path(os.environ.get("VIOT_REF_CKPT_2D", REF.parent / "ckpt" / "exp50_mnist_best.pt"))
MNIST_ROOT = Path(os.environ.get("VIOT_MNIST_ROOT", REF.parent / ".data"))

# md5 of the original files these tests compare against.
MD5 = {
    "fluidot_improved/model_2d.py": "fcc51707",
    "fluidot_improved/train_2d.py": "5cef8c42",
    "fluidot/ops.py": "c466b5e6",
    "fluidot_ops_git_HEAD.py": "d045648c",
    "mnist_grid_chain.py": "00e8a187",
    "eval_2d_heldout.py": "fcfe7faa",
}

SkipTest = unittest.SkipTest


# ---------------------------------------------------------------------------
# Loading the original code
# ---------------------------------------------------------------------------

def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _install_geomloss_shim():
    """fluidot/ops.py imports geomloss at import time but never needs it on our path."""
    try:
        import geomloss  # noqa: F401
    except ImportError:
        mod = types.ModuleType("geomloss")

        class SamplesLoss:  # pragma: no cover - never called
            def __init__(self, *a, **k):
                raise RuntimeError("geomloss shim (parity tests only)")

        mod.SamplesLoss = SamplesLoss
        sys.modules["geomloss"] = mod


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_functions(path, names, namespace):
    """Exec the source of the named (module- or class-level) functions of ``path``."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    consts = {}
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
    missing = set(names) - set(found)
    assert not missing, f"functions {missing} not found in {path}"
    ns = dict(namespace)
    ns.update(consts)
    for n in names:
        code = compile(ast.Module(body=[found[n]], type_ignores=[]), str(path), "exec")
        exec(code, ns)
    return ns


_REF_CACHE = {}


def ref():
    """Return a namespace with the original modules, or skip if they are absent."""
    if "ref" in _REF_CACHE:
        return _REF_CACHE["ref"]
    if not (REF / "fluidot_improved" / "model_2d.py").is_file():
        raise SkipTest(f"reference snapshot not found at {REF} (set VIOT_REF_SNAPSHOT)")
    for rel, prefix in MD5.items():
        p = REF / rel if "/" in rel else REF_EXTRA / rel
        if p.is_file():
            assert _md5(p).startswith(prefix), f"{p} is not the expected original (md5 {prefix}...)"
    _install_geomloss_shim()
    if str(REF) not in sys.path:
        sys.path.insert(0, str(REF))
    import fluidot.ops as o_ops
    import fluidot.train as o_train
    import fluidot_improved.model_2d as o_model
    import fluidot_improved.train_2d as o_train2d
    r = SimpleNamespace(ops=o_ops, train=o_train, model=o_model, train2d=o_train2d,
                        head=None, grid=None, heldout=None, gui=None)
    if (REF_EXTRA / "fluidot_ops_git_HEAD.py").is_file():
        r.head = _load_file("_ref_fluidot_ops_git_head", REF_EXTRA / "fluidot_ops_git_HEAD.py")
    if (REF_EXTRA / "mnist_grid_chain.py").is_file():
        r.grid = _load_file("_ref_mnist_grid_chain", REF_EXTRA / "mnist_grid_chain.py")
    if (REF_EXTRA / "eval_2d_heldout.py").is_file():
        r.heldout = _load_file("_ref_eval_2d_heldout", REF_EXTRA / "eval_2d_heldout.py")
    gui_path = REF / "paint_chain_gui.py"
    if gui_path.is_file():
        # The GUI module needs tkinter and an old matplotlib at import time, so
        # only the two functions under test are compiled from its source.
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from PIL import Image
        ns = _extract_functions(gui_path, ["strokes_to_density", "_rollout_thread"], {
            "np": np, "torch": torch, "F": F, "TF": TF, "Image": Image,
            "_normalize_area_on_grid": o_train._normalize_area_on_grid,
            "MNIST_TARGET_PR_FRAC": o_train.MNIST_TARGET_PR_FRAC,
            "advect": o_ops.advect,
        })
        r.gui = SimpleNamespace(strokes_to_density=ns["strokes_to_density"],
                                rollout_thread=ns["_rollout_thread"],
                                N_INFER_STEPS=ns["N_INFER_STEPS"], RES=ns["RES"])
    _REF_CACHE["ref"] = r
    return r


def need(obj, what):
    if obj is None:
        raise SkipTest(f"{what} not available (see VIOT_REF_EXTRA)")
    return obj


def cuda():
    if not torch.cuda.is_available():
        raise SkipTest("CUDA not available")
    return torch.device("cuda")


def mnist_root():
    if not (MNIST_ROOT / "MNIST" / "raw").is_dir():
        raise SkipTest(f"MNIST not found under {MNIST_ROOT} (set VIOT_MNIST_ROOT)")
    return str(MNIST_ROOT)


def set_orig_mnist(r):
    import torchvision
    r.train._mnist_dataset = torchvision.datasets.MNIST(root=mnist_root(), download=False, train=True)


def assert_equal(a, b, what=""):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype, what
        assert torch.equal(a, b), f"{what}: max |diff| = {(a.double() - b.double()).abs().max().item():.3e}"
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), what
        for i, (x, y) in enumerate(zip(a, b)):
            assert_equal(x, y, f"{what}[{i}]")
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{what}: keys differ {set(a) ^ set(b)}"
        for k in a:
            assert_equal(a[k], b[k], f"{what}[{k}]")
    else:
        assert a == b, f"{what}: {a} != {b}"


def random_pairs(B, H, W, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    r0 = torch.rand(B, 1, H, W, generator=g) ** 4
    r1 = torch.rand(B, 1, H, W, generator=g) ** 4
    r0, r1 = r0 / r0.sum(dim=(-2, -1), keepdim=True), r1 / r1.sum(dim=(-2, -1), keepdim=True)
    return r0.to(device), r1.to(device)


def tiny_models(r, device, res=64, width=16, modes=8, layers=2, seed=0, k_max=0.25):
    """Original and new tiny FNO with identical, non-trivial weights."""
    torch.manual_seed(seed)
    old = r.model.FNOHybridSpectralFlowMatcher2D(max_res=res, k_max=k_max, width=width,
                                                 n_modes=modes, n_layers=layers)
    with torch.no_grad():  # the last layer is zero-initialised; make the velocity non-zero
        old.project[-1].weight.normal_(0.0, 300.0)
        old.project[-1].bias.normal_(0.0, 1.0)
    new = FNO2D(max_res=res, k_max=k_max, width=width, n_modes=modes, n_layers=layers)
    new.load_state_dict(old.state_dict(), strict=True)
    return old.to(device).eval(), new.to(device).eval()


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------

def test_model_alias_and_init_rng():
    r = ref()
    assert FNOHybridSpectralFlowMatcher2D is FNO2D
    for kw in [dict(max_res=64, k_max=0.25, width=16, n_modes=8, n_layers=2),
               dict(max_res=48, k_max=0.0625, width=8, n_modes=4, n_layers=3, time_dim=32)]:
        torch.manual_seed(123)
        a = r.model.FNOHybridSpectralFlowMatcher2D(**kw)
        s_a = torch.get_rng_state()
        torch.manual_seed(123)
        b = FNO2D(**kw)
        assert_equal(a.state_dict(), b.state_dict(), "init state_dict")
        assert list(a.state_dict()) == list(b.state_dict())
        assert torch.equal(s_a, torch.get_rng_state()), "constructor consumed RNG differently"


def _exp50_models(r, device):
    if not REF_CKPT.is_file():
        raise SkipTest(f"checkpoint not found at {REF_CKPT} (set VIOT_REF_CKPT_2D)")
    sd = torch.load(REF_CKPT, map_location="cpu", weights_only=True)
    cfg = dict(max_res=256, k_max=0.25, width=64, n_modes=32, n_layers=8)
    new = FNO2D(**cfg)
    new.load_state_dict(sd, strict=True)
    old = r.model.FNOHybridSpectralFlowMatcher2D(**cfg)
    old.load_state_dict(sd, strict=True)
    assert torch.equal(new.trunc_mask, FNO2D(**cfg).trunc_mask)
    return old.to(device).eval(), new.to(device).eval()


def _check_forward(old, new, device, res_list):
    g = torch.Generator().manual_seed(7)
    for H in res_list:
        rho_t, rho_1 = random_pairs(2, H, H, device, seed=H)
        t = torch.rand(2, generator=g).to(device)
        with torch.no_grad():
            v_old, psi_old = old(rho_t, t, rho_1)
            v_new, psi_new = new(rho_t, t, rho_1)
            v_new2 = new.forward_velocity_only(rho_t, t, rho_1)
        assert v_new.abs().max() > 0
        assert_equal(v_old, v_new, f"v @ {H}")
        assert_equal(psi_old, psi_new, f"psi_hat @ {H}")
        assert_equal(v_old, v_new2, f"forward_velocity_only @ {H}")


def test_model_checkpoint_forward_cpu():
    r = ref()
    old, new = _exp50_models(r, torch.device("cpu"))
    _check_forward(old, new, torch.device("cpu"), [256])


def test_model_checkpoint_forward_cuda():
    r = ref()
    dev = cuda()
    old, new = _exp50_models(r, dev)
    _check_forward(old, new, dev, [256, 128])  # 128 exercises the non-native-resolution mask


def test_model_backward_cpu():
    r = ref()
    old, new = tiny_models(r, "cpu")
    rho_t, rho_1 = random_pairs(2, 64, 64, "cpu", seed=3)
    t = torch.tensor([0.1, 0.7])
    for m in (old, new):
        m.zero_grad()
        m.forward_velocity_only(rho_t, t, rho_1).pow(2).sum().backward()
    assert_equal({k: p.grad for k, p in old.named_parameters()},
                 {k: p.grad for k, p in new.named_parameters()}, "grads")


# ---------------------------------------------------------------------------
# 2. Advection schemes and diagnostics
# ---------------------------------------------------------------------------

def _advection_inputs(device, scale=3.0):
    g = torch.Generator().manual_seed(11)
    rho = torch.rand(3, 1, 48, 40, generator=g).to(device)  # non-square catches H/W swaps
    v = (scale * torch.randn(3, 2, 48, 40, generator=g)).to(device)
    return rho, v


def _check_advection(device):
    r = ref()
    head = need(r.head, "fluidot_ops_git_HEAD.py")
    grid = need(r.grid, "mnist_grid_chain.py")
    rho, v = _advection_inputs(device)
    for dt in (0.1, -0.1, 0.02):
        assert_equal(r.ops.advect(rho, v, dt), ops_2d.advect_maccormack(rho, v, dt), "maccormack")
        assert_equal(r.ops.advect(rho, v, dt), ops_2d.advect(rho, v, dt), "advect default")
        assert_equal(head.advect(rho, v, dt), ops_2d.advect_semi_lagrangian(rho, v, dt), "semi-Lagrangian")
        assert_equal(head.advect(rho, v, dt), ops_2d.advect(rho, v, dt, scheme="semi_lagrangian"), "SL dispatch")
        w_ref = grid.advect_step(rho, v, dt, "weno")
        assert_equal(w_ref, ops_2d.advect_weno(rho, v, dt), "weno")
        assert_equal(w_ref, ops_2d.advect(rho, v, dt, scheme="weno"), "weno dispatch")
    # many sub-steps (hits the max_substeps cap) and the no-clamp variant
    _, v_big = _advection_inputs(device, scale=200.0)
    assert_equal(grid.advect_weno(rho, v_big, 0.1), ops_2d.advect_weno(rho, v_big, 0.1), "weno capped")
    assert_equal(grid.advect_weno(rho, v, 0.1, clamp_nonneg=False),
                 ops_2d.advect_weno(rho, v, 0.1, clamp_nonneg=False), "weno no clamp")
    try:
        ops_2d.advect(rho, v, 0.1, scheme="nope")
        raise AssertionError("unknown scheme accepted")
    except ValueError:
        pass


def test_advection_cpu():
    _check_advection(torch.device("cpu"))


def test_advection_cuda():
    _check_advection(cuda())


def test_advection_gradients_cpu():
    """Training back-propagates through advection: gradients must match too."""
    r = ref()
    head = need(r.head, "fluidot_ops_git_HEAD.py")
    rho0, v0 = _advection_inputs("cpu")
    w = torch.rand_like(rho0)
    for f_old, f_new in [(r.ops.advect, ops_2d.advect_maccormack), (head.advect, ops_2d.advect_semi_lagrangian)]:
        grads = []
        for f in (f_old, f_new):
            rho, v = rho0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
            (f(rho, v, 0.1) * w).sum().backward()
            grads.append((rho.grad, v.grad))
        assert_equal(grads[0], grads[1], f"grad {f_new.__name__}")


def test_diagnostics():
    r = ref()
    held = need(r.heldout, "eval_2d_heldout.py")
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        _, v = _advection_inputs(dev)
        assert_equal(r.train2d.vorticity_2d(v), ops_2d.vorticity_2d(v), "vorticity (train_2d)")
        assert_equal(held.vorticity(v), ops_2d.vorticity_2d(v), "vorticity (eval_2d_heldout)")
        assert_equal(held.divergence(v), ops_2d.divergence_2d(v), "divergence (eval_2d_heldout)")
        assert_equal(r.ops.divergence(v), ops_2d.divergence_2d(v), "divergence (fluidot.ops)")


# ---------------------------------------------------------------------------
# 3. Rollout
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _check_rollout_vs_train2d(r, old, new, rho_0, rho_1, n_steps, scheme, orig_advect):
    with patched(r.train2d, "advect", orig_advect):
        ref_out = r.train2d.rollout(old, rho_0, rho_1, n_steps=n_steps)
    out = ops_2d.rollout_2d(new, rho_0, rho_1, n_steps=n_steps, scheme=scheme,
                            return_frames=True, return_velocities=True)
    assert out["velocities"][-1].abs().max() > 0
    assert_equal(ref_out["frames"], out["frames"], f"frames ({scheme})")
    assert_equal(ref_out["velocities"], out["velocities"], f"velocities ({scheme})")
    assert_equal(ref_out["frames"][-1], out["final"], "final")
    final = ops_2d.rollout_2d(new, rho_0, rho_1, n_steps=n_steps, scheme=scheme)
    assert_equal(ref_out["frames"][-1], final, "final (no frames)")
    only_v = ops_2d.rollout_2d(new, rho_0, rho_1, n_steps=n_steps, scheme=scheme, return_velocities=True)
    assert set(only_v) == {"final", "velocities"}


def test_rollout_cpu():
    r = ref()
    old, new = tiny_models(r, "cpu")
    rho_0, rho_1 = random_pairs(2, 64, 64, "cpu", seed=5)
    _check_rollout_vs_train2d(r, old, new, rho_0, rho_1, 8, "maccormack", r.ops.advect)
    head = need(r.head, "fluidot_ops_git_HEAD.py")
    _check_rollout_vs_train2d(r, old, new, rho_0, rho_1, 8, "semi_lagrangian", head.advect)
    grid = need(r.grid, "mnist_grid_chain.py")
    frames = grid.rollout_segment(old, rho_0, rho_1, 8, "weno", amp=False, cell_batch_size=64)
    out = ops_2d.rollout_2d(new, rho_0, rho_1, n_steps=8, scheme="weno", return_frames=True)
    assert_equal(frames, out["frames"], "weno rollout (mnist_grid_chain.rollout_segment)")


def test_rollout_gui_loop_cpu():
    r = ref()
    gui = need(r.gui, "paint_chain_gui.py")
    old, new = tiny_models(r, "cpu")
    rho_0, rho_1 = random_pairs(1, 64, 64, "cpu", seed=9)
    fake_self = SimpleNamespace(model=old, device=torch.device("cpu"), q=queue.Queue())
    gui.rollout_thread(fake_self, rho_0, rho_1)
    kind, frames = fake_self.q.get_nowait()
    assert kind == "rollout_done", frames
    out = ops_2d.rollout_2d(new, rho_0, rho_1, n_steps=gui.N_INFER_STEPS, return_frames=True)
    assert_equal(frames, out["frames"], "GUI rollout")


def test_rollout_checkpoint_cuda():
    r = ref()
    dev = cuda()
    old, new = _exp50_models(r, dev)
    set_orig_mnist(r)
    torch.manual_seed(42)
    rho_0, rho_1 = data_2d.sample_mnist_pairs(2, 256, 256, dev, root=mnist_root())
    _check_rollout_vs_train2d(r, old, new, rho_0, rho_1, 50, "maccormack", r.ops.advect)
    gui = need(r.gui, "paint_chain_gui.py")
    fake_self = SimpleNamespace(model=old, device=dev, q=queue.Queue())
    gui.rollout_thread(fake_self, rho_0[:1], rho_1[:1])
    kind, frames = fake_self.q.get_nowait()
    assert kind == "rollout_done", frames
    out = ops_2d.rollout_2d(new, rho_0[:1], rho_1[:1], n_steps=50, return_frames=True)
    assert_equal(frames, [f.cpu() for f in out["frames"]], "GUI rollout (exp50, CUDA)")


# ---------------------------------------------------------------------------
# 4. Samplers
# ---------------------------------------------------------------------------

def _check_same_draws(f_old, f_new, seed, what):
    torch.manual_seed(seed)
    a = f_old()
    s_a = torch.get_rng_state()
    torch.manual_seed(seed)
    b = f_new()
    assert_equal(a, b, what)
    assert torch.equal(s_a, torch.get_rng_state()), f"{what}: RNG consumed differently"


def test_mnist_sampler():
    r = ref()
    root = mnist_root()
    set_orig_mnist(r)
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devs:
        dev = torch.device(dev)
        for B, R in [(3, 64), (2, 256)]:
            _check_same_draws(lambda: r.train.sample_density_pair_mnist(B, R, R, dev),
                              lambda: data_2d.sample_mnist_pairs(B, R, R, dev, root=root),
                              seed=B * 100 + R, what=f"mnist B={B} R={R} {dev}")
        # two consecutive batches from the trainer-level samplers
        args = SimpleNamespace(resolution=64, sampler="mnist", data_path=None)
        s_old = r.train2d.get_sampler(args, dev)
        s_new = data_2d.get_sampler("mnist", 64, device=dev, mnist_root=root)
        _check_same_draws(lambda: [s_old(2), s_old(3)], lambda: [s_new(2), s_new(3)], 5, "get_sampler mnist")
    x = data_2d.sample_mnist_pairs(2, 64, 64, "cpu", root=root)[0]
    assert torch.allclose(x.sum(dim=(-2, -1)), torch.ones(2, 1))


def test_normalize_area_on_grid():
    r = ref()
    g = torch.Generator().manual_seed(2)
    img = torch.zeros(1, 64, 64)
    img[:, 20:40, 25:35] = torch.rand(20, 10, generator=g)
    for target in (0.05 * 64 * 64, 0.197 * 64 * 64, 0.4 * 64 * 64):
        assert_equal(r.train._normalize_area_on_grid(img.clone(), target, 64, 64),
                     data_2d.normalize_area_on_grid(img.clone(), target, 64, 64), f"area {target}")
    assert data_2d.MNIST_TARGET_PR_FRAC == r.train.MNIST_TARGET_PR_FRAC


def test_pool_sampler():
    r = ref()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pool.pt")
        g = torch.Generator().manual_seed(4)
        pool = torch.rand(7, 1, 32, 32, generator=g) ** 3
        pool = pool / pool.sum(dim=(-2, -1), keepdim=True)
        torch.save(pool, path)
        devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devs:
            dev = torch.device(dev)
            r.train._chinese_dataset = None  # the original caches the first pool globally
            _check_same_draws(lambda: r.train.sample_density_pair_chinese(5, 32, 32, dev, data_path=path),
                              lambda: data_2d.sample_pool_pairs(5, 32, 32, dev, data_path=path),
                              seed=17, what=f"pool {dev}")
            args = SimpleNamespace(resolution=32, sampler="chinese", data_path=path)
            s_old = r.train2d.get_sampler(args, dev)
            for name in ("pool", "chinese"):
                s_new = data_2d.get_sampler(name, 32, data_path=path, device=dev)
                _check_same_draws(lambda: [s_old(4), s_old(1)], lambda: [s_new(4), s_new(1)], 3,
                                  f"get_sampler {name} {dev}")
        r.train._chinese_dataset = None
    for bad in (lambda: data_2d.get_sampler("pool", 32), lambda: data_2d.get_sampler("v2", 32)):
        try:
            bad()
            raise AssertionError("bad sampler accepted")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 5. Stroke preprocessing
# ---------------------------------------------------------------------------

def test_strokes_to_density():
    r = ref()
    gui = need(r.gui, "paint_chain_gui.py")
    from PIL import Image, ImageDraw
    R = gui.RES
    imgs = []
    img = Image.new("L", (R, R), 0)
    d = ImageDraw.Draw(img)
    d.line([(60, 40), (120, 200), (190, 90)], fill=255, width=18)
    d.ellipse([150, 150, 210, 230], outline=200, width=10)
    imgs.append(img)
    img = Image.new("L", (R, R), 0)  # small, off-centre, wide bbox
    ImageDraw.Draw(img).line([(5, 240), (80, 250)], fill=255, width=6)
    imgs.append(img)
    for i, img in enumerate(imgs):
        a = gui.strokes_to_density(img, torch.device("cpu"))
        assert_equal(a, data_2d.strokes_to_density(img, resolution=R), f"strokes {i} (PIL)")
        assert_equal(a, data_2d.strokes_to_density(np.asarray(img), resolution=R), f"strokes {i} (array)")
    empty = Image.new("L", (R, R), 0)
    assert gui.strokes_to_density(empty, torch.device("cpu")) is None
    assert data_2d.strokes_to_density(empty, resolution=R) is None


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def recorded_losses(store):
    orig = torch.Tensor.backward

    def backward(self, *a, **k):
        store.append(self.detach().clone())
        return orig(self, *a, **k)

    torch.Tensor.backward = backward
    try:
        yield
    finally:
        torch.Tensor.backward = orig


def _run_original_train(r, argv, seed, orig_advect=None):
    old_argv = sys.argv
    sys.argv = ["train_2d"] + argv
    try:
        args = r.train2d.parse_args()
    finally:
        sys.argv = old_argv
    losses = []
    saved_viz = sys.modules.get("fluidot.viz", "missing")
    sys.modules["fluidot.viz"] = None  # the original's optional plots are skipped (ImportError)
    try:
        with contextlib.ExitStack() as st:
            st.enter_context(patched(torch.cuda, "is_available", lambda: False))  # original picks cuda if present
            if orig_advect is not None:
                st.enter_context(patched(r.train2d, "advect", orig_advect))
            st.enter_context(recorded_losses(losses))
            torch.manual_seed(seed)
            r.train2d.train(args)
    finally:
        if saved_viz == "missing":
            del sys.modules["fluidot.viz"]
        else:
            sys.modules["fluidot.viz"] = saved_viz
    return losses


def _run_new_train(argv, seed):
    losses = []
    with recorded_losses(losses):
        train_2d.main(argv + ["--seed", str(seed), "--device", "cpu"])
    return losses


def _compare_run_dirs(d_old, d_new):
    files_old, files_new = sorted(os.listdir(d_old)), sorted(os.listdir(d_new))
    assert files_old == files_new, (files_old, files_new)
    for f in files_old:
        a = torch.load(os.path.join(d_old, f), map_location="cpu", weights_only=False)
        b = torch.load(os.path.join(d_new, f), map_location="cpu", weights_only=False)
        if f == "rollout_results.pt":
            a = {k: v for k, v in a.items() if not k.endswith("_inference_time_ms")}
            b = {k: v for k, v in b.items() if not k.endswith("_inference_time_ms")}
        assert_equal(a, b, f)
    return files_old


def _check_training(sampler_argv, scheme, n_steps, seed, orig_advect_attr=None, extra=()):
    r = ref()
    common = ["--resolution", "64", "--batch-size", "2", "--lr", "1e-4", "--n-steps", str(n_steps),
              "--n-rollout", "3", "--n-test", "2", "--n-infer-steps", "4", "--k-max", "0.25",
              "--arch", "fno", "--fno-width", "16", "--fno-modes", "8", "--fno-layers", "2",
              "--bb-visc-only", "--lambda-ke", "1.0", "--lambda-mu", "0.01", *extra]
    orig_advect = None
    if orig_advect_attr is not None:
        orig_advect = need(r.head, "fluidot_ops_git_HEAD.py").advect
    with tempfile.TemporaryDirectory() as tmp:
        d_old, d_new = os.path.join(tmp, "old"), os.path.join(tmp, "new")
        old_sampler, new_sampler = sampler_argv(tmp)
        r.train._chinese_dataset = None
        l_old = _run_original_train(r, common + old_sampler + ["--save-dir", d_old], seed, orig_advect)
        l_new = _run_new_train(common + new_sampler + ["--save-dir", d_new, "--advection", scheme], seed)
        r.train._chinese_dataset = None
        assert len(l_old) == len(l_new) == n_steps
        assert_equal(l_old, l_new, "per-step losses")
        files = _compare_run_dirs(d_old, d_new)
    return [float(x) for x in l_new], files


def _mnist_argv(tmp):
    r = ref()
    set_orig_mnist(r)
    return ["--sampler", "mnist"], ["--sampler", "mnist", "--mnist-root", mnist_root()]


def _pool_argv(tmp):
    path = os.path.join(tmp, "pool.pt")
    g = torch.Generator().manual_seed(8)
    pool = torch.rand(6, 1, 64, 64, generator=g) ** 4
    torch.save(pool / pool.sum(dim=(-2, -1), keepdim=True), path)
    return ["--sampler", "chinese", "--data-path", path], ["--sampler", "pool", "--data-path", path]


def test_training_mnist_maccormack_cpu():
    """exp50/51/70 path: 3 steps, identical per-step losses, final weights and eval rollout."""
    losses, files = _check_training(_mnist_argv, "maccormack", n_steps=3, seed=1234)
    print("  losses:", losses, "| files:", files)


def test_training_pool_semi_lagrangian_cpu():
    """exp31 path: original trainer with the git-HEAD first-order advect vs --advection semi_lagrangian."""
    losses, files = _check_training(_pool_argv, "semi_lagrangian", n_steps=3, seed=99,
                                    orig_advect_attr="head", extra=["--no-scheduler"])
    print("  losses:", losses, "| files:", files)


def test_training_checkpoint_schedule_cpu():
    """2000 steps: step1000/step2000 checkpoints, model_best rule, final weights (opt-in, slow)."""
    if os.environ.get("VIOT_PARITY_LONG") != "1":
        raise SkipTest("set VIOT_PARITY_LONG=1 to run the 2000-step training parity check")
    losses, files = _check_training(_pool_argv, "maccormack", n_steps=2000, seed=7,
                                    extra=["--log-every", "500"])
    assert "spectral_bb_visc_model_best.pt" in files and "spectral_bb_visc_model_step2000.pt" in files
    print("  final loss:", losses[-1], "| files:", files)


# ---------------------------------------------------------------------------
# Script mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    n_pass = n_skip = n_fail = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
            n_pass += 1
        except SkipTest as e:
            print(f"SKIP  {n}: {e}")
            n_skip += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL  {n}: {e}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
