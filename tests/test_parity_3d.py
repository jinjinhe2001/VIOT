"""Parity of the 3D ``viot`` code with the original research code.

The reference is a copy of the original packages (``fluidot3d/``,
``fluidot_improved/``). Point ``VIOT_REF_SNAPSHOT`` at it; tests that need it
are skipped when it is absent. The checkpoint test needs one released 3D
checkpoint in its original ``.pt`` form (``VIOT_TEST_CKPT_3D``, e.g. the
sphere->airplane model ``spectral_bb_visc_3d_model_best.pt``) and a CUDA GPU
with ~8 GB free.

Everything deterministic is compared bitwise (``torch.equal``). The training
parity runs use the CPU, where PyTorch is deterministic; see
``test_train_amp_cuda_close`` for the (inherently nondeterministic) GPU path.
The last two tests are smoke tests of options the original trainer lacks
(gloo/CPU multi-process fallback, ``--advection maccormack``).

Run with ``python -m pytest tests/test_parity_3d.py -v`` or
``python tests/test_parity_3d.py``.
"""

import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True  # keep the repo and the reference snapshot free of __pycache__
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from viot import data_3d, model_3d, ops_3d, train_3d  # noqa: E402

# Defaults resolve to the release work area next to the repository checkout.
REF_DIR = Path(os.environ.get("VIOT_REF_SNAPSHOT", REPO.parent / "_work" / "gt_snapshot"))
CKPT_3D = Path(os.environ.get("VIOT_TEST_CKPT_3D",
                              REPO.parent / "_work" / "ckpt" / "exp49_airplane_best.pt"))

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref():
    """The original modules, imported from the reference snapshot."""
    if not (REF_DIR / "fluidot_improved" / "train_3d.py").is_file():
        pytest.skip(f"reference snapshot not found at {REF_DIR} (set VIOT_REF_SNAPSHOT)")
    if str(REF_DIR) not in sys.path:
        sys.path.insert(0, str(REF_DIR))
    return SimpleNamespace(
        ops3d=importlib.import_module("fluidot3d.ops"),
        data3d=importlib.import_module("fluidot3d.data"),
        ops_impr=importlib.import_module("fluidot_improved.ops"),
        model=importlib.import_module("fluidot_improved.model_3d"),
        train=importlib.import_module("fluidot_improved.train_3d"),
    )


def _grid(res):
    c = (torch.arange(res, dtype=torch.float32) + 0.5) / res * 2 - 1
    return torch.meshgrid(c, c, c, indexing="ij")


def make_pool(n, res, seed=0):
    """Synthetic voxel pool [n, 1, res, res, res] exercising every sampler branch:
    peaks != 1, negative values (clamp), shapes touching the border (shrink loop)
    and one empty volume."""
    g = torch.Generator().manual_seed(seed)
    z, y, x = _grid(res)
    shapes = []
    for i in range(n):
        kind = i % 6
        if kind == 0:
            s = ((x ** 2 + y ** 2 + z ** 2) < 0.45 ** 2).float()
        elif kind == 1:
            cx, cy, cz = ((torch.rand(3, generator=g) - 0.5) * 0.6).tolist()
            s = (((x - cx) / 0.6) ** 2 + ((y - cy) / 0.3) ** 2 + ((z - cz) / 0.4) ** 2 < 1).float() * 2.5
        elif kind == 2:
            s = ((x.abs() < 0.97) & (y.abs() < 0.2)).float()  # slab reaching the border
        elif kind == 3:
            s = torch.randn(res, res, res, generator=g) * 0.3 + ((x ** 2 + y ** 2 + z ** 2) < 0.25).float()
        elif kind == 4:
            s = torch.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2 + z ** 2) / 0.05) * 3.7
        else:
            s = torch.zeros(res, res, res) if i == 5 else ((x + y + z).abs() < 0.3).float()
        shapes.append(s)
    return torch.stack(shapes).unsqueeze(1)


def rand_density(B, res, seed, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    rho = torch.rand(B, 1, res, res, res, generator=g) ** 4
    return (rho / rho.amax(dim=(-3, -2, -1), keepdim=True)).to(device)


def rand_velocity(B, res, seed, scale=3.0, device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(B, 3, res, res, res, generator=g) * scale
    return v.to(device=device, dtype=dtype)


def small_model_pair(ref, res=32, seed=0, **kw):
    """(reference, new) models with identical random init and a non-zero output layer."""
    cfg = dict(max_res=res, k_max=0.25, width=8, n_modes=4, n_layers=2, time_dim=32)
    cfg.update(kw)
    torch.manual_seed(seed)
    m_ref = ref.model.FNOHybridSpectralFlowMatcher3D(**cfg)
    torch.manual_seed(seed)
    m_new = model_3d.FNO3D(**cfg)
    # the last conv is zero-initialised; give both models the same non-zero weights
    g = torch.Generator().manual_seed(seed + 1)
    w = torch.randn(m_new.project[-1].weight.shape, generator=g) * 50.0
    with torch.no_grad():
        for m in (m_ref, m_new):
            m.project[-1].weight.copy_(w)
    return m_ref, m_new


def assert_state_dicts_equal(a, b):
    assert list(a.keys()) == list(b.keys())
    for k in a:
        assert torch.equal(a[k], b[k]), f"state_dict mismatch at {k}"


def assert_nested_equal(a, b, path="", skip=()):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            if any(k.endswith(s) for s in skip):
                continue
            assert_nested_equal(a[k], b[k], f"{path}/{k}", skip)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_nested_equal(x, y, f"{path}[{i}]", skip)
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, f"{path}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
        assert torch.equal(a, b), f"{path}: max |diff| = {(a.double() - b.double()).abs().max().item():.3e}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# ---------------------------------------------------------------------------
# (1) Model: initialisation, forward, checkpoint loading
# ---------------------------------------------------------------------------

def test_spectral_helpers_equal(ref):
    for D, H, W in [(32, 32, 32), (12, 10, 9)]:
        a = ref.ops_impr._wavenumber_grid_3d(D, H, W, "cpu", torch.float32)
        b = model_3d.wavenumber_grid_3d(D, H, W, "cpu", torch.float32)
        for x, y in zip(a, b):
            assert torch.equal(x, y)
        g = torch.Generator().manual_seed(D)
        A_hat = torch.randn(2, 3, D, H, W // 2 + 1, dtype=torch.complex64, generator=g)
        assert torch.equal(ref.ops_impr.spectral_curl_3d(A_hat, *a[:3]),
                           model_3d.spectral_curl_3d(A_hat, *b[:3]))


def test_model_init_and_forward_cpu(ref):
    """Same seed -> identical parameters (same RNG order); identical forward at
    the training resolution (precomputed mask) and at another one (mask on the fly)."""
    cfg = dict(max_res=32, k_max=0.25, width=8, n_modes=4, n_layers=2, time_dim=32)
    torch.manual_seed(1234)
    m_ref = ref.model.FNOHybridSpectralFlowMatcher3D(**cfg)
    torch.manual_seed(1234)
    m_new = model_3d.FNO3D(**cfg)
    assert_state_dicts_equal(m_ref.state_dict(), m_new.state_dict())
    assert model_3d.FNOHybridSpectralFlowMatcher3D is model_3d.FNO3D

    # RNG consumption of the constructor is identical
    torch.manual_seed(7); ref.model.FNOHybridSpectralFlowMatcher3D(**cfg); r1 = torch.rand(4)
    torch.manual_seed(7); model_3d.FNO3D(**cfg); r2 = torch.rand(4)
    assert torch.equal(r1, r2)

    m_ref, m_new = small_model_pair(ref)
    assert_state_dicts_equal(m_ref.state_dict(), m_new.state_dict())
    for res in (32, 16):
        rho_t, rho_1 = rand_density(2, res, 1), rand_density(2, res, 2)
        t = torch.tensor([0.0, 0.37])
        with torch.no_grad():
            v_a, A_a = m_ref(rho_t, t, rho_1)
            v_b, A_b = m_new(rho_t, t, rho_1)
            assert v_a.abs().max() > 0
            assert torch.equal(v_a, v_b) and torch.equal(A_a, A_b)
            assert torch.equal(m_ref.forward_velocity_only(rho_t, t, rho_1),
                               m_new.forward_velocity_only(rho_t, t, rho_1))

    # gradients
    rho_t, rho_1 = rand_density(2, 32, 3), rand_density(2, 32, 4)
    t = torch.tensor([0.1, 0.9])
    for m in (m_ref, m_new):
        m.zero_grad()
        (m.forward_velocity_only(rho_t, t, rho_1) ** 2).mean().backward()
    for (k, p), (_, q) in zip(m_ref.named_parameters(), m_new.named_parameters()):
        assert torch.equal(p.grad, q.grad), k


@pytest.fixture(scope="module")
def exp49_state_dict():
    if not CKPT_3D.is_file():
        pytest.skip(f"3D checkpoint not found at {CKPT_3D} (set VIOT_TEST_CKPT_3D)")
    return torch.load(CKPT_3D, map_location="cpu", weights_only=True)


EXP49_ARCH = dict(max_res=128, k_max=0.25, width=32, n_modes=16, n_layers=6, time_dim=128)


def _sphere_and_plane(res, device):
    z, y, x = _grid(res)
    sphere = ((x ** 2 + y ** 2 + z ** 2) < 0.4 ** 2).float()
    plane = (((x / 0.8) ** 2 + (y / 0.12) ** 2 + (z / 0.2) ** 2 < 1)
             | ((x.abs() < 0.15) & (y.abs() < 0.05) & (z.abs() < 0.7))).float()
    return sphere[None, None].to(device), plane[None, None].to(device)


@needs_cuda
def test_checkpoint_strict_load_and_forward_cuda(ref, exp49_state_dict):
    sd = exp49_state_dict
    m_new = model_3d.FNO3D(**EXP49_ARCH)
    m_new.load_state_dict(sd, strict=True)
    assert sum(p.numel() for p in m_new.parameters()) == 201_450_019
    # the stored truncation mask is exactly the one this config builds
    assert torch.equal(sd["trunc_mask"], model_3d.FNO3D(**EXP49_ARCH).trunc_mask)
    m_ref = ref.model.FNOHybridSpectralFlowMatcher3D(**EXP49_ARCH)
    m_ref.load_state_dict(sd, strict=True)
    m_new, m_ref = m_new.cuda().eval(), m_ref.cuda().eval()

    rho_0, rho_1 = _sphere_and_plane(128, "cuda")
    t = torch.tensor([0.3], device="cuda")
    with torch.no_grad():
        v_a, A_a = m_ref(rho_0, t, rho_1)
        v_b, A_b = m_new(rho_0, t, rho_1)
        assert v_a.abs().max() > 1e-3
        assert torch.equal(v_a, v_b) and torch.equal(A_a, A_b)
        with torch.autocast("cuda"):
            v_a16 = m_ref.forward_velocity_only(rho_0, t, rho_1)
            v_b16 = m_new.forward_velocity_only(rho_0, t, rho_1)
        assert v_a16.dtype == torch.float16
        assert torch.equal(v_a16, v_b16)

        # (3) full 50-step evaluation rollout with the real checkpoint
        out_ref = ref.train.rollout(m_ref, rho_0, rho_1, n_steps=50)
        out_new = ops_3d.rollout_3d(m_new, rho_0, rho_1, n_steps=50,
                                    return_frames=True, return_velocities=True)
    assert_nested_equal(out_ref["frames"], out_new["frames"])
    assert_nested_equal(out_ref["velocities"], out_new["velocities"])
    assert torch.equal(out_new["final"], out_ref["frames"][-1])
    # the transport actually moved mass toward the target
    err0 = ((rho_0 - rho_1) ** 2).sum().sqrt()
    err1 = ((out_new["final"] - rho_1) ** 2).sum().sqrt()
    assert err1 < err0
    del m_new, m_ref
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# (2) Advection and diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if HAS_CUDA else []))
def test_advection_equal(ref, device):
    for shape, dt in [((2, 1, 24, 20, 16), 0.1), ((1, 1, 32, 32, 32), 0.02)]:
        B, _, D, H, W = shape
        g = torch.Generator().manual_seed(D)
        rho = torch.rand(shape, generator=g).to(device)
        v = (torch.randn(B, 3, D, H, W, generator=g) * 5.0).to(device)

        sl_ref = ref.ops3d.advect_3d(rho, v, dt)
        assert torch.equal(sl_ref, ops_3d.advect_semi_lagrangian_3d(rho, v, dt))
        assert torch.equal(sl_ref, ops_3d.advect_3d(rho, v, dt))
        assert torch.equal(sl_ref, ops_3d.advect_3d(rho, v, dt, scheme="semi_lagrangian"))

        mc_ref = ref.ops_impr.advect_3d(rho, v, dt)
        assert torch.equal(mc_ref, ops_3d.advect_maccormack_3d(rho, v, dt))
        assert torch.equal(mc_ref, ops_3d.advect_3d(rho, v, dt, scheme="maccormack"))
        assert not torch.equal(sl_ref, mc_ref)

        # AMP-like inputs of the training loop: fp32 density, fp16 velocity
        v16 = v.half()
        assert torch.equal(ref.ops3d.advect_3d(rho, v16, dt), ops_3d.advect_3d(rho, v16, dt))

        assert torch.equal(ref.ops3d.vorticity_3d(v), ops_3d.vorticity_3d(v))
        assert torch.equal(ref.ops3d.divergence_3d(v), ops_3d.divergence_3d(v))

    with pytest.raises(ValueError):
        ops_3d.advect_3d(rho, v, 0.1, scheme="weno")


def test_advection_gradients_equal(ref):
    g = torch.Generator().manual_seed(0)
    rho = torch.rand(2, 1, 16, 16, 16, generator=g)
    v = torch.randn(2, 3, 16, 16, 16, generator=g) * 3.0
    grads = []
    for fn in (ref.ops3d.advect_3d, ops_3d.advect_semi_lagrangian_3d):
        r, u = rho.clone().requires_grad_(), v.clone().requires_grad_()
        (fn(r, u, 0.1) ** 2).sum().backward()
        grads.append((r.grad, u.grad))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


# ---------------------------------------------------------------------------
# (3) Rollout
# ---------------------------------------------------------------------------

def test_rollout_equal_cpu(ref, monkeypatch):
    m_ref, m_new = small_model_pair(ref)
    rho_0, rho_1 = rand_density(2, 32, 5), rand_density(2, 32, 6)

    out_ref = ref.train.rollout(m_ref, rho_0, rho_1, n_steps=8)
    out_new = ops_3d.rollout_3d(m_new, rho_0, rho_1, n_steps=8,
                                return_frames=True, return_velocities=True)
    assert_nested_equal(out_ref["frames"], out_new["frames"])
    assert_nested_equal(out_ref["velocities"], out_new["velocities"])
    assert out_ref["velocities"][-1].abs().max() > 0
    final = ops_3d.rollout_3d(m_new, rho_0, rho_1, n_steps=8)
    assert torch.equal(final, out_ref["frames"][-1])
    cpu_out = ops_3d.rollout_3d(m_new, rho_0, rho_1, n_steps=8, return_frames=True,
                                store_device="cpu")
    assert_nested_equal(out_ref["frames"], cpu_out["frames"])

    # MacCormack rollout == the original rollout loop driven by the MacCormack advect
    monkeypatch.setattr(ref.train, "advect_3d", ref.ops_impr.advect_3d)
    out_ref_mc = ref.train.rollout(m_ref, rho_0, rho_1, n_steps=8)
    monkeypatch.undo()
    out_new_mc = ops_3d.rollout_3d(m_new, rho_0, rho_1, n_steps=8, scheme="maccormack",
                                   return_frames=True)
    assert_nested_equal(out_ref_mc["frames"], out_new_mc["frames"])
    assert not torch.equal(out_new_mc["final"], out_new["final"])

    # mass_renorm=False is the clamp-only update of the peak-normalised training rollout
    rho = rho_0.clone()
    with torch.no_grad():
        for s in range(8):
            t = torch.full((2,), s / 8, dtype=rho.dtype)
            rho = ref.ops3d.advect_3d(rho, m_ref.forward_velocity_only(rho, t, rho_1), 1.0 / 8).clamp(min=0)
    assert torch.equal(rho, ops_3d.rollout_3d(m_new, rho_0, rho_1, n_steps=8, mass_renorm=False))


# ---------------------------------------------------------------------------
# (4) Data: loading, pair sampling, augmentation
# ---------------------------------------------------------------------------

def test_ensure_within_domain_equal(ref):
    rho = make_pool(6, 32).clamp(min=0)
    a = ref.train._ensure_within_domain(rho.clone())
    b = data_3d.ensure_within_domain(rho.clone())
    assert torch.equal(a, b)
    # the border-touching slab (index 2) was shrunk; the sphere (index 0) was not
    plain = rho / (rho.sum(dim=(-3, -2, -1), keepdim=True) + 1e-12)
    assert not torch.equal(b[2], plain[2]) and torch.equal(b[0], plain[0])


def test_load_voxels_equal(ref, tmp_path):
    pool = make_pool(6, 32)
    torch.save(pool, tmp_path / "pool5d.pt")
    torch.save(pool[:, 0], tmp_path / "pool4d.pt")
    for name in ("pool5d.pt", "pool4d.pt"):
        a = ref.data3d.load_shapenet_voxels(str(tmp_path / name))
        b = data_3d.load_voxels(str(tmp_path / name))
        assert b.shape == (6, 1, 32, 32, 32) and torch.equal(a, b)


@pytest.mark.parametrize("peak_norm", [True, False])
def test_sample_voxel_pairs_equal(ref, peak_norm):
    src, tgt = make_pool(6, 32, seed=0), make_pool(5, 32, seed=1)
    for seed in range(6):
        for s, t in [(src, tgt), (src, src)]:
            torch.manual_seed(seed)
            a0, a1 = ref.train._sample_shapenet_safe(s, t, 4, "cpu", raw=True, peak_norm=peak_norm)
            ra = torch.rand(3)
            torch.manual_seed(seed)
            b0, b1 = data_3d.sample_voxel_pairs(s, t, 4, "cpu", peak_norm=peak_norm)
            rb = torch.rand(3)
            assert torch.equal(a0, b0) and torch.equal(a1, b1)
            assert torch.equal(ra, rb), "different RNG consumption"
    if peak_norm:
        nz = b0.amax(dim=(-3, -2, -1)) > 0
        assert torch.allclose(b0.amax(dim=(-3, -2, -1))[nz], torch.ones(1))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if HAS_CUDA else []))
def test_sampler_plus_augment_equal(ref, device):
    """The trainer's per-step data path: sample, then augment in place."""
    src = make_pool(6, 32, seed=2)
    torch.manual_seed(42)
    ref_batches = []
    for _ in range(5):
        r0, r1 = ref.train._sample_shapenet_safe(src, src, 3, device, raw=True, peak_norm=True)
        r0, r1, d = ref.data3d.augment_batch_3d(r0, r1, None)
        assert d is None
        ref_batches.append((r0, r1))
    ra = torch.rand(2)
    torch.manual_seed(42)
    for r0, r1 in ref_batches:
        n0, n1 = data_3d.sample_voxel_pairs(src, src, 3, device, peak_norm=True)
        n0, n1 = data_3d.augment_batch_3d(n0, n1)
        assert torch.equal(r0, n0) and torch.equal(r1, n1)
    assert torch.equal(ra, torch.rand(2))


# ---------------------------------------------------------------------------
# (5) Training: ORIGINAL vs NEW trainer, end to end
# ---------------------------------------------------------------------------

def test_cli_flags_and_defaults_match(ref, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_3d"])
    old = vars(ref.train.parse_args())
    new = vars(train_3d.parse_args([]))
    dropped = {"bb_only", "all", "lambda_ke_warmup", "lambda_mass"}
    assert set(old) - set(new) == dropped
    assert set(new) - set(old) == {"advection"} and new["advection"] == "semi_lagrangian"
    for k in set(old) - dropped:
        assert old[k] == new[k], k


_KEY_LINE = re.compile(r"^\s+(step\s+\d+/|\[ckpt\]|Applied init-scale|Parameters:|Missing keys|"
                       r"Unexpected keys|Loaded \d+ params|Loaded spectral|Loaded \d+ (source|target))|"
                       r"^(Terminal Error|Mass Conservation|Mean \|div|Mean Enstrophy|Kinetic Energy)")


def _key_lines(stdout):
    """Lines that must match between the trainers (timings stripped)."""
    out = []
    for line in stdout.splitlines():
        if _KEY_LINE.search(line):
            out.append(re.sub(r" \| [\d.]+s$", "", line.rstrip()))
    return out


def _env(pythonpath, cuda=False, **extra):
    env = dict(os.environ)
    env.update(KMP_DUPLICATE_LIB_OK="TRUE", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(pythonpath), PYTHONIOENCODING="utf-8")
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = "-1"  # (an empty value is dropped on Windows)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _cmd(module, args):
    return [sys.executable, "-B", "-m", module] + [str(a) for a in args]


def _run(module, pythonpath, args, cwd, cuda=False, **extra_env):
    proc = subprocess.run(_cmd(module, args), cwd=cwd, env=_env(pythonpath, cuda, **extra_env),
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=1800)
    assert proc.returncode == 0, f"{module} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout


def _run_both(args, tmp_path, tag, cuda=False, extra_new=()):
    d_ref, d_new = tmp_path / f"{tag}_ref", tmp_path / f"{tag}_new"
    out_ref = _run("fluidot_improved.train_3d", REF_DIR, list(args) + ["--save-dir", d_ref], tmp_path, cuda)
    out_new = _run("viot.train_3d", REPO, list(args) + list(extra_new) + ["--save-dir", d_new], tmp_path, cuda)
    return d_ref, d_new, out_ref, out_new


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=True)


def _assert_runs_equal(d_ref, d_new, out_ref, out_new, files):
    lines_ref, lines_new = _key_lines(out_ref), _key_lines(out_new)
    assert any("step" in l for l in lines_ref) or "Eval-only" in out_ref
    assert any(l.startswith("Terminal Error") for l in lines_ref), out_ref[-2000:]
    assert lines_ref == lines_new
    for f in files:
        assert (d_new / f).is_file(), f
        assert_state_dicts_equal(_load(d_ref / f), _load(d_new / f))
    assert_nested_equal(_load(d_ref / "rollout_results_3d.pt"), _load(d_new / "rollout_results_3d.pt"),
                        skip=("_inference_time_ms",))


COMMON = ["--sampler", "shapenet", "--n-test", 2, "--n-infer-steps", 4, "--k-max", 0.25,
          "--bb-visc-only", "--lambda-terminal", 10, "--grad-clip", 1.0, "--raw-sampler"]
TINY = ["--resolution", 32, "--fno-width", 8, "--fno-modes", 4, "--fno-layers", 2,
        "--batch-size", 2, "--n-rollout", 3, "--log-every", 1]


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    d = tmp_path_factory.mktemp("pools")
    torch.save(make_pool(6, 32, seed=0), d / "src32.pt")
    torch.save(make_pool(5, 32, seed=1), d / "tgt32.pt")
    torch.save(make_pool(6, 8, seed=2), d / "pool8.pt")
    return d


def test_train_fresh_then_continue_then_eval_cpu(ref, pools, tmp_path):
    """exp49-style fresh run, exp59-style continuation, and --eval-only; bitwise on the CPU."""
    fresh = COMMON + TINY + [
        "--data-path-src", pools / "src32.pt", "--data-path-tgt", pools / "tgt32.pt",
        "--lr", 1e-3, "--n-steps", 3, "--lambda-ke", 0.1, "--lambda-mu", 0.003,
        "--lambda-mu-warmup", 2, "--amp", "--checkpoint", "--no-scheduler",
        "--peak-norm", "--init-scale", 10.0]
    d_ref, d_new, o_ref, o_new = _run_both(fresh, tmp_path, "fresh")
    assert "Applied init-scale" in o_new
    _assert_runs_equal(d_ref, d_new, o_ref, o_new, ["spectral_bb_visc_3d_model.pt"])

    # continuation from the fresh final weights: cosine schedule, no checkpointing,
    # sum-normalised densities (rho_scale = R^3, per-step mass renormalisation)
    init = d_ref / "spectral_bb_visc_3d_model.pt"
    cont = COMMON + TINY + [
        "--data-path-src", pools / "src32.pt", "--data-path-tgt", pools / "src32.pt",
        "--lr", 5e-4, "--n-steps", 3, "--lambda-ke", 0, "--lambda-mu", 0.005,
        "--lambda-mu-warmup", 0, "--init-scale", 10.0, "--init-from", init]
    d_ref2, d_new2, o_ref2, o_new2 = _run_both(cont, tmp_path, "cont")
    assert "Missing keys (will keep model defaults): ['trunc_mask']" in o_new2
    assert "Applied init-scale" not in o_new2
    _assert_runs_equal(d_ref2, d_new2, o_ref2, o_new2, ["spectral_bb_visc_3d_model.pt"])

    # --eval-only (no *_best.pt here, so both load the final weights)
    ev = COMMON + TINY + ["--data-path-src", pools / "src32.pt", "--data-path-tgt", pools / "tgt32.pt",
                          "--peak-norm", "--eval-only"]
    for d in ("ev_ref", "ev_new"):
        (tmp_path / d).mkdir()
        shutil.copy(init, tmp_path / d / "spectral_bb_visc_3d_model.pt")
    o_ref3 = _run("fluidot_improved.train_3d", REF_DIR, ev + ["--save-dir", tmp_path / "ev_ref"], tmp_path)
    o_new3 = _run("viot.train_3d", REPO, ev + ["--save-dir", tmp_path / "ev_new"], tmp_path)
    assert "params, final)" in o_new3
    _assert_runs_equal(tmp_path / "ev_ref", tmp_path / "ev_new", o_ref3, o_new3, [])


def test_train_checkpoints_and_best_cpu(ref, pools, tmp_path):
    """2000 tiny steps: step1000/step2000/best/final files and best-selection logic."""
    args = COMMON + [
        "--resolution", 8, "--fno-width", 4, "--fno-modes", 2, "--fno-layers", 1,
        "--batch-size", 2, "--n-rollout", 1, "--log-every", 250,
        "--data-path-src", pools / "pool8.pt", "--data-path-tgt", pools / "pool8.pt",
        "--lr", 3e-3, "--n-steps", 2000, "--lambda-ke", 0, "--lambda-mu", 0.003,
        "--no-scheduler", "--peak-norm", "--init-scale", 10.0]
    d_ref, d_new, o_ref, o_new = _run_both(args, tmp_path, "ckpt")
    files = ["spectral_bb_visc_3d_model_step1000.pt", "spectral_bb_visc_3d_model_step2000.pt",
             "spectral_bb_visc_3d_model_best.pt", "spectral_bb_visc_3d_model.pt"]
    ckpt_lines = [l for l in _key_lines(o_new) if "[ckpt]" in l]
    assert len(ckpt_lines) == 2
    assert "saved best model" in ckpt_lines[0] and "saved checkpoint (best=" in ckpt_lines[1]
    _assert_runs_equal(d_ref, d_new, o_ref, o_new, files)

    # --eval-only prefers *_best.pt
    ev = COMMON + ["--resolution", 8, "--fno-width", 4, "--fno-modes", 2, "--fno-layers", 1,
                   "--data-path-src", pools / "pool8.pt", "--data-path-tgt", pools / "pool8.pt",
                   "--peak-norm", "--eval-only"]
    o_ref2 = _run("fluidot_improved.train_3d", REF_DIR, ev + ["--save-dir", d_ref], tmp_path)
    o_new2 = _run("viot.train_3d", REPO, ev + ["--save-dir", d_new], tmp_path)
    assert "params, best)" in o_new2
    _assert_runs_equal(d_ref, d_new, o_ref2, o_new2, [])


@needs_cuda
def test_train_amp_cuda_close(ref, pools, tmp_path):
    """The released recipe path (CUDA + --amp + --checkpoint).

    Not bitwise: the CUDA backward of ``grid_sample`` accumulates with atomic
    adds, so gradients (and everything after the first optimizer step) vary in
    the last bits from run to run, for the original code as much as for this
    one. The first step's loss (before any update) must match exactly; later
    values must agree to a tight tolerance.
    """
    args = COMMON + TINY + [
        "--data-path-src", pools / "src32.pt", "--data-path-tgt", pools / "tgt32.pt",
        "--lr", 1e-4, "--n-steps", 3, "--lambda-ke", 0, "--lambda-mu", 0.003,
        "--lambda-mu-warmup", 2, "--amp", "--checkpoint", "--no-scheduler",
        "--peak-norm", "--init-scale", 10.0]
    d_ref, d_new, o_ref, o_new = _run_both(args, tmp_path, "amp", cuda=True)
    assert "Training on cuda" in o_new
    steps_ref = [l for l in _key_lines(o_ref) if "step" in l]
    steps_new = [l for l in _key_lines(o_new) if "step" in l]
    assert len(steps_ref) == len(steps_new) == 3
    assert steps_ref[0] == steps_new[0]
    num = re.compile(r"(loss|term|ke|enst)=([-+0-9.e]+)")
    for a, b in zip(steps_ref, steps_new):
        for (k, x), (_, y) in zip(num.findall(a), num.findall(b)):
            # 1e-3 relative, plus one unit of the last printed digit (rounding)
            mant, _, exp = x.partition("e")
            unit = 10.0 ** (int(exp or 0) - len(mant.partition(".")[2]))
            assert abs(float(x) - float(y)) <= 1e-3 * abs(float(x)) + unit, (k, a, b)
    sd_ref = _load(d_ref / "spectral_bb_visc_3d_model.pt")
    sd_new = _load(d_new / "spectral_bb_visc_3d_model.pt")
    for k in sd_ref:
        # Adam moves each weight by at most ~lr per step
        assert torch.allclose(sd_ref[k], sd_new[k], rtol=0, atol=3 * 2 * 1e-4), k


# ---------------------------------------------------------------------------
# New-only options (no counterpart in the original trainer)
# ---------------------------------------------------------------------------

SMOKE = COMMON + TINY + ["--lr", 1e-3, "--n-steps", 3, "--lambda-ke", 0, "--lambda-mu", 0.003,
                         "--no-scheduler", "--peak-norm", "--checkpoint"]


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_distributed_two_ranks_cpu(pools, tmp_path):
    """Two ranks with the environment torchrun sets (gloo backend on the CPU).

    Rank 0 starts from the same weights and batch as a single-process run, so
    its first (pre-update) loss is identical; afterwards the averaged gradients
    of both ranks' batches make the runs differ. Only rank 0 writes files.
    (Launched by hand because torchrun's rendezvous needs libuv, which some
    Windows builds of PyTorch lack.)
    """
    data = ["--data-path-src", pools / "src32.pt", "--data-path-tgt", pools / "tgt32.pt"]
    single = _run("viot.train_3d", REPO, SMOKE + data + ["--save-dir", tmp_path / "single"], tmp_path)

    args = SMOKE + data + ["--save-dir", tmp_path / "ddp"]
    dist_env = dict(MASTER_ADDR="127.0.0.1", MASTER_PORT=_free_port(), WORLD_SIZE=2,
                    USE_LIBUV=0, OMP_NUM_THREADS=1)
    p1 = subprocess.Popen(_cmd("viot.train_3d", args), cwd=tmp_path,
                          env=_env(REPO, RANK=1, LOCAL_RANK=1, **dist_env),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        out0 = _run("viot.train_3d", REPO, args, tmp_path, RANK=0, LOCAL_RANK=0, **dist_env)
        assert p1.wait(timeout=600) == 0, p1.stdout.read().decode(errors="replace")[-3000:]
    finally:
        if p1.poll() is None:
            p1.kill()
    assert "world_size=2 | distributed=True" in out0 and "All done!" in out0
    steps_1 = [l for l in _key_lines(single) if "step" in l]
    steps_2 = [l for l in _key_lines(out0) if "step" in l]
    assert steps_1[0] == steps_2[0] and steps_1[1:] != steps_2[1:]
    assert sorted(p.name for p in (tmp_path / "ddp").iterdir()) == \
        ["rollout_results_3d.pt", "spectral_bb_visc_3d_model.pt"]


def test_train_maccormack_option_cpu(pools, tmp_path):
    data = ["--data-path-src", pools / "src32.pt", "--data-path-tgt", pools / "tgt32.pt"]
    sl = _run("viot.train_3d", REPO, SMOKE + data + ["--save-dir", tmp_path / "sl"], tmp_path)
    mc = _run("viot.train_3d", REPO, SMOKE + data + ["--advection", "maccormack",
                                                     "--save-dir", tmp_path / "mc"], tmp_path)
    assert "advection=maccormack" in mc and "advection=semi_lagrangian" in sl
    a = _load(tmp_path / "sl" / "spectral_bb_visc_3d_model.pt")
    b = _load(tmp_path / "mc" / "spectral_bb_visc_3d_model.pt")
    assert any(not torch.equal(a[k], b[k]) for k in a)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
