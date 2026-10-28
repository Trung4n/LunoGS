# Synthetic depth-sensor degradation for the robustness experiment
# (docs/DECISIONS.md D72). The Replica depth rendered by Blender is exact;
# a real RGB-D sensor is noisy, loses pixels at depth edges and has a
# limited range. This module degrades the SENSOR depth (Camera.depth) the
# dataset hands to SLAM, nothing else: the depth loss, the Gaussian
# placement and every light-model use of the sensor depth see the same
# degraded image. It never reads ground truth beyond that depth image.
#
# Config, optional block (absent = the depth as read, like ParamSpec perturb;
# when present every key is required, no defaults):
#
#   Dataset:
#     depth_noise:
#       seed: <int>                 # noise of frame i drawn from rng(seed, i)
#       max_range_m: <m>            # depth beyond -> 0 (no reading)
#       axial_sigma: {a: <m>, b: <1/m>, z0: <m>}
#                                   # sigma(z) = a + b (z - z0)^2, Gaussian
#       edge: {window_px: <k>, rel_jump: <r>}
#                                   # pixel -> 0 if (max - min) of the clean
#                                   # depth over its (2k+1)^2 window > r * z
#       correlated: {cell_px: <n>, a: <m>, b: <1/m>, z0: <m>}      # OPTIONAL
#                                   # spatially correlated error: one Gaussian
#                                   # value per n x n cell, bilinearly
#                                   # interpolated (rescaled to unit variance),
#                                   # times sigma_c(z) = a + b (z - z0)^2; a
#                                   # smooth "wavy" depth bias inside a frame,
#                                   # which the depth loss cannot average out
#       dropout: {cell_px: <n>, frac: <p>}                       # OPTIONAL
#                                   # n x n blocks lost with probability p
#                                   # (holes like those on dark or shiny
#                                   # surfaces, not isolated pixels)
#
# The two optional parts were added for the heavier level (D75); absent = not
# applied, so the light level of D72 is unchanged (its white noise is drawn
# first from the same generator, the extra parts after it).
#
# The noise of a frame depends only on (seed, frame index), so every config
# of the experiment, and every re-read of the same frame (tracking, keyframe
# placement, rendering evaluation), sees the identical degraded depth.

import numpy as np

from utils.config_utils import require_key

_KEYS = {"seed", "max_range_m", "axial_sigma", "edge", "correlated", "dropout"}


def _number(v, where, positive=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{where} must be a number, got {v!r}")
    if v < 0 or (positive and v == 0):
        raise ValueError(f"{where} must be {'> 0' if positive else '>= 0'}, got {v!r}")
    return float(v)


def _block(parent, key, where, keys):
    b = require_key(parent, key, where)
    if not isinstance(b, dict):
        raise TypeError(f"{where}.{key} must be a block, got {b!r}")
    extra = set(b) - set(keys)
    if extra:
        raise ValueError(f"{where}.{key}: unexpected keys {sorted(extra)}")
    return b


def _cell(block, where):
    n = require_key(block, "cell_px", where)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{where}.cell_px must be an integer >= 1, got {n!r}")
    return n


def read_depth_noise(dataset_cfg):
    """Parsed Dataset.depth_noise as a plain dict, or None when absent."""
    if "depth_noise" not in dataset_cfg:
        return None
    where = "Dataset"
    block = _block(dataset_cfg, "depth_noise", where, _KEYS)
    where = "Dataset.depth_noise"
    seed = require_key(block, "seed", where)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{where}.seed must be an integer >= 0, got {seed!r}")
    sig = _block(block, "axial_sigma", where, ("a", "b", "z0"))
    edge = _block(block, "edge", where, ("window_px", "rel_jump"))
    window = require_key(edge, "window_px", f"{where}.edge")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"{where}.edge.window_px must be an integer >= 1, got {window!r}")
    out = {
        "seed": seed,
        "max_range_m": _number(require_key(block, "max_range_m", where), f"{where}.max_range_m", positive=True),
        "a": _number(require_key(sig, "a", f"{where}.axial_sigma"), f"{where}.axial_sigma.a"),
        "b": _number(require_key(sig, "b", f"{where}.axial_sigma"), f"{where}.axial_sigma.b"),
        "z0": _number(require_key(sig, "z0", f"{where}.axial_sigma"), f"{where}.axial_sigma.z0"),
        "window_px": window,
        "rel_jump": _number(require_key(edge, "rel_jump", f"{where}.edge"), f"{where}.edge.rel_jump",
                            positive=True),
        "correlated": None,
        "dropout": None,
    }
    if "correlated" in block:
        w = f"{where}.correlated"
        cor = _block(block, "correlated", where, ("cell_px", "a", "b", "z0"))
        out["correlated"] = {"cell_px": _cell(cor, w),
                             **{k: _number(require_key(cor, k, w), f"{w}.{k}") for k in ("a", "b", "z0")}}
    if "dropout" in block:
        w = f"{where}.dropout"
        dro = _block(block, "dropout", where, ("cell_px", "frac"))
        frac = _number(require_key(dro, "frac", w), f"{w}.frac")
        if frac >= 1:
            raise ValueError(f"{w}.frac must be < 1, got {frac!r}")
        out["dropout"] = {"cell_px": _cell(dro, w), "frac": frac}
    return out


def edge_mask(depth, window_px, rel_jump):
    """True where the depth jumps by more than rel_jump * z inside the
    (2k+1)^2 window (missing readings, 0, count as jumps: real sensors also
    lose the pixels next to holes)."""
    d = np.asarray(depth, dtype=np.float64)
    k = window_px
    pad = np.pad(d, k, mode="edge")
    h, w = d.shape
    hi = np.full_like(d, -np.inf)
    lo = np.full_like(d, np.inf)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            win = pad[dy:dy + h, dx:dx + w]
            np.maximum(hi, win, out=hi)
            np.minimum(lo, win, out=lo)
    return (hi - lo) > rel_jump * d


def correlated_field(rng, shape, cell_px):
    """Unit-variance Gaussian field, correlated over ~cell_px pixels: values on
    a coarse grid, bilinearly interpolated, divided by the interpolation's
    standard deviation so every pixel has variance 1."""
    h, w = shape
    gy, gx = h // cell_px + 2, w // cell_px + 2
    grid = rng.standard_normal((gy, gx))
    ty = np.arange(h) / cell_px
    tx = np.arange(w) / cell_px
    iy, fy = np.floor(ty).astype(int), ty - np.floor(ty)
    ix, fx = np.floor(tx).astype(int), tx - np.floor(tx)
    rows = grid[iy] * (1 - fy)[:, None] + grid[iy + 1] * fy[:, None]
    field = rows[:, ix] * (1 - fx)[None] + rows[:, ix + 1] * fx[None]
    std = np.sqrt(((1 - fy) ** 2 + fy ** 2)[:, None] * ((1 - fx) ** 2 + fx ** 2)[None])
    return field / std


def block_dropout(rng, shape, cell_px, frac):
    """True on the n x n blocks that are lost."""
    h, w = shape
    lost = rng.random((h // cell_px + 1, w // cell_px + 1)) < frac
    return lost[np.arange(h)[:, None] // cell_px, np.arange(w)[None] // cell_px]


def apply_depth_noise(depth, params, frame_idx):
    """Degraded copy of a metric depth image (H, W; 0 = no reading)."""
    clean = np.asarray(depth, dtype=np.float64)
    valid = clean > 0
    drop = ~valid | (clean > params["max_range_m"])
    drop |= edge_mask(clean, params["window_px"], params["rel_jump"])
    rng = np.random.default_rng([params["seed"], int(frame_idx)])
    sigma = params["a"] + params["b"] * (clean - params["z0"]) ** 2
    noisy = clean + rng.standard_normal(clean.shape) * sigma
    cor = params["correlated"]
    if cor is not None:
        sigma_c = cor["a"] + cor["b"] * (clean - cor["z0"]) ** 2
        noisy += correlated_field(rng, clean.shape, cor["cell_px"]) * sigma_c
    dro = params["dropout"]
    if dro is not None:
        drop |= block_dropout(rng, clean.shape, dro["cell_px"], dro["frac"])
    drop |= noisy <= 0
    noisy[drop] = 0.0
    return noisy
