"""Shape regularizer of the Gaussians in mapping (Training.scale_regularizer,
docs/DECISIONS.md D149). Chosen by name in the config:

  {type: isotropic, weight: w}
      w * mean |s_i - mean(s)| over the 3 scales of every Gaussian (metres):
      pushes Gaussians toward spheres. The term of original MonoGS, which had
      10 written in the code; every base config states 10.
  {type: disc, in_plane_weight: a, flatness_weight: b}
      a * mean |s_max - s_mid| (metres, MonoGS's term on the two largest
      scales: no needles) + b * mean (s_min / s_mid), s_mid detached (only the
      smallest scale is pushed down, by its ratio to the in-plane size, so
      the term does not depend on the scene's scale): flat discs, whose
      shortest axis is a surface normal (normal source shortest_axis).
"""
import torch

from utils.config_utils import require_key

WHERE = "Training.scale_regularizer"


def _number(block, key):
    v = require_key(block, key, WHERE)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise ValueError(f"{WHERE}.{key} must be a number >= 0, got {v!r}")
    return float(v)


class Isotropic:
    name = "isotropic"
    keys = ("weight",)

    def __init__(self, block):
        self.weight = _number(block, "weight")

    def __call__(self, scaling):
        return self.weight * torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean()


class Disc:
    name = "disc"
    keys = ("in_plane_weight", "flatness_weight")

    def __init__(self, block):
        self.in_plane_weight = _number(block, "in_plane_weight")
        self.flatness_weight = _number(block, "flatness_weight")

    def __call__(self, scaling):
        s, _ = scaling.sort(dim=1)
        in_plane = torch.abs(s[:, 2] - s[:, 1]).mean()
        flat = (s[:, 0] / s[:, 1].detach()).mean()
        return self.in_plane_weight * in_plane + self.flatness_weight * flat


# Classes, not closures: the backend is pickled into its own process.
FORMS = {cls.name: cls for cls in (Isotropic, Disc)}


def read_scale_regularizer(config):
    """The regularizer of Training.scale_regularizer: reg(scaling (N,3)) ->
    scalar, reg.name the form."""
    block = require_key(config["Training"], "scale_regularizer", "Training")
    if not isinstance(block, dict):
        raise ValueError(f"{WHERE} must be a block {{type: ..., <weights>}}, got {block!r}")
    name = require_key(block, "type", WHERE)
    if name not in FORMS:
        raise ValueError(f"{WHERE}.type must be one of {sorted(FORMS)}, got {name!r}")
    cls = FORMS[name]
    extra = set(block) - set(cls.keys) - {"type"}
    if extra:
        raise ValueError(f"{WHERE}: unexpected keys {sorted(extra)} for type {name!r}")
    return cls(block)
