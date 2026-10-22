# ParamSpec: the single format of every numeric light-model parameter
# (CLAUDE.md section 4.5):
#
#   <name>:
#     source: params_file | value
#     key: <dotted JSON path>     # required iff source = params_file
#     value: <number or list>     # required iff source = value
#     learnable: true | false
#     lr: <number>                # required iff learnable
#     prior_weight: <number>      # required iff learnable; 0 = no penalty
#     perturb: <block>            # optional, see below
#
# Nothing has a default. Keys that do not belong (e.g. lr on a fixed
# parameter, key with source: value) are errors as well, so a typo cannot
# silently change an ablation.
#
# The params file is the per-scene light_params.json (docs/DECISIONS.md D13,
# D17). Its keys are nested, hence dotted paths such as
# "t_CL_m.estimated_opencv".
#
# perturb (docs/DECISIONS.md D57): a deliberate calibration error applied to
# the value read above, for the sensitivity experiments. Exactly one of
#   {scale: s}                       value * s       (s > 0, number)
#   {add: a}                         value + a       (a: same shape as value)
#   {rotate_deg: g, axis: [x,y,z]}   Rot(axis, g) @ value   (3x3 value only)
# The nominal value still comes from the params file (no copied numbers in
# the YAML); the error is written in the config, hence recorded with the run.
# Absent = the parameter as read: that is not a default VALUE of the model.
# rotate_deg left-multiplies: for R_CL (light -> camera) it rotates the light
# axis a_L = R_CL e_z by g degrees about `axis` in the CAMERA frame.
#
# Parsing does not import torch; ParamSpec.tensor() does, lazily.

import json
import math
import numbers
import os
from dataclasses import dataclass

from utils.config_utils import require_key

_SOURCES = ("params_file", "value")


@dataclass(frozen=True)
class ParamSpec:
    name: str  # dotted config path, for messages
    value: object  # resolved number or nested list of numbers
    source: str
    key: object  # JSON path or None
    learnable: bool
    lr: object  # float or None
    prior_weight: object  # float or None
    nominal: object = None  # value before `perturb`, None if not perturbed
    perturb: object = None  # the perturb block, None if not perturbed

    def tensor(self):
        """CPU float32 tensor of the value (requires_grad iff learnable).
        Kept on CPU so shaders pickle into spawned processes; the shader
        moves it to the data's device when it runs."""
        import torch

        t = torch.tensor(self.value, dtype=torch.float32)
        return t.requires_grad_(self.learnable)


def params_file_path(config):
    """Light.params_file resolved against Dataset.dataset_path (D17)."""
    light = require_key(config, "Light", "")
    rel = require_key(light, "params_file", "Light")
    root = require_key(require_key(config, "Dataset", ""), "dataset_path", "Dataset")
    return os.path.join(root, rel)


def load_params_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"light params file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_dotted(data, key, where):
    node = data
    walked = []
    for part in key.split("."):
        walked.append(part)
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"{where}.key '{key}': '{'.'.join(walked)}' not found in the params file"
            )
        node = node[part]
    return node


def _check_numeric(value, where):
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{where}: expected a number or list of numbers, got {value!r}")
    if isinstance(value, numbers.Real):
        return
    if isinstance(value, (list, tuple)) and value:
        for v in value:
            _check_numeric(v, where)
        return
    raise TypeError(f"{where}: expected a number or list of numbers, got {value!r}")


def _shape(value):
    if isinstance(value, (list, tuple)):
        inner = {_shape(v) for v in value}
        if len(inner) != 1:
            return None  # ragged
        sub_shape = inner.pop()
        return None if sub_shape is None else (len(value),) + sub_shape
    return ()


def _elementwise(a, b, op):
    if isinstance(a, (list, tuple)):
        return [_elementwise(x, y, op) for x, y in zip(a, b)]
    return op(a, b)


def _rotation(axis, angle_deg):
    # Rodrigues: rotation of angle_deg about the unit vector `axis`.
    n = math.sqrt(sum(c * c for c in axis))
    x, y, z = (c / n for c in axis)
    a = math.radians(angle_deg)
    c, s, t = math.cos(a), math.sin(a), 1.0 - math.cos(a)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _is_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _apply_perturb(value, perturb, where):
    """Returns the perturbed value (plain nested lists / floats)."""
    w = f"{where}.perturb"
    if not isinstance(perturb, dict):
        raise TypeError(f"{w} must be a block, got {perturb!r}")
    ops = [k for k in ("scale", "add", "rotate_deg") if k in perturb]
    if len(ops) != 1:
        raise ValueError(f"{w} must hold exactly one of scale / add / rotate_deg, got {sorted(perturb)}")
    op = ops[0]
    allowed = {"rotate_deg", "axis"} if op == "rotate_deg" else {op}
    extra = set(perturb) - allowed
    if extra:
        raise ValueError(f"{w}: unexpected keys {sorted(extra)} for {op}")
    if op == "scale":
        s = perturb["scale"]
        if not _is_number(s) or s <= 0:
            raise ValueError(f"{w}.scale must be a number > 0, got {s!r}")
        return _elementwise(value, value, lambda v, _: v * s) if isinstance(value, list) else value * s
    if op == "add":
        a = perturb["add"]
        _check_numeric(a, f"{w}.add")
        if _shape(a) != _shape(value):
            raise ValueError(f"{w}.add must have the shape of the value {_shape(value)}, got {a!r}")
        return _elementwise(value, a, lambda v, d: v + d)
    g = perturb["rotate_deg"]
    axis = require_key(perturb, "axis", w)
    if not _is_number(g):
        raise ValueError(f"{w}.rotate_deg must be a number, got {g!r}")
    if _shape(axis) != (3,) or not all(_is_number(c) for c in axis) or not any(axis):
        raise ValueError(f"{w}.axis must be a non-zero 3-vector, got {axis!r}")
    if _shape(value) != (3, 3):
        raise ValueError(f"{w}.rotate_deg needs a 3x3 value, got shape {_shape(value)}")
    r = _rotation(axis, g)
    return [[sum(r[i][k] * value[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def resolve_param(block, where, params_data):
    """Parses one ParamSpec block. params_data: the loaded params file (dict),
    or None when no params file is available (then source: params_file fails)."""
    if not isinstance(block, dict):
        raise TypeError(f"{where} must be a ParamSpec block, got {block!r}")
    source = require_key(block, "source", where)
    if source not in _SOURCES:
        raise ValueError(f"{where}.source must be one of {_SOURCES}, got {source!r}")
    learnable = require_key(block, "learnable", where)
    if not isinstance(learnable, bool):
        raise TypeError(f"{where}.learnable must be true or false, got {learnable!r}")

    allowed = {"source", "learnable"}
    if source == "params_file":
        key = require_key(block, "key", where)
        if not isinstance(key, str) or not key:
            raise TypeError(f"{where}.key must be a non-empty string, got {key!r}")
        if params_data is None:
            raise ValueError(f"{where}: source params_file but no params file was loaded")
        value = lookup_dotted(params_data, key, where)
        allowed.add("key")
    else:
        key = None
        value = require_key(block, "value", where)
        allowed.add("value")

    lr = prior_weight = None
    if learnable:
        lr = require_key(block, "lr", where)
        prior_weight = require_key(block, "prior_weight", where)
        for name, v in (("lr", lr), ("prior_weight", prior_weight)):
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or v < 0:
                raise ValueError(f"{where}.{name} must be a number >= 0, got {v!r}")
        allowed.update(("lr", "prior_weight"))

    allowed.add("perturb")  # optional: a deliberate calibration error (D57)
    extra = set(block) - allowed
    if extra:
        raise ValueError(
            f"{where}: unexpected keys {sorted(extra)} "
            f"(source={source}, learnable={learnable} allows {sorted(allowed)})"
        )
    _check_numeric(value, where)
    if "perturb" not in block:
        return ParamSpec(where, value, source, key, learnable, lr, prior_weight)
    perturbed = _apply_perturb(value, block["perturb"], where)
    # Printed so the log of a sensitivity run shows the value actually used.
    from utils.logging_utils import Log

    Log(f"{where} perturbed by {block['perturb']}: {value} -> {perturbed}", tag="Light")
    return ParamSpec(where, perturbed, source, key, learnable, lr, prior_weight,
                     nominal=value, perturb=block["perturb"])
