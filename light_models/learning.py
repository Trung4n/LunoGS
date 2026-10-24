# Learnable light parameters: self-calibration of the light model during
# MAPPING (docs/DECISIONS.md D60). Never during tracking (CLAUDE.md sections
# 1 and 6): the frontend's shader copy has learning switched off and only
# receives the values the backend learned.
#
# Which parameters learn is declared per parameter (ParamSpec learnable, lr,
# prior_weight) and WHEN they start is declared by Light.schedule:
#
#   schedule:
#     - {after_keyframes: <n>, unfreeze: [<path>, ...]}
#
# A path names the parameter inside the Light block: "angular.half_angle_deg",
# "pose.R_CL", "pose.t_CL", "intensity", "ambient.c", "falloff.tau", ...
# A parameter is updated once the map holds MORE than after_keyframes
# keyframes (same convention as the lambert warm-up, D43). Every learnable
# parameter must appear in exactly one entry; a schedule without any
# learnable parameter is an error too (a stale ablation line).
#
# Prior: prior_weight * ||theta - theta_start||^2, theta_start = the value the
# run started from (the calibration, perturbed or not). 0 = no penalty, i.e.
# pure self-calibration.

from dataclasses import dataclass

import torch

from utils.config_utils import require_key


@dataclass
class LearnableParam:
    path: str
    tensor: torch.Tensor  # CPU leaf tensor the shader computes with
    start: torch.Tensor  # detached copy of the starting value (prior target)
    lr: float
    prior_weight: float
    after_keyframes: int = None  # from Light.schedule
    check: object = None  # zero-argument callable raising on an invalid value
    # Zero-argument callable restoring the parameter's constraints after a
    # step (projected gradient, e.g. the monotone spline, D67), or None.
    project: object = None
    # Zero-argument callable returning an extra penalty (tensor) of the
    # parameter, added while it is active (e.g. spline smoothness), or None.
    regularizer: object = None


def make_learnable(path, tensor, spec, check=None):
    """LearnableParam for a component tensor built from a learnable ParamSpec.
    The tensor must be the leaf created by ParamSpec.tensor(); a derived
    tensor (e.g. an RGB view) cannot be optimized in place."""
    if not (tensor.is_leaf and tensor.requires_grad):
        raise NotImplementedError(
            f"Light.{path}: learnable is not supported for this form or shape "
            "(the component does not keep the parameter as a leaf tensor)"
        )
    return LearnableParam(path, tensor, tensor.detach().clone(), float(spec.lr),
                          float(spec.prior_weight), check=check)


def read_schedule(light_cfg, learnable):
    """Fills after_keyframes of every LearnableParam (dict path -> param) from
    Light.schedule, with the checks described at the top of the module."""
    if not learnable:
        if "schedule" in light_cfg:
            raise ValueError("Light.schedule is given but no light parameter is learnable")
        return
    schedule = require_key(light_cfg, "schedule", "Light")
    if not isinstance(schedule, list) or not schedule:
        raise TypeError(f"Light.schedule must be a non-empty list, got {schedule!r}")
    seen = {}
    for i, entry in enumerate(schedule):
        where = f"Light.schedule[{i}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{where} must be a block, got {entry!r}")
        extra = set(entry) - {"after_keyframes", "unfreeze"}
        if extra:
            raise ValueError(f"{where}: unexpected keys {sorted(extra)}")
        after = require_key(entry, "after_keyframes", where)
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError(f"{where}.after_keyframes must be an integer >= 0, got {after!r}")
        paths = require_key(entry, "unfreeze", where)
        if not isinstance(paths, list) or not paths:
            raise TypeError(f"{where}.unfreeze must be a non-empty list, got {paths!r}")
        for path in paths:
            if path not in learnable:
                raise ValueError(
                    f"{where}.unfreeze: '{path}' is not a learnable light parameter "
                    f"(learnable: {sorted(learnable)})"
                )
            if path in seen:
                raise ValueError(f"Light.schedule: '{path}' is unfrozen twice")
            seen[path] = after
    missing = set(learnable) - set(seen)
    if missing:
        raise KeyError(f"Light.schedule: learnable parameters never unfrozen: {sorted(missing)}")
    for path, after in seen.items():
        learnable[path].after_keyframes = after


class LightLearner:
    """Adam over the shader's learnable light parameters, one group per
    parameter (its own lr). Used by BackEnd.map only."""

    def __init__(self, learnable):
        if not learnable:
            raise ValueError("LightLearner needs at least one learnable parameter")
        self.params = learnable
        self.optimizer = torch.optim.Adam(
            [{"params": [p.tensor], "lr": p.lr, "name": path} for path, p in learnable.items()]
        )

    def active(self, n_keyframes):
        return [path for path, p in self.params.items() if n_keyframes > p.after_keyframes]

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def prior_loss(self, n_keyframes):
        """Penalties of the active parameters: the prior (prior_weight > 0)
        and the parameter's own regularizer if any; None when there is none."""
        active = set(self.active(n_keyframes))
        terms = []
        for path, p in self.params.items():
            if path not in active:
                continue
            if p.prior_weight > 0:
                terms.append(p.prior_weight * ((p.tensor - p.start) ** 2).sum())
            if p.regularizer is not None:
                terms.append(p.regularizer())
        return sum(terms) if terms else None

    def step(self, n_keyframes):
        """One Adam step on the active parameters (frozen ones keep their
        value: their gradient is dropped first), then the domain checks."""
        active = set(self.active(n_keyframes))
        for path, p in self.params.items():
            if path not in active:
                p.tensor.grad = None
        if active:
            self.optimizer.step()
            for path in active:
                p = self.params[path]
                if p.project is not None:
                    with torch.no_grad():
                        p.project()
                if p.check is not None:
                    p.check()
        self.zero_grad()
