# Cosine(n, l): the foreshortening term of the direct light.
# Each form returns (factor (H,W), aux) where aux holds per-pixel maps that
# the loss may use (pixel_weight.min_cos_nl, docs/DECISIONS.md D15).

import torch

from light_models.registry import build_component, declare_planned, register


@register("cosine", "none")
class NoCosine:
    """Factor 1: the direct term ignores surface orientation."""

    requires_sensor_depth = False

    def __call__(self, points, valid, to_light, gbuffer, camera):
        return torch.ones_like(valid, dtype=points.dtype), {}

    def set_keyframe_count(self, n):
        pass


@register("cosine", "lambert", options=("normal_source", "warmup_keyframes"))
class Lambert:
    """max(0, n . l), l = unit vector from the point to the light. Pixels
    without a valid normal get factor 0 and are reported in aux so the loss
    can exclude them (they would otherwise be shaded ambient-only).

    warmup_keyframes (docs/DECISIONS.md D43): while the map holds at most
    that many keyframes, the factor is 1 (no n . l) and no n . l map is
    produced, so LightTracking.pixel_weight does not apply. Monocular SLAM
    starts from Gaussians at GUESSED depths: normals from that geometry are
    meaningless, and weighting by n . l removed nearly every tracking pixel,
    so the pose never moved and no keyframe was ever added (2026-09-30).
    FrontEnd and BackEnd report the keyframe count (set_keyframe_count)
    before each tracking / mapping step, each on its own copy of the shader.
    """

    def __init__(self, normal_source, warmup_keyframes):
        if (isinstance(warmup_keyframes, bool) or not isinstance(warmup_keyframes, int)
                or warmup_keyframes < 0):
            raise ValueError(
                "Light.cosine.warmup_keyframes must be an integer >= 0, "
                f"got {warmup_keyframes!r}"
            )
        self.warmup_keyframes = warmup_keyframes
        # Runtime state, not a configuration value: overwritten before every
        # SLAM step. True until then (verification scripts, GUI), i.e. the
        # full model.
        self.active = True
        # A block {type: <normal source>, <its options>}, e.g.
        # {type: depth_fd, stencil_px: 3} (docs/DECISIONS.md D40).
        if not isinstance(normal_source, dict):
            raise TypeError(
                "Light.cosine.normal_source must be a block {type: ..., <options>}, "
                f"got {normal_source!r}"
            )
        self.normals = build_component(
            "normals", normal_source, "Light.cosine.normal_source", None
        )
        self.requires_sensor_depth = self.normals.requires_sensor_depth

    def set_keyframe_count(self, n):
        self.active = n > self.warmup_keyframes

    def graph_state(self):
        # The Python state that changes which kernels __call__ runs (D131).
        return self.active

    def __call__(self, points, valid, to_light, gbuffer, camera):
        if not self.active:
            # Warm-up: no n . l, and no cos map (the loss sees no pixel weight).
            ones = torch.ones_like(valid, dtype=points.dtype)
            return ones, {"cosine_warmup": torch.ones_like(valid)}
        n, n_valid = self.normals(points, valid, gbuffer, camera)
        cos_nl = (n * to_light).sum(0)
        cos_nl = torch.where(n_valid, cos_nl, torch.zeros_like(cos_nl))
        return cos_nl.clamp_min(0.0), {"cos_nl": cos_nl, "normal_valid": n_valid}


declare_planned("cosine")  # no planned forms beyond none / lambert
