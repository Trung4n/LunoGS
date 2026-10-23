# Shaders: the image-space step applied by render() on top of the rasterized
# G-buffer (see gaussian_renderer.render for the call contract). Selected by
# name with Light.shader.type and looked up in _REGISTRY, never by if/else at
# call sites (CLAUDE.md section 2).
#
# Contract: shader(gbuffer, viewpoint_camera) -> dict with
#   "radiance_linear": (3,H,W) predicted linear radiance (required);
#   any other key: an auxiliary per-pixel map (e.g. "light_valid",
#   "light_cos_nl") copied into render_pkg for the losses.
#
# Shader objects are handed to the backend and GUI processes, which are
# started with multiprocessing "spawn": they must be picklable, i.e.
# module-level classes, no closures or lambdas, and only CPU tensors.
#
# This module must not import torch at import time (config tests run without
# it); the colocated shader, which needs torch, is imported only when built.


class IdentityShader:
    """radiance_linear = albedo: unit shading, no light at all.

    Not a light model: the test mode of docs/DECISIONS.md D6. With
    loss_color_space: srgb it is original MonoGS with the color stored in
    linear space (render = linear2sRGB(albedo)), so its ATE must match the
    baseline within the configured tolerance; a larger gap points at the
    color path (storage, conversion, init), not at the light model.
    """

    requires_sensor_depth = False

    def __call__(self, gbuffer, viewpoint_camera):
        return {"radiance_linear": gbuffer["albedo"]}

    # No light parameter at all, hence nothing to learn (D60).
    def learnable_params(self):
        return {}

    def set_learning(self, on):
        pass

    def own_learnable_storage(self):
        pass

    def learnable_state(self):
        return {}

    def load_learnable_state(self, state):
        if state:
            raise ValueError(f"identity shader has no learnable parameter, got {sorted(state)}")

    def set_keyframe_count(self, n):
        pass


def _make_identity(light_cfg, load_params):
    if "schedule" in light_cfg:
        raise ValueError("Light.schedule is meaningless with the identity shader")
    return IdentityShader()


def _make_colocated(light_cfg, load_params):
    from light_models.colocated import ColocatedShader

    return ColocatedShader(light_cfg, load_params())


# type name -> (factory, allowed keys of the Light.shader block besides
# `type`). The factory gets the whole Light block (the colocated model reads
# its components from it) and a zero-argument loader of the params file, so
# shaders that need no light parameters never touch the file.
_REGISTRY = {
    "identity": (_make_identity, frozenset()),
    "colocated": (_make_colocated, frozenset()),
}


def make_shader(light_cfg, load_params):
    """Instantiates the shader described by the Light block."""
    if "shader" not in light_cfg:
        raise KeyError("Missing required config key 'Light.shader'")
    shader_cfg = light_cfg["shader"]
    if not isinstance(shader_cfg, dict) or "type" not in shader_cfg:
        raise KeyError("Missing required config key 'Light.shader.type'")
    name = shader_cfg["type"]
    if name not in _REGISTRY:
        raise ValueError(
            f"Light.shader.type must be one of {tuple(_REGISTRY)}, got {name!r}"
        )
    factory, allowed = _REGISTRY[name]
    unknown = set(shader_cfg) - {"type"} - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in Light.shader for type '{name}' "
            f"(allowed: {sorted(allowed)})"
        )
    return factory(light_cfg, load_params)
