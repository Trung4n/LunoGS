# Albedo of the Gaussians created for a new keyframe (docs/DECISIONS.md D7).
#
# Called by BackEnd.add_next_kf only when a shader is active; original MonoGS
# (shader None) keeps its own color initialization untouched.
#
# Every strategy has the SAME signature, rich enough for all three planned
# strategies so adding median_depth / rendered_depth does not change callers:
#
#   strategy(params, config, viewpoint, placement_depth, *, shader, render_keyframe)
#     params          - Light.init_albedo block minus `strategy` (validated by
#                       the strategy against its declared keys).
#     config          - full config (LightTracking: exposure, color space).
#     viewpoint       - the keyframe Camera: observed image, intrinsics, pose,
#                       tracked exposure_a/b.
#     placement_depth - (H,W) numpy depth used to place the new Gaussians
#                       (FrontEnd.add_new_keyframe): sensor depth, or for
#                       monocular a GUESS (flat plane, or rendered depth with
#                       outliers replaced and noise added). 0 = no Gaussian.
#     shader          - the active shader, to evaluate shading at a depth.
#     render_keyframe - zero-argument callable: render_pkg of the map at this
#                       keyframe BEFORE the new Gaussians are added, or None
#                       if the map is empty (first keyframe). Lazy, so
#                       strategies that do not need it cost no extra render.
#   returns (3,H,W) float tensor: LINEAR albedo per pixel, unclamped (values
#   above 1 are kept; see D1). GaussianModel samples it at each new
#   Gaussian's pixel.
#
# Why several strategies: for monocular the placement depth is a guess, and
# dividing the observation by the shading at a guessed depth makes the
# albedo wrong by the square of the depth ratio (inverse-square falloff).

import torch

from light_models import read_albedo_init, read_light_tracking
from utils.color_space import sRGB2Linear
from utils.slam_utils import get_observed_in_loss_space, undo_exposure_affine
from utils.config_utils import require_key


def observed_radiance_linear(config, viewpoint):
    # The observation brought back to the map's exposure-neutral reference
    # (inverse affine, in the loss space where the affine is defined, D3),
    # then expressed in linear space.
    observed = get_observed_in_loss_space(config, viewpoint)
    observed = undo_exposure_affine(config, observed, viewpoint)
    if read_light_tracking(config).loss_color_space == "srgb":
        observed = sRGB2Linear(observed)
    return observed


def _observed(params, config, viewpoint, placement_depth, *, shader, render_keyframe):
    # Shading taken as 1: the albedo starts as the observed linear radiance.
    # Biased (bright near the lamp axis, dark at the cone edge) but needs no
    # depth at all; mapping then corrects it through the shaded loss.
    return observed_radiance_linear(config, viewpoint)


def shading_at_depth(shader, depth, viewpoint):
    """(3,H,W) radiance the shader predicts for a WHITE (albedo 1), fully
    opaque surface at z-depth `depth` (H,W tensor; 0 = unknown, shaded at
    the shader's fallback depth). The same shader SLAM renders with, so the
    de-shading below is the exact inverse of the image formation."""
    h, w = depth.shape
    gbuffer = {
        "albedo": torch.ones(3, h, w, device=depth.device, dtype=depth.dtype),
        "depth": depth[None],
        "opacity": (depth > 0).to(depth.dtype)[None],
        # The placement depth is indexed like the observed image (Open3D /
        # OpenCV pixel centers), not like a rasterized map (D29).
        "pixel_offset": 0.0,
    }
    # Per-Gaussian light maps (D121): the de-shaded albedo is that of NEW
    # Gaussians, whose per-Gaussian values start at 0 (the global level c
    # for the ambient residual), so their maps are exactly zero here.
    for key in shader.extra_raster_keys():
        gbuffer[key] = torch.zeros(3, h, w, device=depth.device, dtype=depth.dtype)
    return shader(gbuffer, viewpoint)["radiance_linear"]


def _deshade(params, config, viewpoint, depth, shader):
    # albedo = observation / shading. min_shading bounds the division where
    # the model predicts (almost) no light; with a multiplicative ambient c
    # the shading never drops below c, so min_shading < c never triggers.
    #
    # Saturated observations are NOT left empty (docs/DECISIONS.md D32):
    # there the observation is only a LOWER bound on the true radiance, and
    # every step here keeps the inequality (inverse affine: exp(a) > 0;
    # sRGB -> linear: increasing; / shading > 0), so the value computed is
    # the lower bound A >= observed / s (A >= 1/s for a code of 255 without
    # affine). With LightTracking.saturation_mask on, those pixels never
    # reach the loss, so this bound is all the Gaussian starts from.
    observed = observed_radiance_linear(config, viewpoint)
    shading = shading_at_depth(shader, depth.to(observed.device, observed.dtype), viewpoint)
    return observed / shading.clamp_min(params["min_shading"])


def _placement_depth_tensor(placement_depth):
    return torch.from_numpy(placement_depth.astype("float32"))


def _median_plane(placement_depth, strategy):
    # Fronto-parallel plane at the median of the placement depth. For
    # monocular, the placement depth is a GUESS with noise added on purpose
    # (FrontEnd.add_new_keyframe) so the new Gaussians spread out; that noise
    # is not a measurement and must not enter the de-shading: finite
    # differences of it give meaningless normals, n . l ~ 0, shading ~ c and
    # an albedo inflated by 1/c (seen on keyframe 0 of a lambert run,
    # 2026-09-29, docs/DECISIONS.md D33). Only its scale is kept.
    z = _placement_depth_tensor(placement_depth)
    known = z > 0
    if not known.any():
        raise ValueError(f"init_albedo {strategy}: the placement depth has no valid pixel")
    return torch.full_like(z, float(z[known].median()))


def _median_depth(params, config, viewpoint, placement_depth, *, shader, render_keyframe):
    # Shading of the median plane (see _median_plane) everywhere.
    plane = _median_plane(placement_depth, "median_depth")
    return _deshade(params, config, viewpoint, plane, shader)


def _rendered_depth(params, config, viewpoint, placement_depth, *, shader, render_keyframe):
    # Depth rendered from the current map where it is trusted (opacity >=
    # Light.gbuffer.opacity_thr, D5), the median plane elsewhere (always the
    # case on the first keyframe, whose map is empty). Never the noisy
    # placement depth itself (see _median_plane).
    z = _median_plane(placement_depth, "rendered_depth")
    pkg = render_keyframe()
    if pkg is not None:
        thr = require_key(
            require_key(config["Light"], "gbuffer", "Light"), "opacity_thr", "Light.gbuffer"
        )
        opacity = pkg["opacity"][0].detach()
        trusted = opacity >= thr
        rendered = pkg["depth"][0].detach() / torch.where(trusted, opacity, torch.ones_like(opacity))
        z = torch.where(trusted.cpu(), rendered.cpu().to(z.dtype), z)
    return _deshade(params, config, viewpoint, z, shader)


def _placement_depth(params, config, viewpoint, placement_depth, *, shader, render_keyframe):
    # De-shading pixel by pixel at the placement depth itself. Only valid when
    # that depth is MEASURED: RGB-D, where FrontEnd.add_new_keyframe passes
    # the keyframe's sensor depth unchanged (D47). get_albedo_init refuses it
    # for monocular, whose placement depth is a noisy guess (D33). Pixels
    # without a measurement (0) are shaded at the shader's fallback depth;
    # they receive no Gaussian anyway.
    z = _placement_depth_tensor(placement_depth)
    if not (z > 0).any():
        raise ValueError("init_albedo placement_depth: the placement depth has no valid pixel")
    return _deshade(params, config, viewpoint, z, shader)


# Strategies that trust the placement depth pixel by pixel: refused when it
# is a guess (Dataset.sensor_type monocular).
_MEASURED_DEPTH_ONLY = frozenset({"placement_depth"})


def _positive_min_shading(params):
    v = params["min_shading"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"Light.init_albedo.min_shading must be a number > 0, got {v!r}")


# name -> (function, required parameter keys, validator). Every listed key is
# required and no other key is accepted. A name listed in
# light_models.ALBEDO_INIT_STRATEGIES but missing here is planned.
_REGISTRY = {
    "observed": (_observed, frozenset(), None),
    "median_depth": (_median_depth, frozenset({"min_shading"}), _positive_min_shading),
    "rendered_depth": (_rendered_depth, frozenset({"min_shading"}), _positive_min_shading),
    "placement_depth": (_placement_depth, frozenset({"min_shading"}), _positive_min_shading),
}


def get_albedo_init(config):
    """Resolves Light.init_albedo to a callable with the strategy signature
    (params already bound). Raises for unknown, unimplemented or
    mis-parameterized strategies; slam.py calls it at startup to fail fast."""
    settings = read_albedo_init(config)
    if settings.strategy not in _REGISTRY:
        raise NotImplementedError(
            f"Light.init_albedo.strategy '{settings.strategy}' is planned but "
            "not implemented yet"
        )
    fn, keys, validate = _REGISTRY[settings.strategy]
    unknown = set(settings.params) - keys
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in Light.init_albedo for strategy "
            f"'{settings.strategy}' (allowed: {sorted(keys)})"
        )
    missing = keys - set(settings.params)
    if missing:
        raise KeyError(
            f"Missing required config key(s) {sorted('Light.init_albedo.' + k for k in missing)}"
        )
    if validate is not None:
        validate(settings.params)
    if settings.strategy in _MEASURED_DEPTH_ONLY:
        dataset = require_key(config, "Dataset", "")
        if require_key(dataset, "sensor_type", "Dataset") == "monocular":
            raise ValueError(
                f"Light.init_albedo.strategy '{settings.strategy}' needs a measured "
                "depth; with Dataset.sensor_type monocular the placement depth is a "
                "guess (docs/DECISIONS.md D33). Use rendered_depth or median_depth."
            )

    def init_albedo(config, viewpoint, placement_depth, *, shader, render_keyframe):
        with torch.no_grad():
            albedo = fn(
                settings.params,
                config,
                viewpoint,
                placement_depth,
                shader=shader,
                render_keyframe=render_keyframe,
            )
        _, h, w = viewpoint.original_image.shape
        if tuple(albedo.shape) != (3, h, w):
            raise ValueError(
                f"init_albedo '{settings.strategy}' returned shape "
                f"{tuple(albedo.shape)}, expected {(3, h, w)}"
            )
        return albedo

    return init_albedo
