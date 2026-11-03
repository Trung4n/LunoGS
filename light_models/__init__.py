# Entry point of the co-located light model (CLAUDE.md section 4).
#
# build_shader(config) is the ONLY place that decides whether SLAM runs the
# original MonoGS image formation or the lit one. Its result is handed to
# every render() call as the required `shader` keyword (see
# gaussian_renderer.render for the shader contract).
#
# read_light_tracking(config) parses the LightTracking block that the loss
# functions in utils/slam_utils.py depend on.
#
# Every key is read strictly: a missing key raises, nothing falls back to a
# "reasonable" default (CLAUDE.md section 2). This module does not import
# torch, so the config parsing is unit-testable without a GPU environment.

from types import SimpleNamespace

from light_models.params import load_params_file, params_file_path
from light_models.shaders import make_shader
from utils.config_utils import require_key

# Spaces the photometric loss can be computed in (docs/DECISIONS.md D2):
#   srgb   - render_pkg["render"] vs the observed image as stored.
#   linear - render_pkg["radiance_linear"] vs sRGB2Linear(observed).
LOSS_COLOR_SPACES = ("srgb", "linear")


def _light_enabled(config):
    enabled = require_key(require_key(config, "Light", ""), "enabled", "Light")
    if not isinstance(enabled, bool):
        raise TypeError(f"Light.enabled must be true or false, got {enabled!r}")
    return enabled


def build_shader(config, params_data=None):
    """Returns None for original MonoGS (Light.enabled: false), otherwise the
    shader callable used by render().

    params_data: the loaded light params (dict). None = read the per-scene
    file Light.params_file relative to Dataset.dataset_path (D17), and only
    if the shader needs light parameters. The verification scripts pass the
    data explicitly."""
    if not _light_enabled(config):
        return None

    # Albedo is stored in the degree-0 SH coefficient (docs/DECISIONS.md):
    # higher SH bands would let the map bake view-dependent lighting back in.
    if require_key(config["Training"], "spherical_harmonics", "Training"):
        raise ValueError(
            "Light.enabled: true requires Training.spherical_harmonics: false"
        )

    def load_params():
        if params_data is not None:
            return params_data
        return load_params_file(params_file_path(config))

    shader = make_shader(config["Light"], load_params)
    if shader.requires_sensor_depth:
        # A component reads the observed frame's sensor depth (e.g. normal
        # source sensor_depth_fd, D49). In monocular, Camera.depth is still
        # loaded (D20) but is ground truth, not a sensor: refuse.
        dataset = require_key(config, "Dataset", "")
        sensor = require_key(dataset, "sensor_type", "Dataset")
        if sensor == "monocular":
            raise ValueError(
                "The light model reads the sensor depth (normal source "
                "sensor_depth_fd), which needs Dataset.sensor_type depth, got monocular"
            )
    return shader


# Strategies for the albedo of newly created Gaussians (docs/DECISIONS.md D7).
# The implementations live in light_models/albedo_init.py (needs torch); only
# the names are here so the config parsing stays importable without torch.
#   observed       - albedo = linearized observation, shading taken as 1.
#   median_depth   - observation / shading at the keyframe's median depth.
#   rendered_depth - observation / shading at the depth rendered from the map.
#   placement_depth - observation / shading at the measured placement depth
#                    (RGB-D sensor depth); refused for monocular.
ALBEDO_INIT_STRATEGIES = ("observed", "median_depth", "rendered_depth", "placement_depth")


def read_albedo_init(config):
    """Light.init_albedo block: {strategy: <name>, <strategy-specific keys>}.

    Returns a namespace with `strategy` and `params` (every other key of the
    block, validated by the strategy itself in albedo_init.py).
    """
    light_cfg = require_key(config, "Light", "")
    block = require_key(light_cfg, "init_albedo", "Light")
    strategy = require_key(block, "strategy", "Light.init_albedo")
    if strategy not in ALBEDO_INIT_STRATEGIES:
        raise ValueError(
            f"Light.init_albedo.strategy must be one of {ALBEDO_INIT_STRATEGIES}, "
            f"got {strategy!r}"
        )
    params = {k: v for k, v in block.items() if k != "strategy"}
    return SimpleNamespace(strategy=strategy, params=params)


# How the learning rate of the albedo (the SH DC coefficient) is set
# (docs/DECISIONS.md D97):
#   absolute                  - opt_params.feature_lr, as in MonoGS.
#   relative_to_initial_albedo - relative_lr * the mean albedo of the
#                    Gaussians created at map initialization (one value for
#                    the run). Adam takes an absolute step, so the RELATIVE
#                    albedo step, which sets how fast colours fit and how much
#                    the map densifies (D90-D92), depends on the albedo scale,
#                    i.e. on the light intensity K and the scene (D95). This
#                    mode keeps the relative step across scenes and K.
ALBEDO_LR_MODES = ("absolute", "relative_to_initial_albedo")


def read_albedo_lr(config):
    """Light.albedo_lr block: {mode: absolute} or
    {mode: relative_to_initial_albedo, relative_lr: <number > 0>}.
    Returns a namespace with `mode` and `relative_lr` (None if absolute)."""
    light_cfg = require_key(config, "Light", "")
    block = require_key(light_cfg, "albedo_lr", "Light")
    if not isinstance(block, dict):
        raise TypeError(f"Light.albedo_lr must be a block {{mode: ...}}, got {block!r}")
    mode = require_key(block, "mode", "Light.albedo_lr")
    if mode not in ALBEDO_LR_MODES:
        raise ValueError(f"Light.albedo_lr.mode must be one of {ALBEDO_LR_MODES}, got {mode!r}")
    allowed = {"mode"} | ({"relative_lr"} if mode == "relative_to_initial_albedo" else set())
    extra = set(block) - allowed
    if extra:
        raise ValueError(f"Light.albedo_lr: unexpected keys {sorted(extra)} for mode '{mode}'")
    rel = None
    if mode == "relative_to_initial_albedo":
        rel = require_key(block, "relative_lr", "Light.albedo_lr")
        if isinstance(rel, bool) or not isinstance(rel, (int, float)) or not rel > 0:
            raise ValueError(f"Light.albedo_lr.relative_lr must be a number > 0, got {rel!r}")
        rel = float(rel)
    return SimpleNamespace(mode=mode, relative_lr=rel)


def read_light_tracking(config):
    """LightTracking options used by the tracking AND mapping losses.

    Returns a namespace with:
      loss_color_space: "srgb" | "linear" (see LOSS_COLOR_SPACES).
      exposure_affine: bool. If true, exp(a) * I + b is applied to the
        prediction in the loss color space right before the loss
        (docs/DECISIONS.md D3); if false, no affine term at all.
    """
    block = require_key(config, "LightTracking", "")
    space = require_key(block, "loss_color_space", "LightTracking")
    if space not in LOSS_COLOR_SPACES:
        raise ValueError(
            f"LightTracking.loss_color_space must be one of {LOSS_COLOR_SPACES}, "
            f"got {space!r}"
        )
    affine = require_key(block, "exposure_affine", "LightTracking")
    if not isinstance(affine, bool):
        raise TypeError(
            f"LightTracking.exposure_affine must be true or false, got {affine!r}"
        )
    if space == "linear" and not _light_enabled(config):
        # radiance_linear only exists when a shader runs. Linearizing the
        # baseline's sRGB render instead would be a different model, not a
        # silent fallback we want.
        raise ValueError(
            "LightTracking.loss_color_space: linear requires Light.enabled: true"
        )
    return SimpleNamespace(
        loss_color_space=space,
        exposure_affine=affine,
        pixel_weight=_read_pixel_weight(block, config),
        saturation_mask=_read_saturation_mask(block),
    )


# Losses a pixel weight can be applied to.
PIXEL_WEIGHT_TARGETS = ("tracking", "mapping")
# Criteria implemented so far (step 4.2). Step 4.4 adds saturation / dark /
# gradient / opacity thresholds here, each one required when present.
_PIXEL_WEIGHT_KEYS = ("enabled", "apply_to", "min_cos_nl")


def _read_pixel_weight(block, config):
    """LightTracking.pixel_weight: {enabled: false} or
    {enabled: true, apply_to: [tracking|mapping, ...], min_cos_nl: <number>}.
    min_cos_nl keeps pixels whose depth_fd normal is valid and n . l >= it
    (docs/DECISIONS.md D15); it needs cosine: lambert, which produces n . l."""
    pw = require_key(block, "pixel_weight", "LightTracking")
    where = "LightTracking.pixel_weight"
    unknown = set(pw) - set(_PIXEL_WEIGHT_KEYS) if isinstance(pw, dict) else None
    if unknown:
        raise ValueError(f"{where}: unexpected keys {sorted(unknown)}")
    enabled = require_key(pw, "enabled", where)
    if not isinstance(enabled, bool):
        raise TypeError(f"{where}.enabled must be true or false, got {enabled!r}")
    if not enabled:
        return SimpleNamespace(enabled=False, apply_to=(), min_cos_nl=None)
    if not _light_enabled(config):
        raise ValueError(f"{where}.enabled: true requires Light.enabled: true")
    apply_to = require_key(pw, "apply_to", where)
    if (
        not isinstance(apply_to, list)
        or not apply_to
        or len(set(apply_to)) != len(apply_to)
        or not set(apply_to) <= set(PIXEL_WEIGHT_TARGETS)
    ):
        raise ValueError(
            f"{where}.apply_to must be a non-empty list of distinct values from "
            f"{PIXEL_WEIGHT_TARGETS}, got {apply_to!r}"
        )
    min_cos = require_key(pw, "min_cos_nl", where)
    if isinstance(min_cos, bool) or not isinstance(min_cos, (int, float)) or not -1 <= min_cos <= 1:
        raise ValueError(f"{where}.min_cos_nl must be a number in [-1, 1], got {min_cos!r}")
    return SimpleNamespace(enabled=True, apply_to=tuple(apply_to), min_cos_nl=float(min_cos))


# How a saturated observation is excluded (docs/DECISIONS.md D31):
#   pixel   - a pixel saturated in ANY channel is dropped in all three;
#   channel - only the saturated channel values are dropped.
SATURATION_MODES = ("pixel", "channel")
_SATURATION_KEYS = ("enabled", "mode", "threshold_8bit", "apply_to")


def _read_saturation_mask(block):
    """LightTracking.saturation_mask: {enabled: false} or
    {enabled: true, mode: pixel|channel, threshold_8bit: <1..255>,
     apply_to: [tracking|mapping, ...]}.

    Computed on the OBSERVED image only (its 8-bit codes, before any gamma
    removal, CLAUDE.md section 6), never on the prediction: a pixel predicted
    above 1 whose observation is not saturated stays in the loss, since that
    is exactly where the model is wrong. It therefore needs no light model
    and also applies with Light.enabled: false (the baseline + mask control).
    """
    where = "LightTracking.saturation_mask"
    sm = require_key(block, "saturation_mask", "LightTracking")
    if not isinstance(sm, dict):
        raise TypeError(f"{where} must be a block, got {sm!r}")
    unknown = set(sm) - set(_SATURATION_KEYS)
    if unknown:
        raise ValueError(f"{where}: unexpected keys {sorted(unknown)}")
    enabled = require_key(sm, "enabled", where)
    if not isinstance(enabled, bool):
        raise TypeError(f"{where}.enabled must be true or false, got {enabled!r}")
    if not enabled:
        return SimpleNamespace(enabled=False, mode=None, threshold_8bit=None, apply_to=())
    mode = require_key(sm, "mode", where)
    if mode not in SATURATION_MODES:
        raise ValueError(f"{where}.mode must be one of {SATURATION_MODES}, got {mode!r}")
    thr = require_key(sm, "threshold_8bit", where)
    if isinstance(thr, bool) or not isinstance(thr, int) or not 1 <= thr <= 255:
        raise ValueError(f"{where}.threshold_8bit must be an integer in [1, 255], got {thr!r}")
    apply_to = require_key(sm, "apply_to", where)
    if (
        not isinstance(apply_to, list)
        or not apply_to
        or len(set(apply_to)) != len(apply_to)
        or not set(apply_to) <= set(PIXEL_WEIGHT_TARGETS)
    ):
        raise ValueError(
            f"{where}.apply_to must be a non-empty list of distinct values from "
            f"{PIXEL_WEIGHT_TARGETS}, got {apply_to!r}"
        )
    return SimpleNamespace(enabled=True, mode=mode, threshold_8bit=thr, apply_to=tuple(apply_to))


def read_min_gradient(config):
    """LightTracking.min_gradient: {enabled: false} or
    {enabled: true, threshold_8bit: <number > 0>} (docs/DECISIONS.md D37).

    An ABSOLUTE floor on the image gradient of the pixels MonoGS's tracking
    keeps (Camera.compute_grad_mask), in 8-bit gray levels per pixel, gray =
    mean of R, G, B as MonoGS computes it. MonoGS only thresholds relative to
    each 32x32 cell's median gradient; in the dark regions of the lit data
    that median is ~0 and JPEG/quantization noise below one gray level gets
    selected. Observation only, so it also applies with Light.enabled: false.
    Read separately from read_light_tracking: it concerns the tracking pixel
    selection, not the loss.
    """
    block = require_key(config, "LightTracking", "")
    where = "LightTracking.min_gradient"
    mg = require_key(block, "min_gradient", "LightTracking")
    if not isinstance(mg, dict):
        raise TypeError(f"{where} must be a block, got {mg!r}")
    unknown = set(mg) - {"enabled", "threshold_8bit"}
    if unknown:
        raise ValueError(f"{where}: unexpected keys {sorted(unknown)}")
    enabled = require_key(mg, "enabled", where)
    if not isinstance(enabled, bool):
        raise TypeError(f"{where}.enabled must be true or false, got {enabled!r}")
    if not enabled:
        return SimpleNamespace(enabled=False, threshold_8bit=None)
    thr = require_key(mg, "threshold_8bit", where)
    if isinstance(thr, bool) or not isinstance(thr, (int, float)) or thr <= 0:
        raise ValueError(f"{where}.threshold_8bit must be a number > 0, got {thr!r}")
    return SimpleNamespace(enabled=True, threshold_8bit=float(thr))
