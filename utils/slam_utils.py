import torch

from gaussian_splatting.utils.loss_utils import l1_loss, ssim_separable
from light_models import read_light_tracking
from utils.color_space import sRGB2Linear
from utils.config_utils import require_key
from utils.perf import frame_constant

# Loss functions and image helpers shared by tracking (FrontEnd) and mapping
# (BackEnd). All losses are plain masked L1 photometric (+ optional depth)
# errors between a rendered frame and the frame's GT image/depth; the masks
# are what differ between tracking vs. mapping and monocular vs. RGB-D.
#
# Color space (docs/DECISIONS.md D2, D3): the entry points take the whole
# render_pkg and pick the prediction/target pair themselves according to
# LightTracking.loss_color_space (get_loss_images), then apply the exposure
# affine in that same space (apply_exposure_affine). Pixel MASKS, however,
# are always computed on the observed image as stored (sRGB): their
# thresholds were chosen for that encoding (CLAUDE.md section 6).
#
# Shapes used throughout: images are (3,H,W), depth/opacity maps are
# (1,H,W), all on CUDA.


def get_loss_images(config, render_pkg, viewpoint):
    # (prediction, target) in the space named by LightTracking.loss_color_space.
    #   srgb:   "render" (always sRGB, see gaussian_renderer.render) vs the
    #           observed image as stored. Exactly the baseline pair.
    #   linear: "radiance_linear" (only present with a shader) vs the
    #           observed image converted with the ONE sRGB->linear function.
    target = get_observed_in_loss_space(config, viewpoint)
    if read_light_tracking(config).loss_color_space == "srgb":
        return render_pkg["render"], target
    return render_pkg["radiance_linear"], target


def get_observed_in_loss_space(config, viewpoint):
    # The observed image expressed in LightTracking.loss_color_space: as
    # stored (sRGB) or converted with the ONE sRGB->linear function.
    observed = viewpoint.original_image.cuda()
    if read_light_tracking(config).loss_color_space == "srgb":
        # Through frame_constant too: a replayed CUDA graph must read the
        # observation from the frame's static buffer (D130).
        return frame_constant(viewpoint, "observed", lambda: observed)
    return frame_constant(viewpoint, "observed_linear", lambda: sRGB2Linear(observed))


def apply_exposure_affine(config, image, viewpoint):
    # Per-frame affine exposure correction exp(a) * I + b (Camera.exposure_a/b)
    # applied to the PREDICTION, in the loss color space, right before the
    # loss. Correction is applied to the render rather than the GT so the map
    # itself stays exposure-neutral. With LightTracking.exposure_affine: false
    # nothing is applied: a and b get no gradient and stay at their initial
    # identity values.
    if not read_light_tracking(config).exposure_affine:
        return image
    return torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b


def undo_exposure_affine(config, image, viewpoint):
    # Exact inverse of apply_exposure_affine, in the same (loss) color space:
    # the prediction that apply_exposure_affine would map onto `image`. Used
    # to bring an observed image back to the map's exposure-neutral reference
    # (light_models/albedo_init.py). Identity when exposure_affine is false.
    if not read_light_tracking(config).exposure_affine:
        return image
    return (image - viewpoint.exposure_b) / torch.exp(viewpoint.exposure_a)


def get_light_pixel_weight(config, render_pkg, target):
    # (1,H,W) 0/1 weight from LightTracking.pixel_weight for `target`
    # ("tracking" or "mapping"), or None when no weight applies (then the
    # loss is computed exactly as before, op for op). Criterion so far:
    # min_cos_nl, i.e. keep pixels with a valid normal and n . l >= min_cos_nl
    # (docs/DECISIONS.md D15). n . l comes from the shader's lambert term.
    pw = read_light_tracking(config).pixel_weight
    if not pw.enabled or target not in pw.apply_to:
        return None
    if "light_cosine_warmup" in render_pkg:
        # Lambert warm-up (D43): no n . l yet, so no n . l weight.
        return None
    if "light_cos_nl" not in render_pkg:
        raise KeyError(
            "LightTracking.pixel_weight.min_cos_nl needs n . l from the shader "
            "(Light.cosine.type: lambert)"
        )
    keep = (render_pkg["light_cos_nl"] >= pw.min_cos_nl) & render_pkg["light_normal_valid"]
    return keep.to(render_pkg["light_cos_nl"].dtype)


# LightTracking.saturation_mask.mode -> how the per-channel test becomes the
# mask applied to the loss (docs/DECISIONS.md D31).
_SATURATION_REDUCE = {
    "pixel": lambda sat: sat.any(dim=0, keepdim=True),  # (1,H,W): drop all 3
    "channel": lambda sat: sat,  # (3,H,W): drop only the saturated values
}


def saturated_observations(observed, mode, threshold_8bit):
    # Bool mask of saturated OBSERVED values, from the image's 8-bit codes as
    # stored (before any gamma removal, CLAUDE.md section 6). The dataset
    # loader divides the 8-bit image by 255; rounding back to codes avoids
    # float round-off moving a code across the threshold.
    codes = torch.round(observed * 255.0)
    return _SATURATION_REDUCE[mode](codes >= threshold_8bit)


def get_saturation_keep(config, viewpoint, target):
    # 0/1 mask (1,H,W or 3,H,W) keeping the non-saturated observations for
    # `target` ("tracking" or "mapping"), or None when the mask is off. Uses
    # the observation only, never the prediction: a predicted value above 1
    # on a non-saturated observation is a model error and stays in the loss.
    sm = read_light_tracking(config).saturation_mask
    if not sm.enabled or target not in sm.apply_to:
        return None
    observed = viewpoint.original_image.cuda()

    def build():
        sat = saturated_observations(observed, sm.mode, sm.threshold_8bit)
        return (~sat).to(observed.dtype)
    return frame_constant(viewpoint, ("saturation_keep", target), build)


def get_loss_weight(config, render_pkg, viewpoint, target):
    # Product of every extension mask that applies to `target`: saturation
    # (observation only) and the light-model pixel weight (n . l). None when
    # none applies, so the loss is then computed exactly as the baseline's.
    weights = [
        w
        for w in (
            get_saturation_keep(config, viewpoint, target),
            get_light_pixel_weight(config, render_pkg, target),
        )
        if w is not None
    ]
    if not weights:
        return None
    weight = weights[0]
    for w in weights[1:]:
        weight = weight * w
    return weight


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    # Returns per-channel (vertical, horizontal) derivatives, same (C,H,W)
    # shape as the input thanks to the 1-pixel reflect padding.
    # NOTE on naming: the kernel called `conv_y` actually differentiates along
    # x (columns) and produces img_grad_h; `conv_x` differentiates along y
    # (rows) and produces img_grad_v. The returned v/h names are the ones to
    # trust.
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    # 1/32: sum of |kernel weights|, keeps gradient magnitudes in the same
    # range as the input intensities.
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    # groups=c: filter each channel independently (depthwise convolution).
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    # Marks pixels whose ENTIRE 3x3 neighbourhood has |value| > eps, i.e.
    # pixels where a 3x3 gradient can be trusted. A gradient computed next to
    # an invalid (zero/near-black) pixel — image border, black undistortion
    # margin, missing depth — would be a fake edge; this mask removes those.
    # Box-filtering the boolean map and checking "== 9" is a cheap way to ask
    # "are all 9 neighbours valid?".
    # The two returned masks are identical (both kernels are all-ones); two
    # are returned only to mirror image_gradient()'s (v, h) signature.
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    # Edge-aware depth smoothness regularizer: penalizes depth gradients,
    # but down-weights the penalty where the color image itself has strong
    # edges (w = exp(-10 * grad^2)), so depth is allowed to jump at object
    # boundaries and forced to be smooth inside textureless surfaces.
    # NOT CALLED anywhere in this codebase (and `huber_eps`/`mask` are
    # unused) — an unused experiment kept from development.
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, render_pkg, viewpoint):
    # Entry point for the TRACKING loss (called every iteration of
    # FrontEnd.tracking()). Only the camera pose delta (+ exposure if
    # LightTracking.exposure_affine) receive gradients from it — the
    # Gaussians are frozen during tracking, and so are the light parameters
    # (CLAUDE.md section 6).
    #
    # There is no `initialization` case: the initialization frame never goes
    # through tracking — FrontEnd.initialize() assigns it the GT pose and
    # hands it straight to the backend.
    image, gt_image = get_loss_images(config, render_pkg, viewpoint)
    image = apply_exposure_affine(config, image, viewpoint)
    depth, opacity = render_pkg["depth"], render_pkg["opacity"]
    weight = get_loss_weight(config, render_pkg, viewpoint, "tracking")
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(
            config, image, gt_image, depth, opacity, viewpoint, weight
        )
    return get_loss_tracking_rgbd(
        config, image, gt_image, depth, opacity, viewpoint, weight
    )


def get_loss_tracking_rgb(config, image, gt_image, depth, opacity, viewpoint, pixel_weight):
    # Photometric tracking loss (monocular case, and also the RGB term of the
    # RGB-D tracking loss). `image`/`gt_image` are already in the loss color
    # space (see get_loss_tracking). `depth` is unused.
    rgb_pixel_mask = tracking_rgb_mask(config, viewpoint, pixel_weight)
    # Weighting by rendered `opacity` (per-pixel accumulated alpha) makes the
    # loss ignore pixels the map doesn't cover yet: there the rendering is
    # just background, and matching it would drag the pose toward nonsense.
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    # Mean over ALL pixels (masked ones contribute 0), so the loss magnitude
    # scales with the fraction of valid pixels.
    return l1.mean()


def tracking_rgb_mask(config, viewpoint, pixel_weight):
    # The pixel mask of the photometric tracking loss, also used to log the
    # fraction of usable pixels (light_models/diagnostics.py), so the log
    # and the loss cannot drift apart. pixel_weight: get_loss_weight().
    observed = viewpoint.original_image.cuda()

    def build():
        _, h, w = observed.shape
        mask_shape = (1, h, w)
        # rgb_boundary_threshold: drop near-black GT pixels (sum of RGB below
        # threshold) — typically black borders from undistortion/rectification
        # that carry no real scene information. Computed on the observed image
        # as stored, whatever the loss color space.
        rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
        mask = (observed.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        # ...and additionally keep only high-gradient (edge) pixels, see
        # Camera.compute_grad_mask(): flat regions give almost no pose signal.
        return mask * viewpoint.grad_mask
    rgb_pixel_mask = frame_constant(viewpoint, "tracking_rgb_mask", build)
    # Extension masks (get_loss_weight); None = baseline, op for op.
    if pixel_weight is not None:
        rgb_pixel_mask = rgb_pixel_mask * pixel_weight
    return rgb_pixel_mask


def pixel_usage_stats(config, render_pkg, viewpoint):
    # Per-frame fractions for light_models/diagnostics.log_pixel_usage, or
    # None when no extension mask is enabled (nothing to report). Fractions
    # are over the 3 x H x W observed values; "used" means the tracking loss
    # mask is non-zero (MonoGS's own boundary + gradient masks included).
    lt = read_light_tracking(config)
    sm, pw = lt.saturation_mask, lt.pixel_weight
    if not sm.enabled and not pw.enabled:
        return None
    with torch.no_grad():
        observed = viewpoint.original_image.cuda()
        ones = torch.ones_like(observed)

        def frac(mask):
            return float((mask * ones).mean())

        stats = {}
        if sm.enabled:
            stats["frac_saturated_px"] = frac(
                saturated_observations(observed, "pixel", sm.threshold_8bit).float()
            )
            stats["frac_saturated_values"] = frac(
                saturated_observations(observed, "channel", sm.threshold_8bit).float()
            )
            stats["frac_removed_saturation"] = frac(
                saturated_observations(observed, sm.mode, sm.threshold_8bit).float()
            )
        light_w = get_light_pixel_weight(config, render_pkg, "tracking")
        if light_w is not None:
            stats["frac_removed_light_weight"] = 1.0 - frac(light_w)
        # Light-model G-buffer: pixels whose rasterized depth is trusted
        # (opacity >= Light.gbuffer.opacity_thr) and, with lambert, whose
        # normal is valid (D41).
        if "light_valid" in render_pkg:
            stats["frac_light_valid"] = float(render_pkg["light_valid"].float().mean())
        if "light_normal_valid" in render_pkg:
            stats["frac_normal_valid"] = float(render_pkg["light_normal_valid"].float().mean())
            if pw.enabled:
                # Valid normal but n . l below the threshold: the other reason
                # (besides invalid normals) for the light weight to drop a pixel.
                low = render_pkg["light_normal_valid"] & (render_pkg["light_cos_nl"] < pw.min_cos_nl)
                stats["frac_cos_below_min"] = float(low.float().mean())
        # Rendered accumulated opacity: how opaque the SLAM map is (D45); the
        # G-buffer trusts a pixel's depth only above Light.gbuffer.opacity_thr.
        op = render_pkg["opacity"].detach().flatten().float()
        stats["opacity_p10"] = float(op.quantile(0.1))
        stats["opacity_median"] = float(op.median())
        stats["frac_used_monogs_masks"] = frac(tracking_rgb_mask(config, viewpoint, None))
        weight = get_loss_weight(config, render_pkg, viewpoint, "tracking")
        stats["frac_used_tracking"] = frac(tracking_rgb_mask(config, viewpoint, weight))
    return stats


def get_loss_tracking_rgbd(config, image, gt_image, depth, opacity, viewpoint, pixel_weight):
    # RGB-D tracking loss = alpha * photometric + (1 - alpha) * depth L1.
    # alpha (Training.alpha) balances the two terms; TUM RGB-D configs set
    # 0.9 to lean more on depth. Required: original MonoGS silently used 0.95
    # when the key was missing, now written in the configs (D47).
    alpha = require_key(config["Training"], "alpha", "Training")

    gt_depth = frame_constant(viewpoint, "sensor_depth", lambda: torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None])
    # Depth term only where BOTH (a) the sensor returned a valid depth
    # (> 1 cm; 0 means "no reading") and (b) the map is well reconstructed
    # at that pixel (opacity > 0.95), otherwise the rendered depth is
    # meaningless and would bias the pose.
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(
        config, image, gt_image, depth, opacity, viewpoint, pixel_weight
    )
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, render_pkg, viewpoint, initialization=False):
    # Entry point for the MAPPING loss (BackEnd.initialize_map / map()).
    # Here gradients flow into the Gaussians AND into keyframe poses
    # (+ exposure if LightTracking.exposure_affine). Same color space and
    # same exposure convention as the tracking loss.
    #
    # initialization=True (BackEnd.initialize_map, first keyframe only): skip
    # the exposure correction. Numerically this is a no-op — exposure_a/b are
    # still 0 at that point, and exp(0)*I + 0 == I — it only avoids building
    # autograd edges to parameters no optimizer steps during initialization.
    # What actually makes frame 0 the exposure reference is BackEnd skipping
    # it (`current_window[cam_idx] == 0`) when building keyframe_optimizers,
    # so its exposure stays at identity forever.
    image, gt_image = get_loss_images(config, render_pkg, viewpoint)
    if not initialization:
        image = apply_exposure_affine(config, image, viewpoint)
    depth = render_pkg["depth"]
    weight = get_loss_weight(config, render_pkg, viewpoint, "mapping")
    if config["Training"]["monocular"]:
        loss = get_loss_mapping_rgb(config, image, gt_image, depth, viewpoint, weight)
    else:
        loss = get_loss_mapping_rgbd(config, image, gt_image, depth, viewpoint, weight)
    # Regularizer of a per-Gaussian light quantity (the L1 on the ambient
    # residual, D121), present only when the light model renders one: mapping
    # only, never tracking or the offline refinement.
    if "light_map_penalty" in render_pkg:
        loss = loss + render_pkg["light_map_penalty"]
    return loss


def get_loss_mapping_rgb(config, image, gt_image, depth, viewpoint, pixel_weight):
    # Photometric mapping loss (monocular). Differences vs. the tracking
    # version: NO edge (grad_mask) restriction and NO opacity weighting —
    # mapping must reconstruct every pixel, including flat regions and
    # not-yet-covered areas (that's precisely where new Gaussians need
    # gradients to grow/move into). `image`/`gt_image` are in the loss color
    # space, the mask is computed on the observed image as stored. `depth` is
    # unused.
    observed = viewpoint.original_image.cuda()
    _, h, w = observed.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (observed.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    if pixel_weight is not None:
        rgb_pixel_mask = rgb_pixel_mask * pixel_weight
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, gt_image, depth, viewpoint, pixel_weight):
    # RGB-D mapping loss = alpha * photometric + (1 - alpha) * depth L1.
    # Same idea as get_loss_mapping_rgb: no opacity mask on the depth term
    # (unlike tracking), because holes in the map are exactly what mapping
    # should fill using the sensor depth.
    alpha = require_key(config["Training"], "alpha", "Training")
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    observed = viewpoint.original_image.cuda()

    gt_depth = frame_constant(viewpoint, "sensor_depth", lambda: torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None])
    rgb_pixel_mask = frame_constant(
        viewpoint, "mapping_rgb_mask",
        lambda: (observed.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape))
    if pixel_weight is not None:
        rgb_pixel_mask = rgb_pixel_mask * pixel_weight
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_loss_refinement(config, render_pkg, viewpoint, lambda_dssim):
    # Offline color refinement (BackEnd.color_refinement): L1 + D-SSIM in the
    # loss color space, no exposure affine, as the baseline. The mapping weight
    # (saturation mask, n . l weight) applies too: refinement is mapping with
    # fixed poses, and without it the clipped saturated observations pull the
    # albedo down (D87). Masked pixels are zeroed in both images, so they give
    # no gradient. No weight applies (baseline): the baseline loss, op for op.
    image, gt_image = get_loss_images(config, render_pkg, viewpoint)
    weight = get_loss_weight(config, render_pkg, viewpoint, "mapping")
    if weight is not None:
        image, gt_image = image * weight, gt_image * weight
    # SSIM with the Gaussian window as two 1-D passes: same value up to float
    # rounding, several times faster (D136).
    return (1.0 - lambda_dssim) * l1_loss(image, gt_image) + lambda_dssim * (1.0 - ssim_separable(image, gt_image))


# Optimizer groups of GaussianModel.training_setup that the offline color
# refinement may optimize (Results.color_refinement_params, D89).
REFINABLE_GROUPS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")


def read_refinement_params(results_cfg):
    """Results.color_refinement_params as a list of optimizer group names
    (required, non-empty, known names)."""
    params = require_key(results_cfg, "color_refinement_params", "Results")
    if not isinstance(params, list) or not params or not set(params) <= set(REFINABLE_GROUPS):
        raise ValueError(
            f"Results.color_refinement_params must be a non-empty list of {list(REFINABLE_GROUPS)}, got {params!r}"
        )
    return params


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    # Robust "typical scene depth" of a RENDERED depth map, over pixels that
    # have positive depth, are well covered by the map (opacity > 0.95) and
    # pass the optional extra `mask`. Detached: pure statistic, no gradients.
    #
    # Used by the frontend for:
    #   - self.median_depth after tracking: makes keyframe thresholds
    #     (kf_translation, kf_min_translation) relative to scene scale, which
    #     is essential for monocular where the absolute scale is arbitrary;
    #   - add_new_keyframe (monocular): with return_std=True, to replace
    #     outlier depths by the median and pick the noise level used to
    #     initialize the new keyframe's Gaussians.
    # Returns median, or (median, std, valid_mask) if return_std.
    #
    # NOTE: despite the `opacity=None` default, opacity.detach() below runs
    # before the None check and would crash; every caller passes it.
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
