import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from light_models import read_albedo_lr, read_profile_bootstrap
from light_models.profile_bootstrap import bootstrap_spline
from light_models.albedo_init import get_albedo_init
from light_models.diagnostics import log_albedo_stats, log_exposure, log_learned_light, log_timing
from light_models.learning import LightLearner
from utils.config_utils import require_key
from utils.graphed_refinement import read_refinement_graph
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj, drain_queue
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_refinement, read_refinement_params

class BackEnd(mp.Process):
    """Mapping half of the SLAM system (runs in its own process).

    The backend owns the Gaussian map and jointly optimizes, over the
    current keyframe window:
      - all Gaussian parameters (position, color, opacity, scale, rotation),
      - the poses of the newest `pose_window` keyframes,
      - the exposure (a, b) of every keyframe in the window.
    It also adds Gaussians for each new keyframe, densifies/prunes the map,
    and periodically sends a detached copy of the map plus the refined
    keyframe poses back to the frontend (push_to_frontend).

    It is driven by messages on backend_queue ("init", "keyframe", "pause",
    "unpause", "color_refinement", "stop"). When no message is waiting, it
    keeps optimizing the current window in the background (multi-thread
    mode only).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Set from slam.py after construction.
        self.gaussians : GaussianModel = None  # the map (GaussianModel) - owned by the backend
        self.pipeline_params = None
        self.opt_params = None  # 3DGS optimizer/densification hyperparameters
        self.background = None
        # GPU of this process (slam.py --backend_gpu, D128); None = the default.
        self.device = None
        # Optimizer of the learnable light parameters, built in run() (D60).
        self.light_learner = None
        # self.shader is also set from slam.py, deliberately NOT initialized
        # here: None means "original MonoGS", so a forgotten assignment must
        # raise AttributeError instead of silently disabling the light model.
        # Rough scene radius (6.0, hard-coded in slam.py); scales the
        # densify/prune distance thresholds in set_hyperparams().
        self.cameras_extent = None
        self.frontend_queue = None  # backend -> frontend (map snapshots)
        self.backend_queue = None  # frontend/main -> backend (requests)
        # live_mode (RealSense): shorter initial BA, see the "keyframe" handler.
        self.live_mode = False

        self.pause = False
        # device/dtype: not used anywhere in the backend.
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        # iteration_count: total mapping iterations since the last reset.
        # Drives the densify / opacity-reset schedules and the xyz learning
        # rate decay (update_learning_rate).
        self.iteration_count = 0
        # last_sent: mapping iterations since the last push_to_frontend(); the
        # background loop pushes a new snapshot every 10 iterations.
        self.last_sent = 0
        # occ_aware_visibility: {keyframe idx -> (N,) 0/1 tensor} of which
        # Gaussians each WINDOW keyframe sees (n_touched > 0). Recomputed every
        # mapping iteration, used for pruning and sent to the frontend for its
        # keyframe decisions.
        self.occ_aware_visibility = {}
        # viewpoints: {frame idx -> Camera} for EVERY keyframe received since
        # the last reset (not just the window). Keyframes outside the window
        # are still rendered occasionally (2 random ones per iteration) so the
        # map does not forget old regions, and all of them are used by
        # color_refinement(). Their images are kept in memory for this.
        self.viewpoints = {}
        # current_window: the frontend's keyframe window, newest first.
        self.current_window = []
        # Monocular only becomes initialized after the initial BA over a full
        # window (see map(prune=True)).
        self.initialized = not self.monocular
        # keyframe_optimizers: Adam over the window keyframes' pose deltas and
        # exposures. Rebuilt on every "keyframe" message (see run()).
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        # Not used anywhere in the backend.
        self.save_results = self.config["Results"]["save_results"]
        # Light.albedo_lr (D97), read here so a bad block fails before the
        # run starts. None for original MonoGS (feature_lr as configured).
        self.albedo_lr = read_albedo_lr(self.config) if self.shader is not None else None
        self.profile_bootstrap = read_profile_bootstrap(self.config["Light"]) if self.shader is not None else None

        # --- initial map (initialize_map), single frame ---
        # number of iterations
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        # densify/prune every N iterations
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        # iteration at which all opacities are reset to 0.01
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        # min opacity: Gaussians below it are pruned
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        # scene extent for the clone/split threshold
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        # --- regular mapping (map) ---
        # iterations per keyframe (no background mapping / during mono init)
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        # Mapping iterations per new keyframe when the backend also maps in
        # the background (sync_mode parallel / hybrid); MonoGS wrote 10 here
        # (docs/DECISIONS.md D132).
        self.keyframe_mapping_itr_num = require_key(
            self.config["Training"], "keyframe_mapping_itr_num", "Training")
        # densify/prune when iteration_count % every == offset
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        # min opacity: Gaussians below it are pruned
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        # scene extent: clone/split threshold (percent_dense * extent) and
        # world-space size pruning (0.1 * extent)
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        # period (in iterations) of the non-visible opacity reset
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        # max on-screen radius in pixels before a Gaussian is pruned
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        # background_mapping: keep refining the window between keyframes
        # (see Training.sync_mode).
        self.background_mapping = self.config["Training"]["sync_mode"] in (
            "parallel",
            "hybrid",
        )

    def report_keyframe_count(self):
        # Light-model warm-up state (D43): keyframes in the map so far.
        if self.shader is not None:
            self.shader.set_keyframe_count(len(self.viewpoints))

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.report_keyframe_count()
        # Adds new Gaussians for a keyframe: back-projects depth_map (prepared
        # by FrontEnd.add_new_keyframe) with the keyframe's pose, randomly
        # downsampled (pcd_downsample_init if init, else pcd_downsample), and
        # tags them with kf_id=frame_idx (GaussianModel.unique_kfIDs, used by
        # the "slam" pruning below). `scale` only matters when no depth_map
        # is given, which never happens here.
        #
        # With a light model, the new Gaussians' albedo comes from the
        # Light.init_albedo strategy (light_models/albedo_init.py), computed
        # BEFORE they are added so render_keyframe sees the map as it was.
        # Resolved here rather than stored in set_hyperparams: set_hyperparams
        # runs before this object is pickled into its own process, and the
        # resolved strategy is a closure, which does not pickle.
        albedo_map = None
        if self.shader is not None:
            init_albedo = get_albedo_init(self.config)

            def render_keyframe():
                if self.gaussians.get_xyz.shape[0] == 0:
                    return None
                with torch.no_grad():
                    return render(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        shader=self.shader,
                    )

            albedo_map = init_albedo(
                self.config,
                viewpoint,
                depth_map,
                shader=self.shader,
                render_keyframe=render_keyframe,
            )
        self.gaussians.extend_from_pcd_seq(
            viewpoint,
            kf_id=frame_idx,
            init=init,
            scale=scale,
            depthmap=depth_map,
            albedo_map=albedo_map,
        )
        if self.shader is not None:
            new = (self.gaussians.unique_kfIDs == frame_idx).to(
                self.gaussians.get_xyz.device
            )
            self.log_albedo(frame_idx, "new", new)
        log_exposure(self.config["Results"]["save_dir"], frame_idx, "new", viewpoint)

    def set_albedo_lr(self, frame_idx):
        # Light.albedo_lr: relative_to_initial_albedo sets the albedo learning
        # rate to relative_lr * the mean albedo of the map just created
        # (docs/DECISIONS.md D97), once per map initialization. f_rest keeps
        # MonoGS's ratio to f_dc (training_setup).
        if self.albedo_lr is None or self.albedo_lr.mode == "absolute":
            return
        with torch.no_grad():
            scale = float(self.gaussians.get_albedo.mean())
        lr = self.albedo_lr.relative_lr * scale
        for group in self.gaussians.optimizer.param_groups:
            if group["name"] == "f_dc":
                group["lr"] = lr
            elif group["name"] == "f_rest":
                group["lr"] = lr / 20.0
        Log(f"albedo learning rate {lr:.3g} = {self.albedo_lr.relative_lr:.3g} x initial mean albedo "
            f"{scale:.4f} (frame {frame_idx})", tag="Light")

    def bootstrap_light_profile(self, frame_idx, viewpoint):
        # Light.profile_bootstrap: restarts the spline from a coarse cone
        # estimated on this frame (docs/DECISIONS.md D112), before the first
        # Gaussians get their albedo from the shader. The frontend receives the
        # new values with the "init" answer (push_to_frontend).
        if self.profile_bootstrap is None:
            return
        fit = bootstrap_spline(self.shader, viewpoint, self.profile_bootstrap)
        Log(f"emission profile estimated on frame {frame_idx}: theta_in {fit.theta_in:g} deg, "
            f"theta_out {fit.theta_out:g} deg, core / outside {fit.level_core / fit.level_out:.1f}", tag="Light")

    def start_light_learning(self):
        # Learnable light parameters (light_models/learning.py, D60): this
        # process (mapping) is the only one that updates them. None when the
        # run has no light model or nothing learnable.
        self.light_learner = None
        if self.shader is None:
            return
        params = self.shader.learnable_params()
        if params:
            self.shader.own_learnable_storage()
            self.shader.set_learning(True)
            self.light_learner = LightLearner(params)
            Log(f"learning light parameters {sorted(params)} in mapping", tag="Light")

    def log_learned(self, frame_idx):
        if self.light_learner is not None:
            state, large = self.shader.learned_for_log()
            log_learned_light(
                self.config["Results"]["save_dir"], frame_idx, len(self.viewpoints),
                state, self.light_learner.at_bound_counts(), large,
            )

    def log_albedo(self, frame_idx, stage, mask=None):
        # Albedo > 1 diagnostics (light_models/diagnostics.py), light model
        # runs only. mask: optional (N,) bool selecting the Gaussians to count.
        with torch.no_grad():
            albedo = self.gaussians.get_albedo
            if mask is not None:
                albedo = albedo[mask]
            log_albedo_stats(
                self.config["Results"]["save_dir"], frame_idx, stage, albedo
            )

    def reset(self):
        # Wipes the backend state before (re)initialization: all keyframes,
        # the window, and every Gaussian (all have unique_kfIDs >= 0).
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        self.report_keyframe_count()
        # Builds the initial map from ONE frame: plain 3DGS-style optimization
        # of the Gaussians only (the frame is fixed at its GT pose and its
        # exposure is not used, see initialization=True), for init_itr_num
        # iterations.
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                shader=self.shader,
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, render_pkg, viewpoint, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                # Densification statistics: largest on-screen radius seen and
                # accumulated screen-space gradient per Gaussian.
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                # Clone/split high-gradient Gaussians and prune transparent
                # ones (no screen-size pruning here: max_screen_size=None).
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                # 3DGS trick: reset every opacity to 0.01 once; Gaussians that
                # are not needed stay transparent and get pruned at the next
                # densify_and_prune. (Both thresholds are 500 in the configs.)
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        # Visibility of the init keyframe, from the last iteration.
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        self.report_keyframe_count()
        # Joint optimization of the map and the window's keyframes.
        #
        # prune=False: `iters` optimization steps (Gaussians, poses of the
        #   newest pose_window keyframes, exposures), with densify/prune and
        #   opacity reset on their schedules.
        # prune=True: a single forward/backward pass used only to recompute
        #   visibility and remove poorly observed Gaussians (monocular). It
        #   returns inside the first iteration WITHOUT any optimizer step, so
        #   `iters` is irrelevant there.
        #
        # Returns gaussian_split (True if the Gaussian set changed); no caller
        # uses it.
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        # Only the newest pose_window keyframes get their pose updated below.
        # The older window keyframes act as fixed anchors (exposure is still
        # optimized for them).
        frames_to_optimize = self.config["Training"]["pose_window"]

        # All keyframes outside the window: candidates for the 2 random
        # "anti-forgetting" views rendered each iteration.
        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1
            if self.light_learner is not None:
                # Drops the light gradients of anything rendered since the
                # last step (initialization, prune pass, albedo init).
                self.light_learner.zero_grad()

            loss_mapping = 0
            # Per-view render outputs, accumulated for densification stats.
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            # Window views only: used for occ_aware_visibility.
            n_touched_acm = []

            # Built but never used.
            keyframes_opt = []

            # 1) Every keyframe in the window.
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    shader=self.shader,
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(self.config, render_pkg, viewpoint)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            # 2) Two random older keyframes (outside the window), so regions
            # the camera left behind keep being supervised.
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    shader=self.shader,
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, render_pkg, viewpoint)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            # 3) Isotropic regularizer: penalizes each Gaussian's scales for
            # deviating from their mean, pushing Gaussians toward spheres.
            # Avoids long thin "needle" Gaussians that fit the training views
            # but render badly from new viewpoints (which tracking relies on).
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            if self.light_learner is not None:
                prior = self.light_learner.prior_loss(len(self.viewpoints))
                if prior is not None:
                    loss_mapping = loss_mapping + prior.to(loss_mapping.device)
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                # Visibility of each window keyframe from this iteration's
                # renders.
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        # n_obs: number of window keyframes seeing each Gaussian.
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            # i.e. Gaussians created by the 3 newest keyframes
                            # (unique_kfIDs >= 3rd largest window index) that
                            # are seen by <= 3 window keyframes. In monocular,
                            # new Gaussians start from a guessed depth; those
                            # not confirmed by several views are probably at
                            # the wrong depth. Older Gaussians already
                            # survived this test and are left alone. Before
                            # initialization every Gaussian is eligible.
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        # Only applied in MONOCULAR: with RGB-D the depth is
                        # measured, so this covisibility pruning is skipped.
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            # Keep visibility masks aligned with the smaller
                            # Gaussian set.
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        # The first prune pass over a full window ends the
                        # monocular initialization (it follows the initial BA).
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    # NOTE: returns without optimizer.step() or zero_grad().
                    # The gradients from this pass stay on the parameters that
                    # were not replaced by prune_points (the camera deltas
                    # always, the Gaussians too with RGB-D) and get added to
                    # the next map() iteration's gradients.
                    return False

                # Densification statistics from all rendered views.
                for idx in range(len(viewspace_point_tensor_acm)):
                    # torch.where: same values without boolean indexing (D131).
                    self.gaussians.max_radii2D = torch.where(
                        visibility_filter_acm[idx],
                        torch.max(self.gaussians.max_radii2D, radii_acm[idx]),
                        self.gaussians.max_radii2D,
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                # Clone/split + prune (low opacity, too large on screen or in
                # the world) on a fixed schedule.
                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                # Every gaussian_reset iterations: Gaussians seen by NONE of
                # this iteration's views get their opacity set to 0.4 (visible
                # ones keep theirs). Unlike vanilla 3DGS, which resets all
                # opacities to 0.01, this leaves what is currently observed
                # untouched.
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                # Decays the xyz learning rate with the total mapping
                # iteration count.
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.light_learner is not None:
                    # Light parameters active per Light.schedule (D60).
                    self.light_learner.step(len(self.viewpoints))
                # Pose update
                # Fold the stepped deltas into R,T for the newest
                # pose_window keyframes. Frame 0 is never moved: it anchors
                # the world frame.
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self):
        # Offline refinement run once at the end of an --eval run (requested by
        # slam.py between the "before" and "after" rendering metrics): 26000
        # iterations of standard 3DGS training (L1 + D-SSIM with weight
        # lambda_dssim) on one random keyframe at a time. Only the Gaussians
        # are optimized: poses are fixed, no exposure correction (whatever
        # LightTracking.exposure_affine says), no
        # densification (max_radii2D is updated but unused). The xyz learning
        # rate schedule restarts from iteration 1.
        # Only the groups of Results.color_refinement_params are refined (D89):
        # all of them in MonoGS; the albedo only with a light model, whose
        # shading depends on the geometry (refining it without a depth term
        # wrecked the map). The other groups get no gradient.
        refined = read_refinement_params(self.config["Results"])
        Log(f"Starting color refinement of {refined}")
        for group in self.gaussians.optimizer.param_groups:
            if group["name"] not in refined:
                for p in group["params"]:
                    p.requires_grad_(False)
        if self.shader is not None:
            # The light is not refined offline: fixed at its learned value.
            self.shader.set_learning(False)

        iteration_total = 26000
        graph_opts = read_refinement_graph(self.config["Results"])
        if graph_opts is not None:
            # The same iterations replayed as a CUDA graph (D136).
            from utils.graphed_refinement import GraphedRefinement

            keys = list(self.viewpoints.keys())
            GraphedRefinement(
                self.config, self.gaussians, self.shader, self.pipeline_params, self.background,
                self.viewpoints, refined, self.opt_params.lambda_dssim, graph_opts,
            ).run(iteration_total, lambda: random.randint(0, len(keys) - 1))
            Log("Map refinement done")
            return
        for iteration in tqdm(range(1, iteration_total + 1)):
            # Picks one random keyframe (same as random.choice).
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam,
                self.gaussians,
                self.pipeline_params,
                self.background,
                shader=self.shader,
            )
            visibility_filter, radii = (
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # L1 + D-SSIM in LightTracking.loss_color_space with the mapping
            # weight (D87); still no exposure affine, as in the baseline.
            loss = get_loss_refinement(self.config, render_pkg, viewpoint_cam, self.opt_params.lambda_dssim)
            loss.backward()
            with torch.no_grad():
                # torch.where: same values, no wait for the GPU (D136).
                mr = self.gaussians.max_radii2D
                mr.copy_(torch.where(visibility_filter, torch.maximum(mr, radii), mr))
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def use_device(self):
        """Runs this process on self.device (slam.py --backend_gpu, D128):
        "cuda" then means that GPU, the still empty map is rebuilt there and
        every frame received is moved there. None = the default GPU, as
        before."""
        if self.device is None:
            return
        torch.cuda.set_device(self.device)
        old = self.gaussians
        if old.get_xyz.shape[0] != 0:
            raise RuntimeError("the backend can only change GPU before the map has Gaussians")
        gaussians = GaussianModel(old.max_sh_degree, config=self.config)
        gaussians.init_lr(old.spatial_lr_scale)
        if self.shader is not None:
            gaussians.declare_extras(self.shader.per_gaussian_params())
        gaussians.training_setup(self.opt_params)
        self.gaussians = gaussians
        self.background = self.background.to(self.device)

    def on_device(self, viewpoint):
        # A frame sent by the frontend, moved to this process's GPU if needed.
        if self.device is not None:
            viewpoint.to_device(self.device)
        # Constants of the keyframe's observation (sensor depth and normals,
        # masks) built once, not at every mapping render (D131).
        viewpoint.frame_cache = {}
        return viewpoint

    def push_to_frontend(self, tag=None):
        # Sends [tag, map copy, window visibility, window keyframe poses] to
        # the frontend (applied there by FrontEnd.sync_backend). The map is
        # deep-copied with detached tensors (clone_obj), so the frontend gets
        # a frozen snapshot. tag: "init" / "keyframe" answer the matching
        # request; "sync_backend" is a periodic or color-refinement update.
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        # 5th element: learned light parameters ({} when none), applied to
        # the frontend's shader (FrontEnd.sync_backend, D60).
        light_state = self.shader.learnable_state() if self.shader is not None else {}
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes, light_state]
        self.frontend_queue.put(msg)

    def run(self):
        # Process entry point. Messages take priority; when the queue is empty
        # the backend keeps refining the current window.
        self.use_device()
        self.start_light_learning()
        # Background mapping since the last keyframe (timing log, D79).
        bg_iters, bg_ms = 0, 0.0
        save_dir = self.config["Results"]["save_dir"]
        # The process that started this one (None when not run as a child).
        parent = mp.parent_process()
        while True:
            if parent is not None and not parent.is_alive():
                # The main process died (crash, kill): stop instead of mapping
                # on forever and keeping the job alive (D83).
                Log("main process is gone: backend stops")
                break
            if self.backend_queue.empty():
                # Idle cases: paused, nothing to map yet, or no background
                # mapping (mapping then only happens inside the "keyframe"
                # handler).
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if not self.background_mapping:
                    time.sleep(0.01)
                    continue
                # Background mapping: one iteration at a time, so a new
                # message is picked up quickly. Every 10 iterations: a prune
                # pass (a single pass despite iters=10, see map()) and a new
                # snapshot for the frontend.
                t_bg = time.perf_counter()
                self.map(self.current_window)
                torch.cuda.synchronize()
                bg_iters += 1
                bg_ms += 1000 * (time.perf_counter() - t_bg)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    # slam.py waits for the resulting "sync_backend" message.
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    # From FrontEnd.request_init: start a new map from one
                    # frame (at its GT pose).
                    cur_frame_idx = data[1]
                    viewpoint = self.on_device(data[2])
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.bootstrap_light_profile(cur_frame_idx, viewpoint)
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.set_albedo_lr(cur_frame_idx)
                    self.initialize_map(cur_frame_idx, viewpoint)
                    if self.shader is not None:
                        self.log_albedo(cur_frame_idx, "map")
                    self.log_learned(cur_frame_idx)
                    log_exposure(
                        self.config["Results"]["save_dir"], cur_frame_idx, "map", viewpoint
                    )
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    # From FrontEnd.request_keyframe: register the keyframe,
                    # adopt the frontend's new window, add its Gaussians, map,
                    # prune, reply.
                    cur_frame_idx = data[1]
                    viewpoint = self.on_device(data[2])
                    current_window = data[3]
                    depth_map = data[4]
                    log_timing(save_dir, "background", cur_frame_idx, bg_ms, bg_iters)
                    bg_iters, bg_ms = 0, 0.0
                    t_kf = time.perf_counter()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    # With background mapping: only 10 iterations here, the
                    # background loop keeps mapping afterwards. Without: the
                    # full mapping_itr_num, since nothing runs in between.
                    iter_per_kf = (self.keyframe_mapping_itr_num if self.background_mapping
                                   else self.mapping_itr_num)
                    if not self.initialized:
                        # Monocular initialization phase.
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            # Window just became full: initial bundle
                            # adjustment, meant to optimize the poses of all
                            # window keyframes except the oldest.
                            # NOTE: this larger frames_to_optimize only affects
                            # which deltas are put in the optimizer below.
                            # map() folds deltas into R,T only for the first
                            # config pose_window keyframes, and the rasterizer
                            # ignores the delta values in its forward pass, so
                            # the extra keyframes' deltas get stepped by Adam
                            # but never applied to their poses.
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    # Optimizer for the window: pose deltas (half the tracking
                    # learning rate) for the newest frames_to_optimize
                    # keyframes, exposure for all of them. Frame 0 is skipped
                    # entirely: its pose and exposure stay fixed as the
                    # reference.
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    torch.cuda.synchronize()
                    log_timing(save_dir, "keyframe", cur_frame_idx, 1000 * (time.perf_counter() - t_kf), iter_per_kf)
                    if self.shader is not None:
                        self.log_albedo(cur_frame_idx, "map")
                    self.log_learned(cur_frame_idx)
                    log_exposure(
                        self.config["Results"]["save_dir"],
                        cur_frame_idx,
                        "map",
                        self.viewpoints[cur_frame_idx],
                    )
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        # "stop": drain both queues so the process can exit cleanly. Non-
        # blocking (D62): slam.py drains frontend_queue at the same time, and
        # a blocking get() after empty() could wait forever.
        drain_queue(self.backend_queue)
        drain_queue(self.frontend_queue)
        Log("Backend run loop finished")
        return
