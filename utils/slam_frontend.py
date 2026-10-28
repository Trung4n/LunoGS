import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera, build_projection_matrix
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.run_recorder import open_recorder
from light_models.diagnostics import log_pixel_usage
from utils.slam_utils import get_loss_tracking, get_median_depth, pixel_usage_stats


class FrontEnd(mp.Process):
    """Tracking half of the SLAM system.

    For every incoming frame the frontend:
      1. estimates its camera pose by rendering the current map and
         minimizing a photometric (+ depth) loss w.r.t. the pose only
         (tracking());
      2. decides whether the frame should become a keyframe (is_keyframe());
      3. if so, updates the sliding keyframe window (add_to_window()),
         prepares a depth map to seed new Gaussians (add_new_keyframe()) and
         sends the keyframe to the backend (request_keyframe()).

    The map itself is owned and optimized by the BackEnd process. The
    frontend only holds a detached snapshot of it (self.gaussians), which is
    replaced every time the backend pushes an update (sync_backend()).

    Although this subclasses mp.Process, slam.py calls frontend.run()
    directly, so it actually runs inside the main process.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Set from slam.py after construction (same wiring as BackEnd).
        self.background = None
        self.pipeline_params = None
        # self.shader is also set from slam.py, deliberately NOT initialized
        # here: None means "original MonoGS", so a forgotten assignment must
        # raise AttributeError instead of silently disabling the light model.
        self.frontend_queue = None  # backend -> frontend messages (map snapshots)
        self.backend_queue = None  # frontend -> backend requests (init/keyframe/...)
        self.q_main2vis = None  # frontend -> GUI packets
        self.q_vis2main = None  # GUI -> frontend (pause/unpause)

        # initialized: True once the system has a usable map and tracking can
        # run normally. RGB-D is initialized right after the first frame;
        # monocular only after the keyframe window has been filled once (the
        # map from a single RGB frame has no real geometry yet).
        self.initialized = False
        # kf_indices: frame indices of every keyframe created so far, in
        # order. Used for ATE evaluation and to skip keyframes in
        # eval_rendering.
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        # Not used anywhere in the frontend (only reset in initialize()).
        self.iteration_count = 0
        # occ_aware_visibility: {keyframe idx -> (N,) 0/1 tensor} marking which
        # Gaussians each keyframe in the window sees (n_touched > 0). Computed
        # by the backend and copied here in sync_backend(). This is the
        # covisibility information behind every keyframe decision below.
        self.occ_aware_visibility = {}
        # current_window: frame indices of the active keyframes, NEWEST FIRST
        # (current_window[0] is the latest keyframe). The backend optimizes
        # the map and poses over exactly these keyframes.
        self.current_window = []

        # reset: when True, the next frame is used to (re)initialize the whole
        # system. True at start-up; set again if monocular initialization fails.
        self.reset = True
        # requested_init: an "init" request is in flight; the frontend waits
        # until the backend answers before tracking anything.
        self.requested_init = False
        # requested_keyframe: number of keyframe requests the backend hasn't
        # answered yet. In practice 0 or 1: while it is > 0 the frontend keeps
        # tracking but does not create new keyframes.
        self.requested_keyframe = 0
        # Frame stride for picking the tracking initialization; always 1 here.
        self.use_every_n_frames = 1

        # Latest map snapshot received from the backend (None until the
        # backend answers the "init" request).
        self.gaussians = None
        # cameras: {frame idx -> Camera} for EVERY frame processed so far
        # (keyframes and non-keyframes). Non-keyframes are clean()-ed to free
        # their images but kept for their poses (tracking init, ATE, eval).
        self.cameras = dict()
        self.device = "cuda:0"
        # pause: set from the GUI; while True the main loop spins without
        # processing frames.
        self.pause = False

    def set_hyperparams(self):
        # Called from slam.py once config["Results"]["save_dir"] is known.
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        # save_trj_kf_intv: evaluate/log ATE every N keyframes during the run.
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        # tracking_itr_num: max optimizer steps per frame (stops early on
        # convergence, see tracking()).
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        # kf_interval: minimum number of frames between two keyframes (only
        # enforced while the window is not full, or when wait_for_mapping).
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        # wait_for_mapping: stop tracking new frames until the backend has
        # finished mapping the last keyframe (see Training.sync_mode).
        self.wait_for_mapping = self.config["Training"]["sync_mode"] in (
            "hybrid",
            "sequential",
        )

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        # Registers cur_frame_idx as a keyframe (appends it to kf_indices) and
        # returns the (H, W) numpy depth map the backend will back-project
        # into new Gaussians for it (GaussianModel.extend_from_pcd_seq).
        # Pixels set to 0 produce no Gaussians. `init` is unused.
        #
        # - RGB-D: the sensor depth.
        # - Monocular, first frame (depth=None): no geometry is known yet, so
        #   a flat plane at depth 2 plus noise; the initial BA shapes it.
        # - Monocular, later keyframes: the depth RENDERED from the current
        #   map, with outliers replaced and noise added (see below).
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        # Near-black pixels (e.g. undistortion borders) get no Gaussians.
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                # Hard-coded False: the inverse-depth variant below is dead code.
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    # Unreliable pixels: rendered depth outside median +/- std,
                    # or not well covered by the map (see get_median_depth).
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    # Replace them by the median depth...
                    depth[invalid_depth_mask] = median_depth
                    # ...and add noise: larger (0.5 std) where the depth was a
                    # guess, smaller (0.2 std) where it came from the map, so
                    # the new Gaussians spread out and mapping can pull them to
                    # the right depth.
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        # (Re)starts the whole system from cur_frame_idx: clears all frontend
        # state, fixes this frame at its GT pose (this anchors the world
        # frame, so the estimated trajectory is expressed in the GT
        # coordinate system) and asks the backend to build the initial map
        # from it. Tracking resumes once the backend replies with "init".
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        # (stale requests from before a reset must not reach the backend)
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        # Estimates viewpoint's pose (and exposure) against the frozen map.
        # Returns the render package of the last iteration, whose n_touched /
        # depth / opacity drive the keyframe decision and, if needed,
        # add_new_keyframe().
        #
        # Initial guess = previous frame's pose (no motion model).
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)
        if self.shader is not None:
            # Light-model warm-up state (D43): keyframes in the map so far.
            self.shader.set_keyframe_count(len(self.kf_indices))

        # Only this camera's pose delta and exposure are optimized. The map
        # snapshot in self.gaussians is detached (clone_obj), so no gradient
        # reaches the Gaussians here.
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        # A fresh optimizer per frame: Adam state never carries over.
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                shader=self.shader,
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(self.config, render_pkg, viewpoint)
            loss_tracking.backward()

            with torch.no_grad():
                # Adam moves the pose delta, update_pose folds it into R/T
                # (see pose_utils.update_pose).
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            # Live GUI update, every 10 iterations.
            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        # Typical scene depth at this frame. Keyframe distance thresholds are
        # expressed relative to it (see is_keyframe). Computed from the last
        # render, i.e. one pose update before the final pose (negligible once
        # converged).
        if self.shader is not None:
            # Fail loudly instead of tracking on with a broken pose: a lambert
            # run went through all 2000 frames without a single new keyframe
            # after tracking broke down (2026-09-29, docs/DECISIONS.md D33).
            finite = torch.isfinite(loss_tracking) & torch.isfinite(viewpoint.R).all()
            finite = finite & torch.isfinite(viewpoint.T).all()
            if not bool(finite):
                raise RuntimeError(
                    f"Tracking gave a non-finite loss or pose at frame {cur_frame_idx} "
                    "with the light model enabled"
                )
        self.median_depth = get_median_depth(depth, opacity)
        # Fraction of pixels removed by the extension masks (saturation,
        # n . l), from the last tracking render; CSV only, and only when
        # such a mask is enabled (light_models/diagnostics.py).
        usage = pixel_usage_stats(self.config, render_pkg, viewpoint)
        log_pixel_usage(self.save_dir, cur_frame_idx, usage)
        if usage is not None and usage["frac_used_tracking"] == 0.0:
            # Nothing left for the tracking loss: the pose cannot move (D43).
            Log(f"frame {cur_frame_idx}: the tracking loss uses no pixel", tag="Light")
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        # Keyframe test used once the window is full. A frame becomes a
        # keyframe if EITHER
        #   - it moved far from the last keyframe
        #     (dist > kf_translation * median_depth), OR
        #   - it moved at least a little (dist > kf_min_translation * median_depth)
        #     AND it sees a noticeably different set of Gaussians than the last
        #     keyframe (IoU of visible Gaussians < kf_overlap).
        # Distances are scaled by median scene depth so the same thresholds
        # work regardless of scene size (and of the arbitrary monocular scale).
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        # Translation part of the relative pose current <- last keyframe.
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        
        # dist_check = dist > kf_translation * self.median_depth
        if dist > kf_translation * self.median_depth:
            return True
        
        # dist_check2 = dist > kf_min_translation * self.median_depth
        if dist < kf_min_translation * self.median_depth:
            return False

        # Covisibility as Intersection-over-Union of the two sets of visible
        # Gaussians.
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union

        # return (point_ratio_2 < kf_overlap and dist_check2) or dist_check
        return point_ratio_2 < kf_overlap

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        # Inserts the new keyframe at the front of the window and evicts old
        # keyframes. Returns (new_window, removed_frame), removed_frame being
        # the last evicted index or None. The caller only uses removed_frame
        # to detect a failed monocular initialization.
        #
        # window[0] (the new keyframe) and window[1] (the previous newest
        # keyframe) are never evicted.
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        # Step 1: covisibility-based eviction.
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            # overlap = |A ∩ B| / min(|A|, |B|). Unlike IoU, it stays high when
            # one view sees a subset of the other (e.g. after zooming in).
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        # Only ONE low-overlap keyframe is evicted per call: the last one in
        # the list, i.e. the oldest (window is newest-first).
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        # Step 2: if the window is still over capacity, evict by a distance
        # score (the keyframe-marginalization heuristic from DSO). For each
        # candidate i:
        #   score_i = sqrt(dist(i, current)) * sum_j 1 / dist(i, j)
        # High score = close to the other keyframes (redundant) and far from
        # the current frame (less relevant now). The highest score is evicted,
        # which keeps the remaining keyframes spread out in space.
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    # --- frontend -> backend requests (answered through frontend_queue) ---

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        # Backend adds Gaussians from depthmap, runs mapping over the new
        # window, then replies with a "keyframe" message.
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        # Never called. The backend has no handler for "map" either: sending
        # this message would make BackEnd.run() raise "Unprocessed data".
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        # Backend wipes its map, builds a new one from this single frame
        # (BackEnd.initialize_map) and replies with an "init" message.
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        # Applies a backend message [tag, gaussians, occ_aware_visibility,
        # keyframes]: replaces the map snapshot and the covisibility info, and
        # overwrites the poses of the window's keyframes with the
        # BA-refined ones. Frames tracked after those keyframes are NOT
        # corrected retroactively.
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility
        if self.shader is not None:
            # Light parameters learned by the backend (D60); also seen by
            # slam.py's evaluation, which shares this shader object.
            self.shader.load_learnable_state(data[4])

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        # Frees the images/params of a frame that did not become a keyframe
        # (see Camera.clean), and periodically returns cached GPU memory.
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        # Main loop. Each iteration does ONE of three things:
        #   a) handle a GUI pause/unpause message,
        #   b) handle a message from the backend (frontend_queue not empty),
        #      which always takes priority over new frames,
        #   c) process the next dataset frame: track it, then maybe turn it
        #      into a keyframe.
        # Returns when the dataset is exhausted.
        if self.shader is not None:
            # Tracking never optimizes light parameters (CLAUDE.md section
            # 6): they are constants here, updated only from the backend.
            self.shader.set_learning(False)
        # Optional recording for offline replay videos (D76); None = off.
        self.recorder = open_recorder(self.config, self.dataset, self.shader, self.background)
        cur_frame_idx = 0
        # Same intrinsics for every frame: built once, shared by all Cameras.
        projection_matrix = build_projection_matrix(
            self.dataset.fx,
            self.dataset.fy,
            self.dataset.cx,
            self.dataset.cy,
            self.dataset.width,
            self.dataset.height,
            self.device,
        )
        # Per-frame timing, used only for the 3 fps throttle below.
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            # a) GUI pause handling. While paused this busy-waits (no sleep).
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                # c) process the next frame.
                tic.record()
                # End of the sequence: final ATE + map export, then stop. A
                # keyframe request still in flight is not waited for.
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    if self.recorder is not None:
                        self.recorder.finish(
                            self.cameras, self.kf_indices, self.gaussians, cur_frame_idx - 1
                        )
                    break

                # Wait states (sleep, then loop back to check the queues):
                # - the backend is still building the initial map;
                if self.requested_init:
                    time.sleep(0.01)
                    continue

                # - wait_for_mapping: nothing new until the previous
                #   keyframe has been fully mapped;
                if self.wait_for_mapping and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                # - monocular initialization phase: same, the map is too
                #   fragile to track new frames against a stale snapshot.
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                # First frame (or after a failed monocular init): this frame
                # restarts the system instead of being tracked.
                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    if self.recorder is not None:
                        self.recorder.frame(cur_frame_idx, viewpoint, None, len(self.kf_indices))
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # Monocular becomes "initialized" the first time the window is
                # full (the backend runs its initial BA at that moment).
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                if self.recorder is not None:
                    # Before cleanup() frees the frame's images.
                    self.recorder.frame(cur_frame_idx, viewpoint, render_pkg, len(self.kf_indices))

                # GUI: current frame, window keyframes and a full copy of the
                # map snapshot (a deep copy on every frame).
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                # Backend still busy with the previous keyframe: keep tracking
                # but don't consider this frame as a keyframe. This is what
                # limits multi-thread mode to one keyframe in flight.
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # --- Keyframe decision ---
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                # Which Gaussians this frame sees: touched at least one pixel.
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                # Window not full yet (start-up): a simpler rule replaces the
                # result above — enough frames elapsed AND overlap (IoU) with
                # the last keyframe below kf_overlap, no distance condition.
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                # kf_interval is only enforced here and in the start-up branch;
                # in parallel mode with a full window, the spacing comes
                # from the "one keyframe in flight" rule above instead.
                if self.wait_for_mapping:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    # Monocular init must fill the window with keyframes that
                    # all overlap. If one had to be evicted, the camera moved
                    # too much for the initial map: start over from this frame
                    # (cur_frame_idx is not incremented, so it is re-read and
                    # becomes the new init frame).
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                # Periodic ATE logging, every save_trj_kf_intv keyframes.
                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    # (gives the backend time to map the new keyframe; this
                    # sleep is included in the FPS reported by slam.py)
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                # b) message from the backend.
                data = self.frontend_queue.get()
                # Periodic update from the backend's background mapping loop.
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                # Answer to request_keyframe.
                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                    if self.recorder is not None:
                        self.recorder.map_snapshot(len(self.kf_indices), cur_frame_idx, self.gaussians)

                # Answer to request_init.
                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False
                    if self.recorder is not None:
                        self.recorder.map_snapshot(len(self.kf_indices), cur_frame_idx, self.gaussians)

                # Never sent: the backend only emits the three tags above.
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
