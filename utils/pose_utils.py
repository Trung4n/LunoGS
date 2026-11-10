import numpy as np
import torch

# Lie-algebra helpers for camera pose optimization.
#
# Tracking/mapping don't optimize R (3x3, must stay orthonormal) and T
# directly — instead each Camera carries a small 6-DoF tangent vector
# (cam_trans_delta, cam_rot_delta) that gradient descent is free to move
# anywhere in R^6, and update_pose() below "folds" it into the real pose via
# the SE(3) exponential map. This is the standard se(3)/SE(3) trick used to
# do gradient-based pose refinement without ever breaking the rotation's
# orthonormality constraint.
#
# Convention: a twist tau = [rho (translation part), theta (rotation part)],
# each in R^3. theta is an axis-angle vector: its direction is the rotation
# axis, its norm is the rotation angle in radians.


def rt2mat(R, T):
    # Pack a rotation matrix + translation vector into a 4x4 homogeneous
    # transform (numpy). Not called anywhere in this codebase currently —
    # kept as a small standalone utility.
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    # Build the skew-symmetric "hat" matrix [x]_x of a 3-vector x, such that
    # [x]_x @ v == cross(x, v) for any vector v. This is the building block
    # of the so(3) exponential map below (so(3) = skew-symmetric matrices).
    # One stack instead of 6 indexed writes (docs/DECISIONS.md D128).
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    ssm = torch.stack([
        torch.stack([zero, -x[2], x[1]]),
        torch.stack([x[2], zero, -x[0]]),
        torch.stack([-x[1], x[0], zero]),
    ])
    return ssm


def SO3_exp(theta):
    # so(3) -> SO(3) exponential map (Rodrigues' rotation formula): turns an
    # axis-angle vector theta into a proper 3x3 rotation matrix.
    # Uses a first/second-order Taylor expansion near angle=0 to avoid the
    # sin(angle)/angle and (1-cos(angle))/angle^2 terms blowing up (0/0) for
    # very small rotations, which is the common case here since these are
    # per-iteration pose *deltas*, not full rotations.
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    # The branch is chosen on the GPU (torch.where, docs/DECISIONS.md D128):
    # a Python `if` on a CUDA tensor synchronizes, and this runs once per
    # camera per optimization step. Same two formulas as before.
    small = angle < 1e-5
    a = torch.where(small, torch.ones_like(angle), torch.sin(angle) / angle)
    b = torch.where(small, torch.full_like(angle, 0.5), (1 - torch.cos(angle)) / (angle**2))
    return I + a * W + b * W2


def V(theta):
    # SO(3) "left Jacobian". For SE(3), the exponential map does NOT simply
    # copy the translation part rho of the twist into the output translation
    # (that would only be true for pure translations with no rotation) — it
    # must be pre-multiplied by V(theta) to account for the translation
    # sweeping along the rotation. See SE3_exp below. Same small-angle
    # Taylor-series safeguard as SO3_exp.
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    # Branch on the GPU, as in SO3_exp.
    small = angle < 1e-5
    a = torch.where(small, torch.full_like(angle, 0.5), (1.0 - torch.cos(angle)) / (angle**2))
    b = torch.where(small, torch.full_like(angle, 1.0 / 6.0), (angle - torch.sin(angle)) / (angle**3))
    return I + W * a + W2 * b


def SE3_exp(tau):
    # se(3) -> SE(3) exponential map: turns a 6-vector twist tau = [rho, theta]
    # into a 4x4 homogeneous transform T (a small rigid-body motion). This is
    # what turns a raw, unconstrained gradient-descent step (tau, living in
    # flat R^6) into a valid rotation+translation that can be composed with
    # an existing camera pose.
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Norm of the 6-vector step below which tracking stops (MonoGS's value).
CONVERGED_THRESHOLD = 1e-4


def update_pose(camera, converged_threshold=CONVERGED_THRESHOLD):
    # Applies one accumulated optimizer step to a camera's pose, and reports
    # whether tracking has converged. Called after every Adam.step() during
    # both frontend tracking (FrontEnd.tracking) and backend pose refinement
    # (BackEnd.map) — i.e. once per optimization iteration, per camera.
    #
    # tau: the 6-DoF pose delta for this step, as produced by gradient
    # descent on cam_trans_delta/cam_rot_delta (learnable nn.Parameters on
    # the Camera — see camera_utils.Camera).
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    # Current world-to-camera pose, as a 4x4 homogeneous matrix.
    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    # Left-multiply: apply the small delta on top of (i.e. "before", in
    # camera-then-world composition order) the current pose. This keeps the
    # result an exact rotation matrix, unlike naively doing R += dR.
    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    # A tiny step norm means the optimizer has essentially stopped moving
    # the pose -> tracking has converged; callers use this to break out of
    # their fixed-iteration-count optimization loop early.
    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    # The delta has now been absorbed into camera.R/T, so reset it to zero
    # before the next iteration accumulates a fresh one on top.
    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged


def _se3_exp_np(tau):
    """SE3_exp in numpy float64: same formulas and small-angle branches."""
    rho, theta = tau[:3], tau[3:]
    w = np.array([[0.0, -theta[2], theta[1]], [theta[2], 0.0, -theta[0]], [-theta[1], theta[0], 0.0]])
    w2 = w @ w
    angle = np.linalg.norm(theta)
    i = np.eye(3)
    if angle < 1e-5:
        r = i + w + 0.5 * w2
        v = i + 0.5 * w + (1.0 / 6.0) * w2
    else:
        r = i + (np.sin(angle) / angle) * w + ((1 - np.cos(angle)) / (angle**2)) * w2
        v = i + w * ((1.0 - np.cos(angle)) / (angle**2)) + w2 * ((angle - np.sin(angle)) / (angle**3))
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = v @ rho
    return out


def update_pose_host(camera, converged_threshold=CONVERGED_THRESHOLD):
    """update_pose computed on the CPU, for the frontend's tracking loop
    (docs/DECISIONS.md D128). The loop synchronizes with the GPU every
    iteration anyway to test convergence; reading the 6-vector there and doing
    the SE(3) update on the host replaces ~40 tiny GPU kernels. Same formulas,
    in float64, then float32 like update_pose. Returns a bool."""
    # numpy on the host: dozens of tiny CPU torch ops cost ~1 ms (D130).
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0).detach()
    tau = tau.to("cpu", torch.float64).numpy()
    T_w2c = np.eye(4)
    T_w2c[0:3, 0:3] = camera.R.detach().to("cpu", torch.float64).numpy()
    T_w2c[0:3, 3] = camera.T.detach().to("cpu", torch.float64).numpy()
    new_w2c = _se3_exp_np(tau) @ T_w2c
    converged = bool(np.linalg.norm(tau) < converged_threshold)
    # float32 as update_pose gives (its T_w2c is a default-dtype eye), also
    # when the camera started from the float64 ground-truth pose.
    dtype = torch.get_default_dtype()
    camera.update_RT(torch.from_numpy(new_w2c[0:3, 0:3]).to(dtype), torch.from_numpy(new_w2c[0:3, 3]).to(dtype))
    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
