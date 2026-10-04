"""WebXR (Y-up, RH, -Z forward) <-> MuJoCo (Z-up, RH) coordinate transforms."""
from __future__ import annotations

import math

import numpy as np

# Y-up -> Z-up: rotation +90 deg about world X. Quaternion (w, x, y, z).
_S = math.sqrt(0.5)
Q_WM = np.array([_S, _S, 0.0, 0.0], dtype=np.float64)
R_WM = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def webxr_to_mujoco(pos_xyz, quat_xyzw) -> tuple[np.ndarray, np.ndarray]:
    """Convert a WebXR pose to MuJoCo world frame.

    Returns (pos_mujoco[3], quat_mujoco_wxyz[4]). Left-multiply only on the
    rotation, so the body's local axes keep the camera convention
    (-Z forward, +Y up).
    """
    p = np.asarray(pos_xyz, dtype=np.float64)
    q_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    t_m = R_WM @ p
    q_w = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    q_m = quat_mul(Q_WM, q_w)
    n = np.linalg.norm(q_m)
    if n > 1e-9:
        q_m /= n
    return t_m, q_m
