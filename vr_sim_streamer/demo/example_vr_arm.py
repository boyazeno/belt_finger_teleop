"""Example: teleoperate a 6-DOF MuJoCo arm with a VR controller.

Pipeline:
  - VRStreamer publishes stereo video and ingests headset / controller poses.
  - VRControllerLeader turns the relative controller motion into a target
    end-effector pose in the same shape as SpacemouseLeader.get_action().
  - This script runs Jacobian-based delta IK kinematically: each tick it
    computes a joint delta from the current Cartesian error, writes the new
    qpos directly, and calls mj_forward. (No actuator dynamics — appropriate
    for a teleop visualization. Swap in mj_step + position actuators if you
    want full physics.)

Controls (right hand by default):
  - Move the controller in space  -> end effector follows (1:1)
  - B button (rising edge)        -> re-anchor: current controller pose
                                     becomes the new reference, current EE
                                     target stays put (no jump).
  - Trigger                       -> closes the gripper opening (output only;
                                     this scene has no gripper actuator).

Run::

    MUJOCO_GL=egl python demo/example_vr_arm.py \
        --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mujoco  # noqa: E402

from vr_streamer import VRStreamer, VRControllerLeader  # noqa: E402

DEFAULT_SCENE = _REPO_ROOT / "demo" / "scene_arm.xml"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
ACTUATOR_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6"]
HOME_QPOS = np.array([0.0, -0.6, 1.4, 0.0, 0.6, 0.0], dtype=np.float64)


def quat_wxyz_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quaternion -> intrinsic xyz Euler angles."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_orientation_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector from current to target (both wxyz).

    Returns a 3-vector approximately equal to the rotation axis times the
    rotation angle, suitable for use as the angular component of a Cartesian
    error in delta-IK.
    """
    aw, ax, ay, az = q_target
    cw, cx, cy, cz = q_current
    bw, bx, by, bz = cw, -cx, -cy, -cz  # conjugate of current
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    if rw < 0.0:
        rx, ry, rz = -rx, -ry, -rz
    return 2.0 * np.array([rx, ry, rz], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert", "--ssl-certfile", dest="cert", default=None)
    ap.add_argument("--key",  "--ssl-keyfile",  dest="key",  default=None)
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--sim-hz", type=int, default=200)
    ap.add_argument("--pos-scale", type=float, default=1.0)
    ap.add_argument("--rot-scale", type=float, default=1.0)
    ap.add_argument("--kp-pos", type=float, default=8.0)
    ap.add_argument("--kp-rot", type=float, default=4.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("vr-arm")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    # Resolve robot ids.
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    if any(j < 0 for j in joint_ids):
        raise RuntimeError(f"missing joints: {JOINT_NAMES}")
    dof_adr = [int(model.jnt_dofadr[j]) for j in joint_ids]
    qpos_adr = [int(model.jnt_qposadr[j]) for j in joint_ids]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES]
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    target_marker_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vr_target_marker")
    target_marker_mid = int(model.body_mocapid[target_marker_bid])

    # Move to a home pose.
    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = HOME_QPOS[i]
    for i, aid in enumerate(actuator_ids):
        data.ctrl[aid] = HOME_QPOS[i]
    mujoco.mj_forward(model, data)

    # Use the home TCP pose as the leader's anchor point so the EE doesn't jump.
    cur_tcp = data.site_xpos[tcp_site].copy()
    cur_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(cur_quat, data.site_xmat[tcp_site].copy())
    init_euler = quat_wxyz_to_euler_xyz(cur_quat)
    log.info("home TCP pos=%s quat(wxyz)=%s", np.round(cur_tcp, 3), np.round(cur_quat, 3))

    # Start streamer + leader.
    streamer = VRStreamer(model, data, fps=args.video_fps)
    streamer.start(host=args.host, port=args.port,
                   ssl_certfile=args.cert, ssl_keyfile=args.key)
    leader = VRControllerLeader(
        streamer, hand=args.hand,
        initial_pos=cur_tcp, initial_rot_euler=init_euler,
        initial_gripper=0.0,
        pos_scale=args.pos_scale, rot_scale=args.rot_scale,
        reset_button="b",
    )
    log.info("VR teleop ready. Connect from the Quest, hit Connect -> Enter VR. "
             "Press B on the %s controller to re-anchor.", args.hand)

    # Sim loop with delta-IK tracking. Render at video_fps, step at sim_hz.
    period = 1.0 / args.sim_hz
    sync_every = max(1, args.sim_hz // args.video_fps)
    next_t = time.monotonic()
    step = 0
    last_log = 0.0

    try:
        while True:
            action = leader.get_action()
            target_pos = action[:3]
            target_quat = action[3:7]   # (w,x,y,z)
            gripper = action[7]

            # Visualize the live VR target as a green mocap marker.
            data.mocap_pos[target_marker_mid] = target_pos
            data.mocap_quat[target_marker_mid] = target_quat

            # Cartesian error -> twist -> joint vel via Jacobian pinv.
            cur_pos = data.site_xpos[tcp_site]
            cur_q = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(cur_q, data.site_xmat[tcp_site].copy())
            v_lin = np.clip(args.kp_pos * (target_pos - cur_pos), -0.6, 0.6)
            v_ang = np.clip(args.kp_rot * quat_orientation_error(target_quat, cur_q), -2.5, 2.5)
            twist = np.concatenate([v_lin, v_ang])

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, tcp_site)
            J = np.vstack([jacp, jacr])[:, dof_adr]
            qvel = np.linalg.pinv(J, rcond=1e-3) @ twist

            # Kinematic update: write qpos directly, then mj_forward to
            # refresh derived state (xpos, xmat, jacobians).
            for i, adr in enumerate(qpos_adr):
                data.qpos[adr] = data.qpos[adr] + qvel[i] * period
                data.ctrl[actuator_ids[i]] = data.qpos[adr]
            mujoco.mj_forward(model, data)
            step += 1
            if step % sync_every == 0:
                streamer.sync()

            now = time.monotonic()
            if now - last_log > 1.0:
                last_log = now
                err = np.linalg.norm(target_pos - cur_pos)
                log.info("target=%s cur=%s |err|=%.3f gripper=%.3f",
                         np.round(target_pos, 3), np.round(cur_pos, 3), err, gripper)

            next_t += period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        leader.stop()
        streamer.stop()


if __name__ == "__main__":
    main()
