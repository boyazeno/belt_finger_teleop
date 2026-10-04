"""VRControllerLeader — turn synced VR-controller poses into a robot end-effector
target action, in the same shape as `SpacemouseLeader.get_action()`.

The controller's *relative* motion from a captured reference pose drives the
target end-effector pose; the absolute controller pose in MuJoCo world is not
used directly. Pressing the B button re-anchors the reference at the current
controller pose without making the robot target jump (the current desired
target becomes the new base).
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from .transforms import quat_mul

if TYPE_CHECKING:
    from .streamer import VRStreamer


def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Intrinsic xyz Euler -> (w, x, y, z) quaternion. No scipy dependency."""
    cx, cy, cz = np.cos(euler_xyz * 0.5)
    sx, sy, sz = np.sin(euler_xyz * 0.5)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], dtype=np.float64)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _slerp_from_identity(q: np.ndarray, t: float) -> np.ndarray:
    """Slerp from identity (1,0,0,0) to q by factor t in [0,1]."""
    if t == 1.0:
        return q
    qw = float(q[0])
    if qw < 0.0:
        q = -q
        qw = -qw
    if qw > 0.9995:
        return q
    qw = max(-1.0, min(1.0, qw))
    a = np.arccos(qw)
    s = np.sin(a)
    s0 = np.sin((1.0 - t) * a) / s
    s1 = np.sin(t * a) / s
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    out = s0 * identity + s1 * q
    return out / np.linalg.norm(out)


class VRControllerLeader:
    """Teleoperator that translates VR controller motion to an end-effector pose.

    Action layout (8,) — same as ``SpacemouseLeader.get_action()``::

        [x, y, z, qw, qx, qy, qz, gripper_opening]

    Behavior
    --------
    A reference controller pose is captured on first valid sample and
    re-captured every time the configured reset button is pressed. The
    output target pose is computed as::

        target_pos  = base_pos + (ctrl_pos - ref_pos) * pos_scale
        target_quat = slerp(I, ctrl_quat * conj(ref_quat), rot_scale) * base_quat

    where ``base_*`` are the *desired* values at the moment of the last
    re-anchor (so re-anchoring does not make the robot jump). The trigger
    value linearly closes the gripper: ``trigger=0`` → ``gripper_max`` open,
    ``trigger=1`` → ``gripper_min`` closed.

    Parameters
    ----------
    streamer:
        A running `VRStreamer`. Reads `streamer.inputs.<hand>` each tick.
    hand:
        Which hand drives the robot ("left" or "right").
    initial_pos, initial_rot_euler, initial_gripper:
        Starting target pose. Same semantics as `SpacemouseLeader`.
    pos_scale, rot_scale:
        Multipliers on the controller delta. `1.0` is 1:1.
    gripper_min, gripper_max:
        Gripper opening range in metres.
    reset_button:
        Which controller button re-anchors the reference. One of
        "a", "b", "trigger", "squeeze", "thumbstick". Defaults to "b".
    poll_hz:
        Background polling rate.
    """

    def __init__(
        self,
        streamer: "VRStreamer",
        *,
        hand: str = "right",
        initial_pos: Optional[np.ndarray] = None,
        initial_rot_euler: Optional[np.ndarray] = None,
        initial_gripper: float = 0.0,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        gripper_min: float = 0.0,
        gripper_max: float = 0.085,
        reset_button: str = "b",
        poll_hz: float = 100.0,
    ):
        if hand not in ("left", "right"):
            raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
        self._streamer = streamer
        self._hand = hand
        self._reset_button = reset_button.lower()
        self._pos_scale = float(pos_scale)
        self._rot_scale = float(rot_scale)
        self._gripper_min = float(gripper_min)
        self._gripper_max = float(gripper_max)
        self._poll_period = 1.0 / float(poll_hz)

        self._initial_pos = np.array(
            initial_pos if initial_pos is not None else [0.0, 0.0, 0.3], dtype=np.float64)
        self._initial_quat = _euler_xyz_to_quat_wxyz(np.array(
            initial_rot_euler if initial_rot_euler is not None else [0.0, 0.0, 0.0],
            dtype=np.float64))
        self._initial_gripper = float(initial_gripper)

        self._lock = threading.Lock()
        # Anchor: target = base + delta(ctrl, ref).
        self._base_pos = self._initial_pos.copy()
        self._base_quat = self._initial_quat.copy()
        self._gripper = self._initial_gripper
        # Reference controller pose. None until first valid sample.
        self._ref_pos: Optional[np.ndarray] = None
        self._ref_quat: Optional[np.ndarray] = None
        self._prev_reset_pressed = False

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vr-controller-leader", daemon=True)
        self._thread.start()

    # ----- public API -----

    def reset(
        self,
        initial_pos: Optional[np.ndarray] = None,
        initial_rot_euler: Optional[np.ndarray] = None,
        initial_gripper: Optional[float] = None,
    ) -> None:
        """Re-initialise the desired pose (call between episodes).

        Forces re-capture of the reference controller pose on the next tick.
        """
        with self._lock:
            if initial_pos is not None:
                self._initial_pos = np.array(initial_pos, dtype=np.float64)
            if initial_rot_euler is not None:
                self._initial_quat = _euler_xyz_to_quat_wxyz(
                    np.array(initial_rot_euler, dtype=np.float64))
            if initial_gripper is not None:
                self._initial_gripper = float(initial_gripper)
            self._base_pos = self._initial_pos.copy()
            self._base_quat = self._initial_quat.copy()
            self._gripper = self._initial_gripper
            self._ref_pos = None
            self._ref_quat = None

    def set_reference_to_current(self) -> None:
        """Re-anchor: capture current controller pose as the new reference,
        and lock the current desired target as the new base.

        Equivalent to a B-button press, but callable programmatically.
        """
        hi = getattr(self._streamer.inputs, self._hand)
        if hi.kind != "controller" or hi.pos is None or hi.quat is None:
            return
        with self._lock:
            cur_pos, cur_quat = self._compute_target_unsafe(hi.pos, hi.quat)
            self._base_pos = cur_pos
            self._base_quat = cur_quat
            self._ref_pos = np.asarray(hi.pos, dtype=np.float64).copy()
            self._ref_quat = np.asarray(hi.quat, dtype=np.float64).copy()

    def get_action(self) -> np.ndarray:
        """Return the current target action as a length-8 array
        ``[x, y, z, qw, qx, qy, qz, gripper_opening]``.
        """
        hi = getattr(self._streamer.inputs, self._hand)
        with self._lock:
            if hi.kind == "controller" and hi.pos is not None and hi.quat is not None:
                pos, quat = self._compute_target_unsafe(hi.pos, hi.quat)
            else:
                pos, quat = self._base_pos.copy(), self._base_quat.copy()
            gripper = self._gripper
        return np.concatenate([pos, quat, [gripper]])

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    # ----- internals -----

    def _read_reset_button(self, hi) -> bool:
        b = self._reset_button
        if b == "a": return bool(hi.a)
        if b == "b": return bool(hi.b)
        if b == "trigger": return bool(hi.trigger_pressed)
        if b == "squeeze": return bool(hi.squeeze_pressed)
        if b == "thumbstick": return bool(hi.thumbstick_pressed)
        return False

    def _compute_target_unsafe(self, ctrl_pos, ctrl_quat) -> tuple[np.ndarray, np.ndarray]:
        """Caller must hold self._lock."""
        ctrl_pos = np.asarray(ctrl_pos, dtype=np.float64)
        ctrl_quat = np.asarray(ctrl_quat, dtype=np.float64)
        if self._ref_pos is None or self._ref_quat is None:
            return self._base_pos.copy(), self._base_quat.copy()
        d_pos = (ctrl_pos - self._ref_pos) * self._pos_scale
        q_delta = quat_mul(ctrl_quat, _quat_conj(self._ref_quat))
        if self._rot_scale != 1.0:
            q_delta = _slerp_from_identity(q_delta, self._rot_scale)
        target_quat = quat_mul(q_delta, self._base_quat)
        n = np.linalg.norm(target_quat)
        if n > 1e-9:
            target_quat = target_quat / n
        return self._base_pos + d_pos, target_quat

    def _run(self) -> None:
        while not self._stop.is_set():
            hi = getattr(self._streamer.inputs, self._hand)
            connected = (
                hi.kind == "controller" and hi.pos is not None and hi.quat is not None)
            if connected:
                ctrl_pos = np.asarray(hi.pos, dtype=np.float64)
                ctrl_quat = np.asarray(hi.quat, dtype=np.float64)
                pressed = self._read_reset_button(hi)

                # Edge: re-anchor on rising edge of the reset button.
                if pressed and not self._prev_reset_pressed:
                    with self._lock:
                        cur_pos, cur_quat = self._compute_target_unsafe(ctrl_pos, ctrl_quat)
                        self._base_pos = cur_pos
                        self._base_quat = cur_quat
                        self._ref_pos = ctrl_pos.copy()
                        self._ref_quat = ctrl_quat.copy()
                self._prev_reset_pressed = pressed

                # First valid sample: just capture the reference.
                with self._lock:
                    if self._ref_pos is None:
                        self._ref_pos = ctrl_pos.copy()
                        self._ref_quat = ctrl_quat.copy()
                    # Trigger -> gripper opening (linear, trigger pressed = closed).
                    opening = (
                        self._gripper_max
                        - hi.trigger * (self._gripper_max - self._gripper_min))
                    self._gripper = float(np.clip(
                        opening, self._gripper_min, self._gripper_max))
            else:
                self._prev_reset_pressed = False

            time.sleep(self._poll_period)
