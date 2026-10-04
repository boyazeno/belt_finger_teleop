"""VRStreamer — main facade. Owns mocap pose application, rendering, networking."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import uvicorn

from .inputs import HandInput, Inputs
from .renderer import StereoRenderer
from .transforms import webxr_to_mujoco
from .webrtc import build_app

log = logging.getLogger("vr_streamer")

# Default location of the bundled WebXR client (static/index.html).
_DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"


class VRStreamer:
    """Drop-in WebXR teleop + stereo viz for an arbitrary MuJoCo model.

    The host project owns `mj_step`. Call `sync()` once per simulation tick
    to apply the latest WebXR poses to the configured mocap bodies, render
    the eye cameras, and publish the SBS frame to connected peers.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        head_body: str = "head",
        left_eye: str = "left_eye",
        right_eye: str = "right_eye",
        left_controller: Optional[str] = "controller_left",
        right_controller: Optional[str] = "controller_right",
        eye_width: int = 640,
        eye_height: int = 480,
        fps: int = 30,
        static_dir: Optional[Path] = None,
        flying_max_speed: float = 5.0,
        flying_hand: str = "left",
        flying_deadzone: float = 0.15,
    ):
        self._model = model
        self._data = data
        self._fps = fps
        self._static_dir = Path(static_dir) if static_dir else _DEFAULT_STATIC_DIR

        self._head_mocap = self._mocap_id(head_body)
        self._cl_mocap = self._mocap_id(left_controller) if left_controller else -1
        self._cr_mocap = self._mocap_id(right_controller) if right_controller else -1

        # Verify cameras exist before we render.
        for cam in (left_eye, right_eye):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                raise ValueError(
                    f"camera '{cam}' not found. Use attach_stereo_cameras / "
                    f"attach_vr_rig before compiling the model.")

        self._renderer = StereoRenderer(
            model, left_eye=left_eye, right_eye=right_eye,
            eye_width=eye_width, eye_height=eye_height)

        # Pending pose targets posted by the asyncio thread.
        self._pending_lock = threading.Lock()
        self._pending: dict = {}  # {"head": (pos,quat), "left": (...), "right": (...)}

        # Public, read-only-ish input snapshot (updated from sync()).
        self.inputs = Inputs()

        # Latest rendered frame, published by sync(), consumed by webrtc track.
        self._frame_lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

        # Flying locomotion: left/right thumbstick translates the rig in the
        # MuJoCo XY plane on top of tracked head motion. Set max_speed=0 to
        # disable. Public so the user can read/zero it (e.g., on respawn).
        self.flying_max_speed = float(flying_max_speed)
        self._flying_hand = flying_hand if flying_hand in ("left", "right") else "left"
        self._flying_deadzone = float(flying_deadzone)
        self.flying_offset = np.zeros(3, dtype=np.float64)
        self._last_sync_t: Optional[float] = None

        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None

    # ----- model introspection -----

    def _mocap_id(self, body_name: str) -> int:
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"body '{body_name}' not found")
        mid = int(self._model.body_mocapid[bid])
        if mid < 0:
            raise ValueError(
                f"body '{body_name}' is not a mocap body. VRStreamer drives "
                f"poses via mocap_pos/mocap_quat; mark the body mocap=\"true\".")
        return mid

    # ----- network thread (asyncio) -> sim thread handover -----

    def _on_pose_msg(self, msg: dict) -> None:
        head = msg.get("head")
        update: dict = {}
        if isinstance(head, dict) and "pos" in head and "quat" in head:
            update["head"] = webxr_to_mujoco(head["pos"], head["quat"])
        for hand in ("left", "right"):
            c = msg.get(hand)
            if not isinstance(c, dict):
                continue
            entry: dict = {}
            if "pos" in c and "quat" in c:
                entry["pose"] = webxr_to_mujoco(c["pos"], c["quat"])
            entry["raw"] = c
            update[hand] = entry
        with self._pending_lock:
            self._pending.update(update)

    # ----- sim-thread API -----

    def sync(self) -> None:
        """Apply latest poses + button state, render eyes, publish frame.

        Call once per `mj_step` (or every N steps if you want a slower video
        rate; rendering cost dominates).
        """
        with self._pending_lock:
            pending = self._pending
            self._pending = {}

        # 1) Apply tracked head pose.
        head_quat: Optional[np.ndarray] = self.inputs.head_quat
        if "head" in pending:
            t, q = pending["head"]
            self._data.mocap_pos[self._head_mocap] = t
            self._data.mocap_quat[self._head_mocap] = q
            self.inputs.head_pos = t
            self.inputs.head_quat = q
            head_quat = q

        # 2) Apply tracked controller poses + button state.
        for hand, mocap_id in (("left", self._cl_mocap), ("right", self._cr_mocap)):
            entry = pending.get(hand)
            if entry is None:
                continue
            hi: HandInput = getattr(self.inputs, hand)
            if "pose" in entry:
                t, q = entry["pose"]
                if mocap_id >= 0:
                    self._data.mocap_pos[mocap_id] = t
                    self._data.mocap_quat[mocap_id] = q
                hi.pos, hi.quat = t, q
            self._update_hand_input(hi, entry.get("raw") or {})

        # 3) Flying locomotion: integrate the chosen thumbstick into
        # `flying_offset`, then add it to every rig mocap so the user
        # smoothly translates through the world.
        self._integrate_flying(head_quat)
        if not np.all(self.flying_offset == 0.0):
            off = self.flying_offset
            self._data.mocap_pos[self._head_mocap] = self._data.mocap_pos[self._head_mocap] + off
            for mid in (self._cl_mocap, self._cr_mocap):
                if mid >= 0:
                    self._data.mocap_pos[mid] = self._data.mocap_pos[mid] + off

        # 4) Render and publish.
        sbs = self._renderer.render(self._data)
        with self._frame_lock:
            self._frame = sbs.copy()

    def _integrate_flying(self, head_quat: Optional[np.ndarray]) -> None:
        """Update self.flying_offset from the active thumbstick.

        Forward direction = head's local -Z projected onto the MuJoCo XY plane
        (Z-up). Strafe = head's local +X likewise. WebGamepad axes convention:
        axes[2] = stick X (+right), axes[3] = stick Y (+down/back). So
        forward speed scales with `-stick_y`.
        """
        now = time.monotonic()
        if self._last_sync_t is None:
            self._last_sync_t = now
            return
        dt = now - self._last_sync_t
        self._last_sync_t = now

        if self.flying_max_speed <= 0.0 or head_quat is None:
            return
        hi: HandInput = getattr(self.inputs, self._flying_hand)
        if hi.kind != "controller":
            return
        sx, sy = hi.thumbstick
        mag = (sx * sx + sy * sy) ** 0.5
        if mag < self._flying_deadzone:
            return
        # Rescale so deadzone -> 0 and 1.0 stays 1.0, capped at 1.
        s = min(1.0, (mag - self._flying_deadzone) / (1.0 - self._flying_deadzone))
        sx, sy = sx / mag * s, sy / mag * s

        forward = self._head_forward_xy(head_quat)
        if forward is None:
            return
        right = np.array([forward[1], -forward[0], 0.0], dtype=np.float64)

        delta = (forward * (-sy) + right * sx) * (self.flying_max_speed * dt)
        delta[2] = 0.0  # keep Z untouched (XY-plane flying).
        self.flying_offset += delta

    @staticmethod
    def _head_forward_xy(quat_wxyz: np.ndarray) -> Optional[np.ndarray]:
        """World-frame forward direction (head local -Z) projected onto XY."""
        w, x, y, z = quat_wxyz
        # Rotate (0, 0, -1) by quat: this is the third column of R, negated.
        fx = -2.0 * (x * z + w * y)
        fy = -2.0 * (y * z - w * x)
        # No need for fz — we project to XY.
        n = (fx * fx + fy * fy) ** 0.5
        if n < 1e-6:
            return None
        return np.array([fx / n, fy / n, 0.0], dtype=np.float64)

    @staticmethod
    def _update_hand_input(hi: HandInput, raw: dict) -> None:
        kind = raw.get("kind")
        if kind == "controller":
            btns = raw.get("buttons") or {}
            tr = btns.get("trigger") or {}
            sq = btns.get("squeeze") or {}
            ts = btns.get("thumbstick") or {}
            a = btns.get("a") or {}
            b = btns.get("b") or {}
            axes = btns.get("axes") or []
            hi.kind = "controller"
            hi.trigger = float(tr.get("value") or 0.0)
            hi.trigger_pressed = bool(tr.get("pressed") or False)
            hi.squeeze = float(sq.get("value") or 0.0)
            hi.squeeze_pressed = bool(sq.get("pressed") or False)
            hi.thumbstick_pressed = bool(ts.get("pressed") or False)
            hi.axes = tuple(float(x) for x in axes)
            # Quest oculus-touch profile: axes = [touchpad_x, touchpad_y,
            # thumbstick_x, thumbstick_y]. Fall back to [0..1] if shorter.
            if len(axes) >= 4:
                hi.thumbstick = (float(axes[2]), float(axes[3]))
            elif len(axes) >= 2:
                hi.thumbstick = (float(axes[0]), float(axes[1]))
            hi.a = bool(a.get("pressed") or False)
            hi.b = bool(b.get("pressed") or False)
        elif kind == "hand":
            pinch = raw.get("pinch") or {}
            hi.kind = "hand"
            hi.trigger = float(pinch.get("value") or 0.0)
            hi.trigger_pressed = bool(pinch.get("pressed") or False)
            hi.squeeze = 0.0
            hi.squeeze_pressed = False

    def _frame_getter(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._frame is None else self._frame

    # ----- server lifecycle -----

    def start(self, *, host: str = "0.0.0.0", port: int = 8443,
              ssl_certfile: Optional[str] = None,
              ssl_keyfile: Optional[str] = None,
              log_level: str = "info") -> None:
        """Start the FastAPI/aiortc server in a background thread.

        Pass `ssl_certfile` and `ssl_keyfile` to enable HTTPS (required by
        the Quest browser for WebXR on non-localhost origins).
        """
        if self._server is not None:
            raise RuntimeError("VRStreamer already started")

        app, _ = build_app(
            static_dir=self._static_dir,
            frame_getter=self._frame_getter,
            on_pose_msg=self._on_pose_msg,
            fps=self._fps,
            sbs_size=(self._renderer.sbs_h, self._renderer.sbs_w),
        )
        config = uvicorn.Config(
            app, host=host, port=port,
            ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
            log_level=log_level, access_log=False,
        )
        self._server = uvicorn.Server(config)

        def _run() -> None:
            try:
                self._server.run()
            except Exception:
                log.exception("uvicorn server crashed")

        self._server_thread = threading.Thread(target=_run, name="vr-streamer-http", daemon=True)
        self._server_thread.start()
        scheme = "https" if ssl_certfile and ssl_keyfile else "http"
        log.info("VRStreamer serving on %s://%s:%d/", scheme, host, port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
        self._server = None
        self._server_thread = None
        self._renderer.close()
