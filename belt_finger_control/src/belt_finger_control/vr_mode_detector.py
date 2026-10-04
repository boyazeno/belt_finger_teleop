"""Meta Quest 3s VR teleop — drop-in replacement for JoystickModeDetector.

Replaces the Xbox controller + external RealSense/AprilTag camera with a Quest
headset + controllers. WebXR poses and button state stream to the PC over
WebRTC, reusing the signaling server from `vr_sim_streamer` (vr_streamer.webrtc).

The right controller's 6DoF pose drives the arm via frame-to-frame delta,
gated by a clutch (right grip). Gripper DOFs and special buttons are split
across both controllers — see the mapping notes on VRController.
"""
from __future__ import annotations

import logging
import math
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytransform3d.rotations as protations
import uvicorn

import belt_finger_control.filter as bfilter

try:
    from vr_streamer.webrtc import build_app
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "vr_mode_detector requires the `vr_streamer` package (from the "
        "vr_sim_streamer project) to be importable in this environment. "
        "Install it or add it to PYTHONPATH."
    ) from e

log = logging.getLogger("belt_finger_control.vr_mode_detector")

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# CLI audio players tried, in order, to render the vibrate() feedback tone.
# Each entry is the arg list with a trailing WAV file path appended.
_AUDIO_PLAYERS = [
    ["aplay", "-q"],
    ["paplay"],
    ["play", "-q"],          # sox
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["afplay"],              # macOS
]


def _resolve_audio_player() -> Optional[list]:
    for cmd in _AUDIO_PLAYERS:
        if shutil.which(cmd[0]):
            return cmd
    return None


def _detect_lan_ip() -> Optional[str]:
    """Best-effort LAN IP detection (the address a peer on the same network
    would reach this host on). No traffic is actually sent."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _ensure_self_signed_cert() -> tuple[str, str]:
    """Generate (and cache under ~/.cache/belt_finger_control) a self-signed
    cert + key so VRController can serve HTTPS without the user having to run
    openssl manually. WebXR on the Quest requires HTTPS for non-localhost
    origins; the browser will show a security warning on first connect that
    the user must accept once."""
    cache_dir = Path.home() / ".cache" / "belt_finger_control"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cache_dir / "vr_cert.pem"
    key_path = cache_dir / "vr_key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "belt_finger_control VR")])
    san = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    lan_ip = _detect_lan_ip()
    if lan_ip:
        try:
            san.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
        except ValueError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    return str(cert_path), str(key_path)


def _write_tone_wav(path: str, duration: float, freq: float = 880.0,
                    rate: int = 16000) -> None:
    """Write a mono 16-bit sine tone of `duration` seconds to `path`."""
    n = int(rate * max(0.0, duration))
    amp = 18000
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            frames += struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / rate)))
        w.writeframes(bytes(frames))


def _tf_from_pos_quat(pos, quat_xyzw) -> np.ndarray:
    """Build a 4x4 transform from a WebXR pos + xyzw quaternion.

    Stays in the raw WebXR (Y-up) frame: delta math is frame-consistent so no
    Y-up -> Z-up conversion is needed, and this keeps the same pytransform3d
    conventions as the original camera path.
    """
    x, y, z, w = quat_xyzw
    quat_wxyz = np.array([w, x, y, z], dtype=float)
    tf = np.eye(4)
    tf[:3, :3] = protations.matrix_from_quaternion(quat_wxyz)
    tf[:3, 3] = np.asarray(pos, dtype=float)
    return tf

@dataclass
class VRControllerState:
    """Thread-safe per-frame snapshot of WebXR input."""
    tf_head: Optional[np.ndarray] = None
    tf_left: Optional[np.ndarray] = None
    tf_right: Optional[np.ndarray] = None

    # Right controller (drives the arm).
    right_trigger: float = 0.0
    right_grip: float = 0.0
    right_grip_pressed: bool = False  # clutch
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    right_stick_click: bool = False
    right_a: bool = False  # A button
    right_b: bool = False  # B button

    # Left controller.
    left_trigger: float = 0.0
    left_grip: float = 0.0
    left_grip_pressed: bool = False
    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    left_stick_click: bool = False
    left_a: bool = False  # X button
    left_b: bool = False  # Y button

    connected: bool = False

    # A is a toggle; reset/recalibrate are one-shot edges latched by the
    # controller and cleared on read.
    pause: bool = False
    reset_edge: bool = False
    recalibrate_edge: bool = False


@dataclass
class VRSpecialButtonState:
    """Mirrors joystick_mode_detector.SpecialButtonState."""
    start: bool = False
    select: bool = False
    button_y: bool = False
    tactile: bool = False


class _VRPosePublisher:
    """Compact ROS2 debug helper — publishes left/right/head controller poses
    as ``geometry_msgs/PoseStamped`` on ``/vr/{left,right,head}_pose`` so they
    can be streamed and visualised in RViz.

    Becomes a silent no-op when *rclpy* is not importable, so it is safe to
    instantiate unconditionally.
    """

    _TOPICS = {
        "left":  "/vr/left_pose",
        "right": "/vr/right_pose",
        "head":  "/vr/head_pose",
    }

    def __init__(self, frame_id: str = "world"):
        self._ok = False
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from geometry_msgs.msg import PoseStamped
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = Node("vr_pose_debug")
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._pubs = {
                k: self._node.create_publisher(PoseStamped, v, qos)
                for k, v in self._TOPICS.items()
            }
            self._PoseStamped = PoseStamped
            self._frame_id = frame_id
            self._ok = True
            log.info("[_VRPosePublisher] publishing on %s", list(self._TOPICS.values()))
        except Exception:
            log.debug("[_VRPosePublisher] rclpy unavailable; pose debug publishing disabled")

    def publish(self, hand: str, tf4x4: np.ndarray) -> None:
        """Publish a 4x4 homogeneous transform as PoseStamped on the *hand* topic."""
        if not self._ok or hand not in self._pubs:
            return
        try:
            msg = self._PoseStamped()
            msg.header.frame_id = self._frame_id
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.pose.position.x = float(tf4x4[0, 3])
            msg.pose.position.y = float(tf4x4[1, 3])
            msg.pose.position.z = float(tf4x4[2, 3])
            q_wxyz = protations.quaternion_from_matrix(tf4x4[:3, :3])
            msg.pose.orientation.w = float(q_wxyz[0])
            msg.pose.orientation.x = float(q_wxyz[1])
            msg.pose.orientation.y = float(q_wxyz[2])
            msg.pose.orientation.z = float(q_wxyz[3])
            self._pubs[hand].publish(msg)
        except Exception:
            log.debug("[_VRPosePublisher] publish error", exc_info=True)

    def destroy(self) -> None:
        if self._ok:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._ok = False


class VRController:
    """Runs the aiortc/uvicorn WebXR signaling server in a background thread
    and exposes thread-safe getters for the latest Quest controller state.

    WebXR gamepad mapping (from the static client's readGamepad):
      buttons[0]=trigger, buttons[1]=squeeze/grip, buttons[3]=thumbstick click,
      buttons[4]=A/X, buttons[5]=B/Y, axes[2..3]=thumbstick x/y.
    Per controller `a` is A (right) / X (left); `b` is B (right) / Y (left).

    Control mapping:
      - right grip held = clutch (arm tracking engaged)
      - A = pause toggle, B = reset, X = recalibrate, Y = quit
      - left grip = record toggle (special.start)
      - either thumbstick click = tactile toggle (special.tactile)
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._lock = threading.Lock()
        self._state = VRControllerState()
        self.special_button_state = VRSpecialButtonState()
        self._static_dir = _STATIC_DIR

        # Latched one-shot events (consumed by get_state).
        self._reset_pending = False
        self._recalibrate_pending = False
        self._pause = False

        # Previous-frame button values for edge detection.
        self._prev_right_a = False
        self._prev_right_b = False
        self._prev_left_a = False
        self._prev_left_b = False
        self._prev_left_grip_pressed = False
        self._prev_left_stick_click = False
        self._prev_right_stick_click = False

        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None
        self._audio_player = _resolve_audio_player()
        self._vibrate_warned = False
        self._pose_publisher = _VRPosePublisher()

    # ----- network thread (asyncio) callback -----

    def _on_pose_msg(self, msg: dict) -> None:
        """Parse a WebXR pose JSON frame. Runs on the asyncio thread."""
        try:
            state = VRControllerState(connected=True)

            head = msg.get("head")
            if isinstance(head, dict) and "pos" in head and "quat" in head:
                state.tf_head = _tf_from_pos_quat(head["pos"], head["quat"])

            for hand in ("left", "right"):
                c = msg.get(hand)
                if not isinstance(c, dict):
                    continue
                if "pos" in c and "quat" in c:
                    tf = _tf_from_pos_quat(c["pos"], c["quat"])
                    setattr(state, f"tf_{hand}", tf)
                btns = c.get("buttons") or {}
                tr = btns.get("trigger") or {}
                sq = btns.get("squeeze") or {}
                ts = btns.get("thumbstick") or {}
                a = btns.get("a") or {}
                b = btns.get("b") or {}
                axes = btns.get("axes") or []
                stick_x = float(axes[2]) if len(axes) >= 4 else (
                    float(axes[0]) if len(axes) >= 2 else 0.0)
                stick_y = float(axes[3]) if len(axes) >= 4 else (
                    float(axes[1]) if len(axes) >= 2 else 0.0)
                setattr(state, f"{hand}_trigger", float(tr.get("value") or 0.0))
                setattr(state, f"{hand}_grip", float(sq.get("value") or 0.0))
                setattr(state, f"{hand}_grip_pressed", bool(sq.get("pressed") or False))
                setattr(state, f"{hand}_stick_x", stick_x)
                setattr(state, f"{hand}_stick_y", -1*stick_y)
                setattr(state, f"{hand}_stick_click", bool(ts.get("pressed") or False))
                setattr(state, f"{hand}_a", bool(a.get("pressed") or False))
                setattr(state, f"{hand}_b", bool(b.get("pressed") or False))

            # Edge detection.
            if state.right_a and not self._prev_right_a:
                self._pause = not self._pause  # A = pause toggle
            if state.right_b and not self._prev_right_b:
                self._reset_pending = True  # B = reset
            if state.left_a and not self._prev_left_a:
                self._recalibrate_pending = True  # X = recalibrate
            if state.left_b and not self._prev_left_b:
                self.special_button_state.button_y = True  # Y = quit (latched)
            if state.left_grip_pressed and not self._prev_left_grip_pressed:
                self.special_button_state.start = not self.special_button_state.start
            stick_click = state.left_stick_click or state.right_stick_click
            prev_stick_click = self._prev_left_stick_click or self._prev_right_stick_click
            if stick_click and not prev_stick_click:
                self.special_button_state.tactile = not self.special_button_state.tactile

            self._prev_right_a = state.right_a
            self._prev_right_b = state.right_b
            self._prev_left_a = state.left_a
            self._prev_left_b = state.left_b
            self._prev_left_grip_pressed = state.left_grip_pressed
            self._prev_left_stick_click = state.left_stick_click
            self._prev_right_stick_click = state.right_stick_click

            state.pause = self._pause

            # rotate the controller poses (when panel surface align with table surface ) to match the robot base frame (Z-up, X-forward, Y-left).
            state.tf_left[:3,:3] = state.tf_left[:3,:3] @ protations.matrix_from_euler([-np.pi/2-np.pi/4, 0, -np.pi/2], 0,1,2, False)  # rotate right controller to match robot frame
            state.tf_right[:3,:3] = state.tf_right[:3,:3] @ protations.matrix_from_euler([-np.pi/2-np.pi/4, 0, -np.pi/2], 0,1,2, False)  # rotate right controller to match robot frame

            # Publish poses for RViz debug visualisation.
            for _hand in ("left", "right", "head"):
                _tf = getattr(state, f"tf_{_hand}")
                if _tf is not None:
                    self._pose_publisher.publish(_hand, _tf)

            with self._lock:
                self._state = state
        except Exception:
            log.exception("VR pose handler failed")

    # ----- server lifecycle -----

    def start(self, host: str = "0.0.0.0", port: int = 8443,
              ssl_certfile: Optional[str] = None,
              ssl_keyfile: Optional[str] = None,
              log_level: str = "warning") -> None:
        if self._server is not None:
            raise RuntimeError("VRController already started")

        # WebXR on the Quest browser requires HTTPS for non-localhost origins.
        # If the caller didn't supply a cert, generate a self-signed one so
        # the Quest can actually reach the page.
        if not (ssl_certfile and ssl_keyfile):
            try:
                ssl_certfile, ssl_keyfile = _ensure_self_signed_cert()
                log.warning(
                    "No SSL cert provided; using auto-generated self-signed "
                    "cert at %s. The Quest browser will show a security "
                    "warning on first connect — tap Advanced > Proceed.",
                    ssl_certfile)
            except Exception:
                log.exception("Failed to auto-generate self-signed cert; "
                              "serving plain HTTP (WebXR will refuse to start "
                              "from the Quest).")

        app, _ = build_app(
            static_dir=self._static_dir,
            frame_getter=lambda: None,  # no sim video; track falls back to black
            on_pose_msg=self._on_pose_msg,
            fps=60,
            sbs_size=(16, 32),
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

        self._server_thread = threading.Thread(
            target=_run, name="vr-mode-detector-http", daemon=True)
        self._server_thread.start()

        scheme = "https" if ssl_certfile and ssl_keyfile else "http"
        lan_ip = _detect_lan_ip()
        print(f"[VRController] serving on {scheme}://{host}:{port}/")
        if lan_ip and host in ("0.0.0.0", "::", ""):
            print(f"[VRController] reach from Quest at {scheme}://{lan_ip}:{port}/")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
        self._server = None
        self._server_thread = None
        self._pose_publisher.destroy()

    # ----- sim-thread API -----

    def get_state(self) -> VRControllerState:
        with self._lock:
            state = deepcopy(self._state)
            state.reset_edge = self._reset_pending
            state.recalibrate_edge = self._recalibrate_pending
            self._reset_pending = False
            self._recalibrate_pending = False
        return state

    def get_special_botton_state(self) -> VRSpecialButtonState:
        with self._lock:
            return deepcopy(self.special_button_state)

    def reset(self) -> None:
        print("[VRController] Reset done.")

    def vibrate(self, times: int = 1, duration: float = 0.5) -> None:
        """Audio feedback substitute for controller haptics.

        The WebRTC pipeline is one-way (no haptic return channel to the Quest),
        so instead play a tone on the host PC for `duration` seconds, `times`
        times, mirroring JoystickController.vibrate's cadence. Best-effort and
        non-fatal; falls back to the terminal bell if no audio player is found.
        """
        for _ in range(max(1, int(times))):
            played = False
            if self._audio_player is not None:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                        _write_tone_wav(f.name, duration)
                        subprocess.run(self._audio_player + [f.name],
                                       check=False, timeout=duration + 5.0)
                    played = True
                except Exception:
                    log.exception("VRController.vibrate() audio playback failed")
            if not played:
                if not self._vibrate_warned:
                    log.warning("VRController.vibrate(): no audio player found; "
                                "using terminal bell")
                    self._vibrate_warned = True
                print("\a", end="", flush=True)
                time.sleep(duration)
            time.sleep(0.5)


class VRModeDetector:
    """Drop-in replacement for JoystickModeDetector backed by a Quest 3s.

    Same interface: start() / control() / reset() / __del__, and a
    `.joystick_controller` attribute exposing get_special_botton_state() /
    vibrate(). Existing scripts switch with a one-line import change.
    """

    def __init__(self, control_arm: bool = True, *, passthrough: bool = True,
                 host: str = "0.0.0.0", port: int = 8443,
                 ssl_certfile: Optional[str] = None,
                 ssl_keyfile: Optional[str] = None,
                 arm_gain: float = 1.0,
                 pose_translation_min_cutoff: float = 1.5,
                 pose_translation_beta: float = 5.0,
                 pose_rotation_min_cutoff: float = 1.5,
                 pose_rotation_beta: float = 0.5,
                 debug: bool = False):
        # Gripper-DOF scales are unchanged from JoystickModeDetector (their
        # inputs are still trigger 0..1 / stick -1..1). The "robot" scale is
        # new: frame-to-frame deltas are ~10-50x smaller than the original
        # init->current displacement, so the input span is much smaller.
        # TODO: tune "robot" scale and arm_gain on hardware.
        self.scale_max = {
            "robot": (np.array([0.005, 0.005, 0.005, 0.05, 0.05, 0.05]),
                      np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005])),
            "opening": (0.00, 0.08), "transy": (-1.0, 0.5),
            "rotz": (1.0, -0.5), "rotx": (1.0, -0.5),
        }
        self.scale_min = {
            "robot": (np.array([-0.005, -0.005, -0.005, -0.05, -0.05, -0.05]),
                      np.array([-0.005, -0.005, -0.005, -0.005, -0.005, -0.005])),
            "opening": (1.0, 0.0), "transy": (1.0, -0.5),
            "rotz": (-1.0, 0.5), "rotx": (-1.0, 0.5),
        }

        self.debug = debug
        self.control_arm = control_arm
        self.passthrough = passthrough
        self.arm_gain = arm_gain
        self.right_pose_filter = bfilter.OneEuroPoseFilter(
            translation_min_cutoff=pose_translation_min_cutoff,
            translation_beta=pose_translation_beta,
            rotation_min_cutoff=pose_rotation_min_cutoff,
            rotation_beta=pose_rotation_beta)
        self._host = host
        self._port = port
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile

        self.joystick_controller = VRController(debug=self.debug)
        self.filter_dict = bfilter.FilterDict(
            bfilter.MovingAverageFilter,
            # bfilter.ExponentialMovingAverageFilter,
            ["robot", "opening", "transy", "rotz", "rotx"])

        self.tf_ee_2_init = self.get_tf_ee_2_init()

        # Clutch / frame-to-frame delta state.
        self.last_tf_right: Optional[np.ndarray] = None
        self.last_left_trigger: float = 0.0

    def get_tf_ee_2_init(self):
        tf_ee_2_init = np.eye(4)
        tf_ee_2_init[:3, 0] = [-1,0,0]#[0, 1, 0]
        tf_ee_2_init[:3, 1] = [0,0,-1]#[1, 0, 0]
        tf_ee_2_init[:3, 2] = [0,-1,0]#[0, 0, -1]
        return tf_ee_2_init

    def __del__(self):
        try:
            self.joystick_controller.stop()
        except Exception:
            pass

    def start(self):
        self.joystick_controller.start(
            host=self._host, port=self._port,
            ssl_certfile=self._ssl_certfile, ssl_keyfile=self._ssl_keyfile)
        mode = "ar" if self.passthrough else "vr"
        print(f"[VRModeDetector] Open https://<host-ip>:{self._port}/?mode={mode} "
              f"on the Quest browser, tap Connect, then Enter XR.")
        print("[VRModeDetector] Waiting for Quest to connect...")
        while not self.joystick_controller.get_state().connected:
            time.sleep(0.2)
        print("[VRModeDetector] Quest connected.")

    def reset(self):
        self.joystick_controller.reset()
        self.filter_dict.reset()
        self.right_pose_filter.reset()
        self.last_tf_right = None

    def control(self, robot_state):
        state = self.joystick_controller.get_state()
        special = self.joystick_controller.get_special_botton_state()

        # WebXR pose frames arrive asynchronously and contain small position
        # and orientation jitter. Apply adaptive One Euro filtering to the
        # absolute right-controller pose before forming the frame-to-frame
        # transform below. The cutoff rises during deliberate motion, avoiding
        # the fixed delay of an EMA while retaining a valid rigid transform.
        if state.tf_right is None:
            self.right_pose_filter.reset()
            self.last_tf_right = None
        else:
            state.tf_right = self.right_pose_filter.update(state.tf_right)

        if special.button_y:
            print("Quit now!")
            return None, None

        if state.reset_edge:
            self.reset()

        if state.recalibrate_edge:
            # No marker array to recalibrate; re-anchor the arm delta instead.
            self.last_tf_right = None
            self.right_pose_filter.reset()
            self.filter_dict.reset()

        arm_action = np.zeros((6,))
        gripper_action = np.array([0.08, 0.0, 0.0, 0.0])


        A = np.eye(4)
        A[:3,:3] = protations.matrix_from_euler([np.pi/2, np.pi/2, 0.0], i=0, j=1, k=2, extrinsic=False) # tf_w_q #TODO
        tf_w_ee = robot_state.eef_pose

        if not state.pause:
            if self.control_arm:
                clutch = state.right_grip_pressed
                if clutch and state.tf_right is not None:
                    if self.last_tf_right is None:
                        # Re-anchor on (re-)engage: zero action, no jump.
                        self.last_tf_right = state.tf_right # tf_q_h
                    else:
                        # This one doesn't require tf_w_q to be known exactly. Only the rotation of tf_w_q is needed.
                        # Get tf_w_q when you face the robot and reposition the quest. 
                        # Check the rviz output of the Quest world frarm and the robot base frame. The difference should be A (tf_w_q).
                        tf_q_hp = state.tf_right
                        tf_q_h = self.last_tf_right
                        def inv(mat):
                            return np.linalg.inv(mat)
                        tf_ee_eep = inv(tf_w_ee) @ A @ tf_q_hp @ inv(tf_q_h) @ inv(A) @ tf_w_ee

                        rxyz_aa = protations.axis_angle_from_matrix(tf_ee_eep[:3, :3], strict_check=False)
                        rxyz = rxyz_aa[:3]*rxyz_aa[3] # axis-angle to euler
                        rxyz[1:] = -rxyz[1:] # flip y and z to match the robot frame

                        xyz = A @ (tf_q_hp[:, 3] - tf_q_h[:, 3])
                        xyz = xyz[:3] # translation delta

                        print(f"raw xyz: {xyz}, rxyz: {rxyz}")
                        # rxyz[:] = 0.0 #!For testing, disable rotation control
                        arm_action = np.concatenate([xyz, rxyz], axis=0) * self.arm_gain
                        self.last_tf_right = state.tf_right
                else:
                    # Clutch released: freeze and drop the anchor.
                    self.last_tf_right = None

                # arm_action = arm_action * (0.8 if state.right_trigger > 0.5 else 1.0)
                # arm_action = self.normalize_action("robot", arm_action)

            opening = self.normalize_action("opening", state.left_trigger)
            transy = self.normalize_action("transy", state.left_stick_y)
            rotz = self.normalize_action("rotz", state.right_stick_x)
            rotx = self.normalize_action("rotx", state.right_stick_y)
            gripper_action = np.array([opening, transy, rotz, rotx])

            self.last_left_trigger = state.left_trigger
        else:
            opening = self.normalize_action("opening", self.last_left_trigger)
            gripper_action = np.array([opening, 0.0, 0.0, 0.0])

        return arm_action, gripper_action

    def normalize_action(self, key, val, use_filter=True):
        if key in self.scale_max:
            val = np.clip((val - self.scale_min[key][0]) / (self.scale_max[key][0] - self.scale_min[key][0]) * 2 - 1, -1.0, 1.0)  # convert to [-1,1]
            if key == "opening":
                val = (val + 1.0) / 2.0
            else:
                val = (val + 1.0) / 2.0  # dont slow down
            rescaled_val = val * (self.scale_max[key][1] - self.scale_min[key][1]) + self.scale_min[key][1]  # remap to [min,max]
            if use_filter:
                return self.filter_dict.update(key, rescaled_val)
            else:
                return rescaled_val
        else:
            raise Exception(f"Invalid key: {key}")
