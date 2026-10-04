# vr_streamer: WebXR Teleoperation and Stereo Streaming for MuJoCo

`vr_streamer` is a drop-in Python library that connects a Meta Quest headset (via the Quest Browser and WebXR) to a MuJoCo simulation running on a PC:

1. **Head tracking:** the headset streams its 6-DoF pose to the PC, where it drives a pair of stereo "eye" cameras in the MuJoCo scene.
2. **Teleoperation:** controller (or hand-tracking) poses and button states are streamed to the PC and exposed to your simulation loop.
3. **Video streaming:** MuJoCo renders the left and right eye views, which are concatenated side by side and streamed back to the headset over WebRTC for stereoscopic display.

## Architecture

| Component | Technology |
|---|---|
| Web server and signaling | `FastAPI` + `uvicorn` |
| Transport | WebRTC via `aiortc` (video track + `RTCDataChannel` for poses) |
| Simulation and rendering | `mujoco` (offscreen rendering, e.g. `MUJOCO_GL=egl`) |
| Frame processing | `opencv-python` (side-by-side concatenation) |
| Client | WebXR page using `Three.js`, served from `vr_streamer/static/index.html` |

Default stream settings: 2 x 640x480 eye images (1280x480 side-by-side) at 30 FPS.

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

## HTTPS certificate

The Quest Browser only allows WebXR on secure origins (HTTPS or `localhost`). Create a self-signed certificate for the PC, replacing `<PC_IP>` with the PC's IP address on the local network:

```bash
mkdir -p certs && openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout certs/key.pem -out certs/cert.pem -days 365 \
    -subj "/CN=<PC_IP>" -addext "subjectAltName=IP:<PC_IP>,DNS:localhost"
```

Do not commit the generated key. Alternatively, connect the Quest over USB and forward the port with `adb reverse tcp:8443 tcp:8443`, then open the page via `localhost`.

## Running the demos

Run from this directory:

```bash
# Bundled scene: head + controllers drive mocap bodies
MUJOCO_GL=egl python -m demo.run_demo --cert certs/cert.pem --key certs/key.pem --port 8443
# (equivalently, after installation: vr-streamer-demo --cert certs/cert.pem --key certs/key.pem)

# Teleoperate a 6-DoF arm with the right controller (Jacobian-based delta IK)
MUJOCO_GL=egl python demo/example_vr_arm.py --port 8443 --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem

# Inject eye cameras into an XML scene that has none
MUJOCO_GL=egl python -m demo.example_inject_cameras --port 8443 --cert certs/cert.pem --key certs/key.pem
```

Then open `https://<PC_IP>:8443/` in the Quest Browser, accept the certificate warning, press **Connect**, and then **Enter VR**.

## Using the library in your own project

```python
import mujoco
from vr_streamer import VRStreamer, attach_vr_rig

spec = mujoco.MjSpec.from_file("my_robot.xml")
attach_vr_rig(spec)                       # adds head + L/R controllers + eye cameras
model = spec.compile()
data = mujoco.MjData(model)

streamer = VRStreamer(model, data)
streamer.start(host="0.0.0.0", port=8443,
               ssl_certfile="certs/cert.pem", ssl_keyfile="certs/key.pem")

while True:
    mujoco.mj_step(model, data)
    streamer.sync()                       # apply poses, render, publish frame
    # streamer.inputs.right.pos / .quat / .trigger / .a / .b ...
```

If your model already contains the head body, eye cameras and controller bodies, pass their names instead of calling `attach_vr_rig`:

```python
VRStreamer(model, data,
           head_body="my_head", left_eye="cam_l", right_eye="cam_r",
           left_controller="hand_l", right_controller="hand_r")
```

Main public API:

| Name | Purpose |
|---|---|
| `VRStreamer` | Runs the web/WebRTC server in a background thread; `sync()` applies the latest poses, renders and publishes a frame; `inputs` holds the latest head/controller state |
| `VRControllerLeader` | Converts relative controller motion into an end-effector target `[x, y, z, qw, qx, qy, qz, gripper_opening]`, with a re-anchor button |
| `attach_vr_rig`, `attach_stereo_cameras` | Add the head/controller mocap bodies and eye cameras to an `MjSpec` |
| `add_eye_cameras_to_xml`, `merge_vr_rig_xml` | Same, operating on MJCF XML strings or files |
| `webxr_to_mujoco` | Convert a WebXR pose to the MuJoCo frame |

## Coordinate conventions

WebXR uses a right-handed, Y-up frame (-Z forward), while MuJoCo uses a right-handed, Z-up frame. All incoming positions and quaternions are converted with `webxr_to_mujoco` (a +90° rotation about the world X axis) before they are applied to the simulation. Quaternions exposed by `VRStreamer.inputs` are in MuJoCo `(w, x, y, z)` order.

## Notes on latency

Video is streamed without frame buffering, and `sync()` always publishes the most recent frame. Rendering dominates the per-frame cost, so call `sync()` at the desired video rate rather than at every physics step if the simulation runs faster than the stream.
