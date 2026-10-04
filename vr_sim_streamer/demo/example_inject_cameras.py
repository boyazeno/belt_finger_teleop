"""Example: take a MuJoCo XML that has no eye cameras, inject left/right
eye cameras under a chosen body, compile the model, and run the VR streamer.

Run::

    MUJOCO_GL=egl python -m demo.example_inject_cameras \
        --port 8443 --cert certs/cert.pem --key certs/key.pem
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mujoco  # noqa: E402

from vr_streamer import VRStreamer, add_eye_cameras_to_xml  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INPUT_XML = REPO / "demo" / "scene_fresh.xml"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(INPUT_XML),
                   help="path to a MuJoCo XML (head body optional)")
    p.add_argument("--parent-body", default=None,
                   help="body to attach the cameras under. Defaults to 'head'; "
                        "the head mocap body is created if missing.")
    p.add_argument("--ipd", type=float, default=0.064)
    p.add_argument("--fovy", type=float, default=90.0)
    p.add_argument("--save-modified", default=None,
                   help="optional path to write the modified XML for inspection")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--cert", "--ssl-certfile", dest="cert", default=None)
    p.add_argument("--key",  "--ssl-keyfile",  dest="key",  default=None)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("inject-cameras")

    # 1) Inject the two eye cameras into the user's XML.
    modified_xml = add_eye_cameras_to_xml(
        args.input,
        parent_body=args.parent_body,
        ipd=args.ipd,
        fovy=args.fovy,
        output_path=args.save_modified,
    )
    log.info("injected left_eye / right_eye under <body name=%r>", args.parent_body)
    if args.save_modified:
        log.info("modified XML written to %s", args.save_modified)

    # 2) Compile the modified XML directly (no temp file needed).
    model = mujoco.MjModel.from_xml_string(modified_xml)
    data = mujoco.MjData(model)
    log.info("compiled: nbody=%d ncam=%d", model.nbody, model.ncam)

    # 3) Hand off to the streamer. The fresh scene has no controller mocap
    # bodies, so disable them; only the head is driven from VR.
    streamer = VRStreamer(
        model, data, fps=args.fps,
        left_controller=None, right_controller=None,
    )
    streamer.start(host=args.host, port=args.port,
                   ssl_certfile=args.cert, ssl_keyfile=args.key)

    period = 1.0 / args.fps
    next_t = time.monotonic()
    last_log = 0.0
    prev_pressed = {"left": False, "right": False}
    try:
        while True:
            mujoco.mj_step(model, data)
            streamer.sync()

            now = time.monotonic()
            for side in ("left", "right"):
                hi = getattr(streamer.inputs, side)
                if hi.trigger_pressed != prev_pressed[side]:
                    log.info("trigger %s %s (kind=%s, value=%.2f)",
                             side, "DOWN" if hi.trigger_pressed else "UP",
                             hi.kind, hi.trigger)
                    prev_pressed[side] = hi.trigger_pressed
            if now - last_log > 0.5:
                last_log = now
                L, R = streamer.inputs.left, streamer.inputs.right
                if L.kind != "none" or R.kind != "none":
                    log.info(
                        "L[%s] tr=%.2f stick=(%+.2f,%+.2f) | "
                        "R[%s] tr=%.2f stick=(%+.2f,%+.2f)",
                        L.kind, L.trigger, L.thumbstick[0], L.thumbstick[1],
                        R.kind, R.trigger, R.thumbstick[0], R.thumbstick[1],
                    )

            next_t += period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
