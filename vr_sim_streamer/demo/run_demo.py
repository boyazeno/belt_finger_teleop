"""Runnable demo: load demo/scene.xml, start VRStreamer, drive the sim loop.

    MUJOCO_GL=egl python -m demo.run_demo \
        --cert certs/cert.pem --key certs/key.pem --port 8443

Open https://<server-ip>:<port>/ from the Quest browser, accept the cert,
press Connect, then Enter VR.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python demo/run_demo.py` without installing the package: put the
# repo root on sys.path so `vr_streamer` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mujoco  # noqa: E402

from vr_streamer import VRStreamer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO / "demo" / "scene.xml"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--cert", "--ssl-certfile", dest="cert", default=None, help="ssl cert pem path")
    p.add_argument("--key",  "--ssl-keyfile",  dest="key",  default=None, help="ssl key pem path")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scene", default=str(SCENE_PATH))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("demo")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    streamer = VRStreamer(model, data, fps=args.fps)
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

            # Edge-triggered trigger prints + a half-second heartbeat.
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
                        "L[%s] tr=%.2f sq=%.2f stick=(%+.2f,%+.2f) | "
                        "R[%s] tr=%.2f sq=%.2f stick=(%+.2f,%+.2f)",
                        L.kind, L.trigger, L.squeeze, L.thumbstick[0], L.thumbstick[1],
                        R.kind, R.trigger, R.squeeze, R.thumbstick[0], R.thumbstick[1],
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
