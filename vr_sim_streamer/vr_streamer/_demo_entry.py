"""Console-script entry point: runs the bundled demo scene with VRStreamer.

Installed as `vr-streamer-demo` via [project.scripts] in pyproject.toml.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mujoco

from . import VRStreamer

DEFAULT_SCENE = Path(__file__).resolve().parent / "scene.xml"


def run_demo() -> None:
    p = argparse.ArgumentParser(prog="vr-streamer-demo")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--cert", "--ssl-certfile", dest="cert", default=None)
    p.add_argument("--key",  "--ssl-keyfile",  dest="key",  default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("vr-streamer-demo")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    streamer = VRStreamer(model, data, fps=args.fps)
    streamer.start(host=args.host, port=args.port,
                   ssl_certfile=args.cert, ssl_keyfile=args.key)

    period = 1.0 / args.fps
    next_t = time.monotonic()
    try:
        while True:
            mujoco.mj_step(model, data)
            streamer.sync()
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
    run_demo()
