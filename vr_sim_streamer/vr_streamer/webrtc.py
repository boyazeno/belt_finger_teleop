"""FastAPI app + aiortc signaling, factored to be created on demand by VRStreamer."""
from __future__ import annotations

import asyncio
import fractions
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger("vr_streamer.webrtc")

VIDEO_CLOCK_RATE = 90_000


class _Offer(BaseModel):
    sdp: str
    type: str


class StereoFrameTrack(VideoStreamTrack):
    """Pulls the latest SBS frame from a thread-safe getter at fixed fps."""

    kind = "video"

    def __init__(self, frame_getter: Callable[[], Optional[np.ndarray]],
                 *, fps: int = 30, fallback_size: tuple[int, int] = (480, 1280)):
        super().__init__()
        self._get = frame_getter
        self._fps = fps
        self._period = 1.0 / float(fps)
        self._tick = VIDEO_CLOCK_RATE // fps
        self._n = 0
        self._t0: Optional[float] = None
        h, w = fallback_size
        self._black = np.zeros((h, w, 3), dtype=np.uint8)

    async def recv(self) -> VideoFrame:
        if self._t0 is None:
            self._t0 = time.monotonic()
        target = self._t0 + (self._n + 1) * self._period
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        arr = self._get()
        if arr is None:
            arr = self._black
        frame = VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = self._n * self._tick
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
        self._n += 1
        return frame


def build_app(
    *,
    static_dir: Path,
    frame_getter: Callable[[], Optional[np.ndarray]],
    on_pose_msg: Callable[[dict], None],
    fps: int = 30,
    sbs_size: tuple[int, int] = (480, 1280),
) -> tuple[FastAPI, set[RTCPeerConnection]]:
    """Build a FastAPI app that serves `static_dir/index.html` and offers the
    given frame source over WebRTC. `on_pose_msg` is invoked from the asyncio
    thread for every parsed pose JSON received on the 'pose' DataChannel."""
    app = FastAPI(title="vr_streamer")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    pcs: set[RTCPeerConnection] = set()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/offer")
    async def offer(params: _Offer) -> dict[str, str]:
        pc = RTCPeerConnection()
        pcs.add(pc)
        log.info("PeerConnection created (total=%d)", len(pcs))

        @pc.on("connectionstatechange")
        async def _state() -> None:
            log.info("connection state: %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                pcs.discard(pc)

        @pc.on("datachannel")
        def _dc(channel) -> None:
            log.info("DataChannel opened: label=%s", channel.label)

            @channel.on("message")
            def _msg(message) -> None:
                if not isinstance(message, str) or channel.label != "pose":
                    return
                try:
                    on_pose_msg(json.loads(message))
                except Exception:
                    log.exception("pose handler failed")

        pc.addTrack(StereoFrameTrack(frame_getter, fps=fps, fallback_size=sbs_size))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=params.sdp, type=params.type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await asyncio.gather(*(pc.close() for pc in pcs), return_exceptions=True)
        pcs.clear()

    return app, pcs
