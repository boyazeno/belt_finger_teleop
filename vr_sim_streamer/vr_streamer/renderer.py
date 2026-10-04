"""Offscreen stereo renderer producing a 1280x480 SBS BGR frame."""
from __future__ import annotations

import logging
from typing import Optional

import mujoco
import numpy as np

log = logging.getLogger("vr_streamer.renderer")


class StereoRenderer:
    """Wraps two `mujoco.Renderer` calls into one SBS frame."""

    def __init__(self, model: mujoco.MjModel, *,
                 left_eye: str = "left_eye",
                 right_eye: str = "right_eye",
                 eye_width: int = 640,
                 eye_height: int = 480):
        self._model = model
        self._left = left_eye
        self._right = right_eye
        self.eye_w = eye_width
        self.eye_h = eye_height
        self.sbs_w = eye_width * 2
        self.sbs_h = eye_height
        self._renderer: Optional[mujoco.Renderer] = None
        self._sbs = np.empty((self.sbs_h, self.sbs_w, 3), dtype=np.uint8)

    def lazy_init(self) -> None:
        if self._renderer is not None:
            return
        # MuJoCo's offscreen framebuffer is sized by `<visual><global
        # offwidth/offheight>` in the model XML. If the requested eye
        # resolution exceeds it, Renderer.__init__ fails and its destructor
        # then errors with "no attribute '_gl_context'". Grow the model's
        # offscreen size to fit before constructing the Renderer.
        g = self._model.vis.global_
        need_w, need_h = self.eye_w, self.eye_h
        if g.offwidth < need_w:
            log.info("growing model offwidth %d -> %d", g.offwidth, need_w)
            g.offwidth = need_w
        if g.offheight < need_h:
            log.info("growing model offheight %d -> %d", g.offheight, need_h)
            g.offheight = need_h
        self._renderer = mujoco.Renderer(
            self._model, height=self.eye_h, width=self.eye_w)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """Render both eyes and return an SBS BGR frame (sbs_h, sbs_w, 3)."""
        self.lazy_init()
        assert self._renderer is not None
        self._renderer.update_scene(data, camera=self._left)
        left = self._renderer.render()
        self._renderer.update_scene(data, camera=self._right)
        right = self._renderer.render()
        # RGB -> BGR for aiortc VideoFrame ("bgr24").
        self._sbs[:, :self.eye_w, :] = left[:, :, ::-1]
        self._sbs[:, self.eye_w:, :] = right[:, :, ::-1]
        return self._sbs

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
