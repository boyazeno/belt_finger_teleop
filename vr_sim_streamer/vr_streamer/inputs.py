"""Public input snapshot dataclasses, surfaced via VRStreamer.inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass
class HandInput:
    kind: Literal["controller", "hand", "none"] = "none"
    pos: Optional[np.ndarray] = None       # MuJoCo world (3,)
    quat: Optional[np.ndarray] = None      # MuJoCo (w,x,y,z)
    trigger: float = 0.0                   # 0..1, or pinch strength for hands
    trigger_pressed: bool = False
    squeeze: float = 0.0
    squeeze_pressed: bool = False
    thumbstick: tuple[float, float] = (0.0, 0.0)
    thumbstick_pressed: bool = False
    axes: tuple[float, ...] = ()  # full gamepad.axes array as sent by the browser
    a: bool = False
    b: bool = False


@dataclass
class Inputs:
    head_pos: Optional[np.ndarray] = None
    head_quat: Optional[np.ndarray] = None
    left: HandInput = field(default_factory=HandInput)
    right: HandInput = field(default_factory=HandInput)
