"""vr_streamer — drop-in WebXR teleop + stereo viz for MuJoCo projects.

Typical usage in a host project::

    import mujoco
    from vr_streamer import VRStreamer, attach_vr_rig

    spec = mujoco.MjSpec.from_file("my_robot.xml")
    attach_vr_rig(spec)                       # adds head + L/R controllers + eye cams
    model = spec.compile()
    data = mujoco.MjData(model)

    streamer = VRStreamer(model, data)
    streamer.start(host="0.0.0.0", port=8443,
                   ssl_certfile="certs/cert.pem", ssl_keyfile="certs/key.pem")

    while True:
        mujoco.mj_step(model, data)
        streamer.sync()                       # apply poses + render + publish frame
        # streamer.inputs.left.trigger / .pinch / .pos / .quat ...

If your model already has the bodies/cameras, you can pass their names::

    VRStreamer(model, data,
               head_body="my_head", left_eye="cam_l", right_eye="cam_r",
               left_controller="hand_l", right_controller="hand_r")
"""
from .cameras import (
    attach_stereo_cameras,
    attach_vr_rig,
    add_eye_cameras_to_xml,
    merge_vr_rig_xml,
    VR_RIG_XML_SNIPPET,
)
from .inputs import HandInput, Inputs
from .streamer import VRStreamer
from .teleop import VRControllerLeader
from .transforms import webxr_to_mujoco

__all__ = [
    "VRStreamer",
    "VRControllerLeader",
    "attach_stereo_cameras",
    "attach_vr_rig",
    "add_eye_cameras_to_xml",
    "merge_vr_rig_xml",
    "VR_RIG_XML_SNIPPET",
    "Inputs",
    "HandInput",
    "webxr_to_mujoco",
]
