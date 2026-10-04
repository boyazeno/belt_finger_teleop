"""Helpers to inject stereo cameras / VR rig into a MuJoCo model.

Background: a compiled `mujoco.MjModel` is immutable — you cannot add bodies
or cameras to it at runtime. The supported paths are:

    1. **Programmatic** (preferred, mujoco >= 3.2): build / mutate a
       `mujoco.MjSpec`, then `spec.compile()`. Use `attach_stereo_cameras`
       or `attach_vr_rig` below.
    2. **XML-time**: paste `VR_RIG_XML_SNIPPET` into your scene XML before
       compiling.
    3. **String surgery** on user XML: `merge_vr_rig_xml(xml_str)` injects
       the snippet into the first <worldbody> of an existing XML string.

Choose (1) if available; fall back to (2) or (3) otherwise.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Optional, Union

import mujoco

PathOrStr = Union[str, "os.PathLike[str]"]

# Default IPD (Quest 3 average) and FOV.
DEFAULT_IPD = 0.064
DEFAULT_FOVY = 90.0


VR_RIG_XML_SNIPPET = """\
<!-- vr_streamer rig: paste inside <worldbody> -->
<body name="head" mocap="true" pos="0 0 1.6">
  <geom type="sphere" size="0.04" rgba="1 1 1 0.25" contype="0" conaffinity="0"/>
  <camera name="left_eye"  pos="-0.032 0 0" fovy="90"/>
  <camera name="right_eye" pos=" 0.032 0 0" fovy="90"/>
</body>
<body name="controller_left" mocap="true" pos="-0.2 0.2 1.2">
  <geom type="box" size="0.025 0.025 0.07" rgba="1.0 0.55 0.1 1" contype="0" conaffinity="0"/>
</body>
<body name="controller_right" mocap="true" pos="0.2 0.2 1.2">
  <geom type="box" size="0.025 0.025 0.07" rgba="0.2 0.7 1.0 1" contype="0" conaffinity="0"/>
</body>
"""


def _require_mjspec():
    if not hasattr(mujoco, "MjSpec"):
        raise RuntimeError(
            "mujoco.MjSpec not available. Upgrade to mujoco>=3.2 or use the "
            "XML path: paste VR_RIG_XML_SNIPPET into your <worldbody>, or "
            "call merge_vr_rig_xml(xml_str)."
        )


def attach_stereo_cameras(
    spec: "mujoco.MjSpec",
    parent_body: str,
    *,
    ipd: float = DEFAULT_IPD,
    fovy: float = DEFAULT_FOVY,
    left_name: str = "left_eye",
    right_name: str = "right_eye",
) -> None:
    """Add two eye cameras to an existing body in `spec`.

    The body must move with the user's head. If you don't have one, call
    `attach_vr_rig` instead, which adds the head as a mocap body.
    """
    _require_mjspec()
    body = spec.body(parent_body)
    if body is None:
        raise ValueError(f"body '{parent_body}' not found in spec")
    body.add_camera(name=left_name,  pos=[-ipd / 2.0, 0.0, 0.0], fovy=fovy)
    body.add_camera(name=right_name, pos=[ ipd / 2.0, 0.0, 0.0], fovy=fovy)


def attach_vr_rig(
    spec: "mujoco.MjSpec",
    *,
    head_pos: tuple[float, float, float] = (0.0, 0.0, 1.6),
    ipd: float = DEFAULT_IPD,
    fovy: float = DEFAULT_FOVY,
    head_name: str = "head",
    left_controller: str = "controller_left",
    right_controller: str = "controller_right",
    left_eye: str = "left_eye",
    right_eye: str = "right_eye",
) -> None:
    """Add a head (mocap) with two stereo cameras and two controller mocaps.

    All five bodies become drivable from VRStreamer by name.
    """
    _require_mjspec()
    world = spec.worldbody

    head = world.add_body(name=head_name, mocap=True, pos=list(head_pos))
    head.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04, 0, 0],
                  rgba=[1, 1, 1, 0.25], contype=0, conaffinity=0)
    head.add_camera(name=left_eye,  pos=[-ipd / 2.0, 0, 0], fovy=fovy)
    head.add_camera(name=right_eye, pos=[ ipd / 2.0, 0, 0], fovy=fovy)

    cl = world.add_body(name=left_controller, mocap=True, pos=[-0.2, 0.2, 1.2])
    cl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.07],
                rgba=[1.0, 0.55, 0.1, 1], contype=0, conaffinity=0)

    cr = world.add_body(name=right_controller, mocap=True, pos=[0.2, 0.2, 1.2])
    cr.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.07],
                rgba=[0.2, 0.7, 1.0, 1], contype=0, conaffinity=0)


def _looks_like_xml(s: str) -> bool:
    return s.lstrip().startswith("<")


def _load_xml(xml_or_path: PathOrStr) -> str:
    if isinstance(xml_or_path, (bytes, bytearray)):
        return bytes(xml_or_path).decode("utf-8")
    s = os.fspath(xml_or_path) if not isinstance(xml_or_path, str) else xml_or_path
    if _looks_like_xml(s):
        return s
    with open(s, "r", encoding="utf-8") as f:
        return f.read()


def add_eye_cameras_to_xml(
    xml_or_path: PathOrStr,
    *,
    parent_body: Optional[str] = None,
    create_head_if_missing: bool = True,
    head_name: str = "head",
    head_pos: tuple[float, float, float] = (0.0, 0.0, 1.6),
    head_marker: bool = True,
    ipd: float = DEFAULT_IPD,
    fovy: float = DEFAULT_FOVY,
    left_name: str = "left_eye",
    right_name: str = "right_eye",
    output_path: Optional[PathOrStr] = None,
) -> str:
    """Add left/right eye cameras to a MuJoCo XML and return the modified string.

    For a fresh scene that doesn't yet know about VR, the typical call is just
    ``add_eye_cameras_to_xml(xml)``: a `head` **mocap** body is created in
    `<worldbody>` (so VRStreamer can drive it from the headset pose) and the
    two cameras are attached inside it.

    Parameters
    ----------
    xml_or_path:
        Either an XML string (auto-detected by leading ``<``) or a path to an
        XML file.
    parent_body:
        Name of the body the cameras should be attached to. If None, the
        target is `head_name` (default "head").
    create_head_if_missing:
        If the target body doesn't exist, create it as a mocap body at
        `head_pos` in `<worldbody>`. Set False to require the body to
        pre-exist (errors out if not found).
    head_name, head_pos, head_marker:
        Used only when creating the head. `head_marker=True` adds a small
        translucent sphere geom so the head is visible in MuJoCo viewers.
    ipd, fovy:
        Stereo separation (meters) and per-eye vertical FOV (degrees).
    left_name, right_name:
        Camera names. These are what `VRStreamer` looks up by default.
    output_path:
        If given, also write the modified XML to this path.

    Returns
    -------
    The modified XML as a string.
    """
    xml = _load_xml(xml_or_path)

    # Preserve any XML declaration / leading comments verbatim by capturing
    # the prologue, then re-emitting it after ElementTree serialization.
    prologue = ""
    m = re.match(r"^(\s*<\?xml[^?]*\?>\s*)", xml)
    if m:
        prologue = m.group(1)
        xml_body = xml[m.end():]
    else:
        xml_body = xml

    root = ET.fromstring(xml_body)
    if root.tag != "mujoco":
        raise ValueError(f"expected root <mujoco>, got <{root.tag}>")

    target_name = parent_body or head_name
    parent = _find_body(root, target_name)

    if parent is None:
        if not create_head_if_missing:
            raise ValueError(
                f"body '{target_name}' not found in XML and "
                f"create_head_if_missing=False")
        # Make sure <worldbody> exists, then create the head mocap body.
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        parent = ET.SubElement(worldbody, "body", {
            "name": target_name,
            "mocap": "true",
            "pos": f"{_fmt(head_pos[0])} {_fmt(head_pos[1])} {_fmt(head_pos[2])}",
        })
        if head_marker:
            ET.SubElement(parent, "geom", {
                "type": "sphere",
                "size": "0.04",
                "rgba": "1 1 1 0.25",
                "contype": "0",
                "conaffinity": "0",
            })
    else:
        # The body must be drivable. mocap bodies are the supported case;
        # otherwise the user owns its kinematics and we just attach cameras.
        existing = {c.get("name") for c in parent.findall("camera")}
        if left_name in existing or right_name in existing:
            raise ValueError(
                f"camera name(s) already present under '{target_name}': "
                f"{existing & {left_name, right_name}}")

    half = ipd / 2.0
    fovy_str = _fmt(fovy)
    ET.SubElement(parent, "camera", {
        "name": left_name,
        "pos": f"{_fmt(-half)} 0 0",
        "fovy": fovy_str,
    })
    ET.SubElement(parent, "camera", {
        "name": right_name,
        "pos": f"{_fmt(half)} 0 0",
        "fovy": fovy_str,
    })

    out = prologue + ET.tostring(root, encoding="unicode")
    if output_path is not None:
        with open(os.fspath(output_path), "w", encoding="utf-8") as f:
            f.write(out)
    return out


def _find_body(root: ET.Element, name: str) -> Optional[ET.Element]:
    """DFS for a <body name="..."> anywhere under <mujoco>."""
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _fmt(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def merge_vr_rig_xml(xml: str) -> str:
    """Inject VR_RIG_XML_SNIPPET into the first <worldbody> of `xml`.

    Use this when you only have an XML string and `mujoco.MjSpec` is not
    available. Returns a new XML string ready for `mujoco.MjModel.from_xml_string`.
    """
    m = re.search(r"<worldbody[^>]*>", xml)
    if not m:
        raise ValueError("no <worldbody> tag found in XML")
    insert_at = m.end()
    return xml[:insert_at] + "\n" + VR_RIG_XML_SNIPPET + xml[insert_at:]
