import os
import re
import xml.etree.ElementTree as ET


def preprocess_urdf(
    urdf_path: str, fixed_joints: list[str] | None = None
) -> str:
    """Preprocess a URDF file for use with Isaac Lab.

    Removes the world link and its associated joint (if present),
    resolves mesh filenames to absolute paths, and optionally fixes
    specified joints. The result is written to a cache directory so
    the original file is not modified.
    """
    if fixed_joints is None:
        fixed_joints = []

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Remove world link
    for child in root.findall(".//link[@name='world']"):
        root.remove(child)

    for child in root.findall(".//joint"):
        parent = child.find("parent")
        if parent is not None and parent.get("link") == "world":
            root.remove(child)

    # Fix specified joints
    for child in root.findall(".//joint"):
        for regexp in fixed_joints:
            if re.fullmatch(regexp, child.attrib["name"]):
                child.attrib["type"] = "fixed"

    # Resolve mesh paths to absolute
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    for child in root.findall(".//mesh"):
        filename = child.attrib.get("filename", "")
        # Strip package:// prefix if present
        filename = filename.replace("package://", "")
        if not os.path.isabs(filename):
            filename = os.path.join(urdf_dir, filename)
        child.attrib["filename"] = filename

    # Resolve <mujoco><compiler meshdir="..."> to absolute path so MuJoCo
    # can find meshes even when the preprocessed URDF is in a different dir.
    for compiler in root.findall(".//mujoco/compiler"):
        meshdir = compiler.get("meshdir")
        if meshdir and not os.path.isabs(meshdir):
            compiler.set("meshdir", os.path.join(urdf_dir, meshdir))

    # Write preprocessed URDF to cache
    ET.indent(root)
    tree_out = ET.ElementTree(root)
    cache_dir = os.path.join(
        os.path.expanduser("~/.cache/compliant_control_tools"),
        os.path.basename(os.path.dirname(os.path.abspath(urdf_path))),
    )
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "tmp_" + os.path.basename(urdf_path))
    tree_out.write(out_path)
    return out_path
