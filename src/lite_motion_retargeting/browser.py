"""The motions and the robot that the motion browser, ``site/index.html``, plays.

The page is static, so any file server that answers range requests can host it, the
Hugging Face Hub included. The page shows the robot layer of each motion in ``dataset/``.
Training reads the same layer, so the page shows the motion that the robot tracks:

    site/data/index.json.gz             the motions: name, source, category, length, ...
    site/data/robot.json                the link tree of the URDF, and a mesh per visual
    site/data/robot.bin                 those meshes: float32 xyz vertices, then uint32 triangles
    site/data/packs/<source>-<n>.bin    the motions of a source, end to end

For each motion, a pack holds float32 rows at the rate of the robot layer. Then it holds
the rows of the busiest few seconds again, as the preview that a card plays. An index
entry gives the location of each as ``[pack, offset, bytes]``, and the page reads it as a
byte range. A file per motion would put more files in a Hub repo than the Hub accepts.

A row is the pelvis position (xyz), the pelvis rotation (xyzw, the three.js order), then
the angle of each joint that the motion moves. Most motions move the same joints, so an
index entry names its joints by a position in ``joint_sets``. A joint that the entry does
not name, such as a finger, stays at zero. A motion no longer than a preview has no
preview, and its card plays the whole motion.

An entry also names the joints that its source drives, by a position in ``active_sets``.
The index lists the joints of each joint group in ``groups`` and of each Lite variant in
``robots``. Thus the page can show which joints of each robot a motion can train.
"""

import gzip
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from . import DATASET_ROOT, SITE_ROOT
from .hub import VIEWER, layer_url
from .retargeting.g1 import G1_DRIVEN
from .robots import joint_names, lite_urdf_path, load_lite_model
from .sources import SOURCES, clip_dir, clip_key, motion_name, source_of
from .store import BASE_LAYER

SITE_DATA: Path = SITE_ROOT / "data"

#: Seconds. A card loops this much of its motion, where the joints move most.
PREVIEW_SECONDS: float = 4.0

#: Bytes. A pack ends when the next motion would take it past this size.
PACK_BYTES: int = 256 * 2**20

#: The joint groups that the page names, each as a pattern that matches its joint names.
GROUPS: dict[str, str] = {
    "Legs": r".*_(hip_.*|knee_pitch|ankle_pitch|ankle_roll)",
    "Ankle yaw": r".*_ankle_yaw",
    "Waist": r"waist_.*",
    "Arms": r".*_(shoulder_.*|elbow_pitch)",
    "Wrists": r".*_wrist_.*",
    "Neck": r"neck_.*",
    "Fingers": r".*_finger.*",
}

#: The Lite variants that a motion can train, by package name, with their names on the page.
ROBOTS: dict[str, str] = {
    "lite_biped": "Lite Biped",
    "lite_bimanual": "Lite Bimanual",
    "lite_dummy": "Lite Dummy",
    "lite": "Lite",
    "lite_pro": "Lite Pro",
}

def build(sources: list[str] | None = None, out: Path = SITE_DATA) -> None:
    """Write ``site/data/`` from the robot layer of ``dataset/``.

    Args:
        sources: The keys of ``sources.SOURCES`` to include. The default is every source.
            For a site that others can open, include only ``lafan1`` and the ``kimodo``
            sources. The licenses of the other sources do not let you share their motions.
        out: The directory to write. ``uv run publish`` writes its upload to a different
            directory, so the local site keeps every source.
    """
    urdf = lite_urdf_path()
    robot = ET.parse(urdf).getroot().get("name")
    layer = DATASET_ROOT / robot
    sources = sources or list(SOURCES)
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        raise SystemExit(f"Unknown sources {unknown}. Choose from {sorted(SOURCES)}.")
    paths = sorted(path for path in layer.glob("*.rrd") if source_of(path) in sources)
    if not paths:
        raise SystemExit(f"No motions of {sources} under {layer}. Run `uv run retarget`.")
    # The page lists a source only if it has a motion to show.
    present = {source_of(path) for path in paths}
    sources = [source for source in sources if source in present]

    # All of the data is derived, so a file that is already there is stale: start empty.
    # site/data can be a link to another disk, and rmtree does not follow a link, so clear
    # the target of the link.
    target = out.resolve()
    shutil.rmtree(target, ignore_errors=True)
    (target / "packs").mkdir(parents=True)
    mesh_mb = write_robot(urdf, out)

    clips = {source: clip_paths(source) for source in sources}
    layer_joints: list[str] = []
    # Each set maps to its position, in the order that the motions first use it.
    joint_sets: dict[tuple[int, ...], int] = {}
    active_sets: dict[tuple[int, ...], int] = {}
    motions = []
    packs = Packs(out)
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(paths))) as pool:
        exported = pool.map(partial(export_motion, robot=robot), paths, chunksize=16)
        for path, entry in tqdm(zip(paths, exported, strict=True), total=len(paths), unit="motion"):
            names = entry.pop("joint_names")
            if not layer_joints:
                layer_joints = names
            if names != layer_joints:
                raise SystemExit(f"{path}: its joints differ from the joints of the other motions.")
            joints, active = entry.pop("joints"), entry.pop("active")
            rows, preview = entry.pop("rows"), entry.pop("preview")
            source = source_of(path)
            clip = clips[source].get(path.stem)
            motions.append({
                "name": path.stem,
                "source": source,
                "clip": clip,
                "category": category(clip) if clip else "Other",
                **entry,
                "joint_set": joint_sets.setdefault(joints, len(joint_sets)),
                "active_set": active_sets.setdefault(active, len(active_sets)),
                "motion": packs.add(source, rows),
                "preview": packs.add(source, preview) if preview else None,
            })

    index = {
        "robot": robot,
        "viewer": VIEWER,
        "joint_names": layer_joints,
        "joint_sets": list(joint_sets),
        "active_sets": list(active_sets),
        "groups": [
            {"title": title, "joints": [i for i, name in enumerate(layer_joints) if re.fullmatch(pattern, name)]}
            for title, pattern in GROUPS.items()
        ],
        "robots": [
            {"title": title, "joints": [layer_joints.index(name) for name in joint_names(load_lite_model(variant))]}
            for variant, title in ROBOTS.items()
        ],
        "packs": packs.names,
        "sources": {
            key: {
                "title": SOURCES[key].title,
                "license": SOURCES[key].license,
                "skeleton": SOURCES[key].skeleton,
                "url": SOURCES[key].page or f"https://huggingface.co/datasets/{SOURCES[key].repo_id}",
                "hub": layer_url(key, robot),
            }
            for key in sources
        },
        "motions": motions,
    }
    # The index of every source is tens of MB, and gzip makes it a few MB. The Hub serves
    # a large file as stored, so the page decompresses it. With mtime=0, the same motions
    # give the same bytes, and thus the same upload.
    text = json.dumps(index, separators=(",", ":")).encode()
    (out / "index.json.gz").write_bytes(gzip.compress(text, compresslevel=9, mtime=0))

    data_mb = sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) / 1e6
    print(f"{len(motions)} motions of {', '.join(sources)} -> {out}")
    print(f"  {data_mb:.1f} MB in {len(packs.names)} packs, and {mesh_mb:.1f} MB robot meshes")
    print("  serve: uv run serve")


class Packs:
    """Writes motions end to end into ``packs/`` under a directory, a run of packs per source."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.names: list[str] = []
        self.source: str | None = None
        self.size = 0

    def add(self, source: str, data: bytes) -> list[int]:
        """Write ``data``, and return its location as ``[pack, offset, bytes]``.

        A pack holds only one source, so a server can serve the motions of a source without
        the others.
        """
        if source != self.source or self.size + len(data) > PACK_BYTES:
            count = sum(name.startswith(f"packs/{source}-") for name in self.names)
            self.names.append(f"packs/{source}-{count:03d}.bin")
            self.source = source
            self.size = 0
        with open(self.out / self.names[-1], "ab") as file:
            file.write(data)
        offset = self.size
        self.size += len(data)
        return [len(self.names) - 1, offset, len(data)]


def export_motion(path: Path, robot: str) -> dict:
    """Read the motion in ``path``, and return its index fields and its bytes.

    The caller removes the fields that are not for the index. ``rows`` holds the bytes of
    the motion, and ``preview`` holds the bytes of its preview, or None if the motion is no
    longer than a preview. ``joint_names`` lists the joints of the layer. ``joints`` lists
    the joints that move, and ``active`` lists the joints that the source drives, both as
    positions in ``joint_names``.
    """
    fps, names, active, rows = read_robot_layer(path, robot)
    # A joint that stays at zero, such as a finger, adds bytes and no motion.
    moving = np.flatnonzero(np.abs(rows[:, 7:]).max(axis=0) > 0.0)
    rows = np.hstack([rows[:, :7], rows[:, 7 + moving]])
    # The preview is the window in which the joints travel furthest.
    window = round(PREVIEW_SECONDS * fps)
    preview = None
    if len(rows) > window:
        speed = np.abs(np.diff(rows[:, 7:], axis=0)).sum(axis=1)
        travelled = np.concatenate([[0.0], np.cumsum(speed)])
        start = int(np.argmax(travelled[window - 1:] - travelled[:-(window - 1)]))
        preview = rows[start:start + window].tobytes()
    travel = np.linalg.norm(np.diff(rows[:, :2], axis=0), axis=1).sum()
    return {
        "fps": fps,
        "frames": len(rows),
        "seconds": round((len(rows) - 1) / fps, 2),
        "source_fps": read_source_fps(DATASET_ROOT / BASE_LAYER / path.name),
        "travel": round(float(travel), 2),
        "rows": rows.tobytes(),
        "preview": preview,
        "joint_names": names,
        "joints": tuple(moving.tolist()),
        "active": tuple(names.index(name) for name in active),
    }


def read_robot_layer(path: Path, robot: str) -> tuple[int, list[str], list[str], np.ndarray]:
    """Read the rate, joint names, driven joints and ``(T, 7 + J)`` rows of a robot layer.

    The pelvis pose comes from the pelvis goal, which is the base where the retarget placed
    the robot. Thus the pose is exactly where the joint chain in the file puts the root. A
    layer without ``active_joint_names`` is older than that property. Such a layer comes
    from a G1 source, so it drives the G1 joints.
    """
    positions = f"/{robot}/joint/positions"
    pelvis = f"/{robot}/goal/poses/pelvis"
    columns: dict[str, list[pa.RecordBatch]] = {}
    joint_names, active, fps = None, None, None
    for chunk in RrdReader(path).stream():
        entity = chunk.entity_path
        if entity == f"/__properties/{robot}":
            properties = chunk.to_record_batch().to_pydict()
            joint_names, fps = properties["joint_names"][0], properties["fps"][0][0]
            active = properties.get("active_joint_names", [G1_DRIVEN])[0]
        elif entity in (positions, pelvis):
            columns.setdefault(entity, []).append(chunk.to_record_batch())
    if joint_names is None:
        raise SystemExit(f"{path}: no /__properties/{robot}, so the file is not a robot layer.")

    def values(entity: str, component: str, width: int) -> np.ndarray:
        table = pa.Table.from_batches(columns[entity]).sort_by("frame")
        flat = table.column(component).combine_chunks().flatten()
        if pa.types.is_fixed_size_list(flat.type):
            flat = flat.flatten()
        return flat.to_numpy().astype(np.float32).reshape(-1, width)

    rows = np.hstack([
        values(pelvis, "Transform3D:translation", 3),
        values(pelvis, "Transform3D:quaternion", 4),
        values(positions, "Scalars:scalars", len(joint_names)),
    ])
    return int(fps), list(joint_names), list(active), rows


def read_source_fps(path: Path) -> int | None:
    """Return the rate of the source clip from its base layer, or None if the store has none."""
    if not path.exists():
        return None
    for chunk in RrdReader(path).stream():
        if chunk.entity_path == "/__properties/motion":
            return int(chunk.to_record_batch().to_pydict()["fps"][0][0])
    return None


def clip_paths(source: str) -> dict[str, str]:
    """Map each motion name of ``source`` to the path of its clip, from the cache.

    A motion name flattens the path of its clip, so only the downloaded source keeps the
    folders that give the category. The map leaves out each clip that is not in the cache.
    """
    return {
        motion_name(source, path): clip_key(source, path)
        for path in clip_dir(source).glob(SOURCES[source].clip_glob)
    }


def category(clip: str) -> str:
    """Return the category of a clip.

    A taxonomy of numbered folders, such as the one of MotionDecode, gives its section at
    the second level: ``1.14.Dance_and_Performance/1.14.5.Ballet/...`` gives ``Ballet``. A
    clip one level deep, ``4. Martial_Arts/...``, gives ``Martial arts``. Otherwise the top
    folder is the category: ``CMU/07/07_01`` gives ``CMU``, and
    ``bowing_forward/episode_00001`` gives ``Bowing forward``. A flat LAFAN1 take,
    ``fallAndGetUp2_subject3``, gives the action at the start of its name, ``Fall and get up``.
    """
    folders = clip.split("/")[:-1]
    numbered = [folder for folder in folders[:2] if re.match(r"\d+(\.\d+)*\.\s*[A-Za-z]", folder)]
    if numbered:
        return re.sub(r"^[\d.\s]+", "", numbered[-1]).replace("_", " ").capitalize()
    if folders:
        label = folders[0].replace("_", " ")
        return label[:1].upper() + label[1:]
    action = re.match(r"[A-Za-z]*", clip).group()
    return " ".join(re.findall(r"[A-Z]?[a-z]+", action)).capitalize() or clip


def write_robot(urdf: Path, out: Path) -> float:
    """Write ``robot.json`` and ``robot.bin`` from a URDF, and return the mesh size in MB.

    ``robot.bin`` holds each visual mesh in the frame of its link, with its repeated vertices
    merged, so the page places the mesh with the pose of the link alone. The page shades the
    mesh flat, as an STL implies with its normal per face.
    """
    root = ET.parse(urdf).getroot()
    # A visual can name a material that the URDF defines once, at the top level.
    materials = {material.get("name"): material for material in root.findall("material")}
    blocks: list[bytes] = []
    offset = 0
    links = []
    for link in root.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, rpy = _origin(visual.find("origin"))
            scale = np.array(mesh.get("scale", "1 1 1").split(), dtype=np.float64)
            vertices, triangles = _read_stl(urdf.parent / mesh.get("filename"))
            vertices = (vertices * scale) @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
            vertex_bytes = vertices.astype("<f4").tobytes()
            triangle_bytes = triangles.astype("<u4").tobytes()
            visuals.append({
                "offset": offset,
                "vertices": len(vertices),
                "triangles": len(triangles),
                "color": _color(visual.find("material"), materials),
            })
            blocks += [vertex_bytes, triangle_bytes]
            offset += len(vertex_bytes) + len(triangle_bytes)
        links.append({"name": link.get("name"), "visuals": visuals})

    joints = []
    for joint in root.findall("joint"):
        xyz, rpy = _origin(joint.find("origin"))
        axis = joint.find("axis")
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "xyz": xyz.tolist(),
            "quat": Rotation.from_euler("xyz", rpy).as_quat().tolist(),  # xyzw
            "axis": [float(v) for v in (axis.get("xyz") if axis is not None else "1 0 0").split()],
        })
    children = {joint["child"] for joint in joints}
    roots = [link["name"] for link in links if link["name"] not in children]
    if len(roots) != 1:
        raise SystemExit(f"{urdf}: expected one root link, found {roots}.")

    description = {"name": root.get("name"), "root": roots[0], "links": links, "joints": joints}
    (out / "robot.json").write_text(json.dumps(description, separators=(",", ":")))
    (out / "robot.bin").write_bytes(b"".join(blocks))
    return offset / 1e6


def _origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``xyz`` and ``rpy`` of a URDF origin, which are zero by default."""
    if element is None:
        return np.zeros(3), np.zeros(3)
    return (
        np.array(element.get("xyz", "0 0 0").split(), dtype=np.float64),
        np.array(element.get("rpy", "0 0 0").split(), dtype=np.float64),
    )


def _color(material: ET.Element | None, materials: dict[str, ET.Element]) -> list[float]:
    """Return the RGBA of a visual's material, inline or by name, or light grey without one."""
    if material is not None and material.find("color") is None:
        material = materials.get(material.get("name"))
    if material is None or material.find("color") is None:
        return [0.8, 0.8, 0.8, 1.0]
    return [float(v) for v in material.find("color").get("rgba").split()]


def _read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL as merged ``(V, 3)`` vertices and ``(F, 3)`` vertex indices."""
    count = int(np.fromfile(path, dtype="<u4", count=1, offset=80)[0])
    if path.stat().st_size != 84 + 50 * count:
        raise SystemExit(f"{path}: not a binary STL. Export the mesh as binary.")
    facets = np.fromfile(
        path, dtype=[("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")],
        offset=84,
    )
    vertices, triangles = np.unique(
        facets["corners"].reshape(-1, 3), axis=0, return_inverse=True
    )
    return vertices.astype(np.float64), triangles.reshape(-1, 3)
