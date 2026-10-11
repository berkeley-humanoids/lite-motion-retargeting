"""Build the motion browser: write what ``site/index.html`` plays into ``site/data/``.

The page is static, so any file server that answers range requests can host it, the
Hugging Face Hub included. It reads the robot layer of each motion in ``dataset/``, which
is what training reads, so the page shows the motion the robot tracks:

    site/data/index.json.gz             the motions: name, source, category, length, ...
    site/data/robot.json                the link tree of the URDF, and a mesh per visual
    site/data/robot.bin                 those meshes: float32 xyz vertices, then uint32 triangles
    site/data/packs/<source>-<n>.bin    the motions of a source, end to end

A pack holds, for each motion, float32 rows at the rate of the robot layer, then the same
rows over its busiest few seconds, the preview a card plays. An index entry gives each
as ``[pack, offset, bytes]``, and the page reads it as a byte range. A file per motion
would leave the Hub with far more files than it takes in a repo.

A row is the pelvis position (xyz), the pelvis rotation (xyzw, as three.js takes it),
then the angle of each joint the motion moves. An index entry names those joints by one
of ``joint_sets``, since most motions move the same ones. A joint it does not list, such
as a finger, stays at zero. A motion no longer than a preview has no preview, and its
card plays the whole motion.

Usage:
    uv run scripts/build_site.py
    uv run scripts/build_site.py --sources lafan1 kimodo kimodolh2 kimodolh3  # what others may see
    uv run scripts/serve.py                            # then open http://localhost:8000
"""

import gzip
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import tyro
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_ROOT, DATASET_REPO_ID, PUBLIC_SOURCE, lite_urdf_path  # noqa: E402
from sources import SOURCES, clip_key, motion_name  # noqa: E402
from store import BASE_LAYER  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_ROOT: Path = REPO_ROOT / "dataset"
SITE_DATA: Path = REPO_ROOT / "site" / "data"

#: Seconds. A card loops this much of its motion, from where the joints move most.
PREVIEW_SECONDS: float = 4.0

#: Bytes. A pack ends once the next motion would take it past this.
PACK_BYTES: int = 256 * 2**20

#: The Rerun web viewer, and where it finds a published motion.
VIEWER: str = "https://rerun.io/viewer?url="
HUB_FILES: str = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/"


def main(sources: list[str] | None = None, out: Path = SITE_DATA) -> None:
    """Write ``site/data/`` from the robot layer of ``dataset/``.

    Args:
        sources: Keys of ``sources.SOURCES`` to include. Every source by default. A site
            others can reach should hold ``lafan1`` and the ``kimodo`` sources only: the
            licenses of the others bar sharing their motions.
        out: The directory to write. ``scripts/publish.py`` writes the one it uploads
            elsewhere, so the local site keeps every source.
    """
    urdf = lite_urdf_path()
    robot = ET.parse(urdf).getroot().get("name")
    layer = DATASET_ROOT / robot
    sources = sources or list(SOURCES)
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        raise SystemExit(f"Unknown sources {unknown}. Expected some of {sorted(SOURCES)}.")
    paths = sorted(
        path for path in layer.glob("*.rrd") if path.stem.split("_", 1)[0] in sources
    )
    if not paths:
        raise SystemExit(f"No motions of {sources} under {layer}. Run scripts/retarget.py.")
    # The page lists a source only if it has a motion to show.
    present = {path.stem.split("_", 1)[0] for path in paths}
    sources = [source for source in sources if source in present]

    # The data is derived, so a stale file is only ever left over: start empty. site/data can
    # be a link to another disk, and rmtree does not follow a link, so clear what it points to.
    target = out.resolve()
    shutil.rmtree(target, ignore_errors=True)
    (target / "packs").mkdir(parents=True)
    mesh_mb = write_robot(urdf, out)

    clips = {source: clip_paths(source) for source in sources}
    joint_names: list[str] = []
    joint_sets: list[list[int]] = []
    motions = []
    packs = Packs(out)
    export = partial(export_motion, robot=robot)
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(paths))) as pool:
        exported = pool.map(export, paths, chunksize=16)
        for path, entry in tqdm(zip(paths, exported, strict=True), total=len(paths), unit="motion"):
            names = entry.pop("joint_names")
            if not joint_names:
                joint_names = names
            if names != joint_names:
                raise SystemExit(f"{path}: its joints differ from those of the other motions.")
            joints = entry.pop("joints")
            if joints not in joint_sets:
                joint_sets.append(joints)
            source = path.stem.split("_", 1)[0]
            clip = clips[source].get(path.stem)
            rows, preview = entry.pop("rows"), entry.pop("preview")
            motions.append({
                "name": path.stem,
                "source": source,
                "clip": clip,
                "category": category(clip) if clip else "Other",
                **entry,
                "joint_set": joint_sets.index(joints),
                "motion": packs.add(source, rows),
                "preview": packs.add(source, preview) if preview else None,
            })

    index = {
        "robot": robot,
        "viewer": VIEWER,
        "joint_names": joint_names,
        "joint_sets": joint_sets,
        "packs": packs.names,
        "sources": {
            key: {
                "title": SOURCES[key].title,
                "license": SOURCES[key].license,
                "url": f"https://huggingface.co/datasets/{SOURCES[key].repo_id}",
                # The Hub repo holds this source alone, so only its motions have a URL the
                # web viewer can open.
                "hub": f"{HUB_FILES}{robot}/" if key == PUBLIC_SOURCE else None,
            }
            for key in sources
        },
        "motions": motions,
    }
    # The index of every source runs to tens of MB, which gzip takes to a few. The Hub
    # serves a large file as stored, so the page unpacks it itself. mtime=0 keeps the
    # bytes, and so the upload, the same when the motions are.
    text = json.dumps(index, separators=(",", ":")).encode()
    (out / "index.json.gz").write_bytes(gzip.compress(text, compresslevel=9, mtime=0))

    data_mb = sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) / 1e6
    print(f"{len(motions)} motions of {', '.join(sources)} -> {out}")
    print(f"  {data_mb:.1f} MB in {len(packs.names)} packs, and {mesh_mb:.1f} MB robot meshes")
    print("  serve: uv run scripts/serve.py")


class Packs:
    """Writes motions end to end into ``packs/`` under a directory, a run of packs per source."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.names: list[str] = []
        self.source = None
        self.size = 0

    def add(self, source: str, data: bytes) -> list[int]:
        """Write ``data`` and return where it went, as ``[pack, offset, bytes]``.

        A pack holds one source, so the motions of a source can be served apart from the
        others.
        """
        if not self.names or source != self.source or self.size + len(data) > PACK_BYTES:
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

    ``rows`` holds the bytes of the motion and ``preview`` those of its preview, or None
    when the motion is no longer than one. It also returns the joint names of the layer,
    and the indexes of the joints that move, which the caller checks and gathers across
    motions.
    """
    fps, names, rows = read_robot_layer(path, robot)
    # A joint that never leaves zero, such as a finger, adds bytes and no motion.
    moving = np.flatnonzero(np.abs(rows[:, 7:]).max(axis=0) > 0.0)
    rows = np.ascontiguousarray(np.hstack([rows[:, :7], rows[:, 7 + moving]]))
    start, stop = preview_window(rows[:, 7:], round(PREVIEW_SECONDS * fps))
    preview = rows[start:stop].tobytes() if stop - start < len(rows) else None
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
        "joints": moving.tolist(),
    }


def read_robot_layer(path: Path, robot: str) -> tuple[int, list[str], np.ndarray]:
    """Read the rate, the joint names, and the ``(T, 7 + J)`` rows of a robot layer.

    The pelvis pose comes from its goal, which is the base the retarget placed the robot
    at, so it is exactly where the joint chain in the file puts the root.
    """
    columns: dict[str, list[pa.RecordBatch]] = {}
    joint_names, fps = None, None
    for chunk in RrdReader(path).stream():
        entity = chunk.entity_path
        if entity == f"/__properties/{robot}":
            properties = chunk.to_record_batch().to_pydict()
            joint_names, fps = properties["joint_names"][0], properties["fps"][0][0]
        elif entity in (f"/{robot}/joint/positions", f"/{robot}/goal/poses/pelvis"):
            columns.setdefault(entity, []).append(chunk.to_record_batch())
    if joint_names is None:
        raise SystemExit(f"{path}: no /__properties/{robot}. Is it a robot layer?")

    def values(entity: str, component: str, width: int) -> np.ndarray:
        table = pa.Table.from_batches(columns[entity]).sort_by("frame")
        flat = table.column(component).combine_chunks().flatten()
        if pa.types.is_fixed_size_list(flat.type):
            flat = flat.flatten()
        return flat.to_numpy().astype(np.float32).reshape(-1, width)

    joint_pos = values(f"/{robot}/joint/positions", "Scalars:scalars", len(joint_names))
    pelvis = f"/{robot}/goal/poses/pelvis"
    rows = np.hstack([
        values(pelvis, "Transform3D:translation", 3),
        values(pelvis, "Transform3D:quaternion", 4),
        joint_pos,
    ])
    return int(fps), list(joint_names), rows


def read_source_fps(path: Path) -> int | None:
    """The rate of the source clip, from the base layer, if the store holds one."""
    if not path.exists():
        return None
    for chunk in RrdReader(path).stream():
        if chunk.entity_path == "/__properties/motion":
            return int(chunk.to_record_batch().to_pydict()["fps"][0][0])
    return None


def clip_paths(source: str) -> dict[str, str]:
    """Map each motion name of ``source`` back to the path of its clip, from the cache.

    A motion name flattens the path, so the directories a category comes from survive
    only in the downloaded source. A motion whose source is not in the cache maps to
    nothing.
    """
    spec = SOURCES[source]
    clip_dir = CACHE_ROOT / source / spec.clip_dir
    return {
        motion_name(source, clip_dir, path): clip_key(clip_dir, path)
        for path in clip_dir.glob(spec.clip_glob)
    }


def category(clip: str) -> str:
    """The group a clip belongs to.

    A taxonomy of numbered folders, as MotionDecode's, groups by its section, the second level:
    ``1.14.Dance_and_Performance/1.14.5.Ballet/...`` gives ``Ballet``, and a clip one level
    deep, ``4. Martial_Arts/...``, gives ``Martial arts``. Otherwise the top folder names the
    group: ``CMU/07/07_01`` gives ``CMU``, and ``bowing_forward/episode_00001`` gives
    ``Bowing forward``. A flat LAFAN1 take, ``fallAndGetUp2_subject3``, gives the action its
    name opens with, ``Fall and get up``.
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


def preview_window(joint_pos: np.ndarray, frames: int) -> tuple[int, int]:
    """The ``frames`` long window in which the joints travel furthest, as ``[start, stop)``."""
    if len(joint_pos) <= frames:
        return 0, len(joint_pos)
    speed = np.abs(np.diff(joint_pos, axis=0)).sum(axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(speed)])
    start = int(np.argmax(travelled[frames - 1:] - travelled[:-(frames - 1)]))
    return start, start + frames


def write_robot(urdf: Path, out: Path) -> float:
    """Write ``robot.json`` and ``robot.bin`` from a URDF, and return the size in MB.

    Each visual mesh is moved into the frame of its link and has its repeated vertices
    merged, so the page places it with the link alone. The page shades it flat, which
    an STL, a mesh with a normal per face, implies.
    """
    root = ET.parse(urdf).getroot()
    # A visual can name a material the robot defines once, at the top of the URDF.
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
            color = _color(visual.find("material"), materials)
            vertex_bytes = vertices.astype("<f4").tobytes()
            triangle_bytes = triangles.astype("<u4").tobytes()
            visuals.append({
                "offset": offset,
                "vertices": len(vertices),
                "triangles": len(triangles),
                "color": color,
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
    """The ``xyz`` and ``rpy`` of a URDF origin, which default to zero."""
    if element is None:
        return np.zeros(3), np.zeros(3)
    return (
        np.array(element.get("xyz", "0 0 0").split(), dtype=np.float64),
        np.array(element.get("rpy", "0 0 0").split(), dtype=np.float64),
    )


def _color(material: ET.Element | None, materials: dict[str, ET.Element]) -> list[float]:
    """The RGBA of a visual's material, inline or named, and light grey without one."""
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


if __name__ == "__main__":
    tyro.cli(main)
