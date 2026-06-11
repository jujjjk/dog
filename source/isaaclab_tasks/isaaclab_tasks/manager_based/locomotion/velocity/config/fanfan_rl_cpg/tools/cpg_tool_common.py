from __future__ import annotations

import ast
import csv
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


LEGS = ("FR", "FL", "RR", "RL")
JOINT_TYPES = ("hip", "thigh", "calf")
JOINT_ORDER = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in JOINT_TYPES)


def package_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "fanfan" / "fanfan" / "urdf" / "fanfan.urdf").exists():
            return parent
    for parent in current.parents:
        for candidate_root in (parent.parent, parent.parent.parent if parent.parent != parent else parent.parent):
            if (candidate_root / "fanfan" / "fanfan" / "urdf" / "fanfan.urdf").exists():
                return candidate_root
    return current.parents[-2]


def default_urdf_path() -> Path:
    env_path = os.environ.get("FANFAN_URDF_PATH")
    if env_path:
        return Path(env_path)
    return workspace_root() / "fanfan" / "fanfan" / "urdf" / "fanfan.urdf"


def default_robot_cfg_path() -> Path:
    return package_dir() / "fanfan_robot_cfg.py"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def parse_vector(text: str | None, count: int = 3) -> list[float]:
    if not text:
        return [0.0] * count
    values = [float(x) for x in text.split()]
    return values + [0.0] * max(0, count - len(values))


def load_default_joint_pos(robot_cfg: Path | None = None) -> dict[str, float]:
    path = robot_cfg or default_robot_cfg_path()
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FANFAN_TEXT_STAND_JOINT_POS":
                    return {str(k): float(v) for k, v in ast.literal_eval(node.value).items()}
    return {}


def parse_urdf(urdf_path: Path) -> dict[str, Any]:
    root = ET.parse(urdf_path).getroot()
    links: dict[str, dict[str, Any]] = {}
    joints: dict[str, dict[str, Any]] = {}

    for link in root.findall("link"):
        name = link.attrib["name"]
        item: dict[str, Any] = {"name": name}
        inertial = link.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            origin = inertial.find("origin")
            item["mass"] = float(mass.attrib["value"]) if mass is not None else None
            item["inertia"] = {k: float(inertia.attrib.get(k, 0.0)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")} if inertia is not None else None
            item["inertial_origin_xyz"] = parse_vector(origin.attrib.get("xyz") if origin is not None else None)
            item["inertial_origin_rpy"] = parse_vector(origin.attrib.get("rpy") if origin is not None else None)
        else:
            item["mass"] = None
            item["inertia"] = None

        collision = link.find("collision/geometry")
        geom: dict[str, Any] = {}
        if collision is not None:
            box = collision.find("box")
            cylinder = collision.find("cylinder")
            sphere = collision.find("sphere")
            if box is not None:
                geom = {"type": "box", "size": parse_vector(box.attrib.get("size"))}
            elif cylinder is not None:
                geom = {
                    "type": "cylinder",
                    "radius": float(cylinder.attrib.get("radius", 0.0)),
                    "length": float(cylinder.attrib.get("length", 0.0)),
                }
            elif sphere is not None:
                geom = {"type": "sphere", "radius": float(sphere.attrib.get("radius", 0.0))}
        item["collision_geometry"] = geom
        links[name] = item

    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        joints[name] = {
            "name": name,
            "type": joint.attrib.get("type", ""),
            "parent": joint.find("parent").attrib.get("link") if joint.find("parent") is not None else None,
            "child": joint.find("child").attrib.get("link") if joint.find("child") is not None else None,
            "origin_xyz": parse_vector(origin.attrib.get("xyz") if origin is not None else None),
            "origin_rpy": parse_vector(origin.attrib.get("rpy") if origin is not None else None),
            "axis": parse_vector(axis.attrib.get("xyz") if axis is not None else None),
            "limit": {
                "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
                "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
                "effort": float(limit.attrib["effort"]) if limit is not None and "effort" in limit.attrib else None,
                "velocity": float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else None,
            },
            "dynamics": {
                "damping": float(dynamics.attrib["damping"]) if dynamics is not None and "damping" in dynamics.attrib else None,
                "friction": float(dynamics.attrib["friction"]) if dynamics is not None and "friction" in dynamics.attrib else None,
            },
        }
    return {"links": links, "joints": joints}


def leg_report(urdf: dict[str, Any], default_joint_pos: dict[str, float]) -> tuple[dict[str, Any], list[str]]:
    joints = urdf["joints"]
    warnings: list[str] = []
    legs: dict[str, Any] = {}
    for leg in LEGS:
        names = [f"{leg}_{j}_joint" for j in JOINT_TYPES]
        missing = [name for name in names if name not in joints]
        if missing:
            warnings.append(f"{leg}: missing joints {missing}")
            continue
        thigh_origin = joints[f"{leg}_calf_joint"]["origin_xyz"]
        foot_fixed = joints.get(f"{leg}_foot_fixed", {})
        foot_origin = foot_fixed.get("origin_xyz", [0.0, 0.0, 0.0])
        thigh_len = abs(float(thigh_origin[2]))
        calf_len = abs(float(foot_origin[2]))
        legs[leg] = {
            "joints": names,
            "foot_link": f"{leg}_foot" if f"{leg}_foot" in urdf["links"] else None,
            "thigh_length_m": thigh_len,
            "calf_length_m": calf_len,
            "nominal_leg_length_m": thigh_len + calf_len,
            "nominal_foot_position_m": [
                joints[f"{leg}_hip_joint"]["origin_xyz"][0],
                joints[f"{leg}_hip_joint"]["origin_xyz"][1],
                -(thigh_len + calf_len),
            ],
            "joint_defaults": {name: default_joint_pos.get(name) for name in names},
        }
    return legs, warnings


def quantile(values: list[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    idx = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return clean[idx]


def csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_array_columns(row: dict[str, str], prefix: str, count: int = 12) -> list[float] | None:
    if prefix in row and row[prefix]:
        parts = re.split(r"[;,\s]+", row[prefix].strip("[] "))
        vals = [float(x) for x in parts if x != ""]
        if len(vals) >= count:
            return vals[:count]
    vals = []
    for i in range(count):
        for key in (f"{prefix}_{i}", f"{prefix}[{i}]", f"{prefix}{i}"):
            if key in row and row[key] != "":
                vals.append(float(row[key]))
                break
        else:
            return None
    return vals
