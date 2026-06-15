from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

HEAVY_URDF_NAME = "fanfan_mass_scaled_only_trunk_plus_800g.urdf"
LEGACY_URDF_NAME = "fanfan.urdf"
HEAVY_USD_NAME = "fanfan_mass_scaled_only_trunk_plus_800g_no_merge.usd"
EXPECTED_TOTAL_MASS_KG = 7.242158331537168
EXPECTED_TRUNK_MASS_KG = 2.76230213761
EXPECTED_THIGH_LENGTH_M = 0.15606
EXPECTED_CALF_LENGTH_M = 0.14894
EXPECTED_EFFORT_LIMIT_NM = 17.0
EXPECTED_VELOCITY_LIMIT_RAD_S = 44.0
EXPECTED_SIM_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
EXPECTED_JOINT_ORIGINS = {
    "FR_hip_joint": (0.19, -0.0451, 0.00075),
    "FR_thigh_joint": (0.0, -0.076, 0.0),
    "FR_calf_joint": (0.0, 0.0005, -0.15606),
    "FL_hip_joint": (0.19, 0.0451, 0.00075),
    "FL_thigh_joint": (0.0, 0.076, 0.0),
    "FL_calf_joint": (0.0, -0.0005, -0.15606),
    "RR_hip_joint": (-0.19, -0.0451, 0.00075),
    "RR_thigh_joint": (0.0, -0.076, 0.0),
    "RR_calf_joint": (0.0, 0.0005, -0.15606),
    "RL_hip_joint": (-0.19, 0.0451, 0.00075),
    "RL_thigh_joint": (0.0, 0.076, 0.0),
    "RL_calf_joint": (0.0, -0.0005, -0.15606),
}
EXPECTED_JOINT_LIMITS = {
    "front_thigh": (-1.570796326795, 0.645771823238),
    "rear_thigh": (-0.645771823238, 1.570796326795),
    "hip": (-0.314159265359, 0.698131700798),
    "calf": (-2.443460952792, 2.443460952792),
}


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    origin_xyz: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    lower: float
    upper: float
    effort: float
    velocity: float


@dataclass(frozen=True)
class FanfanUrdfModel:
    path: Path
    joint_order: tuple[str, ...]
    joints: dict[str, UrdfJoint]
    total_mass_kg: float
    trunk_mass_kg: float
    thigh_length_m: float
    calf_length_m: float


def _matmul(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]):
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4))
        for row in range(4)
    )


def _transform_xyz(xyz: tuple[float, float, float]):
    return (
        (1.0, 0.0, 0.0, xyz[0]),
        (0.0, 1.0, 0.0, xyz[1]),
        (0.0, 0.0, 1.0, xyz[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation(axis: tuple[float, float, float], angle: float):
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return (
        (x * x * one_c + c, x * y * one_c - z * s, x * z * one_c + y * s, 0.0),
        (y * x * one_c + z * s, y * y * one_c + c, y * z * one_c - x * s, 0.0),
        (z * x * one_c - y * s, z * y * one_c + x * s, z * z * one_c + c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def forward_foot_position(
    model: FanfanUrdfModel,
    leg: str,
    joint_angles: tuple[float, float, float],
) -> tuple[float, float, float]:
    transform = _transform_xyz((0.0, 0.0, 0.0))
    for joint_type, angle in zip(("hip", "thigh", "calf"), joint_angles, strict=True):
        joint = model.joints[f"{leg}_{joint_type}_joint"]
        transform = _matmul(transform, _transform_xyz(joint.origin_xyz))
        transform = _matmul(transform, _rotation(joint.axis_xyz, angle))
    transform = _matmul(transform, _transform_xyz((0.0, 0.0, -model.calf_length_m)))
    return transform[0][3], transform[1][3], transform[2][3]


def resolve_heavy_urdf_path() -> Path:
    for variable in ("FANFAN_HEAVY_URDF_PATH", "FANFAN_URDF_PATH"):
        value = os.environ.get(variable)
        if value:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{variable} points to a missing URDF: {path}")
            return path

    current_file = Path(__file__).resolve()
    searched_paths: list[Path] = []
    for parent in current_file.parents:
        urdf_dir = parent / "fanfan" / "urdf"
        for filename in (HEAVY_URDF_NAME, LEGACY_URDF_NAME):
            candidate = urdf_dir / filename
            searched_paths.append(candidate)
            if candidate.is_file():
                return candidate

    searched = "\n  ".join(str(path) for path in searched_paths)
    raise FileNotFoundError(
        f"Could not locate {HEAVY_URDF_NAME} or a compatible {LEGACY_URDF_NAME}.\n"
        f"Searched:\n  {searched}\n"
        "Set FANFAN_HEAVY_URDF_PATH to the 7.242 kg URDF explicitly."
    )


def _vector(element: ET.Element, attribute: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in element.attrib[attribute].split())
    if len(values) != 3:
        raise ValueError(f"Expected three values in {attribute}, got {values}.")
    return values


def load_fanfan_urdf_model(path: str | Path | None = None) -> FanfanUrdfModel:
    urdf_path = Path(path).expanduser().resolve() if path is not None else resolve_heavy_urdf_path()
    root = ET.parse(urdf_path).getroot()
    masses: dict[str, float] = {}
    for link in root.findall("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            masses[link.attrib["name"]] = float(mass.attrib["value"])

    joints: dict[str, UrdfJoint] = {}
    joint_order: list[str] = []
    fixed_origins: dict[str, tuple[float, float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        origin = joint.find("origin")
        if origin is None:
            raise ValueError(f"URDF joint {name} has no origin.")
        if joint.attrib.get("type") == "fixed":
            fixed_origins[name] = _vector(origin, "xyz")
            continue
        if joint.attrib.get("type") != "revolute":
            continue
        axis = joint.find("axis")
        limit = joint.find("limit")
        if axis is None or limit is None:
            raise ValueError(f"Revolute joint {name} is missing axis or limit.")
        joints[name] = UrdfJoint(
            name=name,
            origin_xyz=_vector(origin, "xyz"),
            axis_xyz=_vector(axis, "xyz"),
            lower=float(limit.attrib["lower"]),
            upper=float(limit.attrib["upper"]),
            effort=float(limit.attrib["effort"]),
            velocity=float(limit.attrib["velocity"]),
        )
        joint_order.append(name)

    calf_origins = [abs(joints[f"{leg}_calf_joint"].origin_xyz[2]) for leg in ("FR", "FL", "RR", "RL")]
    foot_origins = [abs(fixed_origins[f"{leg}_foot_fixed"][2]) for leg in ("FR", "FL", "RR", "RL")]
    return FanfanUrdfModel(
        path=urdf_path,
        joint_order=tuple(joint_order),
        joints=joints,
        total_mass_kg=sum(masses.values()),
        trunk_mass_kg=masses["Trunk"],
        thigh_length_m=sum(calf_origins) / len(calf_origins),
        calf_length_m=sum(foot_origins) / len(foot_origins),
    )


def validate_fanfan_urdf(model: FanfanUrdfModel) -> None:
    errors: list[str] = []
    if model.joint_order != EXPECTED_SIM_JOINT_NAMES:
        errors.append(f"joint order {model.joint_order} != {EXPECTED_SIM_JOINT_NAMES}")
    if abs(model.total_mass_kg - EXPECTED_TOTAL_MASS_KG) > 1.0e-6:
        errors.append(f"total mass {model.total_mass_kg:.9f} kg")
    if abs(model.trunk_mass_kg - EXPECTED_TRUNK_MASS_KG) > 1.0e-6:
        errors.append(f"Trunk mass {model.trunk_mass_kg:.9f} kg")
    if abs(model.thigh_length_m - EXPECTED_THIGH_LENGTH_M) > 1.0e-5:
        errors.append(f"thigh length {model.thigh_length_m:.9f} m")
    if abs(model.calf_length_m - EXPECTED_CALF_LENGTH_M) > 5.0e-6:
        errors.append(f"calf length {model.calf_length_m:.9f} m")

    for name in EXPECTED_SIM_JOINT_NAMES:
        joint = model.joints.get(name)
        if joint is None:
            errors.append(f"missing joint {name}")
            continue
        expected_axis = (1.0, 0.0, 0.0) if "_hip_joint" in name else (0.0, 1.0, 0.0)
        if joint.axis_xyz != expected_axis:
            errors.append(f"{name} axis {joint.axis_xyz} != {expected_axis}")
        expected_origin = EXPECTED_JOINT_ORIGINS[name]
        if any(abs(actual - expected) > 1.0e-8 for actual, expected in zip(joint.origin_xyz, expected_origin)):
            errors.append(f"{name} origin {joint.origin_xyz} != {expected_origin}")
        if "_hip_joint" in name:
            expected_limits = EXPECTED_JOINT_LIMITS["hip"]
        elif "_calf_joint" in name:
            expected_limits = EXPECTED_JOINT_LIMITS["calf"]
        elif name.startswith(("FR_", "FL_")):
            expected_limits = EXPECTED_JOINT_LIMITS["front_thigh"]
        else:
            expected_limits = EXPECTED_JOINT_LIMITS["rear_thigh"]
        if abs(joint.lower - expected_limits[0]) > 1.0e-8 or abs(joint.upper - expected_limits[1]) > 1.0e-8:
            errors.append(f"{name} limits [{joint.lower}, {joint.upper}] != {expected_limits}")
        if abs(joint.effort - EXPECTED_EFFORT_LIMIT_NM) > 1.0e-6:
            errors.append(f"{name} effort {joint.effort}")
        if abs(joint.velocity - EXPECTED_VELOCITY_LIMIT_RAD_S) > 1.0e-6:
            errors.append(f"{name} velocity {joint.velocity}")
        if not joint.lower < joint.upper:
            errors.append(f"{name} invalid limits [{joint.lower}, {joint.upper}]")
    if errors:
        raise ValueError("Heavy Fanfan URDF validation failed: " + "; ".join(errors))


def make_heavy_fanfan_cfg():
    from copy import deepcopy

    from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.fanfan_robot_cfg import (
        FANFAN_CFG,
    )

    model = load_fanfan_urdf_model()
    validate_fanfan_urdf(model)
    cfg = deepcopy(FANFAN_CFG)
    cfg.spawn.asset_path = str(model.path)
    cfg.spawn.usd_dir = str(model.path.parents[1] / "USD")
    cfg.spawn.usd_file_name = HEAVY_USD_NAME
    return cfg, model
