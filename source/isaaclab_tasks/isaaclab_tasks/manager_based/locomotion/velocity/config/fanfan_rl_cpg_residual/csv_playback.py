from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from .joint_semantics import POLICY_JOINT_NAMES


def _finite_float(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {row_number}: invalid {field} value {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"CSV row {row_number}: {field} must be finite, got {result}.")
    return result


def _time_field(fieldnames: list[str]) -> str:
    if "elapsed" in fieldnames:
        return "elapsed"
    if "time" in fieldnames:
        return "time"
    raise ValueError("CSV must contain a 'time' or 'elapsed' column.")


def _validate_times(times: list[float]) -> None:
    if len(times) < 2:
        raise ValueError("CSV playback requires at least two distinct frames.")
    origin = times[0]
    for index in range(len(times)):
        times[index] -= origin
    for previous, current in zip(times, times[1:]):
        if current <= previous:
            raise ValueError("CSV frame times must be strictly increasing.")


def load_joint_csv(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load policy-order or real-motor-order joint targets from a CSV file."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Fanfan CSV playback file does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"Fanfan CSV playback file is empty: {path}")

    time_key = _time_field(fieldnames)
    policy_columns = [f"q_policy_{index}" for index in range(12)]
    real_columns = [f"q_real_{index}" for index in range(12)]
    if all(column in fieldnames for column in policy_columns):
        times = [_finite_float(row[time_key], time_key, number) for number, row in enumerate(rows, 2)]
        values = [
            [_finite_float(row[column], column, number) for column in policy_columns]
            for number, row in enumerate(rows, 2)
        ]
        value_space = "policy"
    elif all(column in fieldnames for column in real_columns):
        times = [_finite_float(row[time_key], time_key, number) for number, row in enumerate(rows, 2)]
        values = [
            [_finite_float(row[column], column, number) for column in real_columns]
            for number, row in enumerate(rows, 2)
        ]
        value_space = "real"
    elif {"policy_joint_name", "q_target_policy"}.issubset(fieldnames):
        grouped: dict[float, dict[str, float]] = {}
        order: list[float] = []
        for number, row in enumerate(rows, 2):
            frame_time = _finite_float(row[time_key], time_key, number)
            joint_name = row["policy_joint_name"].strip()
            if joint_name not in POLICY_JOINT_NAMES:
                raise ValueError(f"CSV row {number}: unknown policy_joint_name {joint_name!r}.")
            if frame_time not in grouped:
                grouped[frame_time] = {}
                order.append(frame_time)
            if joint_name in grouped[frame_time]:
                raise ValueError(f"CSV time {frame_time}: duplicate joint {joint_name}.")
            grouped[frame_time][joint_name] = _finite_float(
                row["q_target_policy"], "q_target_policy", number
            )
        times = order
        values = []
        for frame_time in times:
            missing = [name for name in POLICY_JOINT_NAMES if name not in grouped[frame_time]]
            if missing:
                raise ValueError(f"CSV time {frame_time}: missing joints {missing}.")
            values.append([grouped[frame_time][name] for name in POLICY_JOINT_NAMES])
        value_space = "policy"
    else:
        raise ValueError(
            "Unsupported Fanfan CSV format. Expected q_policy_0..11, q_real_0..11, "
            "or ROS2 long-form policy_joint_name/q_target_policy columns."
        )

    _validate_times(times)
    return (
        torch.tensor(times, dtype=torch.float32),
        torch.tensor(values, dtype=torch.float32),
        value_space,
    )


class LoopingJointCsvPlayback:
    """Linearly interpolate a joint trajectory and loop at its final timestamp."""

    def __init__(self, times: torch.Tensor, values: torch.Tensor, *, device: torch.device):
        if times.ndim != 1 or values.ndim != 2 or values.shape != (times.numel(), 12):
            raise ValueError(
                f"Expected times [N] and values [N, 12], got {tuple(times.shape)} and {tuple(values.shape)}."
            )
        self.times = times.to(device=device)
        self.values = values.to(device=device)
        self.duration = float(self.times[-1])
        if self.duration <= 0.0:
            raise ValueError("CSV playback duration must be positive.")

    def sample(self, playback_time: torch.Tensor) -> torch.Tensor:
        wrapped = torch.remainder(playback_time, self.duration)
        upper = torch.searchsorted(self.times, wrapped, right=True)
        upper = torch.clamp(upper, 1, self.times.numel() - 1)
        lower = upper - 1
        t0 = self.times[lower]
        t1 = self.times[upper]
        blend = ((wrapped - t0) / torch.clamp(t1 - t0, min=1.0e-8)).unsqueeze(1)
        return self.values[lower] + blend * (self.values[upper] - self.values[lower])
