#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_onnx_session(path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    obs_dim = int(shape[1]) if len(shape) >= 2 and isinstance(shape[1], int) else None
    return sess, input_name, obs_dim


def _run_onnx(sess, input_name: str, obs: np.ndarray) -> np.ndarray:
    out = sess.run(None, {input_name: obs.astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float32)


def _run_jit(path: str, obs: np.ndarray) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for --jit comparison") from exc
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(obs.astype(np.float32)))
    return out.detach().cpu().numpy().astype(np.float32)


def _default_obs(obs_dim: int, cmd_x: float) -> np.ndarray:
    obs = np.zeros((1, obs_dim), dtype=np.float32)
    obs[0, 6:9] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    obs[0, 9:12] = np.array([cmd_x, 0.0, 0.0], dtype=np.float32)
    if obs_dim >= 50:
        obs[0, 48:50] = np.array([0.0, 1.0], dtype=np.float32)
    return obs


def _obs_from_csv(path: str, limit: int | None = None) -> list[np.ndarray]:
    groups: dict[str, dict[int, dict[str, str]]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode", "policy") != "policy":
                continue
            t = row.get("time", "")
            i = int(float(row["policy_index"]))
            groups.setdefault(t, {})[i] = row
            if limit is not None and len(groups) >= limit:
                break

    obs_list = []
    for _, rows in groups.items():
        if len(rows) < 12:
            continue
        first = rows[min(rows)]
        obs_dim = 50 if "obs_last_action_policy" in first else 36
        obs = np.zeros((1, obs_dim), dtype=np.float32)
        obs[0, 0:3] = [float(first["obs_base_lin_x"]), float(first["obs_base_lin_y"]), float(first["obs_base_lin_z"])]
        obs[0, 3:6] = [float(first["obs_base_ang_x"]), float(first["obs_base_ang_y"]), float(first["obs_base_ang_z"])]
        obs[0, 6:9] = [float(first["obs_gravity_x"]), float(first["obs_gravity_y"]), float(first["obs_gravity_z"])]
        obs[0, 9:12] = [float(first["obs_cmd_x"]), float(first["obs_cmd_y"]), float(first["obs_cmd_wz"])]
        for i in range(12):
            row = rows[i]
            obs[0, 12 + i] = float(row["obs_joint_pos_policy"])
            obs[0, 24 + i] = float(row["obs_joint_vel_policy"])
            if obs_dim >= 48:
                obs[0, 36 + i] = float(row["obs_last_action_policy"])
        if obs_dim >= 50:
            # CSV currently does not store gait phase explicitly.  Use the same
            # neutral phase used for default checks; live CSV gate remains the
            # authority for real deployment readiness.
            obs[0, 48:50] = np.array([0.0, 1.0], dtype=np.float32)
        obs_list.append(obs)
    return obs_list


def _print_action_stats(label: str, action: np.ndarray):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    print(
        f"{label}: raw_action_abs_max={np.max(np.abs(action)):.4f} "
        f"mean={np.mean(action):+.4f} std={np.std(action):.4f} "
        f"action={np.array2string(action, precision=4, separator=', ')}"
    )


def main():
    parser = argparse.ArgumentParser(description="Verify exported Fanfan ONNX actions on deployment-like observations.")
    parser.add_argument("--onnx", required=True, help="Path to exported policy.onnx")
    parser.add_argument("--jit", default="", help="Optional exported policy.pt for JIT-vs-ONNX comparison")
    parser.add_argument("--csv", default="", help="Optional real-policy CSV to replay observations from")
    parser.add_argument("--csv-limit", type=int, default=128)
    parser.add_argument("--warn-action-abs", type=float, default=2.0)
    args = parser.parse_args()

    sess, input_name, obs_dim = _load_onnx_session(args.onnx)
    if obs_dim is None:
        obs_dim = 50
        print("[WARN] Could not infer ONNX obs dim; assuming 50.")
    print(f"[ONNX] input={input_name} obs_dim={obs_dim}")

    test_obs = [
        ("default_cmd0", _default_obs(obs_dim, 0.0)),
        ("default_cmd015", _default_obs(obs_dim, 0.15)),
    ]
    if args.csv:
        for idx, obs in enumerate(_obs_from_csv(args.csv, limit=args.csv_limit)):
            if obs.shape[1] == obs_dim:
                test_obs.append((f"csv_{idx:04d}", obs))

    max_diff = 0.0
    max_abs = 0.0
    for label, obs in test_obs:
        onnx_action = _run_onnx(sess, input_name, obs)
        max_abs = max(max_abs, float(np.max(np.abs(onnx_action))))
        _print_action_stats(label, onnx_action)
        if args.jit:
            jit_action = _run_jit(args.jit, obs)
            diff = np.abs(jit_action - onnx_action)
            max_diff = max(max_diff, float(np.max(diff)))
            print(f"{label}: jit_vs_onnx max_abs_diff={np.max(diff):.6g} mean_abs_diff={np.mean(diff):.6g}")

    if args.jit:
        print(f"[SUMMARY] max_abs_diff={max_diff:.6g}")
        if max_diff > 1.0e-3:
            print("[FAIL] JIT and ONNX differ more than 1e-3.")
    print(f"[SUMMARY] max_raw_action_abs={max_abs:.4f}")
    if max_abs > args.warn_action_abs:
        print("[WARN] Policy action is large on deployment-like obs; do not rush to real walking.")


if __name__ == "__main__":
    main()
