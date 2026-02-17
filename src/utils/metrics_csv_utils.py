"""
Utilities for CSV metrics export in evaluation scripts.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_CSV_FIELDS: List[str] = [
    "sample_id",
    "step_index",
    "guidance_scale",
    "part_index",
    "num_parts",
    "chamfer_distance",
    "fscore",
    "scene_iou",
    "bops_billions",
    "abw_weights_bits",
    "abw_activations_bits",
    "latency_seconds",
    "peak_vram_gb",
    "token_throughput",
]


def aggregate_series(
    series_per_part: Optional[List[List[float]]],
    num_timesteps: int,
    offset: int,
    default_value: float,
) -> List[float]:
    values = [default_value] * num_timesteps
    if not series_per_part:
        return values
    last_val = default_value
    for step in range(num_timesteps):
        idx = step - offset
        if idx >= 0:
            step_samples = [
                part[idx]
                for part in series_per_part
                if len(part) > idx
            ]
            if step_samples:
                last_val = float(np.mean(step_samples))
        values[step] = last_val
    return values


def assign_bitwidth_schedule(
    avg_diffs: List[float],
    avg_curvatures: List[float],
    num_timesteps: int,
    diff_threshold: float,
    curvature_threshold: float,
    valley_start: int = 20,
    valley_end: int = 30,
) -> Tuple[List[int], List[int]]:
    w_bits, a_bits = [], []
    for step in range(num_timesteps):
        if step < valley_start:
            w_bits.append(8)
            a_bits.append(8)
            continue
        if valley_start <= step <= valley_end:
            diff_val = avg_diffs[step]
            curvature_val = avg_curvatures[step]
            if diff_val < diff_threshold:
                if curvature_val >= curvature_threshold:
                    w_bits.append(16)
                    a_bits.append(4)
                else:
                    w_bits.append(4)
                    a_bits.append(16)
            else:
                w_bits.append(8)
                a_bits.append(8)
            continue
        w_bits.append(4)
        a_bits.append(8)
    return w_bits, a_bits


def compute_bops_from_schedule(
    w_bits: List[int],
    a_bits: List[int],
    total_gflops: float,
    num_timesteps: int,
) -> float:
    if not w_bits or not a_bits or num_timesteps <= 0:
        return 0.0
    total_macs = total_gflops * 1e9
    macs_per_step = total_macs / num_timesteps
    bits_product_sum = sum(w * a for w, a in zip(w_bits, a_bits))
    return (macs_per_step * bits_product_sum) / 1e9


def compute_abw_weights(high_ratio: float, high_bits: int, low_bits: int) -> float:
    high_ratio = min(max(high_ratio, 0.0), 1.0)
    return high_ratio * high_bits + (1.0 - high_ratio) * low_bits


def compute_peak_vram_gb(snapshots: List[Dict[str, Any]]) -> float:
    peak = 0.0
    for snap in snapshots:
        for key in ("gpu_memory_allocated_gb", "gpu_memory_reserved_gb"):
            val = snap.get(key)
            if isinstance(val, (int, float)):
                peak = max(peak, float(val))
        mem_mb = snap.get("gpu_memory_used_mb")
        if isinstance(mem_mb, (int, float)):
            peak = max(peak, float(mem_mb) / 1024.0)
    return peak


def _nanmean(values: List[Any]) -> float:
    filtered: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(val):
            continue
        filtered.append(val)
    return float(np.mean(filtered)) if filtered else float("nan")


def build_summary_rows(
    part_rows: List[Dict[str, Any]],
    object_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not part_rows:
        return []
    summary_rows: List[Dict[str, Any]] = []
    guidance_values = sorted({row["guidance_scale"] for row in object_rows})
    for guidance in guidance_values + ["ALL"]:
        if guidance == "ALL":
            part_subset = part_rows
            obj_subset = object_rows
            label = "GLOBAL_AVG"
            guidance_value: Any = "ALL"
        else:
            part_subset = [row for row in part_rows if row["guidance_scale"] == guidance]
            obj_subset = [row for row in object_rows if row["guidance_scale"] == guidance]
            label = f"GS_{guidance:.1f}_AVG"
            guidance_value = guidance
        if not part_subset and not obj_subset:
            continue
        summary_rows.append({
            "sample_id": label,
            "step_index": "avg",
            "guidance_scale": guidance_value,
            "part_index": "all",
            "num_parts": _nanmean([row.get("num_parts") for row in obj_subset]),
            "chamfer_distance": _nanmean([row["chamfer_distance"] for row in part_subset]),
            "fscore": _nanmean([row["fscore"] for row in part_subset]),
            "scene_iou": _nanmean([row.get("scene_iou") for row in obj_subset]),
            "bops_billions": _nanmean([row.get("bops_billions") for row in obj_subset]),
            "abw_weights_bits": _nanmean([row.get("abw_weights_bits") for row in obj_subset]),
            "abw_activations_bits": _nanmean([row.get("abw_activations_bits") for row in obj_subset]),
            "latency_seconds": _nanmean([row.get("latency_seconds") for row in obj_subset]),
            "peak_vram_gb": _nanmean([row.get("peak_vram_gb") for row in obj_subset]),
            "token_throughput": _nanmean([row.get("token_throughput") for row in obj_subset]),
        })
    return summary_rows