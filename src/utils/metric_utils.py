from src.utils.typing_utils import *

import os
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv

import torch

from extensions.chamfer_dist import ChamferDistanceL2
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points

import math
from typing import Any, Dict, List, Optional, Tuple

def sample_from_mesh(
    mesh: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
):
    if num_samples is None:
        return mesh.vertices
    else:
        return mesh.sample(num_samples)

def sample_two_meshes(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
):
    points1 = sample_from_mesh(mesh1, num_samples)
    points2 = sample_from_mesh(mesh2, num_samples)
    return points1, points2

def compute_nearest_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    # Compute nearest neighbor distance from points1 to points2
    nn = NearestNeighbors(n_neighbors=1, leaf_size=30, algorithm='kd_tree', metric=metric).fit(points2)
    min_dist = nn.kneighbors(points1)[0]
    return min_dist

def compute_mutual_nearest_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    min_1_to_2 = compute_nearest_distance(points1, points2, metric=metric)
    min_2_to_1 = compute_nearest_distance(points2, points1, metric=metric)
    return min_1_to_2, min_2_to_1

def compute_mutual_nearest_distance_for_meshes(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
    metric: str = 'l2'
) -> Tuple[np.ndarray, np.ndarray]:
    points1 = sample_from_mesh(mesh1, num_samples)
    points2 = sample_from_mesh(mesh2, num_samples)
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance(points1, points2, metric=metric)
    return min_1_to_2, min_2_to_1

def compute_chamfer_distance(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: int = 10000,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    return chamfer_dist

def compute_f_score(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: int = 10000,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return fscore

def compute_cd_and_f_score(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return chamfer_dist, fscore

# def compute_cd_and_f_score(
#     mesh1: trimesh.Trimesh,
#     mesh2: trimesh.Trimesh,
#     num_samples: Optional[int] = 10000,
#     threshold: float = 0.1,
#     metric: str = 'l2'
# ):
#     # min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
#     # chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
#     chamfer_dist = ChamferDistanceL2().cuda()(torch.tensor(mesh1, device='cuda').unsqueeze(0), torch.tensor(mesh2.vertices, device='cuda').unsqueeze(0)).item()
#     # precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
#     # precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
#     # fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
#     fscore = 0.0
#     return chamfer_dist, fscore

# OUR IMPLIMENTATION
def compute_cd_and_f_score_cuda(
    gt_points: torch.Tensor,
    pred_mesh: trimesh.Trimesh,
    num_samples: int = 204800,
    threshold: float = 0.1,
):
    device = gt_points.device
    # if pred_verts.device != device or pred_faces.device != device:
    #     raise ValueError("All input tensors must be on the same CUDA device.")

    if gt_points.shape[-1] > 3:
        gt_points = gt_points[:, :3]
    if gt_points.shape[0] < num_samples:
        gt_sampled_points = gt_points
    else:
        perm = torch.randperm(gt_points.shape[0], device=device)
        idx = perm[:num_samples]
        gt_sampled_points = gt_points[idx]
        
    gt_sampled_points = gt_sampled_points.unsqueeze(0)

    pred_sampled_points = sample_from_mesh(pred_mesh, num_samples)
    pred_sampled_points = torch.from_numpy(pred_sampled_points).float().to(device).unsqueeze(0)
    knn_gt_to_pred = knn_points(gt_sampled_points, pred_sampled_points, K=1)
    dists_gt_to_pred = torch.sqrt(knn_gt_to_pred.dists.squeeze(-1))
    
    knn_pred_to_gt = knn_points(pred_sampled_points, gt_sampled_points, K=1)
    dists_pred_to_gt = torch.sqrt(knn_pred_to_gt.dists.squeeze(-1))

    chamfer_dist = torch.mean(dists_gt_to_pred) + torch.mean(dists_pred_to_gt)

    precision = torch.mean((dists_gt_to_pred < threshold).float())
    recall = torch.mean((dists_pred_to_gt < threshold).float())
    fscore = 2 * precision * recall / (precision + recall + 1e-8) 

    return chamfer_dist.item(), fscore.item()

def compute_cd_and_f_score_in_training(
    gt_surface: np.ndarray,
    pred_mesh: trimesh.Trimesh,
    num_samples: int = 204800,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    gt_points = gt_surface[:, :3]
    num_samples = max(num_samples, gt_points.shape[0])
    gt_points = gt_points[np.random.choice(gt_points.shape[0], num_samples, replace=False)]
    pred_points = sample_from_mesh(pred_mesh, num_samples)
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance(gt_points, pred_points, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return chamfer_dist, fscore

def get_voxel_set(
    mesh: trimesh.Trimesh,
    num_grids: int = 64,
    scale: float = 2.0,
):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("mesh must be a trimesh.Trimesh object")
    pitch = scale / num_grids
    voxel_girds: trimesh.voxel.base.VoxelGrid = mesh.voxelized(pitch=pitch).fill()
    voxels = set(map(tuple, np.round(voxel_girds.points / pitch).astype(int)))
    return voxels

def compute_IoU(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_grids: int = 64,
    scale: float = 2.0,
):
    if not isinstance(mesh1, trimesh.Trimesh) or not isinstance(mesh2, trimesh.Trimesh):
        raise ValueError("mesh1 and mesh2 must be trimesh.Trimesh objects")
    voxels1 = get_voxel_set(mesh1, num_grids, scale)
    voxels2 = get_voxel_set(mesh2, num_grids, scale)
    intersection = voxels1 & voxels2
    union = voxels1 | voxels2
    iou = len(intersection) / len(union) if len(union) > 0 else 0.0
    return iou

def compute_IoU_for_scene(
    scene: Union[trimesh.Scene, List[trimesh.Trimesh]],
    num_grids: int = 64,
    scale: float = 2.0,
    return_type: Literal["iou", "iou_list"] = "iou",
):
    if isinstance(scene, trimesh.Scene):
        scene = scene.dump()
    if isinstance(scene, list) and len(scene) > 1 and isinstance(scene[0], trimesh.Trimesh):
        meshes = scene
    else:
        raise ValueError("scene must be a trimesh.Scene object or a list of trimesh.Trimesh objects")
    ious = []
    for i in range(len(meshes)):
        for j in range(i+1, len(meshes)):
            iou = compute_IoU(meshes[i], meshes[j], num_grids, scale)
            ious.append(iou)
    if return_type == "iou":
        return np.mean(ious)
    elif return_type == "iou_list":
        return ious
    else:
        raise ValueError("return_type must be 'iou' or 'iou_list'")


def compute_pae(gt_part_surfaces, pred_part_meshes, num_samples=204800, threshold=0.1):
    """
    Compute Part Alignment Error (PAE) as the average Chamfer distance over parts.
    
    Args:
        gt_part_surfaces: List of GT surface points (np.ndarray).
        pred_part_surfaces: List of predicted surface points or meshes.
        num_samples: Number of samples for distance computation.
        threshold: Threshold for F-score (not used here).
    
    Returns:
        float: Average Chamfer distance.
    """
    cd_list = []
    for gt_surf, pred_mesh in zip(gt_part_surfaces, pred_part_meshes):
        if pred_mesh is not None and len(pred_mesh.vertices) > 0:
            cd, _ = compute_cd_and_f_score_in_training(gt_surf, pred_mesh, num_samples, threshold)
            cd_list.append(cd)
    if cd_list:
        return np.mean(cd_list)
    else:
        return float('nan')


def save_aggregate_metrics_csv(
    eval_dir: str,
    metrics_csv_name: str,
    latencies: List[float],
    bops_list: List[float],
    abw_weights_list: List[float],
    abw_activations_list: List[float],
    metrics_summary: Dict,
    logger
):
    # print("here")
    if not latencies:
        # print("here2")
        return
    avg_latency = np.mean(latencies)
    avg_bops = np.mean(bops_list)
    avg_abw_weights = np.mean(abw_weights_list)
    avg_abw_activations = np.mean(abw_activations_list)
    overall_cd = np.mean([cd for gs, metrics in metrics_summary.items() for cd in metrics["chamfer"]])
    ious = [iou for gs, metrics in metrics_summary.items() for iou in metrics["iou"] if iou is not None]
    overall_iou = np.mean(ious) if ious else float('nan')
    overall_f1 = np.nanmean([f1 for gs, metrics in metrics_summary.items() for f1 in metrics["f1_score"]])
    
    row = {
        "latency_seconds": avg_latency,
        "bops_billions": avg_bops,
        "abw_weights_bits": avg_abw_weights,
        "abw_activations_bits": avg_abw_activations,
        "iou": overall_iou,
        "f_score": overall_f1,
        "chamfer_distance": overall_cd,
    }
    fieldnames = ["latency_seconds", "bops_billions", "abw_weights_bits", "abw_activations_bits", "iou", "f_score", "chamfer_distance"]
    csv_path = os.path.join(eval_dir, metrics_csv_name)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"Aggregate metrics saved to {csv_path}")


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