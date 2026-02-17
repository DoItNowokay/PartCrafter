"""
Test script for evaluating PartCrafter model on part segmentation and generation.
"""

import warnings
warnings.filterwarnings("ignore")
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()

import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from src.utils.typing_utils import *
from accelerate.utils import set_seed
from src.utils.render_utils import (
    render_views_around_mesh,
    render_normal_views_around_mesh,
    export_renderings,
    make_grid_for_images_or_videos,
    save_mesh_and_renderings
)
from src.utils.data_utils import get_colored_mesh_composition

# Standard library
import argparse
import csv
import logging
import math
from collections import defaultdict
import trimesh
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import random
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import wandb
import torch
import torch.nn as nn
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from torchvision import transforms

from transformers import BitImageProcessor, Dinov2Model
from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartCrafterDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.train_utils import get_configs, save_experiment_params, save_model_architecture
# from src.utils.metric_utils import compute_cd_and_f_score_cuda
from src.utils.metric_utils import *
from src.utils.resource_utils import record_resource_snapshot, compute_peak_vram_gb
from src.models.briarmbg import BriaRMBG
from src.utils.image_utils import prepare_image
from src.utils.resource_utils import get_gpu_stats
from huggingface_hub import snapshot_download

from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from torch.utils.data import DataLoader
from src.utils.gradient_analysis_utils import GradientSensitivityAnalyzer
from src.utils.metrics_csv_utils import (
    DEFAULT_CSV_FIELDS,
    aggregate_series,
    assign_bitwidth_schedule,
    compute_bops_from_schedule,
    compute_abw_weights,
    compute_peak_vram_gb,
    _nanmean,
    build_summary_rows,
)

def sanitize_artifact_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    sanitized = sanitized.strip("_")
    return sanitized or "artifact"

def save_outputs(
    local_eval_dir: str,
    pred_part_meshes: List[Optional[trimesh.Trimesh]],
    input_image_pil: Image.Image,
    configs: Dict,
    args: argparse.Namespace,
    guidance_scale: float,
    step: int,
    logger: logging.Logger
):
    """
    Save predicted part meshes, renderings, and log media to Weights & Biases.

    Args:
        local_eval_dir: Directory to save outputs.
        pred_part_meshes: List of predicted meshes for each part.
        input_image_pil: Input image.
        configs: Configuration dictionary.
        args: Parsed arguments.
        guidance_scale: Guidance scale used.
        step: Current evaluation step.
        logger: Logger instance.
    """
    os.makedirs(local_eval_dir, exist_ok=True)
    
    input_image_pil.save(os.path.join(local_eval_dir, "input_image.png"))
    for i, mesh in enumerate(pred_part_meshes):
        if mesh: 
            mesh.export(os.path.join(local_eval_dir, f"part_{i:02d}.glb"))

    valid_meshes = pred_part_meshes
    
    if valid_meshes:
        merged_mesh = get_colored_mesh_composition(valid_meshes)
        save_mesh_and_renderings(
            merged_mesh,
            local_eval_dir,
            mesh_filename="object.glb",
            rendering_prefix="rendering",
            render_cfg=configs['test']['rendering'],
            input_image_pil=input_image_pil,
        )
        # Also log media to wandb if enabled
        if not args.no_wandb:
            wandb.log({
                f"evaluation/gs_{guidance_scale:.1f}/input_image": wandb.Image(input_image_pil),
                f"evaluation/gs_{guidance_scale:.1f}/render_video": wandb.Video(
                    os.path.join(local_eval_dir, "rendering.gif"),
                    fps=configs['test']['rendering'].get('fps', 18),
                    format="gif"
                ),
            }, step=step)
    else:
        logger.warning(
            f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | "
            "No valid meshes to merge for visualization."
        )


@torch.no_grad()
def run_evaluation(
    dataloader: DataLoader,
    pipeline: PartCrafterPipeline,
    accelerator: Accelerator,
    logger: logging.Logger,
    args: argparse.Namespace,
    configs: Dict,
    eval_dir: str,
    rmbg_net: BriaRMBG
):
    """
    Run evaluation on the test dataset, computing metrics and saving outputs.

    Args:
        dataloader: DataLoader for test data.
        pipeline: PartCrafter pipeline.
        accelerator: Accelerator for distributed training.
        logger: Logger instance.
        args: Parsed arguments.
        configs: Configuration dictionary.
        eval_dir: Evaluation directory.
        rmbg_net: Background removal network.
    """
    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None

    metrics_summary = defaultdict(lambda: defaultdict(list))
    resource_metrics = []
    csv_rows: List[Dict[str, Any]] = []
    object_level_metrics: List[Dict[str, Any]] = []
    num_timesteps = configs["test"]["num_inference_steps"]
    num_tokens = configs["model"]["vae"]["num_tokens"]
    test_cfg = configs["test"]
    csv_fieldnames = None
    if hasattr(test_cfg, "get"):
        csv_fieldnames = test_cfg.get("metrics_csv_fields", None)
    if not csv_fieldnames:
        csv_fieldnames = DEFAULT_CSV_FIELDS

    def record_resource_snapshot(
        step_idx: int,
        guidance: Optional[float],
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "timestamp": time.time(),
            "step": step_idx,
            "guidance_scale": guidance,
            "phase": phase,
        }
        if extra:
            snapshot.update(extra)
        snapshot.update(get_gpu_stats())
        resource_metrics.append(snapshot)
        return snapshot

    analyzer = None
    if args.analyze_sensitivity:
        # logger.info(f"Initializing Gradient Sensitivity Analyzer (Method: {args.gradient_analysis_method})")
        analyzer = GradientSensitivityAnalyzer(methods=args.gradient_analysis_method)
        pipeline.transformer.requires_grad_(True)

    progress_bar = tqdm(
        enumerate(dataloader),
        total=args.max_test_steps,
        desc="Evaluating",
        disable=not accelerator.is_main_process
    )
    
    for step, batch in progress_bar:
        step_start_stats = record_resource_snapshot(step, None, "step_start")
        save_iteration = accelerator.is_main_process and args.save_ratio > 0 and random.random() < args.save_ratio
        if step >= args.max_test_steps:
            break

        if configs["test"]["batch_size_per_gpu"] != 1:
            logger.warning("Warning: Evaluation is designed for a batch size of 1.")

        image_path = batch["images"][0]
        
        # Create generator per step for reproducibility independent of order
        if args.seed >= 0:
            # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + step)
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        else:
            generator = None

        # Apply background removal
        # Use .module if using accelerator (DDP)
        _rmbg_net = accelerator.unwrap_model(rmbg_net)
        image_pil = prepare_image(
            image_path,
            bg_color=np.array([1.0, 1.0, 1.0]),
            rmbg_net=_rmbg_net
        )
        
        # Fallback if prepare_image fails and returns None (e.g., from empty contour fix)
        if image_pil is None:
            logger.warning(
                f"Step: {step:04d} | SKIPPING. Image preprocessing failed "
                "(e.g., empty mask)."
            )
            continue

        gt_part_surfaces = batch["part_surfaces"]
        gt_part_surfaces = gt_part_surfaces[0].cpu().numpy() # (1, N, P, 6) -> (N, P, 6)
        num_parts = batch["num_parts"][0]
        
        # This limit is from inference_partcrafter.py (MAX_NUM_PARTS = 16)
        MAX_MODEL_PARTS = 16 
        if num_parts > MAX_MODEL_PARTS:
            logger.warning(
                f"Step: {step:04d} | SKIPPING. num_parts ({num_parts}) > "
                f"max ({MAX_MODEL_PARTS})."
            )
            continue  # Skip this batch and go to the next one
        
        # Log the num_parts being used for this item
        logger.info(f"Step: {step:04d} | Processing with num_parts: {num_parts}")
        
        input_image_pil = image_pil 

        for guidance_scale in sorted(args.test_guidance_scales):
            local_resource_snapshots = [step_start_stats]
            local_resource_snapshots.append(
                record_resource_snapshot(
                    step,
                    guidance_scale,
                    f"pre_generation_gs_{guidance_scale}",
                )
            )
            start_time = time.time()
            save_intermediates = accelerator.is_main_process and args.save_intermediates and save_iteration
            log_tokens_diff = accelerator.is_main_process and args.tokens_diff and save_iteration
            log_curvature = accelerator.is_main_process and args.curvature and save_iteration
            log_entropy = accelerator.is_main_process and args.entropy and save_iteration
            save_intermediate_dir = os.path.join(eval_dir, f"gs_{guidance_scale:.1f}", f"step_{step:04d}")
            
            if analyzer:
                analyzer.reset_for_new_step(save_intermediate_dir)

            with torch.no_grad():
                output = pipeline(
                    [image_pil] * num_parts,
                    attention_kwargs={"num_parts": num_parts},
                    num_tokens=configs['model']['vae']['num_tokens'],
                    generator=generator,
                    num_inference_steps=configs['test']['num_inference_steps'],
                    guidance_scale=guidance_scale,
                    max_num_expanded_coords=configs['test']['max_num_expanded_coords'],
                    use_flash_decoder=configs['test']['use_flash_decoder'],
                    save_intermediates=save_intermediates,
                    save_tokens_diff=log_tokens_diff,
                    save_curvature=log_curvature,
                    save_entropy=log_entropy,
                    save_intermediate_dir=save_intermediate_dir, 
                    collect_dynamics_stats=True,
                    configs=configs,
                    analyzer=analyzer 
                )
            
            if analyzer:
                analyzer.plot_results()

            pred_part_meshes = output.meshes
            end_time = time.time()
            generation_time = end_time - start_time
            post_stats = record_resource_snapshot(
                step,
                guidance_scale,
                f"post_generation_gs_{guidance_scale}",
                {"generation_time_seconds": generation_time},
            )
            local_resource_snapshots.append(post_stats)
            logger.info(
                f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | "
                f"Generation Time: {generation_time:.2f} seconds"
            )

            peak_vram_gb = compute_peak_vram_gb(local_resource_snapshots)

            local_eval_dir = os.path.join(eval_dir, f"gs_{guidance_scale:.1f}", f"step_{step:04d}")
            for n in range(num_parts):
                if pred_part_meshes[n] is None:
                    # If the generated mesh is None (decoing error), use a dummy mesh
                    pred_part_meshes[n] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                # pred_part_meshes[n].export(os.path.join(local_eval_dir, f"{n:02d}.glb"))
            batch_cds, batch_f_scores = [], []
            for i in range(num_parts):
                pred_mesh = pred_part_meshes[i]
                gt_surface = gt_part_surfaces[i]

                if pred_mesh is None or len(pred_mesh.vertices) == 0:
                    if accelerator.is_main_process:
                        logger.warning(
                            f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | "
                            f"Part: {i:02d} | SKIPPED: Predicted mesh is None or empty. "
                            "Using default metrics."
                        )
                    
                    part_cd = torch.tensor(configs['test']['metric']['default_cd'])
                    part_f = torch.tensor(configs['test']['metric']['default_f1'])
                    pred_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                    pred_part_meshes[i] = pred_mesh  # Replace None with dummy mesh for saving later
                else:
                    # cd, f_score = compute_cd_and_f_score_cuda(
                    #     gt_surface, pred_mesh,
                    #     num_samples=configs['test']['metric']['cd_num_samples'],
                    #     threshold=configs['test']['metric']['f1_score_threshold'],
                    # )
                    # cd, f_score = compute_cd_and_f_score(
                    #     gt_surface.cpu().numpy(), pred_mesh
                    # )
                    cd, f_score = compute_cd_and_f_score_in_training(
                        gt_surface, pred_mesh,
                        num_samples=configs['test']['metric']['cd_num_samples'],
                        threshold=configs['test']['metric']['f1_score_threshold'],
                        metric=configs['test']['metric']['cd_metric']
                    )
                    part_cd = cd.cpu() if isinstance(cd, torch.Tensor) else torch.tensor(cd)
                    part_f = f_score.cpu() if isinstance(f_score, torch.Tensor) else torch.tensor(f_score)
                
                if accelerator.is_main_process:
                    logger.info(
                        f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | "
                        f"Part: {i:02d} | CD: {part_cd.item():.4f} | F1: {part_f.item():.4f}"
                    )

                batch_cds.append(part_cd.item())
                batch_f_scores.append(part_f.item())
            
            # calculate IoU for the whole scene (merged mesh) if possible
            IoU = None
            if (num_parts > 1):
                IoU = compute_IoU_for_scene(pred_part_meshes)
            if accelerator.is_main_process:
                logger.info(
                    f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | "
                    f"IoU for merged scene: {IoU:.4f}" if IoU is not None else "IoU for merged scene: N/A"
                )

            token_diffs = getattr(output, "token_diffs", None)
            curvature_traces = getattr(output, "curvature", None)
            avg_token_diffs = aggregate_series(token_diffs, num_timesteps, offset=1, default_value=1.0)
            avg_curvatures = aggregate_series(curvature_traces, num_timesteps, offset=2, default_value=0.0)
            w_schedule, a_schedule = assign_bitwidth_schedule(
                avg_token_diffs,
                avg_curvatures,
                num_timesteps,
                diff_threshold=args.token_diff_threshold,
                curvature_threshold=args.curvature_spike_threshold,
            )
            bops_billions = compute_bops_from_schedule(
                w_schedule,
                a_schedule,
                args.total_inference_gflops,
                num_timesteps,
            )
            abw_weights_bits = compute_abw_weights(
                args.abw_high_bit_ratio,
                args.abw_high_bit_width,
                args.abw_low_bit_width,
            )
            abw_activations_bits = float(np.mean(a_schedule)) if a_schedule else 0.0
            
            metrics_summary[guidance_scale]["chamfer"].extend(batch_cds)
            metrics_summary[guidance_scale]["f1_score"].extend(batch_f_scores)
            metrics_summary[guidance_scale]["iou"].append(IoU)

            if accelerator.is_main_process:
                scene_iou_value = float(IoU) if IoU is not None else float("nan")
                token_count = int(num_parts * num_tokens)
                token_throughput = token_count / max(generation_time, 1e-6)
                sample_id = os.path.splitext(os.path.basename(image_path))[0]
                base_metrics = {
                    "sample_id": sample_id,
                    "step_index": int(step),
                    "guidance_scale": guidance_scale,
                    "num_parts": int(num_parts),
                    "scene_iou": scene_iou_value,
                    "bops_billions": bops_billions,
                    "abw_weights_bits": abw_weights_bits,
                    "abw_activations_bits": abw_activations_bits,
                    "latency_seconds": generation_time,
                    "peak_vram_gb": peak_vram_gb,
                    "token_throughput": token_throughput,
                }
                for part_idx, (cd_value, f_value) in enumerate(zip(batch_cds, batch_f_scores)):
                    row = base_metrics.copy()
                    row.update({
                        "part_index": part_idx,
                        "chamfer_distance": float(cd_value),
                        "fscore": float(f_value),
                    })
                    csv_rows.append(row)
                object_level_metrics.append({
                    **base_metrics,
                    "part_index": "object",
                    "chamfer_distance": float(np.mean(batch_cds)) if batch_cds else float("nan"),
                    "fscore": float(np.mean(batch_f_scores)) if batch_f_scores else float("nan"),
                })

            # Log per-item metrics/media to Weights & Biases (main process only)
            if accelerator.is_main_process and (not args.no_wandb):
                item_logs = {
                    f"evaluation/cd_cfg{guidance_scale:.1f}": float(np.mean(batch_cds)),
                    f"evaluation/f1_cfg{guidance_scale:.1f}": float(np.mean(batch_f_scores)),
                    f"evaluation/iou_cfg{guidance_scale:.1f}": IoU if IoU is not None else float('nan'),
                    "evaluation/num_parts": int(num_parts),
                }
                wandb.log(item_logs, step=step)

            if accelerator.is_main_process and args.save_ratio > 0 and save_iteration:
                save_outputs(local_eval_dir, pred_part_meshes, input_image_pil, configs, args, guidance_scale, step, logger)
            
    if accelerator.is_main_process:
        logger.info("\n" + "="*60 + "\n                 Evaluation Results Summary\n" + "="*60)
        report_path = os.path.join(eval_dir, "results.txt")
        with open(report_path, "w") as f:
            f.write("Evaluation Results Summary\n" + "="*60 + "\n")
            for guidance_scale, metrics in sorted(metrics_summary.items()):
                avg_cd = np.mean(metrics["chamfer"])
                avg_f1 = np.mean(metrics["f1_score"])
                log_msg = (
                    f"Guidance Scale: {guidance_scale:<4.1f} | "
                    f"Avg Chamfer Distance: {avg_cd:.4f} | "
                    f"Avg F1-Score: {avg_f1:.4f} | "
                    f"IoU: {np.mean([x for x in metrics['iou'] if x is not None]) if any(x is not None for x in metrics['iou']) else float('nan'):.4f}"
                )
                logger.info(log_msg)
                f.write(log_msg + "\n")
        logger.info(f"\nResults saved to {report_path}")
        # Log aggregate metrics and artifacts to wandb
        if not args.no_wandb:
            aggregate_logs = {}
            for guidance_scale, metrics in sorted(metrics_summary.items()):
                aggregate_logs.update({
                    f"evaluation/avg_cd_cfg{guidance_scale:.1f}": (
                        float(np.mean(metrics["chamfer"])) if len(metrics["chamfer"]) else float('nan')
                    ),
                    f"evaluation/avg_f1_cfg{guidance_scale:.1f}": (
                        float(np.mean(metrics["f1_score"])) if len(metrics["f1_score"]) else float('nan')
                    ),
                    f"evaluation/avg_iou_cfg{guidance_scale:.1f}": (
                        float(np.mean([x for x in metrics["iou"] if x is not None])) if any(x is not None for x in metrics["iou"]) else float('nan')
                    ),
                })
            if aggregate_logs:
                wandb.log(aggregate_logs)
            # Attach result file as an artifact
            artifact_name = f"{sanitize_artifact_name(args.tag)}_eval"
            arti = wandb.Artifact(artifact_name, type="evaluation")
            arti.add_file(report_path)
            log_path = os.path.join(eval_dir, "log.txt")
            if os.path.exists(log_path):
                arti.add_file(log_path)
            wandb.log_artifact(arti)
        if csv_rows:
            csv_path = os.path.join(eval_dir, args.metrics_csv_name)
            summary_rows = build_summary_rows(csv_rows, object_level_metrics)
            fieldnames = list(csv_fieldnames or DEFAULT_CSV_FIELDS)
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in csv_rows + summary_rows:
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
            logger.info(f"Structured metrics saved to {csv_path}")
        
    # Save resource metrics to JSON for analysis
    resource_path = os.path.join(eval_dir, "resource_metrics.json")
    with open(resource_path, "w") as f:
        json.dump(resource_metrics, f, indent=2)
    logger.info(f"Resource metrics saved to {resource_path}")
        
def main():
    """
    Main entry point for the evaluation script.
    Parses arguments, sets up models and data, and runs evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a PartCrafter model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save results.")
    parser.add_argument("--tag", type=str, default="test_run", help="A specific tag for this run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_test_steps", type=int, default=None, help="Max number of batches to evaluate.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers. Set to 0 to avoid I/O bottlenecks.")
    parser.add_argument("--test_guidance_scales", type=float, nargs="+", default=[7.0], help="List of CFG scales to test.")
    parser.add_argument("--save_ratio", type=float, default=0.1, help="Ratio of outputs to save randomly (e.g., 0.1 saves 10%).")
    parser.add_argument("--tokens_diff", action="store_true", help="Whether to save intermediate token differences for debugging.")
    parser.add_argument("--curvature", action="store_true", help="Whether to save intermediate curvature metrics for debugging.")
    parser.add_argument("--entropy", action="store_true", help="Whether to save intermediate shanon entropy for debugging.")
    parser.add_argument("--save_intermediates", action="store_true", help="Whether to save intermediate meshes and renderings during generation.")
    parser.add_argument("--offline_wandb", action="store_true", help="Use offline WandB for experiment tracking")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB for experiment tracking")
    parser.add_argument("--analyze_sensitivity", action="store_true", help="Enable gradient sensitivity analysis")
    parser.add_argument("--gradient_analysis_method", type=str,nargs="+", default=["gradient_norm"], choices=["gradient_norm", "fisher", "weight_gradient"], help="Method for gradient sensitivity analysis")
    parser.add_argument("--token_diff_threshold", type=float, default=0.008, help="Delta z_t threshold used to trigger valley-phase bit allocation.")
    parser.add_argument("--curvature_spike_threshold", type=float, default=0.12, help="Curvature threshold for Anchor vs Diffuse bit assignment.")
    parser.add_argument("--abw_high_bit_ratio", type=float, default=0.1, help="Ratio of weights kept at the high bit-width when computing ABW.")
    parser.add_argument("--abw_high_bit_width", type=int, default=16, help="High bit-width used for load-bearing weights.")
    parser.add_argument("--abw_low_bit_width", type=int, default=4, help="Low bit-width used for fill weights.")
    parser.add_argument("--total_inference_gflops", type=float, default=105.0, help="Total GFLOPs per full inference pass (used for BOPs).")
    parser.add_argument("--metrics_csv_name", type=str, default="metrics.csv", help="Filename for the exported per-sample metrics table.")
    
    args, extras = parser.parse_known_args()
    configs = get_configs(args.config, extras)

    precisions = ""
    for prec in configs.test.bit_precision:
        if prec == "fp16":
            precisions += "fp16_"
        elif prec == "bf16":
            precisions += "bf16_"
        elif prec == "fp32":
            precisions += "fp32_"
        else:
            raise ValueError(f"Unsupported precision: {prec}")
    max_num_samples = configs.dataset.get("max_num_samples", "all")
    num_tokens = configs.model.vae.num_tokens
    num_inference_steps = configs.test.num_inference_steps
    dataset = configs.dataset.config[0].split("/")[-2]
    args.tag = f"{dataset}_{max_num_samples}/num_tokens_{num_tokens}/diffusion_steps_{num_inference_steps}/{precisions}/{args.tag}"
    args.wandb_tag = args.tag.replace("/", "_")  # For cleaner WandB tags
    eval_dir = os.path.join(args.output_dir, f"{args.tag}/{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(eval_dir, exist_ok=True)
    
    accelerator = Accelerator()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = get_accelerate_logger(__name__, log_level="INFO")
    
    if accelerator.is_main_process:
        fh = logging.FileHandler(os.path.join(eval_dir, "log.txt"))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.logger.addHandler(fh)

    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)

    # Save experiment parameters to eval_dir for reproducibility and W&B
    exp_params = save_experiment_params(args, configs, eval_dir)

    logger.info("Downloading base models and RMBG...")
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    logger.info("Loading RMBG model...")
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(accelerator.device)
    rmbg_net.eval() 

    logger.info(f"Loading base models from: {partcrafter_weights_dir}")
    # set weight dtype based on config
    assert len(configs.test.bit_precision) == 1, "Only single precision supported for testing as of now."
    if configs.test.bit_precision[0] == "fp32":
        weight_dtype = torch.float32
    elif configs.test.bit_precision[0] == "fp16":
        weight_dtype = torch.float16    
    else:
        raise ValueError(f"Unsupported precision: {configs.test.bit_precision[0]}")

    pipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir, torch_dtype=weight_dtype)
    pipeline.to(accelerator.device, weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    set_seed(args.seed)
    # Save model architecture summary and initialize Weights & Biases
    if accelerator.is_main_process:
        try:
            save_model_architecture(pipeline.transformer, eval_dir)
        except Exception:
            pass
    if accelerator.is_main_process and (not args.no_wandb):
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="PartCrafter", name=args.wandb_tag, config=exp_params, dir=eval_dir, resume=True)
        arti_exp_info = wandb.Artifact(args.wandb_tag, type="eval_info")
        params_path = os.path.join(eval_dir, "params.yaml")
        model_path = os.path.join(eval_dir, "model.txt")
        log_path = os.path.join(eval_dir, "log.txt")
        if os.path.exists(params_path):
            arti_exp_info.add_file(params_path)
        if os.path.exists(model_path):
            arti_exp_info.add_file(model_path)
        if os.path.exists(log_path):
            arti_exp_info.add_file(log_path)
        wandb.log_artifact(arti_exp_info)
    
    logger.info("Loading test dataset...")
    test_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs["test"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_eval
    )
    
    # Prepare pipeline and rmbg_net with accelerator
    test_loader, rmbg_net, pipeline = accelerator.prepare(
        test_loader, rmbg_net, pipeline
    )

    if args.max_test_steps is None:
        args.max_test_steps = len(test_loader)

    logger.info(f"Loaded {len(test_dataset)} test samples. Evaluating for {args.max_test_steps} steps.")

    run_evaluation(
        dataloader=test_loader,
        pipeline=pipeline,
        accelerator=accelerator,
        logger=logger,
        args=args,
        configs=configs,
        eval_dir=eval_dir,
        rmbg_net=rmbg_net
    )

    # Finish W&B run
    if accelerator.is_main_process and (not args.no_wandb):
        wandb.finish()

if __name__ == "__main__":
    main()