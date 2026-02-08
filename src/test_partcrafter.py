"""
Test script for evaluating PartCrafter model on part segmentation and generation.
"""

import warnings
warnings.filterwarnings("ignore")
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()

import sys
import os
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
import logging
from collections import defaultdict
import trimesh
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import random
import json

# Third-party
import wandb
import torch
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
from src.utils.metric_utils import compute_cd_and_f_score_cuda
from src.models.briarmbg import BriaRMBG
from src.utils.image_utils import prepare_image
from src.utils.resource_utils import get_gpu_stats
from huggingface_hub import snapshot_download

from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from torch.utils.data import DataLoader


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
    resource_metrics = []  # List to collect structured resource data

    progress_bar = tqdm(
        enumerate(dataloader),
        total=args.max_test_steps,
        desc="Evaluating",
        disable=not accelerator.is_main_process
    )
    
    for step, batch in progress_bar:
        resource_metrics.append({
            "timestamp": time.time(),
            "step": step,
            "guidance_scale": None,
            "phase": "step_start",
            **get_gpu_stats(),
        })
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

        gt_part_surfaces = batch["part_surfaces"][0]
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
            resource_metrics.append({
                "timestamp": time.time(),
                "step": step,
                "guidance_scale": guidance_scale,
                "phase": f"pre_generation_gs_{guidance_scale}",
                **get_gpu_stats(),
            })
            start_time = time.time()
            save_intermediates = accelerator.is_main_process and args.save_intermediates and save_iteration
            save_tokens_diff = accelerator.is_main_process and args.tokens_diff and save_iteration
            save_intermediate_dir = os.path.join(eval_dir, f"gs_{guidance_scale:.1f}", f"step_{step:04d}")

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
                save_tokens_diff=save_tokens_diff,
                save_intermediate_dir=save_intermediate_dir,
                configs=configs,
            )
            pred_part_meshes = output.meshes
            end_time = time.time()
            generation_time = end_time - start_time
            resource_metrics.append({
                "timestamp": time.time(),
                "step": step,
                "guidance_scale": guidance_scale,
                "phase": f"post_generation_gs_{guidance_scale}",
                "generation_time_seconds": generation_time,
                **get_gpu_stats(),
            })
            logger.info(
                f"Step: {step:04d} | GS: {guidance_scale:<4.1f} | "
                f"Generation Time: {generation_time:.2f} seconds"
            )

            batch_cds, batch_f_scores = [], []
            for i in range(num_parts):
                pred_mesh = pred_part_meshes[i]
                gt_surface = gt_part_surfaces[i].to(accelerator.device, dtype=torch.float32) 

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
                    cd, f_score = compute_cd_and_f_score_cuda(
                        gt_surface, pred_mesh,
                        num_samples=configs['test']['metric']['cd_num_samples'],
                        threshold=configs['test']['metric']['f1_score_threshold'],
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
            
            metrics_summary[guidance_scale]["chamfer"].extend(batch_cds)
            metrics_summary[guidance_scale]["f1_score"].extend(batch_f_scores)

            # Log per-item metrics/media to Weights & Biases (main process only)
            if accelerator.is_main_process and (not args.no_wandb):
                item_logs = {
                    f"evaluation/cd_cfg{guidance_scale:.1f}": float(np.mean(batch_cds)),
                    f"evaluation/f1_cfg{guidance_scale:.1f}": float(np.mean(batch_f_scores)),
                    "evaluation/num_parts": int(num_parts),
                }
                wandb.log(item_logs, step=step)

            if accelerator.is_main_process and args.save_ratio > 0 and save_iteration:
                local_eval_dir = os.path.join(eval_dir, f"gs_{guidance_scale:.1f}", f"step_{step:04d}")
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
                    f"Avg F1-Score: {avg_f1:.4f}"
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
                })
            if aggregate_logs:
                wandb.log(aggregate_logs)
            # Attach result file as an artifact
            arti = wandb.Artifact(args.tag + "_eval", type="evaluation")
            arti.add_file(report_path)
            log_path = os.path.join(eval_dir, "log.txt")
            if os.path.exists(log_path):
                arti.add_file(log_path)
            wandb.log_artifact(arti)
        
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
    parser.add_argument("--save_intermediates", action="store_true", help="Whether to save intermediate meshes and renderings during generation.")
    parser.add_argument("--offline_wandb", action="store_true", help="Use offline WandB for experiment tracking")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB for experiment tracking")
    args, extras = parser.parse_known_args()
    configs = get_configs(args.config, extras)

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
    weight_dtype = torch.float16

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
        wandb.init(project="PartCrafter", name=args.tag, config=exp_params, dir=eval_dir, resume=True)
        arti_exp_info = wandb.Artifact(args.tag, type="eval_info")
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