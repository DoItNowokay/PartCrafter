"""
Test script for evaluating PartCrafter model with Integrated Sensitivity Analysis.
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
import torch.nn as nn
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from torchvision import transforms

from src.models.autoencoders import TripoSGVAEModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.train_utils import get_configs, save_experiment_params, save_model_architecture
from src.utils.metric_utils import *
from src.models.briarmbg import BriaRMBG
from src.utils.image_utils import prepare_image
from src.utils.resource_utils import get_gpu_stats
from huggingface_hub import snapshot_download

from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from torch.utils.data import DataLoader

# =========================================================================
# INTEGRATED SENSITIVITY ANALYZER (UPDATED)
# =========================================================================
class IntegratedSensitivityAnalyzer:
    def __init__(self, method="gradient_norm"):
        self.results = defaultdict(lambda: defaultdict(list))
        # We store raw timesteps just for reference, but we will plot by index
        self.timesteps_recorded = [] 
        self.method = method
        self.output_dir = None

    def reset_for_new_step(self, output_dir):
        """
        Resets results and points to the current object's folder.
        output_dir example: .../gs_7.0/step_0001/
        """
        self.results = defaultdict(lambda: defaultdict(list))
        self.timesteps_recorded = []
        self.output_dir = output_dir

    def analyze_step(self, model, latents, t, encoder_hidden_states, text_pooled, text_hidden_states, attention_kwargs, do_classifier_free_guidance):
        """
        Runs a shadow forward/backward pass to record gradient norms.
        """
        if self.method != "gradient_norm":
            return

        # 1. Shadow Forward Pass with Gradients Enabled
        with torch.enable_grad():
            model.zero_grad()
            
            # Forward pass to build the graph
            output = model(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                text_pooled=text_pooled,
                text_hidden_states=text_hidden_states,
                attention_kwargs=attention_kwargs,
                return_dict=True
            ).sample

            # 2. Identify the Conditional Output
            if do_classifier_free_guidance:
                output_conditional = output.chunk(2)[1]
            else:
                output_conditional = output

            num_parts = attention_kwargs.get("num_parts", 1)

            # 3. Part-Wise Backward Pass
            for part_idx in range(num_parts):
                model.zero_grad()
                
                # Isolate output for this specific part
                part_output = output_conditional[part_idx].unsqueeze(0)
                
                # Proxy Loss: Norm of the output
                loss = part_output.norm()
                
                # Backward Pass
                loss.backward(retain_graph=True)

                # Record Gradients for ALL Linear layers
                for name, layer in model.named_modules():
                    if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                        grad_norm = layer.weight.grad.norm(2).item()
                        self.results[part_idx][name].append(grad_norm)
            
            del output

        # Record the raw timestep for debugging, but we plot by call count
        self.timesteps_recorded.append(t[0].item())

    def plot_results(self):
        """
        Saves one plot per layer into the step folder.
        Uses Step Index (0...N) for X-axis.
        """
        if not self.output_dir:
            return

        save_dir = os.path.join(self.output_dir, self.method)
        os.makedirs(save_dir, exist_ok=True)
        
        recorded_parts = sorted(list(self.results.keys()))
        if not recorded_parts:
            return

        reference_part = recorded_parts[0]
        available_layers = list(self.results[reference_part].keys())
        colormap = plt.get_cmap('tab10') 

        # Generate X-axis based on how many steps actually ran
        # This will be [0, 1, 2, ... 49] if num_inference_steps=50
        num_steps_ran = len(self.timesteps_recorded)
        steps_axis = list(range(num_steps_ran))


        for name in available_layers:
            plt.figure(figsize=(10, 6))
            has_data = False
            
            for i, part_idx in enumerate(recorded_parts):
                values = self.results[part_idx][name]
                if not values: continue
                has_data = True
                
                plt.plot(steps_axis, values, 
                         label=f"Part {part_idx}", 
                         marker='.', markersize=3, linewidth=1.0, 
                         color=colormap(i % 10), alpha=0.8)

            if not has_data:
                plt.close()
                continue

            plt.title(f"Sensitivity: {name}")
            plt.xlabel(f"Inference Step (0 to {num_steps_ran-1})") # Explicitly shows 0-50 logic
            plt.ylabel("Gradient Norm")
            
            # NOTE: We do NOT invert axis here. Step 0 is start, Step 50 is finish.
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            safe_layer_name = name.replace(".", "_")
            plt.savefig(os.path.join(save_dir, f"{safe_layer_name}.png"))
            plt.close()

# =========================================================================
# STANDARD EVALUATION FUNCTIONS
# =========================================================================

def save_outputs(local_eval_dir, pred_part_meshes, input_image_pil, configs, args, guidance_scale, step, logger):
    os.makedirs(local_eval_dir, exist_ok=True)
    input_image_pil.save(os.path.join(local_eval_dir, "input_image.png"))
    for i, mesh in enumerate(pred_part_meshes):
        if mesh: mesh.export(os.path.join(local_eval_dir, f"part_{i:02d}.glb"))

    valid_meshes = [m for m in pred_part_meshes if m is not None]
    if valid_meshes:
        merged_mesh = get_colored_mesh_composition(valid_meshes)
        save_mesh_and_renderings(
            merged_mesh, local_eval_dir, mesh_filename="object.glb",
            rendering_prefix="rendering", render_cfg=configs['test']['rendering'],
            input_image_pil=input_image_pil,
        )

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
    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None

    metrics_summary = defaultdict(lambda: defaultdict(list))
    
    # --- ANALYZER SETUP ---
    analyzer = None
    if args.analyze_sensitivity:
        logger.info(f"Initializing Gradient Sensitivity Analyzer (Method: {args.gradient_analysis_method})")
        analyzer = IntegratedSensitivityAnalyzer(method=args.gradient_analysis_method)
        pipeline.transformer.requires_grad_(True)
    # ----------------------

    progress_bar = tqdm(enumerate(dataloader), total=args.max_test_steps, desc="Evaluating", disable=not accelerator.is_main_process)
    
    for step, batch in progress_bar:
        if step >= args.max_test_steps: break

        image_path = batch["images"][0]
        
        if args.seed >= 0:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        else:
            generator = None

        _rmbg_net = accelerator.unwrap_model(rmbg_net)
        with torch.no_grad():
            image_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=_rmbg_net)
        
        if image_pil is None: continue

        gt_part_surfaces = batch["part_surfaces"][0].cpu().numpy()
        num_parts = batch["num_parts"][0]
        
        if num_parts > 16: continue
        
        logger.info(f"Step: {step:04d} | Processing sample with num_parts: {num_parts}")
        
        for guidance_scale in sorted(args.test_guidance_scales):
            save_intermediates = accelerator.is_main_process and args.save_intermediates
            save_token_diff = accelerator.is_main_process and args.token_diff
            save_curvature = accelerator.is_main_process and args.curvature
            
            local_eval_dir = os.path.join(eval_dir, f"gs_{guidance_scale:.1f}", f"step_{step:04d}")

            # --- ANALYZER RESET ---
            if analyzer:
                analyzer.reset_for_new_step(local_eval_dir)

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
                    save_token_diff=save_token_diff,
                    save_curvature=save_curvature,
                    save_intermediate_dir=local_eval_dir, 
                    configs=configs,
                    analyzer=analyzer 
                )
            
            # --- SAVE PLOTS ---
            if analyzer:
                analyzer.plot_results()

            pred_part_meshes = output.meshes
            
            batch_cds, batch_f_scores = [], []
            for i in range(num_parts):
                pred_mesh = pred_part_meshes[i]
                gt_surface = gt_part_surfaces[i]
                if pred_mesh is None or len(pred_mesh.vertices) == 0:
                    pred_part_meshes[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                    part_cd = torch.tensor(configs['test']['metric']['default_cd'])
                    part_f = torch.tensor(configs['test']['metric']['default_f1'])
                else:
                    cd, f_score = compute_cd_and_f_score_in_training(
                        gt_surface, pred_mesh,
                        num_samples=configs['test']['metric']['cd_num_samples'],
                        threshold=configs['test']['metric']['f1_score_threshold'],
                        metric=configs['test']['metric']['cd_metric']
                    )
                    part_cd = cd.cpu() if isinstance(cd, torch.Tensor) else torch.tensor(cd)
                    part_f = f_score.cpu() if isinstance(f_score, torch.Tensor) else torch.tensor(f_score)
                
                batch_cds.append(part_cd.item())
                batch_f_scores.append(part_f.item())

            IoU = None
            if (num_parts > 1):
                IoU = compute_IoU_for_scene(pred_part_meshes)

            metrics_summary[guidance_scale]["chamfer"].extend(batch_cds)
            metrics_summary[guidance_scale]["f1_score"].extend(batch_f_scores)
            metrics_summary[guidance_scale]["iou"].append(IoU)

            if accelerator.is_main_process and args.save_ratio > 0 and random.random() < args.save_ratio:
                save_outputs(local_eval_dir, pred_part_meshes, image_pil, configs, args, guidance_scale, step, logger)
            
    if accelerator.is_main_process:
        report_path = os.path.join(eval_dir, "results.txt")
        with open(report_path, "w") as f:
            for gs, metrics in metrics_summary.items():
                f.write(f"GS: {gs} | CD: {np.mean(metrics['chamfer']):.4f} | F1: {np.mean(metrics['f1_score']):.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a PartCrafter model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Dir to save results")
    parser.add_argument("--tag", type=str, default="test_run", help="Tag for run")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_test_steps", type=int, default=None, help="Max batches")
    parser.add_argument("--num_workers", type=int, default=0, help="Workers")
    parser.add_argument("--test_guidance_scales", type=float, nargs="+", default=[7.0], help="CFG scales")
    parser.add_argument("--save_ratio", type=float, default=0.1, help="Save ratio")
    parser.add_argument("--token_diff", action="store_true", help="Save token diffs")
    parser.add_argument("--curvature", action="store_true", help="Save curvature metrics")
    parser.add_argument("--save_intermediates", action="store_true", help="Save intermediates")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--offline_wandb", action="store_true", help="Offline WandB")
    
    # --- ANALYSIS ARGUMENTS ---
    parser.add_argument("--analyze_sensitivity", action="store_true", help="Enable gradient sensitivity analysis")
    parser.add_argument("--gradient_analysis_method", type=str, default="gradient_norm", choices=["gradient_norm"], help="Method for gradient sensitivity analysis")

    args, extras = parser.parse_known_args()
    configs = get_configs(args.config, extras)

    eval_dir = os.path.join(args.output_dir, f"{args.tag}/{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(eval_dir, exist_ok=True)
    
    accelerator = Accelerator()
    logging.basicConfig(level=logging.INFO)
    logger = get_accelerate_logger(__name__, log_level="INFO")

    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(accelerator.device).eval()
    
    weight_dtype = torch.float32 if configs.test.bit_precision[0] == "fp32" else torch.float16
    pipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir, torch_dtype=weight_dtype)
    pipeline.to(accelerator.device, weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    set_seed(args.seed)

    test_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs["test"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        collate_fn=collate_fn_eval
    )
    test_loader, rmbg_net, pipeline = accelerator.prepare(test_loader, rmbg_net, pipeline)

    if args.max_test_steps is None: args.max_test_steps = len(test_loader)

    run_evaluation(
        dataloader=test_loader, pipeline=pipeline, accelerator=accelerator,
        logger=logger, args=args, configs=configs, eval_dir=eval_dir, rmbg_net=rmbg_net
    )

if __name__ == "__main__":
    main()