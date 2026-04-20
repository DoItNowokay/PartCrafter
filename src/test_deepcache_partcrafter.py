"""
Test script for evaluating DeepCache acceleration on PartCrafter.
This script compares baseline PartCrafter generation with DeepCache-accelerated generation.
"""

import warnings
warnings.filterwarnings("ignore")
import diffusers.utils.logging as diffusion_logging
from diffusers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
diffusion_logging.set_verbosity_error()

import sys
import os
import matplotlib.pyplot as plt
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from torch.utils.data import DataLoader

# Local imports
from src.utils.typing_utils import *
from src.models.transformers import PartCrafterDiTModel
from src.models.autoencoders import TripoSGVAEModel
from accelerate.utils import set_seed
from src.utils.render_utils import (
    render_views_around_mesh,
    render_normal_views_around_mesh,
    export_renderings,
    make_grid_for_images_or_videos,
    save_mesh_and_renderings
)
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
import json
import yaml
import wandb
import torch
import torch.nn as nn
import accelerate
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Dinov2Model, BitImageProcessor

from torchvision import transforms

from transformers import BitImageProcessor, Dinov2Model, BitsAndBytesConfig
from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartCrafterDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.train_utils import get_configs, save_experiment_params, save_model_architecture
from src.utils.metric_utils import compute_cd_and_f_score_in_training
from src.utils.resource_utils import record_resource_snapshot, compute_peak_vram_gb
from src.models.briarmbg import BriaRMBG
from src.utils.image_utils import prepare_image
from src.utils.resource_utils import get_gpu_stats
from huggingface_hub import snapshot_download

from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from torch.utils.data import DataLoader

class DiTCacheHelper:
    """
    Custom DeepCache helper for DiT (Diffusion Transformer) models like PartCrafter.
    Caches hidden states at specific block indices to skip computation in later blocks.
    """
    
    def __init__(self, cache_interval: int = 2, cache_branch_id: int = 0, num_timesteps: int = 50):
        self.cache_interval = cache_interval
        self.cache_branch_id = cache_branch_id  # Not used in DiT, but kept for compatibility
        self.num_timesteps = num_timesteps
        
        # Define skip range: cache at block 4, skip blocks 5-15
        self.skip_block_range = (4, 16)  # Start at 4, skip to 16 (so blocks 5-15 are skipped)
        
        # Cache dictionary to store outputs (not deltas)
        self.cache_dict = {}
        
        # Current timestep for within-timestep caching
        self.current_timestep = None
    
    def is_cache_step(self, timestep: int) -> bool:
        """Check if we should cache outputs within this timestep."""
        # For within-timestep caching, we always cache at the designated layer
        # The cache_interval controls which layers we skip within each timestep
        return True  # We'll handle skipping logic differently
    
    def should_skip_layer(self, layer_idx: int, timestep: int) -> bool:
        """Check if we should skip this layer within the current timestep."""
        # For cache_interval=1, compute all layers normally
        if self.cache_interval == 1:
            return False
            
        # Only apply caching within the skip block range
        if layer_idx < self.skip_block_range[0]:
            return False  # Always compute layers before the cache point
            
        if layer_idx >= self.skip_block_range[1]:
            return False  # Always compute layers after the skip range
            
        # Within the skip range, apply caching logic
        # Find the nearest cached layer that this layer should reuse
        # Layers are cached at positions: skip_block_range[0], skip_block_range[0]+cache_interval, etc.
        
        # Calculate position relative to cache start
        relative_pos = layer_idx - self.skip_block_range[0]
        
        # If this relative position is a multiple of cache_interval, compute it
        if relative_pos % self.cache_interval == 0:
            return False
            
        # Otherwise, skip it and reuse from the previous cached layer
        return True
    
    def get_retrieval_key(self, layer_idx: int, timestep: int) -> str:
        """Get the cache key to retrieve for a given layer at the current timestep."""
        # Within the skip range, find which cached layer this layer should reuse
        if layer_idx >= self.skip_block_range[0] and layer_idx < self.skip_block_range[1]:
            # Calculate position relative to cache start
            relative_pos = layer_idx - self.skip_block_range[0]
            # Find the cached layer index
            cached_relative_pos = (relative_pos // self.cache_interval) * self.cache_interval
            cached_layer_idx = self.skip_block_range[0] + cached_relative_pos
        else:
            # For layers outside skip range, they should compute themselves
            cached_layer_idx = layer_idx
            
        return self.get_cache_key(cached_layer_idx, timestep)
    
    def get_cache_key(self, layer_idx: int, timestep: int) -> str:
        """Get cache key for a layer and timestep."""
        return f"{layer_idx}_{timestep}"
    
    def clear_cache_for_new_timestep(self, timestep: int):
        """Clear cache when moving to a new timestep."""
        if self.current_timestep != timestep:
            self.cache_dict.clear()
            self.current_timestep = timestep
    
    def clear_cache(self):
        """Clear the cache dictionary."""
        self.cache_dict.clear()
        self.current_timestep = None


def run_baseline_generation(
    pipeline: PartCrafterPipeline,
    image_pil: Image.Image,
    num_parts: int,
    configs: Dict,
    seed: int = 42
) -> Tuple[List[Optional[trimesh.Trimesh]], float, List[Dict], List[float]]:
    """
    Run baseline PartCrafter generation without DeepCache.

    Args:
        pipeline: PartCrafter pipeline.
        image_pil: Input image.
        num_parts: Number of parts to generate.
        configs: Configuration dictionary.
        seed: Random seed.

    Returns:
        Tuple of (meshes, latency_seconds)
    """
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    # Profiling setup
    profile_times = []
    cache_times = []

    start_time = time.time()
    with torch.no_grad():
        output = pipeline(
            [image_pil] * num_parts,
            attention_kwargs={
                "num_parts": num_parts,
                "profile_times": profile_times,
                "cache_times": cache_times
            },
            num_tokens=configs['model']['vae']['num_tokens'],
            generator=generator,
            num_inference_steps=50,
            guidance_scale=3.0,
            max_num_expanded_coords=10000,
            use_flash_decoder=True,
            save_intermediates=False,
            collect_dynamics_stats=False,
            configs=configs,
        )
    end_time = time.time()
    latency = end_time - start_time

    pred_part_meshes = output.meshes

    return pred_part_meshes, latency, profile_times, cache_times


def run_deepcache_generation(
    pipeline: PartCrafterPipeline,
    image_pil: Image.Image,
    num_parts: int,
    configs: Dict,
    cache_interval: int = 2,
    cache_branch_id: int = 0,
    seed: int = 42
) -> Tuple[List[Optional[trimesh.Trimesh]], float, List[Dict], List[float]]:
    """
    Run PartCrafter generation with DeepCache acceleration.

    Args:
        pipeline: PartCrafter pipeline.
        image_pil: Input image.
        num_parts: Number of parts to generate.
        configs: Configuration dictionary.
        cache_interval: DeepCache cache interval.
        cache_branch_id: DeepCache cache branch ID.
        seed: Random seed.

    Returns:
        Tuple of (meshes, latency_seconds)
    """
    # Create custom DiT cache helper
    deepcache_helper = DiTCacheHelper(
        cache_interval=cache_interval, 
        cache_branch_id=cache_branch_id,
        num_timesteps=50  # Match the reduced steps
    )
    deepcache_helper.clear_cache()  # Reset for new generation

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    # Profiling setup
    profile_times = []
    cache_times = []

    start_time = time.time()
    with torch.no_grad():
        output = pipeline(
            [image_pil] * num_parts,
            attention_kwargs={
                "num_parts": num_parts,
                "deepcache_helper": deepcache_helper,
                "profile_times": profile_times,
                "cache_times": cache_times
            },
            num_tokens=configs['model']['vae']['num_tokens'],
            generator=generator,
            num_inference_steps=50,
            guidance_scale=3.0,
            max_num_expanded_coords=10000,
            use_flash_decoder=True,
            save_intermediates=False,
            collect_dynamics_stats=False,
            configs=configs,
        )
    end_time = time.time()
    latency = end_time - start_time

    pred_part_meshes = output.meshes

    return pred_part_meshes, latency, profile_times, cache_times


def save_meshes(
    output_dir: str,
    baseline_meshes: List[Optional[trimesh.Trimesh]],
    deepcache_meshes: List[Optional[trimesh.Trimesh]],
    input_image_pil: Image.Image,
    configs: Dict,
    save_ratio: float,
    logger: logging.Logger
):
    """
    Save baseline and DeepCache meshes with renderings.

    Args:
        output_dir: Base output directory.
        baseline_meshes: List of baseline meshes.
        deepcache_meshes: List of DeepCache meshes.
        input_image_pil: Input image.
        configs: Configuration dictionary.
        save_ratio: Ratio to save meshes (if > 0, save).
        logger: Logger instance.
    """
    if save_ratio <= 0:
        logger.info("Skipping mesh saving as save_ratio <= 0")
        return
    # Save baseline meshes
    baseline_dir = os.path.join(output_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    
    input_image_pil.save(os.path.join(baseline_dir, "input_image.png"))
    for i, mesh in enumerate(baseline_meshes):
        if mesh:
            mesh.export(os.path.join(baseline_dir, f"part_{i:02d}.glb"))
    
    valid_baseline_meshes = [m for m in baseline_meshes if m is not None]
    if valid_baseline_meshes:
        merged_mesh = get_colored_mesh_composition(valid_baseline_meshes)
        save_mesh_and_renderings(
            merged_mesh,
            baseline_dir,
            mesh_filename="object.glb",
            rendering_prefix="rendering",
            render_cfg=configs['test']['rendering'],
            input_image_pil=input_image_pil,
        )
        logger.info(f"Baseline meshes saved to {baseline_dir}")
    
    # Save DeepCache meshes
    deepcache_dir = os.path.join(output_dir, "deepcache")
    os.makedirs(deepcache_dir, exist_ok=True)
    
    input_image_pil.save(os.path.join(deepcache_dir, "input_image.png"))
    for i, mesh in enumerate(deepcache_meshes):
        if mesh:
            mesh.export(os.path.join(deepcache_dir, f"part_{i:02d}.glb"))
    
    valid_deepcache_meshes = [m for m in deepcache_meshes if m is not None]
    if valid_deepcache_meshes:
        merged_mesh = get_colored_mesh_composition(valid_deepcache_meshes)
        save_mesh_and_renderings(
            merged_mesh,
            deepcache_dir,
            mesh_filename="object.glb",
            rendering_prefix="rendering",
            render_cfg=configs['test']['rendering'],
            input_image_pil=input_image_pil,
        )
        logger.info(f"DeepCache meshes saved to {deepcache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test DeepCache acceleration on PartCrafter")
    parser.add_argument("--config", type=str, default="configs/mp16_nt1024_test.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save results.")
    parser.add_argument("--tag", type=str, default="deepcache_baseline", help="A specific tag for this run.")
    parser.add_argument("--cache_interval", type=int, default=2, help="DeepCache cache interval")
    parser.add_argument("--cache_branch_id", type=int, default=0, help="DeepCache cache branch ID")
    parser.add_argument("--num_parts", type=int, default=4, help="Number of parts to generate")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to test from dataset")
    parser.add_argument("--save_ratio", type=float, default=0.1, help="Ratio of samples to save meshes for (0.0 to 1.0)")
    parser.add_argument("--image_path", type=str, default="assets/chair.png", help="Path to input image")

    args = parser.parse_args()

    # Load configs
    configs = get_configs(args.config, [])

    # Create output directory with timestamp
    eval_dir = os.path.join(args.output_dir, f"{args.tag}/{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(eval_dir, exist_ok=True)

    # Save experiment parameters
    import yaml
    params_path = os.path.join(eval_dir, "params.yaml")
    with open(params_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # Set up logging with file handler
    fh = logging.FileHandler(os.path.join(eval_dir, "log.txt"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    # Set seed
    set_seed(args.seed)

    # Load pipeline
    logger.info("Loading PartCrafter pipeline...")
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)

    weight_dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load all components separately and move to device immediately
    transformer = PartCrafterDiTModel.from_pretrained(
        partcrafter_weights_dir,
        subfolder="transformer",
        low_cpu_mem_usage=True,
    )
    if hasattr(transformer, 'to_empty'):
        transformer.to_empty(device=device)
        transformer.to(dtype=weight_dtype)
    else:
        transformer.to(device, weight_dtype)
    
    vae = TripoSGVAEModel.from_pretrained(
        partcrafter_weights_dir,
        subfolder="vae",
        low_cpu_mem_usage=True,
    )
    if hasattr(vae, 'to_empty'):
        vae.to_empty(device=device)
        vae.to(dtype=weight_dtype)
    else:
        vae.to(device, weight_dtype)
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        partcrafter_weights_dir,
        subfolder="scheduler",
    )
    
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        partcrafter_weights_dir,
        subfolder="image_encoder_dinov2",
        low_cpu_mem_usage=True,
    )
    if hasattr(image_encoder_dinov2, 'to_empty'):
        image_encoder_dinov2.to_empty(device=device)
        image_encoder_dinov2.to(dtype=weight_dtype)
    else:
        image_encoder_dinov2.to(device, weight_dtype)
    
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        partcrafter_weights_dir,
        subfolder="feature_extractor_dinov2",
    )
    
    # Create pipeline manually with all components already on device
    pipeline = PartCrafterPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        image_encoder_dinov2=image_encoder_dinov2,
        feature_extractor_dinov2=feature_extractor_dinov2,
    )
    
    pipeline.set_progress_bar_config(disable=True)

    # Load test dataset and get samples
    logger.info("Loading test dataset...")
    test_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    num_samples = min(args.max_samples, len(test_dataset))
    logger.info(f"Testing on {num_samples} samples from dataset")
    
    # Collect results across samples
    all_baseline_latencies = []
    all_deepcache_latencies = []
    all_chamfer_distances = []
    all_speedups = []
    
    for sample_idx in range(num_samples):
        sample = test_dataset[sample_idx]
        image_pil = Image.open(sample['image']).convert('RGB')
        logger.info(f"Processing sample {sample_idx + 1}/{num_samples}: {sample['image']}")
        
        # Run baseline
        logger.info("Running baseline generation...")
        baseline_meshes, baseline_latency, baseline_profile_times, baseline_cache_times = run_baseline_generation(
            pipeline, image_pil, args.num_parts, configs, seed=args.seed
        )
        
        # Run DeepCache
        logger.info("Running DeepCache generation...")
        deepcache_meshes, deepcache_latency, deepcache_profile_times, deepcache_cache_times = run_deepcache_generation(
            pipeline, image_pil, args.num_parts, configs,
            cache_interval=args.cache_interval, cache_branch_id=args.cache_branch_id,
            seed=args.seed
        )
        
        # Compute metrics for this sample
        chamfer_distances = []
        for i in range(args.num_parts):
            baseline_mesh = baseline_meshes[i] if i < len(baseline_meshes) else None
            deepcache_mesh = deepcache_meshes[i] if i < len(deepcache_meshes) else None
            if baseline_mesh and deepcache_mesh:
                cd = compute_mesh_similarity(baseline_mesh, deepcache_mesh)
                chamfer_distances.append(cd)
            else:
                chamfer_distances.append(float('inf'))
        
        avg_chamfer_distance = np.mean([cd for cd in chamfer_distances if cd != float('inf')])
        speedup = baseline_latency / deepcache_latency if deepcache_latency > 0 else 0
        
        # Collect results
        all_baseline_latencies.append(baseline_latency)
        all_deepcache_latencies.append(deepcache_latency)
        all_chamfer_distances.append(avg_chamfer_distance)
        all_speedups.append(speedup)
        
        logger.info(f"Sample {sample_idx + 1}: Baseline {baseline_latency:.4f}s, DeepCache {deepcache_latency:.4f}s, Speedup {speedup:.2f}x, CD {avg_chamfer_distance:.6f}")
        
        # Save meshes only for first sample or based on save_ratio
        if sample_idx == 0 or random.random() < args.save_ratio:
            sample_tag = f"sample_{sample_idx:04d}"
            sample_dir = os.path.join(eval_dir, sample_tag)
            save_meshes(sample_dir, baseline_meshes, deepcache_meshes, image_pil, configs, 1.0, logger)
    
    # Compute aggregate results
    avg_baseline_latency = np.mean(all_baseline_latencies)
    avg_deepcache_latency = np.mean(all_deepcache_latencies)
    avg_chamfer_distance = np.mean([cd for cd in all_chamfer_distances if cd != float('inf')])
    avg_speedup = np.mean(all_speedups)
    
    # Compute profiling metrics (aggregate)
    # For simplicity, use the last sample's profiling
    skipped_layers_time = sum(
        t['attn1_time'] + t['attn2_time'] + t['ff_time']
        for t in baseline_profile_times
        if 5 <= t['layer_idx'] <= 15
    )
    cache_overhead_time = sum(deepcache_cache_times)
    time_saved_ratio = skipped_layers_time / cache_overhead_time if cache_overhead_time > 0 else float('inf')
    
    logger.info(f"Profiling Results:")
    logger.info(f"  Time spent on layers 5-15 (MHSA + MLP) in baseline: {skipped_layers_time:.6f}s")
    logger.info(f"  Time spent on cache overhead (retrieval + storage): {cache_overhead_time:.6f}s")
    logger.info(f"  Ratio (Time Saved / Overhead): {time_saved_ratio:.2f}")

    # Save results
    results = {
        "baseline": {
            "avg_latency_seconds": avg_baseline_latency,
        },
        "deepcache": {
            "avg_latency_seconds": avg_deepcache_latency,
            "cache_interval": args.cache_interval,
            "cache_branch_id": args.cache_branch_id,
        },
        "comparison": {
            "avg_chamfer_distance": avg_chamfer_distance,
        },
        "profiling": {
            "skipped_layers_time": skipped_layers_time,
            "cache_overhead_time": cache_overhead_time,
            "time_saved_ratio": time_saved_ratio,
        },
        "speedup": avg_speedup,
        "num_samples": num_samples,
    }

    # Save to JSON file
    results_path = os.path.join(eval_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Average Baseline latency: {avg_baseline_latency:.4f}s, Average DeepCache latency: {avg_deepcache_latency:.4f}s")
    logger.info(f"Average Speedup: {avg_speedup:.2f}x, Average CD: {avg_chamfer_distance:.6f}")


if __name__ == "__main__":
    main()