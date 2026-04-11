#!/usr/bin/env python3
"""
Calibration and reparameterization script for PartCrafter DiT model.

This script:
1. Loads the model.
2. Collects layer statistics using a calibration dataset.
3. Computes salience balancing scales using SSC.
4. Reparameterizes the model by folding the scales into weights and adjusting preceding biases/shifts.
5. Saves the reparameterized model.

Usage: python scripts/calibrate_and_reparameterize.py --model_path <path> --data_path <path> --output_path <path>
"""

import argparse
import torch
import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.transformers import PartCrafterDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from quantization.layers import compute_ptq4dit_scales
from src.datasets import ObjaversePartEvalDataset
from torch.utils.data import DataLoader
from src.datasets import collate_fn_eval


def collect_layer_statistics(model, dataloader, num_steps):
    """
    Collect activation statistics for all linear layers in the model.
    """
    import torch.nn as nn
    from quantization.layers import PTQ4DiTLinear
    import numpy as np
    from src.utils.image_utils import prepare_image
    
    hooks = []
    stats = {}
    
    def hook_fn(module, input, output):
        name = f"Linear_{id(module)}"
        if name not in stats:
            stats[name] = {'weight': module.weight.abs().max(dim=0)[0], 'act': {}}
        # input[0] is the activation tensor, shape (batch, ..., in_features)
        act_salience = input[0].abs().flatten(0, -2).max(dim=0)[0]  # per in_feature
        stats[name]['act'][len(stats[name]['act'])] = act_salience
    
    # Register hooks on all linear layers in the transformer
    for module in model.transformer.modules():
        if isinstance(module, (nn.Linear, PTQ4DiTLinear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    model.transformer.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break
            # Prepare input for the pipeline
            image_path = batch["images"][0]
            num_parts = batch["num_parts"][0]
            from PIL import Image
            image_pil = Image.open(image_path).convert('RGB')
            try:
                _ = model(
                    [image_pil] * num_parts,
                    attention_kwargs={"num_parts": num_parts},
                    num_tokens=1024,
                    num_inference_steps=1,  # Minimal steps for calibration
                    guidance_scale=7.0,
                    output_type="latent",
                )
            except Exception as e:
                print(f"Error running model on batch {i}: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return stats


def reparameterize_with_scales(model, scales):
    """
    Fold the computed scales into the model weights and adjust preceding biases/shifts.
    
    Args:
        model: PartCrafterPipeline instance.
        scales: Dict from compute_ptq4dit_scales.
    """
    for block in model.transformer.blocks:
        # Handle MLP layers
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            
            # linear1: preceded by norm2
            if hasattr(mlp, 'linear1'):
                layer = mlp.linear1
                layer_name = f"Linear_{id(layer)}"
                if layer_name in scales:
                    scale = scales[layer_name]
                    layer.weight.data *= scale.unsqueeze(0)  # (out_features, in_features)
                    if hasattr(block, 'norm2'):
                        block.norm2.shift.data /= scale
            
            # linear2: preceded by linear1
            if hasattr(mlp, 'linear2'):
                layer = mlp.linear2
                layer_name = f"Linear_{id(layer)}"
                if layer_name in scales:
                    scale = scales[layer_name]
                    layer.weight.data *= scale.unsqueeze(0)
                    if hasattr(mlp, 'linear1') and mlp.linear1.bias is not None:
                        mlp.linear1.bias.data /= scale
        
        # Handle attention layers: preceded by norm1
        if hasattr(block, 'attn') and hasattr(block, 'norm1'):
            attn = block.attn
            for attr in ['to_q', 'to_k', 'to_v', 'to_out', 'q_proj', 'k_proj', 'v_proj', 'out_proj']:
                if hasattr(attn, attr):
                    layer = getattr(attn, attr)
                    layer_name = f"Linear_{id(layer)}"
                    if layer_name in scales:
                        scale = scales[layer_name]
                        layer.weight.data *= scale.unsqueeze(0)
                        block.norm1.shift.data /= scale


def main():
    parser = argparse.ArgumentParser(description="Calibrate and reparameterize PartCrafter DiT model.")
    parser.add_argument("--config", type=str, default="configs/mp16_nt1024_test.yaml", help="Config file for dataset.")
    parser.add_argument("--max_test_steps", type=int, default=64, help="Number of calibration steps.")
    parser.add_argument("--quant_method", type=str, choices=['none', 'ptq4dit', 'qdit', 'team-p'], default='ptq4dit', help="Quantization method.")
    parser.add_argument("--weight_bit", type=int, default=8, help="Weight bit width.")
    parser.add_argument("--act_bit", type=int, default=8, help="Activation bit width.")
    parser.add_argument("--output_dir", type=str, default="calibration_output", help="Output directory.")
    parser.add_argument("--tag", type=str, default="ptq4dit_calibration", help="Tag for the run.")
    
    args = parser.parse_args()
    
    # Load config
    from src.utils.train_utils import get_configs
    configs = get_configs(args.config, [])
    
    # Set model path
    model_path = "pretrained_weights/PartCrafter"
    output_path = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_path, exist_ok=True)
    
    # Load dataset
    print("Loading calibration dataset...")
    dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn_eval)
    
    # Load model
    print(f"Loading model from {model_path}")
    from src.models.autoencoders import TripoSGVAEModel
    weight_dtype = torch.float32  # Use float32 for calibration to avoid dtype issues
    transformer = PartCrafterDiTModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        quant_method=args.quant_method,
        weight_bit=args.weight_bit,
        act_bit=args.act_bit,
        torch_dtype=weight_dtype,
    )
    vae = TripoSGVAEModel.from_pretrained(
        os.path.join(model_path, "vae"),
        quant_method=args.quant_method,
        weight_bit=args.weight_bit,
        act_bit=args.act_bit,
        torch_dtype=weight_dtype,
    )
    model = PartCrafterPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        torch_dtype=weight_dtype,
    )
    
    # Ensure the transformer has blocks (for quantized models)
    if not hasattr(model.transformer, 'blocks'):
        raise AttributeError("Transformer does not have 'blocks' attribute")
    
    # Collect statistics
    print("Collecting layer statistics...")
    stats = collect_layer_statistics(model, dataloader, num_steps=args.max_test_steps)
    
    # Compute scales
    print("Computing salience balancing scales...")
    scales = compute_ptq4dit_scales(stats)
    
    # Reparameterize
    print("Reparameterizing model...")
    reparameterize_with_scales(model, scales)
    
    # Save the reparameterized model
    print(f"Saving reparameterized model to {output_path}")
    model.save_pretrained(output_path)
    
    # Save scales for reference
    scales_path = os.path.join(args.output_dir, "calibrated_scales.pt")
    torch.save(scales, scales_path)
    
    print("Calibration and reparameterization complete!")


if __name__ == "__main__":
    main()