#!/usr/bin/env python3
"""
Calibration script for Point4Bit quantization using G_KWQ_Logic.

This script:
1. Loads the PartCrafter model.
2. Runs a forward/backward pass on 128 samples from the test dataset.
3. Computes gradients using a loss that includes SDF reconstruction loss (approximated by Chamfer Distance).
4. Uses G_KWQ_Logic to determine 'Key Channels' (top 80% sensitivity).
5. Saves binary masks for each layer indicating Key Channels.

Usage: python scripts/calibrate_point4bit.py --model_path <path> --output_path <path>
"""

import argparse
import torch
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.point4bit import G_KWQ_Logic
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from torch.utils.data import DataLoader
from src.utils.train_utils import get_configs


def main():
    parser = argparse.ArgumentParser(description="Calibrate Point4Bit key channels.")
    parser.add_argument('--model_path', type=str, default='pretrained_weights/PartCrafter', help="Path to pretrained model.")
    parser.add_argument('--output_path', type=str, default='./checkpoints/point4bit_masks.pt', help="Output path for masks.")
    parser.add_argument('--num_samples', type=int, default=128, help="Number of calibration samples.")
    parser.add_argument('--m2_weight_ratio', type=float, default=0.8, help="Ratio of key channels (top sensitivity).")
    parser.add_argument('--config', type=str, default='configs/mp16_nt1024_test.yaml', help="Path to config file.")
    parser.add_argument('--quant_method', type=str, default='point4bit', help="Quantization method.")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}")
    pipeline = PartCrafterPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.to(device)
    
    # Set to eval mode
    pipeline.vae.eval()
    pipeline.transformer.eval()
    pipeline.image_encoder_dinov2.eval()
    if pipeline.text_encoder is not None:
        pipeline.text_encoder.eval()
    if pipeline.condition_processor is not None:
        pipeline.condition_processor.eval()

    # Load dataset
    configs = get_configs(args.config, [])
    test_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn_eval)

    # Collect sensitivities for averaging
    sensitivities = {}
    count = 0

    print(f"Running calibration on {args.num_samples} samples")
    for batch in test_loader:
        if count >= args.num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Load images if they are paths
        if 'images' in batch:
            img_data = batch['images']
            if isinstance(img_data, str):
                from PIL import Image
                batch['images'] = Image.open(img_data).convert("RGB")
            elif isinstance(img_data, list) and len(img_data) > 0 and isinstance(img_data[0], str):
                from PIL import Image
                batch['images'] = [Image.open(img).convert("RGB") for img in img_data]

        # Forward pass with gradient computation
        with torch.enable_grad():
            output = pipeline(
                image=batch['images'],
                captions=batch.get('captions', batch.get('caption')),
                num_inference_steps=10,
                guidance_scale=7.0,
                attention_kwargs={'num_parts': 1},
                return_dict=True
            )
            pred_meshes = output.meshes
            gt_meshes = batch.get('gt_meshes', [])

            # Compute loss including SDF reconstruction loss (using Chamfer Distance)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for pred_mesh, gt_mesh in zip(pred_meshes, gt_meshes):
                if pred_mesh is not None and gt_mesh is not None:
                    cd, _ = compute_cd_and_f_score_cuda(
                        pred_mesh.vertices, gt_mesh.vertices,
                        pred_mesh.faces, gt_mesh.faces
                    )
                    loss = loss + cd

            if loss != 0:
                loss.backward()

        # Collect sensitivities
        for name, module in pipeline.transformer.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                sens = G_KWQ_Logic(module.weight.grad)
                if name not in sensitivities:
                    sensitivities[name] = []
                sensitivities[name].append(sens.detach().cpu())

        count += 1
        if count % 10 == 0:
            print(f"Processed {count} samples")

    # Compute average sensitivities and create masks
    masks = {}
    for name, sens_list in sensitivities.items():
        if sens_list:
            avg_sens = torch.stack(sens_list).mean(dim=0)
            num_key = int(args.m2_weight_ratio * len(avg_sens))
            _, indices = avg_sens.topk(num_key)
            mask = torch.zeros(len(avg_sens), dtype=torch.bool)
            mask[indices] = True
            masks[name] = mask

    # Save masks
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(masks, args.output_path)
    print(f"Saved key channel masks to {args.output_path}")


if __name__ == "__main__":
    main()