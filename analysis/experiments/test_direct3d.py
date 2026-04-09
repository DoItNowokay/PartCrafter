import os
import sys
import time
import random
import argparse
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(BASE_DIR, "Direct3D"))


# Import Direct3D Pipeline
from direct3d.pipeline import Direct3dPipeline

# Import Dataset and Utils from PartCrafter infrastructure
from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from src.utils.train_utils import get_configs
from torch.utils.data import DataLoader

# Import Analyzers
from analysis.hook_manager import HookManager
from analysis.analysers.gradient_analyzer import GradientSensitivityAnalyzer
from analysis.analysers.token_diff_analyzer import TokenDiffAnalyzer
from analysis.analysers.curvature_analyzer import CurvatureAnalyzer
from analysis.analysers.entropy_analyzer import EntropyAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Run Direct3D analysis with PartCrafter Dataloader")
    parser.add_argument("--pipeline_path", type=str, default="dreamtechai/Direct3D", help="Path to local or HF model")
    
    # Replaced --image_path with --config to load the dataset
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file for dataset")
    parser.add_argument("--max_test_steps", type=int, default=10, help="Max number of batches/images to evaluate.")
    
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--tag", type=str, default="direct3d_test")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    
    parser.add_argument("--save_ratio", type=float, default=1.0, help="Ratio of samples to save plots for")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument(
        "--gradient_analysis_method", 
        nargs="*", 
        # default=["gradient_norm"], 
        choices=["gradient_norm", "fisher", "noise_amplification"],
    )
    
    parser.add_argument("--token_diff", action="store_true")
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    
    args, extras = parser.parse_known_args()
    return args, extras

def main():
    args, extras = parse_args()
    
    # Load configs exactly like test_partcrafter.py
    configs = get_configs(args.config, extras)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.tag = f"Direct3D/{args.tag}"
    eval_dir = os.path.join(args.output_dir, args.tag, timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Saving results to: {eval_dir}")
    print("Loading Direct3D Pipeline...")
    
    pipeline = Direct3dPipeline.from_pretrained(args.pipeline_path)
    pipeline.to(device)
    
    # Force gradients to be tracked for the DiT weights
    pipeline.dit.requires_grad_(True)
    
    class Direct3DAnalyzerWrapper(torch.nn.Module):
        def __init__(self, dit, semantic_cond, pixel_cond):
            super().__init__()
            self.dit = dit
            self.semantic_cond = semantic_cond
            self.pixel_cond = pixel_cond

        def forward(self, hidden_states, timestep, **kwargs):
            return self.dit(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=self.semantic_cond,
                pixel_hidden_states=self.pixel_cond
            )

    if args.gradient_analysis_method and len(args.gradient_analysis_method) > 0:
        gradient_analyzer = GradientSensitivityAnalyzer(methods=args.gradient_analysis_method)
    else:
        gradient_analyzer = None

    token_analyzer = TokenDiffAnalyzer() if args.token_diff else None
    curvature_analyzer = CurvatureAnalyzer() if args.curvature else None
    entropy_analyzer = EntropyAnalyzer() if args.entropy else None
    
    hook_manager = HookManager(pipeline.dit) if args.entropy else None
    if hook_manager:
        hook_manager.register_attention_hooks()

    do_classifier_free_guidance = args.guidance_scale > 0
    
    # ---------------------------------------------------------
    # DATASET LOADING
    # ---------------------------------------------------------
    print("Loading test dataset...")
    test_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # Evaluating one image at a time
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_eval
    )
    
    progress_bar = tqdm(enumerate(test_loader), total=args.max_test_steps, desc="Evaluating Dataset")
    
    for step, batch in progress_bar:
        if step >= args.max_test_steps:
            break
            
        print(f"\n--- Generating Sample {step+1}/{args.max_test_steps} ---")
        
        # Extract the image path exactly as done in PartCrafter
        image_path = batch["images"][0]
        
        save_this_sample = (random.random() < args.save_ratio)
        sample_dir = None
        if save_this_sample:
            sample_dir = os.path.join(eval_dir, f"step_{step:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            print(f"  -> Plots will be saved to: {sample_dir}")
            
        if gradient_analyzer: gradient_analyzer.reset_for_new_step(sample_dir)
        if token_analyzer: token_analyzer.reset_for_new_step(sample_dir)
        if curvature_analyzer: curvature_analyzer.reset_for_new_step(sample_dir)
        if entropy_analyzer: entropy_analyzer.reset_for_new_step(sample_dir)
            
        # Prepare Image from Dataloader Path
        image = pipeline.prepare_image(image_path, rmbg=True)
        semantic_cond, pixel_cond = pipeline.encode_image(image, do_classifier_free_guidance)
        
        generator = torch.Generator(device=device).manual_seed(args.seed + step)
        latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=pipeline.vae.latent_shape[0],
            height=pipeline.vae.latent_shape[1],
            width=pipeline.vae.latent_shape[2],
            dtype=image.dtype,
            device=device,
            generator=generator,
        )

        pipeline.scheduler.set_timesteps(args.steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        analyzer_model = Direct3DAnalyzerWrapper(pipeline.dit, semantic_cond, pixel_cond)

        for i, t in enumerate(tqdm(timesteps, desc="Diffusion Steps", leave=False)):
            
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            t_expand = t.expand(latent_model_input.shape[0])

            latent_model_input.requires_grad_(True)
            latents.requires_grad_(True)

            if gradient_analyzer:
                gradient_analyzer.analyze_step(
                    model=analyzer_model,
                    current_timestep_val=i,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    hidden_states=latent_model_input,
                    timestep=t_expand
                )

            noise_pred = analyzer_model(hidden_states=latent_model_input, timestep=t_expand)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = pipeline.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]
            latents = latents.detach()

            if token_analyzer: token_analyzer.step(latents)
            if curvature_analyzer: curvature_analyzer.step(latents)
            if entropy_analyzer and hook_manager:
                entropy_analyzer.step(hook_manager.attention_maps)
                hook_manager.clear()

        # Save plots at the end of the sample's diffusion loop
        if save_this_sample:
            if gradient_analyzer:
                gradient_analyzer.plot_results()
                gradient_analyzer.plot_mixed_precision_allocation(method=args.gradient_analysis_method[0])
            if token_analyzer: token_analyzer.plot_results()
            if curvature_analyzer: curvature_analyzer.plot_results()
            if entropy_analyzer: entropy_analyzer.plot_results()

    if hook_manager:
        hook_manager.remove_hooks()

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()