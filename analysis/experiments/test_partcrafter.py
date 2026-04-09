import os
import sys
import time
import random
import argparse
import torch
import numpy as np
import trimesh
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# PartCrafter Imports
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
from src.utils.train_utils import get_configs
from torch.utils.data import DataLoader
from src.models.briarmbg import BriaRMBG
from src.utils.image_utils import prepare_image
from src.utils.inference_utils import hierarchical_extract_geometry

# Analyzer Imports (The modular external classes)
from analysis.hook_manager import HookManager
from analysis.analysers.gradient_analyzer import GradientSensitivityAnalyzer
from analysis.analysers.token_diff_analyzer import TokenDiffAnalyzer
from analysis.analysers.curvature_analyzer import CurvatureAnalyzer
from analysis.analysers.entropy_analyzer import EntropyAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PartCrafter with External Analyzers")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save results.")
    parser.add_argument("--tag", type=str, default="partcrafter_external")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_test_steps", type=int, default=10, help="Max number of batches to evaluate.")
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--save_ratio", type=float, default=1.0, help="Ratio of outputs to save plots for.")
    
    parser.add_argument("--gradient_analysis_method", type=str, nargs="*", choices=["gradient_norm", "fisher", "noise_amplification"])
    parser.add_argument("--token_diff", action="store_true")
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    
    args, extras = parser.parse_known_args()
    return args, extras

def main():
    args, extras = parse_args()
    configs = get_configs(args.config, extras)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.tag = f"PartCrafter/{args.tag}"
    eval_dir = os.path.join(args.output_dir, args.tag, timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Saving results to: {eval_dir}")
    
    # 1. Load Background Removal Model
    print("Loading RMBG model...")
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()
    
    # 2. Load PartCrafter Pipeline
    print("Loading PartCrafter Pipeline...")
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    weight_dtype = torch.float16
    pipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir, torch_dtype=weight_dtype)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # CRITICAL: Force PyTorch to track gradients for the transformer weights
    pipeline.transformer.requires_grad_(True)
    
    # 3. Model Wrapper (To make it behave like standard DiT for the Gradient Analyzer)
    class PartCrafterAnalyzerWrapper(torch.nn.Module):
        def __init__(self, transformer, image_embeds, attention_kwargs):
            super().__init__()
            self.transformer = transformer
            self.image_embeds = image_embeds
            self.attention_kwargs = attention_kwargs

        def forward(self, hidden_states, timestep, **kwargs):
            return self.transformer(
                hidden_states,
                timestep,
                encoder_hidden_states=self.image_embeds,
                attention_kwargs=self.attention_kwargs,
                return_dict=False
            )

    # 4. Initialize External Analyzers
    if args.gradient_analysis_method and len(args.gradient_analysis_method) > 0:
        gradient_analyzer = GradientSensitivityAnalyzer(methods=args.gradient_analysis_method)
    else:
        gradient_analyzer = None

    token_analyzer = TokenDiffAnalyzer() if args.token_diff else None
    curvature_analyzer = CurvatureAnalyzer() if args.curvature else None
    entropy_analyzer = EntropyAnalyzer() if args.entropy else None
    
    hook_manager = HookManager(pipeline.transformer) if args.entropy else None
    if hook_manager:
        hook_manager.register_attention_hooks()

    # 5. Load Dataset
    print("Loading test dataset...")
    test_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True, collate_fn=collate_fn_eval)
    
    progress_bar = tqdm(enumerate(test_loader), total=args.max_test_steps, desc="Evaluating Dataset")
    
    for step, batch in progress_bar:
        if step >= args.max_test_steps:
            break
            
        print(f"\n--- Generating Sample {step+1}/{args.max_test_steps} ---")
        
        image_path = batch["images"][0]
        num_parts = int(batch["num_parts"][0])
        
        save_this_sample = (random.random() < args.save_ratio)
        sample_dir = None
        if save_this_sample:
            sample_dir = os.path.join(eval_dir, f"step_{step:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            print(f"  -> Outputs will be saved to: {sample_dir}")
            
        # Reset analyzers for the new sample
        if gradient_analyzer: gradient_analyzer.reset_for_new_step(sample_dir)
        if token_analyzer: token_analyzer.reset_for_new_step(sample_dir)
        if curvature_analyzer: curvature_analyzer.reset_for_new_step(sample_dir)
        if entropy_analyzer: entropy_analyzer.reset_for_new_step(sample_dir)

        # ---------------------------------------------------------
        # PIPELINE SETUP (Replicating logic before the loop)
        # ---------------------------------------------------------
        image_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        if image_pil is None:
            continue
            
        do_classifier_free_guidance = args.guidance_scale > 1.0
        attention_kwargs = {"num_parts": num_parts}
        
        # Encode Image Condition (Matching your specific 4-return pipeline)
        encoded_outputs = pipeline.encode_image(image_pil, device, num_parts)
        image_embeds = encoded_outputs[0]
        negative_image_embeds = encoded_outputs[1]
        
        if do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        pipeline.scheduler.set_timesteps(configs['test']['num_inference_steps'], device=device)
        timesteps = pipeline.scheduler.timesteps

        generator = torch.Generator(device=device).manual_seed(args.seed + step)
        num_channels_latents = pipeline.transformer.config.in_channels
        num_tokens = configs['model']['vae']['num_tokens']
        
        latents = pipeline.prepare_latents(
            batch_size=num_parts,
            num_tokens=num_tokens,
            num_channels_latents=num_channels_latents,
            dtype=image_embeds.dtype,
            device=device,
            generator=generator,
        )

        analyzer_model = PartCrafterAnalyzerWrapper(pipeline.transformer, image_embeds, attention_kwargs)


        for i, t in enumerate(tqdm(timesteps, desc="Diffusion Steps", leave=False)):
            
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            t_expand = t.expand(latent_model_input.shape[0])

            # Explicitly require gradients for analysis
            latent_model_input.requires_grad_(True)
            latents.requires_grad_(True)

            # 1. Gradient Analysis Step
            if gradient_analyzer:
                gradient_analyzer.analyze_step(
                    model=analyzer_model,
                    current_timestep_val=i,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    hidden_states=latent_model_input,
                    timestep=t_expand,
                    attention_kwargs=attention_kwargs
                )

            # 2. Forward Pass for actual generation
            with torch.no_grad():
                noise_pred = analyzer_model(hidden_states=latent_model_input, timestep=t_expand)[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_image - noise_pred_uncond)
                
                latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents = latents.detach()

            # 3. Step external analyzers
            if token_analyzer: token_analyzer.step(latents)
            if curvature_analyzer: curvature_analyzer.step(latents)
            if entropy_analyzer and hook_manager:
                entropy_analyzer.step(hook_manager.attention_maps)
                hook_manager.clear()

        # ---------------------------------------------------------
        # PLOTTING & DECODING
        # ---------------------------------------------------------
        if save_this_sample:
            # Let the analyzers plot themselves!
            if gradient_analyzer:
                gradient_analyzer.plot_results()
                gradient_analyzer.plot_mixed_precision_allocation(method=args.gradient_analysis_method[0])
            if token_analyzer: token_analyzer.plot_results()
            if curvature_analyzer: curvature_analyzer.plot_results()
            if entropy_analyzer: entropy_analyzer.plot_results()

            print("  -> Decoding generated latents into 3D meshes...")
            pipeline.vae.set_flash_decoder()
            
            for n in range(num_parts):
                geometric_func = lambda x: pipeline.vae.decode(latents[n].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        dense_octree_depth=8,
                        hierarchical_octree_depth=9,
                        max_num_expanded_coords=1e8,
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                    mesh.export(os.path.join(sample_dir, f"part_{n:02d}.glb"))
                except Exception as e:
                    print(f"     Failed to extract mesh for part {n}: {e}")

    if hook_manager:
        hook_manager.remove_hooks()

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()