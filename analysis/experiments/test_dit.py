import os
import sys
import time
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(BASE_DIR, "DiT"))

from models import DiT_models
from diffusion import create_diffusion
from download import find_model

from analysis.hook_manager import HookManager
from analysis.analysers.gradient_analyzer import GradientSensitivityAnalyzer
from analysis.analysers.token_diff_analyzer import TokenDiffAnalyzer
from analysis.analysers.curvature_analyzer import CurvatureAnalyzer
from analysis.analysers.entropy_analyzer import EntropyAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Run DiT analysis")
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--tag", type=str, default="dit_test")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion timesteps")
    
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images/batches to generate")
    parser.add_argument("--save_ratio", type=float, default=0.2, help="Ratio of samples to save plots for")
    parser.add_argument("--seed", type=int, default=42)
    
    # Changed nargs="+" to nargs="*" so it can accept an empty list to disable it
    parser.add_argument(
        "--gradient_analysis_method", 
        nargs="*", 
        # default=["gradient_norm"], 
        choices=["gradient_norm", "fisher", "noise_amplification"],
        help="Pass empty string to disable (e.g., --gradient_analysis_method )"
    )
    
    parser.add_argument("--token_diff", action="store_true")
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.tag = f"DiT/{args.tag}"
    eval_dir = os.path.join(args.output_dir, args.tag, timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    
    print("Saving results to:", eval_dir)
    print("Loading DiT model...")
    
    model = DiT_models["DiT-XL/2"](input_size=32).to(device)
    state_dict = find_model("DiT-XL-2-256x256.pt")
    model.load_state_dict(state_dict)
    
    model.requires_grad_(True)
    model.eval()
    diffusion = create_diffusion(str(args.steps))
    wrapped_model = diffusion._wrap_model(model)
    
    # Pass the RAW model, not the wrapped_model
    class DiTAnalyzerWrapper(torch.nn.Module):
        def __init__(self, raw_model, diffusion_obj, y):
            super().__init__()
            self.raw_model = raw_model
            self.y = y
            
            # Save the timestep mapping directly as a tensor
            self.register_buffer(
                "timestep_map", 
                torch.tensor(diffusion_obj.timestep_map, dtype=torch.long)
            )

        def forward(self, sample, timestep, **kwargs):
            # Map the 0-49 spaced timesteps back to the 0-1000 original timesteps
            mapped_timestep = self.timestep_map[timestep]
            
            # Pass everything to the raw DiT model
            return self.raw_model(sample, mapped_timestep, y=self.y)

    # Initialize gradient analyzer ONLY if methods are provided
    if args.gradient_analysis_method and len(args.gradient_analysis_method) > 0:
        gradient_analyzer = GradientSensitivityAnalyzer(methods=args.gradient_analysis_method)
    else:
        gradient_analyzer = None
        
    # print(f"Gradient analysis methods: {args.gradient_analysis_method if gradient_analyzer else 'None'}")

    token_analyzer = TokenDiffAnalyzer() if args.token_diff else None
    curvature_analyzer = CurvatureAnalyzer() if args.curvature else None
    entropy_analyzer = EntropyAnalyzer() if args.entropy else None
    
    hook_manager = HookManager(model) if args.entropy else None
    if hook_manager:
        hook_manager.register_attention_hooks()

    batch_size = 1
    
    for sample_idx in range(args.num_samples):
        print(f"\n--- Generating Sample {sample_idx+1}/{args.num_samples} ---")
        
        save_this_sample = (random.random() < args.save_ratio)
        
        sample_dir = None
        if save_this_sample:
            sample_dir = os.path.join(eval_dir, f"sample_{sample_idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            print(f"  -> Plots will be saved to: {sample_dir}")
            
        # Reset analyzers at the start of the NEW SAMPLE
        if gradient_analyzer: gradient_analyzer.reset_for_new_step(sample_dir)
        if token_analyzer: token_analyzer.reset_for_new_step(sample_dir)
        if curvature_analyzer: curvature_analyzer.reset_for_new_step(sample_dir)
        if entropy_analyzer: entropy_analyzer.reset_for_new_step(sample_dir)
            
        latents = torch.randn(batch_size, 4, 32, 32).to(device)
        class_labels = torch.randint(0, 1000, (batch_size,), dtype=torch.long).to(device)
        # analyzer_model = DiTAnalyzerWrapper(wrapped_model, class_labels)
        analyzer_model = DiTAnalyzerWrapper(model, diffusion, class_labels).to(device)

        for step, i in enumerate(tqdm(reversed(range(args.steps)), desc="Diffusion Steps", leave=False)):
            
            t = torch.full((latents.shape[0],), i, device=device, dtype=torch.long)
            latents.requires_grad_(True)

            if gradient_analyzer:
                gradient_analyzer.analyze_step(
                    analyzer_model,         
                    i,                       
                    sample=latents,          
                    timestep=t               
                )

            out = diffusion.p_sample(
                model,
                latents,
                t,
                model_kwargs={"y": class_labels}
            )

            latents = out["sample"].detach()

            if token_analyzer: token_analyzer.step(latents)
            if curvature_analyzer: curvature_analyzer.step(latents)
            if entropy_analyzer and hook_manager:
                entropy_analyzer.step(hook_manager.attention_maps)
                hook_manager.clear()

        # Plot the results for this sample
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