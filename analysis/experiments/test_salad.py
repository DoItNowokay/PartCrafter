import os
import sys
import time
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(BASE_DIR, "SALAD"))

# Analyzer Imports (Your Modular External Classes)
from analysis.hook_manager import HookManager
from analysis.analysers.gradient_analyzer import GradientSensitivityAnalyzer
from analysis.analysers.token_diff_analyzer import TokenDiffAnalyzer
from analysis.analysers.curvature_analyzer import CurvatureAnalyzer
from analysis.analysers.entropy_analyzer import EntropyAnalyzer

def load_salad_model(category, model_class, device):
    """Loads the model based on the logic found in SALAD's demo notebooks"""
    c = OmegaConf.load(f"checkpoints/{category}/{model_class}/hparams.yaml")
    model = hydra.utils.instantiate(c)
    ckpt = torch.load(f"checkpoints/{category}/{model_class}/state_only.ckpt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    # CRITICAL: Unlike the demo notebook which sets requires_grad_(False), 
    # we MUST set this to True so the Gradient Analyzer works!
    model.net.requires_grad_(True)
    
    model = model.to(device)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SALAD Phase 1 with Analyzers")
    parser.add_argument("--category", type=str, default="airplane", help="Category: airplane, chair, table")
    parser.add_argument("--model_class", type=str, default="phase1", help="Model phase: phase1, phase2")
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--tag", type=str, default="salad_test")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--gradient_analysis_method", type=str, nargs="*", choices=["gradient_norm", "fisher", "noise_amplification"])
    parser.add_argument("--token_diff", action="store_true")
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.tag = f"SALAD_{args.category}_{args.model_class}/{args.tag}"
    eval_dir = os.path.join(args.output_dir, args.tag, timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Saving results to: {eval_dir}")
    print("Loading SALAD Model...")
    
    # 1. Load the Model
    model = load_salad_model(args.category, args.model_class, device)
    
    # 2. Wrapper for Gradient Analyzer
    class SaladAnalyzerWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x_t, beta, **kwargs):
            return self.net(x_t, beta=beta)

    analyzer_model = SaladAnalyzerWrapper(model.net)

    # 3. Initialize Analyzers
    if args.gradient_analysis_method and len(args.gradient_analysis_method) > 0:
        gradient_analyzer = GradientSensitivityAnalyzer(methods=args.gradient_analysis_method)
    else:
        gradient_analyzer = None

    token_analyzer = TokenDiffAnalyzer() if args.token_diff else None
    curvature_analyzer = CurvatureAnalyzer() if args.curvature else None
    entropy_analyzer = EntropyAnalyzer() if args.entropy else None
    
    hook_manager = HookManager(model.net) if args.entropy else None
    if hook_manager:
        hook_manager.register_attention_hooks()
        
    for sample_idx in range(args.num_samples):
        print(f"\n--- Generating Sample {sample_idx+1}/{args.num_samples} ---")
        
        save_this_sample = (random.random() < args.save_ratio)
        sample_dir = None
        if save_this_sample:
            sample_dir = os.path.join(eval_dir, f"sample_{sample_idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
        if gradient_analyzer: gradient_analyzer.reset_for_new_step(sample_dir)
        if token_analyzer: token_analyzer.reset_for_new_step(sample_dir)
        if curvature_analyzer: curvature_analyzer.reset_for_new_step(sample_dir)
        if entropy_analyzer: entropy_analyzer.reset_for_new_step(sample_dir)
        
        # ---------------------------------------------------------
        # MANUAL DIFFUSION LOOP (Extracted from phase1.py)
        # ---------------------------------------------------------
        batch_size = args.batch_size
        x_T = torch.randn([batch_size, 16, 16]).to(device)
        x_t = x_T.clone()
        
        num_steps = model.var_sched.num_steps
        
        # SALAD steps downwards from num_steps to 1
        for t in tqdm(range(num_steps, 0, -1), desc="Diffusion Steps", leave=False):
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            alpha = model.var_sched.alphas[t]
            alpha_bar = model.var_sched.alpha_bars[t]
            sigma = model.var_sched.get_sigmas(t, flexibility=0)
            
            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            beta = model.var_sched.betas[[t] * batch_size].to(device)
            
            x_t.requires_grad_(True)
            
            # Step Gradient Analyzer
            if gradient_analyzer:
                gradient_analyzer.analyze_step(
                    model=analyzer_model,
                    current_timestep_val=num_steps - t,  # Track index 0 to T for clean plots
                    do_classifier_free_guidance=False,
                    x_t=x_t,
                    beta=beta
                )
                
            # Forward pass and Scheduler step
            with torch.no_grad():
                e_theta = analyzer_model(x_t=x_t, beta=beta)
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                x_t = x_next.detach()
                
            # Step External Analyzers
            if token_analyzer: token_analyzer.step(x_t)
            if curvature_analyzer: curvature_analyzer.step(x_t)
            if entropy_analyzer and hook_manager:
                entropy_analyzer.step(hook_manager.attention_maps)
                hook_manager.clear()
                
        # ---------------------------------------------------------
        # PLOTTING
        # ---------------------------------------------------------
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