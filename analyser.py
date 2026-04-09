import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import numpy as np
import sys

# Add the src directory to the python path so imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from src.models.transformers import PartCrafterDiTModel
    from src.models.briarmbg import BriaRMBG
    from src.utils.image_utils import prepare_image
    from src.utils.train_utils import get_configs
    from src.datasets import ObjaversePartEvalDataset, collate_fn_eval
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.logging import get_logger
except ImportError as e:
    print(f"Error: Could not import PartCrafter modules. Make sure you are running this from the project root. Detail: {e}")
    exit(1)

class PartCrafterSensitivityAnalyzer:
    def __init__(self, model):
        self.model = model
        # Structure: results[part_idx][layer_name] = [list of values over time]
        self.results = defaultdict(lambda: defaultdict(list))
        self.timesteps_recorded = []

    def clear_history(self):
        """Resets the recorded results for a new object analysis."""
        self.results = defaultdict(lambda: defaultdict(list))
        self.timesteps_recorded = []

    def analyze_step(self, latents, t, encoder_hidden_states, num_parts):
        """
        Performs gradient analysis for a single timestep, separated by PART.
        """
        # 1. Prepare Model
        self.model.eval()
        
        # 2. Forward Pass (Run ONCE for the whole object/batch)
        # We need gradients enabled for the backward pass later
        with torch.enable_grad():
            self.model.zero_grad()
            
            attention_kwargs = {"num_parts": num_parts}
            
            # EXPANSION FIX: Ensure timestep matches batch size (num_parts)
            # latents shape is (num_parts, num_tokens, channels)
            if t.shape[0] != latents.shape[0]:
                t = t.expand(latents.shape[0])

            # Forward pass produces output of shape (num_parts, num_tokens, channels)
            output = self.model(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=attention_kwargs,
                return_dict=True
            ).sample

            # 3. Part-Wise Backward Pass
            # We iterate through each part to measure its specific contribution to the weights.
            
            for part_idx in range(num_parts):
                # a. Zero gradients from previous part iteration
                self.model.zero_grad()
                
                # b. Select only the output for the current part
                # Shape: (1, num_tokens, channels)
                part_output = output[part_idx].unsqueeze(0)
                
                # c. Calculate Proxy Loss for THIS part
                loss = part_output.norm()
                
                # d. Backward Pass
                # retain_graph=True is crucial because we need to reuse the 'output' graph 
                # for the next part iteration without re-running the forward pass.
                loss.backward(retain_graph=True)

                # e. Record Gradients for THIS part
                for name, layer in self.model.named_modules():
                    if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                        grad_norm = layer.weight.grad.norm(2).item()
                        self.results[part_idx][name].append(grad_norm)
        
        # Cleanup graph to free memory after all parts are done
        del output
        self.timesteps_recorded.append(t[0].item()) # Record just the scalar time value

    def plot_results(self, object_id, method_name="gradient_norm", base_output_dir="result_plots"):
        """
        Plots results for ALL PARTS on the SAME graph for a given layer.
        Structure: result_plots/{object_id}/{method_name}/{layer_name}.png
        """
        # Sanitize object_id
        safe_object_id = "".join([c for c in str(object_id) if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip()
        
        # Create directory for this object and method
        save_dir = os.path.join(base_output_dir, safe_object_id, method_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving combined part plots to: {save_dir}")

        # Get all parts we recorded
        recorded_parts = sorted(list(self.results.keys()))
        
        if not recorded_parts:
            print("No data recorded.")
            return

        # Get list of all layers (assuming same layers for all parts)
        # We take the keys from the first part as a reference
        reference_part = recorded_parts[0]
        available_layers = list(self.results[reference_part].keys())
        
        # Use a colormap to distinguish parts clearly
        colormap = plt.get_cmap('tab10') # Good for up to 10 distinct parts
        
        for name in available_layers:
            plt.figure(figsize=(12, 7))
            
            has_data = False
            
            # Loop through all parts and plot their line on this layer's figure
            for i, part_idx in enumerate(recorded_parts):
                values = self.results[part_idx][name]
                
                if not values:
                    continue
                
                has_data = True
                
                # Determine color cyclically
                color = colormap(i % 10)
                
                # Plot
                plt.plot(self.timesteps_recorded, values, 
                         label=f"Part {part_idx}", 
                         marker='.', 
                         markersize=4, 
                         linewidth=1.5,
                         color=color,
                         alpha=0.8) # Slight transparency to see overlaps

            if not has_data:
                plt.close()
                continue

            plt.title(f"Part-Wise Sensitivity: {name}", fontsize=14)
            plt.xlabel("Timestep (t) [1000 = High Noise -> 0 = Clean]", fontsize=12)
            plt.ylabel(f"{method_name} Magnitude", fontsize=12)
            plt.gca().invert_xaxis() 
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Parts")
            plt.tight_layout()
            
            # Save plot
            safe_layer_name = name.replace(".", "_")
            file_path = os.path.join(save_dir, f"{safe_layer_name}.png")
            plt.savefig(file_path, dpi=150)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze weight sensitivity of PartCrafter DiT")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained PartCrafter model folder")
    parser.add_argument("--rmbg_path", type=str, default="pretrained_weights/RMBG-1.4", help="Path to RMBG model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., configs/mp16_nt1024.yaml)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20, help="Number of timesteps to analyze")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of data samples to use for analysis")
    parser.add_argument("--method", type=str, default="gradient_norm", choices=["gradient_norm"], help="Analysis method")
    args = parser.parse_args()

    # Load Configs
    configs = get_configs(args.config, [])
    
    print(f"Loading models...")
    
    # 1. Load Models
    try:
        pipeline = PartCrafterPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
        pipeline.to(args.device)
        transformer = pipeline.transformer
        
        # Load RMBG for background removal
        rmbg_net = BriaRMBG.from_pretrained(args.rmbg_path).to(args.device)
        rmbg_net.eval()
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Setup Analyzer
    analyzer = PartCrafterSensitivityAnalyzer(transformer)
    
    # 3. Load Data
    print("Loading evaluation dataset...")
    eval_dataset = ObjaversePartEvalDataset(configs=configs, mode='test')
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=2, 
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_eval
    )
    
    # 4. Run Analysis Loop
    timesteps = list(range(1000, 0, -int(1000/args.steps)))
    print(f"Starting analysis on {len(timesteps)} timesteps using {args.num_samples} samples...")
    
    samples_processed = 0
    
    for batch in eval_loader:
        if samples_processed >= args.num_samples:
            break
            
        print(f"Processing sample {samples_processed + 1}/{args.num_samples}")
        
        # Extract ID
        # Try to find a real ID if available in batch
        for key in ['uid', 'uids', 'obj_id', 'name']:
            if key in batch:
                object_id = batch[key][0]
                break
        
        image_path = batch["images"][0]
        num_parts = batch["num_parts"][0]
        
        object_id = os.path.basename(os.path.dirname(image_path))
        
        print(f"  - Object ID: {object_id}")
        print(f"  - Number of Parts: {num_parts}")

        # Preprocess Image
        image_pil = prepare_image(
            image_path,
            bg_color=np.array([1.0, 1.0, 1.0]),
            rmbg_net=rmbg_net
        )
        
        if image_pil is None:
            print("Skipping sample due to image processing failure.")
            continue
            
        # Encode Image
        with torch.no_grad():
            image_embeds, negative_image_embeds, _, _ = pipeline.encode_image(
                image_pil, args.device, num_images_per_prompt=1
            )
            encoder_hidden_states = image_embeds 
            
            # Fix: Expand encoder_hidden_states to (num_parts, ...)
            # This is needed because the model expects batch_size = num_parts
            # and cross-attn expects matching batch dims or broadcasting.
            encoder_hidden_states_expanded = encoder_hidden_states.repeat(num_parts, 1, 1)
            
            num_tokens = configs['model']['vae']['num_tokens']
            num_channels_latents = transformer.config.in_channels
            
            initial_latents = torch.randn(
                num_parts, num_tokens, num_channels_latents, 
                device=args.device, dtype=torch.float32
            )

        # Clear history
        analyzer.clear_history()

        # Time Loop
        for t_val in tqdm(timesteps, desc=f"Sample {object_id}"):
            t = torch.tensor([t_val], device=args.device)
            
            # Noise interpolation
            noise = torch.randn_like(initial_latents)
            alpha = t_val / 1000.0
            current_latents = (1 - alpha) * initial_latents + alpha * noise
            
            # Run Part-Wise Analysis
            analyzer.analyze_step(
                latents=current_latents, 
                t=t, 
                encoder_hidden_states=encoder_hidden_states_expanded,
                num_parts=num_parts
            )
            
        # 5. Save Results
        analyzer.plot_results(object_id=object_id, method_name=args.method)
        samples_processed += 1

if __name__ == "__main__":
    main()