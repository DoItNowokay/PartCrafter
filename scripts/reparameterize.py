#!/usr/bin/env python3
"""
Reparameterization script for calibrated PartCrafter DiT model.

This script iterates through the DiT blocks, for each PTQ4DiTLinear layer:
- Multiplies its weights by the salience_scale.
- Adjusts the bias/shift of the preceding layer (adaLN shift or previous Linear bias) to absorb the 1/scale factor.
- Sets salience_scale to 1.0 after reparameterization.

Usage: python scripts/reparameterize.py --model_path <path_to_model>
"""

import argparse
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.transformers import PartCrafterDiTModel
from quantization.layers import PTQ4DiTLinear


def reparameterize_dit(model):
    """
    Reparameterize the DiT model by absorbing salience_scale into weights and preceding biases/shifts,
    and replace PTQ4DiTLinear with standard nn.Linear.
    """
    for block in model.blocks:
        ff = block.ff
        
        # Handle linear1 (first linear in FeedForward)
        if hasattr(ff, 'net') and len(ff.net) >= 3 and isinstance(ff.net[0], PTQ4DiTLinear):
            layer = ff.net[0]
            s = layer.salience_scale
            
            # Fold salience_scale into weights
            new_weight = layer.weight.data * s.unsqueeze(0)
            new_bias = layer.bias.data if layer.bias is not None else None
            
            # Adjust preceding adaLN shift
            if hasattr(block, 'norm3'):
                block.norm3.shift.data *= s
            
            # Create new nn.Linear
            new_layer = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
            new_layer.weight.data = new_weight
            if new_bias is not None:
                new_layer.bias.data = new_bias
            
            # Replace in the sequential
            ff.net[0] = new_layer
        
        # Handle linear2 (second linear in FeedForward)
        if hasattr(ff, 'net') and len(ff.net) >= 3 and isinstance(ff.net[2], PTQ4DiTLinear):
            layer = ff.net[2]
            s = layer.salience_scale
            
            # Fold salience_scale into weights
            new_weight = layer.weight.data * s.unsqueeze(0)
            new_bias = layer.bias.data if layer.bias is not None else None
            
            # Adjust preceding linear1 bias
            if hasattr(ff, 'net') and len(ff.net) >= 3 and hasattr(ff.net[0], 'bias') and ff.net[0].bias is not None:
                ff.net[0].bias.data *= s
            
            # Create new nn.Linear
            new_layer = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
            new_layer.weight.data = new_weight
            if new_bias is not None:
                new_layer.bias.data = new_bias
            
            # Replace in the sequential
            ff.net[2] = new_layer
    
    # Handle attention layers if they contain PTQ4DiTLinear
    for block in model.blocks:
        if hasattr(block, 'attn'):
            attn = block.attn
            
            # Assuming attn has to_q, to_k, to_v, to_out or similar
            for attr in ['to_q', 'to_k', 'to_v', 'to_out', 'q_proj', 'k_proj', 'v_proj', 'out_proj']:
                if hasattr(attn, attr):
                    layer = getattr(attn, attr)
                    if isinstance(layer, PTQ4DiTLinear):
                        s = layer.salience_scale
                        
                        # Fold salience_scale into weights
                        new_weight = layer.weight.data * s.unsqueeze(0)
                        new_bias = layer.bias.data if layer.bias is not None else None
                        
                        # Adjust preceding norm1 shift
                        if hasattr(block, 'norm1'):
                            block.norm1.shift.data *= s
                        
                        # Create new nn.Linear
                        new_layer = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
                        new_layer.weight.data = new_weight
                        if new_bias is not None:
                            new_layer.bias.data = new_bias
                        
                        # Replace
                        setattr(attn, attr, new_layer)


def main():
    parser = argparse.ArgumentParser(description="Reparameterize calibrated PartCrafter DiT model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the calibrated model directory.")
    parser.add_argument("--scales_path", type=str, help="Path to the scales file (.pt or .json).")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the reparameterized model. If not provided, overwrites the input.")
    parser.add_argument("--quant_method", type=str, choices=['ptq4dit'], default='ptq4dit', help="Quantization method.")
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = PartCrafterDiTModel.from_pretrained(args.model_path, quant_method=args.quant_method)
    
    if args.scales_path:
        print(f"Loading scales from {args.scales_path}")
        if args.scales_path.endswith('.pt'):
            scales = torch.load(args.scales_path)
        elif args.scales_path.endswith('.json'):
            import json
            with open(args.scales_path, 'r') as f:
                scales = json.load(f)
            scales = {k: torch.tensor(v) for k, v in scales.items()}
        else:
            raise ValueError("Scales file must be .pt or .json")
        
        # Set salience_scale on layers
        for module in model.modules():
            if isinstance(module, PTQ4DiTLinear):
                name = f"Linear_{id(module)}"
                if name in scales:
                    module.salience_scale.data = scales[name]
                else:
                    print(f"Warning: No scale for {name}")
    
    # Reparameterize
    print("Reparameterizing model...")
    reparameterize_dit(model)
    
    # Save the reparameterized model
    output_path = args.output_path if args.output_path else args.model_path
    print(f"Saving reparameterized model to {output_path}")
    model.save_pretrained(output_path)
    
    print("Reparameterization complete!")


if __name__ == "__main__":
    main()