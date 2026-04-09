"""
Generalized Layer-wise sensitivity analyzer.
Supports multiple methods simultaneously:
    - gradient_norm
    - fisher
    - weight_gradient
    - fake_quant
    - noise_amplification
"""

import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class GradientSensitivityAnalyzer:
    def __init__(self, methods=None, quant_bits=8, noise_std=1e-3):
        if methods is None:
            methods = ["gradient_norm"]

        self.methods = methods
        self.quant_bits = quant_bits
        self.noise_std = noise_std

        # Simplified dictionary: method -> layer_name -> list of values
        self.results = defaultdict(lambda: defaultdict(list))
        self.timesteps_recorded = []
        self.output_dir = None

    def reset_for_new_step(self, output_dir):
        self.results = defaultdict(lambda: defaultdict(list))
        self.timesteps_recorded = []
        self.output_dir = output_dir

    def fake_quantize(self, weight):
        qmax = 2 ** (self.quant_bits - 1) - 1
        scale = weight.abs().max() / qmax + 1e-8
        q_weight = torch.round(weight / scale) * scale
        return q_weight

    def compute_gradient_scores(self, layer):
        if layer.weight.grad is None:
            return None

        scores = {}
        grad = layer.weight.grad
        weight = layer.weight

        if "gradient_norm" in self.methods:
            scores["gradient_norm"] = grad.norm(2).item()

        if "fisher" in self.methods:
            scores["fisher"] = grad.pow(2).mean().item()

        if "weight_gradient" in self.methods:
            scores["weight_gradient"] = (weight * grad).abs().mean().item()

        return scores

    def _get_tensor_output(self, raw_output):
        """Helper to extract the tensor whether the model returns a raw tensor or a HuggingFace Output object."""
        if hasattr(raw_output, "sample"):
            return raw_output.sample
        if isinstance(raw_output, tuple):
            return raw_output[0]
        return raw_output

    def analyze_step(self, model, current_timestep_val, *args, **kwargs):
        """
        Generic analyze step. 
        Pass your model inputs directly into *args and **kwargs.
        """
        with torch.enable_grad():
            model.zero_grad()

            # 1. Forward Pass (Clean)
            raw_output = model(*args, **kwargs)
            clean_output = self._get_tensor_output(raw_output)

            # Compute a generic loss (L2 norm of the output) to generate gradients
            loss = clean_output.norm()
            loss.backward()

            # 2. Collect Gradients
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Linear):
                    grad_scores = self.compute_gradient_scores(layer)
                    if grad_scores:
                        for m, val in grad_scores.items():
                            self.results[m][name].append(val)

            # 3. Fake Quant & Noise Amplification (Requires modifying weights temporarily)
            if "fake_quant" in self.methods or "noise_amplification" in self.methods:
                with torch.no_grad():
                    for name, layer in model.named_modules():
                        if not isinstance(layer, nn.Linear):
                            continue

                        original_weight = layer.weight.data.clone()

                        # Fake Quantization
                        if "fake_quant" in self.methods:
                            layer.weight.data = self.fake_quantize(original_weight)
                            q_out = self._get_tensor_output(model(*args, **kwargs))
                            diff = (clean_output - q_out).norm().item()
                            self.results["fake_quant"][name].append(diff)

                        # Noise Amplification
                        if "noise_amplification" in self.methods:
                            noise = torch.randn_like(original_weight) * self.noise_std
                            layer.weight.data = original_weight + noise
                            n_out = self._get_tensor_output(model(*args, **kwargs))
                            amplification = ((n_out - clean_output).norm() / (noise.norm() + 1e-8)).item()
                            self.results["noise_amplification"][name].append(amplification)

                        # Restore original weights
                        layer.weight.data = original_weight

        self.timesteps_recorded.append(current_timestep_val)

    def plot_results(self):
        if not self.output_dir:
            return

        steps_axis = list(range(len(self.timesteps_recorded)))

        for method in self.methods:
            save_dir = os.path.join(self.output_dir, method)
            os.makedirs(save_dir, exist_ok=True)

            available_layers = list(self.results[method].keys())

            for name in available_layers:
                values = self.results[method][name]
                if not values:
                    continue

                plt.figure(figsize=(10, 6))
                plt.plot(steps_axis, values, marker='.', markersize=3, linewidth=1.0, color='blue')
                plt.title(f"{method} Sensitivity: {name}")
                plt.xlabel("Timestep (Index)")
                plt.ylabel("Score")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Generic safe filename
                safe_name = name.replace(".", "_")
                save_path = os.path.join(save_dir, f"{safe_name}.png")
                plt.savefig(save_path)
                plt.close()

    def plot_mixed_precision_allocation(self, method="fisher"):
        if method not in self.results or not self.results[method]:
            print(f"Method {method} not found or no results recorded.")
            return

        layer_scores = {}
        for layer_name, values in self.results[method].items():
            if len(values) > 0:
                layer_scores[layer_name] = np.mean(values)

        if not layer_scores:
            return

        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        layers = [x[0] for x in sorted_layers]
        scores = [x[1] for x in sorted_layers]

        n = len(layers)
        fp16_cutoff = int(0.2 * n)
        int8_cutoff = int(0.6 * n)

        plt.figure(figsize=(12, 6))
        plt.bar(range(n), scores)
        plt.axvline(fp16_cutoff, linestyle='--', color='red', label='FP16 Cutoff')
        plt.axvline(int8_cutoff, linestyle='--', color='orange', label='INT8 Cutoff')
        plt.title("Sensitivity-Guided Mixed Precision Allocation")
        plt.xlabel("Layer Rank (High → Low Sensitivity)")
        plt.ylabel(f"{method} Score")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"mixed_precision_allocation_{method}.png")
        plt.savefig(save_path)
        plt.close()