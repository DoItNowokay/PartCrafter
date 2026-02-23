"""
Layer-wise sensitivity analyzer for quantization.
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


class GradientSensitivityAnalyzer:
    def __init__(self, methods=None, quant_bits=8, noise_std=1e-3):
        """
        methods: list of str
        quant_bits: int (for fake quant)
        noise_std: float (for noise amplification)
        """

        if methods is None:
            methods = ["gradient_norm"]

        self.methods = methods
        self.quant_bits = quant_bits
        self.noise_std = noise_std

        self.results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        self.timesteps_recorded = []
        self.output_dir = None

    # ---------------------------------------------------
    # Reset
    # ---------------------------------------------------
    def reset_for_new_step(self, output_dir):
        self.results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.timesteps_recorded = []
        self.output_dir = output_dir

    # ---------------------------------------------------
    # Fake Quantization
    # ---------------------------------------------------
    def fake_quantize(self, weight):
        qmax = 2 ** (self.quant_bits - 1) - 1
        scale = weight.abs().max() / qmax + 1e-8
        q_weight = torch.round(weight / scale) * scale
        return q_weight

    # ---------------------------------------------------
    # Gradient-based scores
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # Main Analysis Step
    # ---------------------------------------------------
    def analyze_step(
        self,
        model,
        latents,
        t,
        encoder_hidden_states,
        text_pooled,
        text_hidden_states,
        attention_kwargs,
        do_classifier_free_guidance,
    ):

        # -------------------------
        # 1️⃣ Forward pass (clean)
        # -------------------------
        with torch.enable_grad():
            model.zero_grad()

            clean_output = model(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                text_pooled=text_pooled,
                text_hidden_states=text_hidden_states,
                attention_kwargs=attention_kwargs,
                return_dict=True
            ).sample

            if do_classifier_free_guidance:
                clean_output = clean_output.chunk(2)[1]

            num_parts = attention_kwargs.get("num_parts", 1)

            for part_idx in range(num_parts):
                model.zero_grad()

                part_output = clean_output[part_idx].unsqueeze(0)
                loss = part_output.norm()
                loss.backward(retain_graph=True)

                for name, layer in model.named_modules():
                    if isinstance(layer, nn.Linear):

                        grad_scores = self.compute_gradient_scores(layer)
                        if grad_scores:
                            for m, val in grad_scores.items():
                                self.results[m][part_idx][name].append(val)

            if "fake_quant" in self.methods or "noise_amplification" in self.methods:

                with torch.no_grad():

                    for name, layer in model.named_modules():
                        if not isinstance(layer, nn.Linear):
                            continue

                        original_weight = layer.weight.data.clone()

                        # Fake Quant
                        if "fake_quant" in self.methods:
                            layer.weight.data = self.fake_quantize(original_weight)

                            quant_output = model(
                                hidden_states=latents,
                                timestep=t,
                                encoder_hidden_states=encoder_hidden_states,
                                text_pooled=text_pooled,
                                text_hidden_states=text_hidden_states,
                                attention_kwargs=attention_kwargs,
                                return_dict=True
                            ).sample

                            if do_classifier_free_guidance:
                                quant_output = quant_output.chunk(2)[1]

                            diff = (clean_output - quant_output).norm().item()

                            for part_idx in range(num_parts):
                                self.results["fake_quant"][part_idx][name].append(diff)

                        # Noise Amplification
                        if "noise_amplification" in self.methods:
                            noise = torch.randn_like(original_weight) * self.noise_std
                            layer.weight.data = original_weight + noise

                            noisy_output = model(
                                hidden_states=latents,
                                timestep=t,
                                encoder_hidden_states=encoder_hidden_states,
                                text_pooled=text_pooled,
                                text_hidden_states=text_hidden_states,
                                attention_kwargs=attention_kwargs,
                                return_dict=True
                            ).sample

                            if do_classifier_free_guidance:
                                noisy_output = noisy_output.chunk(2)[1]

                            amplification = (
                                (noisy_output - clean_output).norm() /
                                (noise.norm() + 1e-8)
                            ).item()

                            for part_idx in range(num_parts):
                                self.results["noise_amplification"][part_idx][name].append(amplification)

                        # Restore
                        layer.weight.data = original_weight

            del clean_output

        self.timesteps_recorded.append(t[0].item())


    def plot_results(self):

        if not self.output_dir:
            return

        num_steps_ran = len(self.timesteps_recorded)
        steps_axis = list(range(num_steps_ran))
        colormap = plt.get_cmap('tab10')

        for method in self.methods:

            save_dir = os.path.join(self.output_dir, method)
            os.makedirs(save_dir, exist_ok=True)

            recorded_parts = sorted(list(self.results[method].keys()))
            if not recorded_parts:
                continue

            reference_part = recorded_parts[0]
            available_layers = list(self.results[method][reference_part].keys())

            for name in available_layers:
                plt.figure(figsize=(10, 6))
                has_data = False

                for i, part_idx in enumerate(recorded_parts):
                    values = self.results[method][part_idx][name]
                    if not values:
                        continue

                    has_data = True

                    plt.plot(
                        steps_axis,
                        values,
                        label=f"Part {part_idx}",
                        marker='.',
                        markersize=3,
                        linewidth=1.0,
                        color=colormap(i % 10),
                        alpha=0.8
                    )

                if not has_data:
                    plt.close()
                    continue

                plt.title(f"{method} Sensitivity: {name}")
                plt.xlabel("Timestep")
                plt.ylabel("Score")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                # --------------------------------------------------
                # Custom block folder logic
                # --------------------------------------------------

                parts = name.split(".")

                save_path = None

                if len(parts) >= 3 and parts[0] == "blocks":
                    # blocks.3.attn2.to_q
                    block_id = parts[1]
                    block_folder = f"block{block_id}"

                    block_path = os.path.join(save_dir, block_folder)
                    os.makedirs(block_path, exist_ok=True)

                    # Remove blocks.<id>.
                    remaining_name = "_".join(parts[2:])
                    save_path = os.path.join(block_path, f"{remaining_name}.png")

                else:
                    # No block prefix → save normally
                    safe_name = name.replace(".", "_")
                    save_path = os.path.join(save_dir, f"{safe_name}.png")

                plt.savefig(save_path)
                plt.close()
                
    def plot_mixed_precision_allocation(self, method="fisher"):

        if method not in self.results:
            print(f"Method {method} not found.")
            return

        if not self.results[method]:
            print("No results recorded.")
            return

        import numpy as np

        # Use first part (single object case)
        part_idx = sorted(self.results[method].keys())[0]

        layer_scores = {}

        # Aggregate mean over timesteps
        for layer_name, values in self.results[method][part_idx].items():
            if len(values) > 0:
                layer_scores[layer_name] = np.mean(values)

        if len(layer_scores) == 0:
            print("No layer scores available.")
            return

        # Sort layers by sensitivity (descending)
        sorted_layers = sorted(layer_scores.items(),
                               key=lambda x: x[1],
                               reverse=True)

        layers = [x[0] for x in sorted_layers]
        scores = [x[1] for x in sorted_layers]

        n = len(layers)

        fp16_cutoff = int(0.2 * n)
        int8_cutoff = int(0.6 * n)

        plt.figure(figsize=(12, 6))
        plt.bar(range(n), scores)

        plt.axvline(fp16_cutoff, linestyle='--')
        plt.axvline(int8_cutoff, linestyle='--')

        plt.title("Sensitivity-Guided Mixed Precision Allocation")
        plt.xlabel("Layer Rank (High → Low Sensitivity)")
        plt.ylabel(f"{method} Score")

        plt.tight_layout()

        save_path = os.path.join(self.output_dir,
                                 f"mixed_precision_allocation_{method}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Mixed precision plot saved to: {save_path}")

