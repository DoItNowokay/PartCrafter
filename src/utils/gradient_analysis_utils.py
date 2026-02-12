"""
Utilities for gradient analysis and sensitivity testing.
"""

import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GradientSensitivityAnalyzer:
    def __init__(self, method="gradient_norm"):
        self.results = defaultdict(lambda: defaultdict(list))
        # We store raw timesteps just for reference, but we will plot by index
        self.timesteps_recorded = []
        self.method = method
        self.output_dir = None

    def reset_for_new_step(self, output_dir):
        """
        Resets results and points to the current object's folder.
        output_dir example: .../gs_7.0/step_0001/
        """
        self.results = defaultdict(lambda: defaultdict(list))
        self.timesteps_recorded = []
        self.output_dir = output_dir

    def analyze_step(self, model, latents, t, encoder_hidden_states, text_pooled, text_hidden_states, attention_kwargs, do_classifier_free_guidance):
        """
        Runs a shadow forward/backward pass to record gradient norms.
        """
        if self.method != "gradient_norm":
            return

        # 1. Shadow Forward Pass with Gradients Enabled
        with torch.enable_grad():
            model.zero_grad()

            # Forward pass to build the graph
            output = model(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                text_pooled=text_pooled,
                text_hidden_states=text_hidden_states,
                attention_kwargs=attention_kwargs,
                return_dict=True
            ).sample

            # 2. Identify the Conditional Output
            if do_classifier_free_guidance:
                output_conditional = output.chunk(2)[1]
            else:
                output_conditional = output

            num_parts = attention_kwargs.get("num_parts", 1)

            # 3. Part-Wise Backward Pass
            for part_idx in range(num_parts):
                model.zero_grad()

                # Isolate output for this specific part
                part_output = output_conditional[part_idx].unsqueeze(0)

                # Proxy Loss: Norm of the output
                loss = part_output.norm()

                # Backward Pass
                loss.backward(retain_graph=True)

                # Record Gradients for ALL Linear layers
                for name, layer in model.named_modules():
                    if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                        grad_norm = layer.weight.grad.norm(2).item()
                        self.results[part_idx][name].append(grad_norm)

            del output

        # Record the raw timestep for debugging, but we plot by call count
        self.timesteps_recorded.append(t[0].item())

    def plot_results(self):
        """
        Saves one plot per layer into the step folder.
        Uses Step Index (0...N) for X-axis.
        """
        if not self.output_dir:
            return

        save_dir = os.path.join(self.output_dir, self.method)
        os.makedirs(save_dir, exist_ok=True)

        recorded_parts = sorted(list(self.results.keys()))
        if not recorded_parts:
            return

        reference_part = recorded_parts[0]
        available_layers = list(self.results[reference_part].keys())
        colormap = plt.get_cmap('tab10')

        # Generate X-axis based on how many steps actually ran
        # This will be [0, 1, 2, ... 49] if num_inference_steps=50
        num_steps_ran = len(self.timesteps_recorded)
        steps_axis = list(range(num_steps_ran))

        for name in available_layers:
            plt.figure(figsize=(10, 6))
            has_data = False

            for i, part_idx in enumerate(recorded_parts):
                values = self.results[part_idx][name]
                if not values: continue
                has_data = True

                plt.plot(steps_axis, values,
                         label=f"Part {part_idx}",
                         marker='.', markersize=3, linewidth=1.0,
                         color=colormap(i % 10), alpha=0.8)

            if not has_data:
                plt.close()
                continue

            plt.title(f"Sensitivity: {name}")
            plt.xlabel(f"Timestep") # Explicitly shows 0-50 logic
            plt.ylabel("Gradient Norm")

            # NOTE: We do NOT invert axis here. Step 0 is start, Step 50 is finish.

            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            safe_layer_name = name.replace(".", "_")
            plt.savefig(os.path.join(save_dir, f"{safe_layer_name}.png"))
            plt.close()