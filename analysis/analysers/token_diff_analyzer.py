import os
import matplotlib.pyplot as plt
import torch


class TokenDiffAnalyzer:

    def __init__(self):

        self.prev_latents = None
        self.values = []
        self.output_dir = None

    def reset_for_new_step(self, output_dir):

        self.prev_latents = None
        self.values = []
        self.output_dir = output_dir

    def step(self, latents):

        latents = latents.detach()

        if self.prev_latents is None:
            self.prev_latents = latents
            return

        diff = torch.mean(torch.abs(latents - self.prev_latents)).item()

        self.values.append(diff)
        self.prev_latents = latents
        
    def set_results(self, token_diffs):
        self.results = token_diffs

    def plot_results(self):

        if self.output_dir is None or len(self.values) == 0:
            print(self.output_dir, self.values)
            print("No values to plot or output directory not set.")
            return

        plt.figure(figsize=(8,5))

        plt.plot(self.values, marker=".")
        plt.xlabel("Timestep")
        plt.ylabel("Mean |Δ tokens|")
        plt.title("Token Difference")

        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "token_diff_n.png")
        plt.savefig(save_path)
        plt.close()