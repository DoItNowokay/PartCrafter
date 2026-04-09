import os
import matplotlib.pyplot as plt
import torch


class EntropyAnalyzer:

    def __init__(self):

        self.values = []
        self.output_dir = None

    def reset_for_new_step(self, output_dir):

        self.values = []
        self.output_dir = output_dir

    def step(self, attention_maps):

        if len(attention_maps) == 0:
            return

        entropies = []

        for attn in attention_maps:

            p = attn.softmax(dim=-1)

            entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
            entropies.append(entropy.mean().item())

        self.values.append(sum(entropies) / len(entropies))
        
    def set_results(self, entropy):
        self.results = entropy

    def plot_results(self):

        if self.output_dir is None or len(self.values) == 0:
            return

        plt.figure(figsize=(8,5))

        plt.plot(self.values, marker=".")
        plt.xlabel("Timestep")
        plt.ylabel("Entropy")
        plt.title("Attention Entropy")

        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "entropy_n.png")
        plt.savefig(save_path)
        plt.close()