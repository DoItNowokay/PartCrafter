import os
import matplotlib.pyplot as plt
import torch


class CurvatureAnalyzer:

    def __init__(self):

        self.prev_latents = None
        self.prev_delta = None
        self.values = []
        self.output_dir = None

    def reset_for_new_step(self, output_dir):

        self.prev_latents = None
        self.prev_delta = None
        self.values = []
        self.output_dir = output_dir

    def step(self, latents):

        latents = latents.detach()

        if self.prev_latents is None:
            self.prev_latents = latents
            return

        delta = latents - self.prev_latents

        if self.prev_delta is not None:

            a = delta.flatten()
            b = self.prev_delta.flatten()

            cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)

            curvature = 1 - cos.item()
            self.values.append(curvature)

        self.prev_delta = delta
        self.prev_latents = latents
        
    def set_results(self, curvature):
        self.results = curvature

    def plot_results(self):

        if self.output_dir is None or len(self.values) == 0:
            return

        plt.figure(figsize=(8,5))

        plt.plot(self.values, marker=".")
        plt.xlabel("Timestep")
        plt.ylabel("1 - cos(θ)")
        plt.title("Diffusion Curvature")

        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "curvature_n.png")
        plt.savefig(save_path)
        plt.close()