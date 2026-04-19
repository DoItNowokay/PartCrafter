import torch
import torch.nn as nn
import numpy as np


class Point4BitQuantizer:
    """
    A class that handles symmetric uniform quantization.
    Quantizes tensors to a specified bit width symmetrically around zero.
    """
    def __init__(self, bits=4):
        self.bits = bits

    def quantize(self, x, bits):
        """
        Quantize the input tensor x using symmetric uniform quantization.

        Args:
            x (torch.Tensor): Input tensor to quantize.
            bits (int or list): Bit width(s). If int, quantize the entire tensor.
                                If list, quantize per group (assumes group_dim=0).

        Returns:
            torch.Tensor: Quantized tensor.
        """
        if isinstance(bits, int):
            # Per-tensor quantization
            scale = x.abs().max() / (2**(bits - 1) - 1)
            if scale == 0:
                return x
            quant = torch.round(x / scale)
            quant = torch.clamp(quant, -2**(bits - 1), 2**(bits - 1) - 1)
            return quant * scale
        elif isinstance(bits, list):
            # Per-group quantization, assumes x.shape[0] == len(bits)
            result = []
            for i, b in enumerate(bits):
                group = x[i]
                scale = group.abs().max() / (2**(b - 1) - 1)
                if scale == 0:
                    result.append(group)
                else:
                    quant = torch.round(group / scale)
                    quant = torch.clamp(quant, -2**(b - 1), 2**(b - 1) - 1)
                    result.append(quant * scale)
            return torch.stack(result, dim=0)
        else:
            raise ValueError("bits must be int or list")

    def quantize_adaptive(self, x, bin_edges_list):
        """
        Quantize using adaptive bin edges with equal probability mass.

        Args:
            x (torch.Tensor): Input tensor of shape (num_groups, C).
            bin_edges_list (list of torch.Tensor): List of bin edges for each group, each of shape (m+1,).

        Returns:
            torch.Tensor: Quantized tensor.
        """
        result = []
        for i, bin_edges in enumerate(bin_edges_list):
            group = x[i]
            quantized = []
            for val in group.flatten():
                # Find the bin
                assigned = False
                for j in range(len(bin_edges) - 1):
                    if bin_edges[j] <= val < bin_edges[j + 1]:
                        mid = (bin_edges[j] + bin_edges[j + 1]) / 2
                        quantized.append(mid)
                        assigned = True
                        break
                if not assigned:
                    # Clamp to the last bin
                    quantized.append(bin_edges[-1])
            quantized = torch.tensor(quantized, device=x.device, dtype=x.dtype).view(group.shape)
            result.append(quantized)
        return torch.stack(result, dim=0)


def FA_PAQ_Logic(activation, m1, m):
    """
    Identifies the top m1 'foreground' tokens based on channel-wise mean magnitude.
    Computes m quantization intervals using CDF-based partitioning (equal probability mass).

    Args:
        activation (torch.Tensor): Activation tensor of shape (B, N, C).
        m1 (float): Ratio of foreground tokens.
        m (int): Number of quantization intervals.

    Returns:
        fg_bin_edges (torch.Tensor): Bin edges for foreground, shape (m+1,).
        bg_bin_edges (torch.Tensor): Bin edges for background, shape (m+1,).
        fg_mask (torch.Tensor): Boolean mask for foreground tokens, shape (B, N).
    """
    # Compute channel-wise mean magnitude
    scores = activation.abs().mean(dim=-1)  # (B, N)
    num_fg = int(m1 * activation.shape[1])
    _, indices = scores.topk(num_fg, dim=1)
    fg_mask = torch.zeros_like(scores, dtype=torch.bool)
    fg_mask.scatter_(1, indices, True)

    # Collect values for foreground and background
    fg_expanded = fg_mask.unsqueeze(-1).expand_as(activation)
    fg_values = activation[fg_expanded].flatten()
    bg_values = activation[~fg_expanded].flatten()

    def compute_bin_edges(values, m):
        if len(values) == 0 or m <= 0:
            # Default symmetric bin edges
            max_val = 1.0
            return torch.linspace(-max_val, max_val, m + 1, device=values.device if len(values) > 0 else torch.device('cpu'))
        sorted_values = torch.sort(values)[0]
        bin_edges = []
        for i in range(m + 1):
            idx = min(i * (len(sorted_values) - 1) // m, len(sorted_values) - 1)
            bin_edges.append(sorted_values[idx])
        return torch.stack(bin_edges)

    fg_bin_edges = compute_bin_edges(fg_values, m)
    bg_bin_edges = compute_bin_edges(bg_values, m)
    return fg_bin_edges, bg_bin_edges, fg_mask


def G_KWQ_Logic(grad):
    """
    Computes a sensitivity score for weights by averaging the absolute gradients per output channel.

    Args:
        grad (torch.Tensor): Gradient tensor of shape (out_channels, in_channels).

    Returns:
        torch.Tensor: Sensitivity scores of shape (out_channels,).
    """
    return grad.abs().mean(dim=1)


class Point4BitLinear(nn.Linear):
    """
    A simple wrapper for nn.Linear that uses Point4Bit utilities in its forward pass
    to apply different bit-widths to foreground vs. background tokens.
    """
    def __init__(self, in_features, out_features, fg_ratio=0.2, weight_ratio=0.8,
                 high_bits=8, low_bits=4, bias=True, token_diff_threshold=0.008, curvature_spike_threshold=0.12, masks=None):
        super().__init__(in_features, out_features, bias)
        self.fg_ratio = fg_ratio
        self.weight_ratio = weight_ratio
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.quantizer = Point4BitQuantizer()
        self.token_diff_threshold = token_diff_threshold
        self.curvature_spike_threshold = curvature_spike_threshold
        self.m = 2 ** high_bits  # Number of intervals for CDF
        self.masks = masks  # Precomputed binary masks for key channels

    def forward(self, x, step=None, token_diff=None, curvature=None):
        # Handle different input shapes
        original_shape = x.shape
        if x.dim() == 2:
            # Assume (seq, dim), treat as (1, seq, dim)
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            # For other shapes, flatten to 2D
            x = x.view(-1, x.shape[-1])
            x = x.unsqueeze(0)  # (1, total, dim)
        
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        # Determine if to use CDF-based intervals
        use_cdf = (step is not None and 20 <= step <= 30 and
                   ((token_diff is not None and token_diff > self.token_diff_threshold) or
                    (curvature is not None and curvature > self.curvature_spike_threshold)))

        if use_cdf:
            # Use FA-PAQ CDF intervals
            fg_bin_edges, bg_bin_edges, fg_mask = FA_PAQ_Logic(x, self.fg_ratio, self.m)
            bin_edges_list = [fg_bin_edges if fg_mask.view(-1)[i] else bg_bin_edges for i in range(B * N)]
            x_quant = self.quantizer.quantize_adaptive(x_flat, bin_edges_list)
        else:
            # Use uniform quantization with different bits
            scores = x_flat.abs().mean(dim=1)  # (B*N,)
            num_fg = int(self.fg_ratio * B * N)
            _, indices = scores.topk(num_fg)
            fg_mask = torch.zeros(B * N, dtype=torch.bool, device=x.device)
            fg_mask[indices] = True
            bits_list = [self.high_bits if fg else self.low_bits for fg in fg_mask]
            x_quant = self.quantizer.quantize(x_flat, bits_list)

        # Assign bits to weight channels based on sensitivity
        if self.masks is not None:
            weight_bits = [self.high_bits if self.masks[i] else self.low_bits for i in range(self.out_features)]
        elif self.weight.grad is not None:
            sensitivity = G_KWQ_Logic(self.weight.grad)
            num_high = int(self.weight_ratio * self.out_features)
            _, indices = sensitivity.topk(num_high)
            weight_bits = [self.low_bits] * self.out_features
            for i in indices.tolist():
                weight_bits[i] = self.high_bits
        else:
            weight_bits = [self.high_bits] * self.out_features

        weight_quant = self.quantizer.quantize(self.weight, weight_bits)

        # Perform linear operation
        out = torch.matmul(x_quant, weight_quant.t())
        if self.bias is not None:
            out = out + self.bias

        # Reshape back
        out = out.reshape(B, N, self.out_features)
        if original_shape != out.shape:
            if len(original_shape) == 2:
                out = out.squeeze(0)
            elif len(original_shape) > 3:
                out = out.view(original_shape[:-1] + (self.out_features,))
        return out