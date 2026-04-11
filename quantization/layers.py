import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


class PTQ4DiTLinear(nn.Linear):
    def __init__(self, *args, bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = bits
        self.register_buffer('salience_scale', torch.ones(self.in_features))

    def quantize(self, x, s):
        """
        Implements standard symmetric uniform quantization.
        x_int = clamp(round(x / s))
        For symmetric quantization, zero-point z=0.
        """
        max_int = 2**(self.bits - 1) - 1
        min_int = -2**(self.bits - 1)
        x_int = torch.clamp(torch.round(x / s), min_int, max_int)
        return x_int

    def forward(self, x):
        # Apply salience scale to activations
        x = x / self.salience_scale
        
        # Compute scale for activations
        s_act = x.abs().max() / (2**(self.bits - 1) - 1)
        
        # Quantize activations
        x_q = self.quantize(x, s_act)
        x_deq = x_q * s_act
        
        # Scale weights
        weight_scaled = self.weight * self.salience_scale
        
        # Compute scale for weights
        s_weight = weight_scaled.abs().max() / (2**(self.bits - 1) - 1)
        
        # Quantize weights
        weight_q = self.quantize(weight_scaled, s_weight)
        weight_deq = weight_q * s_weight
        
        # Perform linear operation
        return F.linear(x_deq, weight_deq, self.bias)


def calibrate_ptq4dit(model, dataloader, num_timesteps=10):
    """
    Calibrate PTQ4DiTLinear layers using SSC (Spearman's rho-guided) logic.
    
    For each layer:
    - Collect salience for weights and activations across timesteps.
    - Compute balanced salience per channel using SSC weighting.
    - Update the salience_scale buffer.
    """
    layers = []
    hooks = []
    activations = {}
    
    def hook_fn(module, input, output):
        activations[module] = input[0].detach()
    
    for module in model.modules():
        if isinstance(module, PTQ4DiTLinear):
            layers.append(module)
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    act_salience_list = {layer: [] for layer in layers}
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_timesteps:
                break
            _ = model(batch)
            for layer in layers:
                act = activations[layer]
                # act.shape assumed (batch, ..., in_features)
                salience = act.abs().flatten(0, -2).max(dim=0)[0]  # per in_feature
                act_salience_list[layer].append(salience)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    for layer in layers:
        weight_salience = layer.weight.abs().max(dim=0)[0]  # per in_feature
        act_salience_ts = act_salience_list[layer]  # list of tensors
        
        # Compute Spearman's rho weights
        weights = []
        for act_s in act_salience_ts:
            rho, _ = spearmanr(weight_salience.cpu().numpy(), act_s.cpu().numpy())
            weights.append(rho)
        
        weights = torch.tensor(weights, device=weight_salience.device, dtype=torch.float)
        
        # Aggregate activation salience with SSC weights
        sum_w = weights.sum()
        if sum_w == 0:
            s_X = torch.stack(act_salience_ts).mean(dim=0)
        else:
            s_X = sum(w * s for w, s in zip(weights, act_salience_ts)) / sum_w
        
        # Compute balanced salience
        balanced = torch.sqrt(s_X * weight_salience)
        
        # Update salience_scale
        layer.salience_scale.copy_(balanced)


def collect_layer_statistics(model, dataloader, num_steps=50):
    """
    Collect layer statistics for calibration.
    
    Registers forward hooks on every nn.Linear layer inside the joint_blocks.
    For each layer and each 'timestep' (batch index), records the max(abs(activations)) per channel,
    aggregated over the denoising process for that batch.
    Records the max(abs(weights)) per channel.
    
    Args:
        model: PartCrafterPipeline instance.
        dataloader: DataLoader yielding batches (dicts) for model(**batch).
        num_steps: Number of batches to process.
    
    Returns:
        dict: {layer_name: {'act': {timestep: tensor of max abs per channel}, 'weight': tensor of max abs per channel}}
    """
    statistics = {}
    hooks = []
    
    def hook_fn(module, input, output):
        act = input[0].detach()
        # act.shape assumed (batch, seq, channels)
        max_act = act.abs().flatten(0, -2).max(dim=0)[0]  # per channel
        
        layer_name = f"{module.__class__.__name__}_{id(module)}"
        
        if layer_name not in statistics:
            statistics[layer_name] = {
                'act': {},
                'weight': module.weight.abs().max(dim=0)[0]  # per channel
            }
        
        t = getattr(model, '_current_t', 0)  # batch index as timestep
        
        if t not in statistics[layer_name]['act']:
            statistics[layer_name]['act'][t] = max_act
        else:
            statistics[layer_name]['act'][t] = torch.max(statistics[layer_name]['act'][t], max_act)
    
    # Register hooks on nn.Linear in joint_blocks
    for block in model.transformer.joint_blocks:
        for module in block.modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break
            model._current_t = i
            _ = model(**batch)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return statistics


def compute_ptq4dit_scales(stats):
    """
    Compute Salience Balancing Scales using SSC (Spearman's rho-guided Salience Calibration).
    
    For each layer:
    - Aggregate activation salience across timesteps using SSC weights.
    - Compute balanced salience per channel as sqrt(s_X * s_W).
    
    Args:
        stats: Dictionary from collect_layer_statistics.
    
    Returns:
        dict: {layer_name: tensor of per-channel scaling factors}
    """
    scales = {}
    
    for layer_name, layer_stats in stats.items():
        weight_salience = layer_stats['weight']  # tensor per channel
        act_stats = layer_stats['act']  # {t: tensor per channel}
        
        # Compute SSC weights
        weights = []
        act_salience_ts = []
        for t, act_s in act_stats.items():
            rho, _ = spearmanr(weight_salience.cpu().numpy(), act_s.cpu().numpy())
            weights.append(rho)
            act_salience_ts.append(act_s)
        
        weights = torch.tensor(weights, device=weight_salience.device, dtype=torch.float)
        
        # Aggregate activation salience with SSC weights
        sum_w = weights.sum()
        if sum_w == 0:
            s_X = torch.stack(act_salience_ts).mean(dim=0)
        else:
            s_X = sum(w * s for w, s in zip(weights, act_salience_ts)) / sum_w
        
        # Compute balanced salience
        balanced = torch.sqrt(s_X * weight_salience)
        
        scales[layer_name] = balanced
    
    return scales