import sys
sys.path.append('/data1/simroop/github/Q-DiT')
sys.path.append('/data1/simroop/github/PartCrafter')

import sys
import torch
import torch.nn as nn
from copy import deepcopy

# Core Q-DiT Imports
from qdit.qLinearLayer import QLinearLayer
from qdit.quant import Quantizer

# PartCrafter Imports
from src.models.transformers.partcrafter_transformer import DiTBlock as PartCrafterBlock

class QuantMlpPartCrafter(nn.Module):
    def __init__(self, mlp, args):
        super().__init__()
        self.args = deepcopy(args)
        self.input_quant = Quantizer(args=self.args)
        self.act_quant = Quantizer(args=self.args)
        
        # 1. Identify the GEGLU structure
        # mlp.net[0] is typically the GEGLU module
        # mlp.net[2] is the output Linear layer
        
        print("\n===== MLP STRUCTURE =====")
        for i, layer in enumerate(mlp.net):
            print(i, layer)
        print("========================\n")
        self.is_geglu = hasattr(mlp.net[0], 'proj')
        
        if self.is_geglu:
            # # Wrap the expansion projection (2048 -> 8192)
            # self.fc1 = QLinearLayer(mlp.net[0].proj, self.args)
            # self.act_module = mlp.net[0]
            # # Wrap the contraction projection (4096 -> 2048)
            # self.fc2 = QLinearLayer(mlp.net[2], self.args)
            pass
            
        else:
            # Standard MLP path fallback
            linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
            self.fc1 = QLinearLayer(linears[0], self.args)
            self.fc2 = QLinearLayer(linears[-1], self.args)
            self.act_module = next((m for m in mlp.net if not isinstance(m, nn.Linear)), nn.Identity())

        self.norm = getattr(mlp, "norm", nn.Identity())

    def forward(self, x):
        x = self.input_quant(x)
        
        if self.is_geglu:
            # Project to expansion space (e.g. 8192)
            hidden_states = self.fc1(x)
            
            # --- THE CRITICAL FIX: MANUAL CHUNK ---
            # Bypass self.act_module(hidden_states) to avoid double-projection crash.
            # We perform the exact GEGLU math here.
            x, gate = hidden_states.chunk(2, dim=-1)
            x = x * torch.nn.functional.gelu(gate)
        else:
            x = self.fc1(x)
            x = self.act_module(x)
            
        x = self.norm(x)
        x = self.act_quant(x)
        
        # Project back to model dim (e.g. 2048)
        x = self.fc2(x)
        return x

class QuantAttentionPartCrafter(nn.Module):
    def __init__(self, attn, args):
        super().__init__()
        self.args = deepcopy(args)
        
        # 1. Mirror Metadata
        for name, value in attn.__dict__.items():
            if not isinstance(value, (nn.Module, nn.ModuleList, nn.Parameter)) and not name.startswith('_'):
                setattr(self, name, value)
        
        # 2. Mirror Norms and Processor
        self.processor = attn.processor
        self.spatial_norm = getattr(attn, "spatial_norm", None)
        self.group_norm = getattr(attn, "group_norm", None)
        self.norm_q = getattr(attn, "norm_q", None)
        self.norm_k = getattr(attn, "norm_k", None)

        # 3. Wrap Projections
        self.to_q = QLinearLayer(attn.to_q, self.args)
        self.to_k = QLinearLayer(attn.to_k, self.args)
        self.to_v = QLinearLayer(attn.to_v, self.args)
        
        self.to_out = nn.ModuleList()
        for module in attn.to_out:
            if isinstance(module, nn.Linear):
                self.to_out.append(QLinearLayer(module, self.args))
            else:
                self.to_out.append(module)
        
        self.input_quant = Quantizer(args=self.args)
        self.act_quant = Quantizer(args=self.args)

    def forward(self, x, **kwargs):
        # ✅ THE FIX: Filter kwargs based on the processor type
        # TripoSGAttnProcessor2_0 does not accept 'num_parts', 'compute_entropy', etc.
        if "TripoSGAttnProcessor" in self.processor.__class__.__name__:
            # List of standard arguments supported by the base TripoSG processor
            supported_keys = ['encoder_hidden_states', 'attention_mask', 'image_rotary_emb']
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_keys}
            return self.processor(self, x, **filtered_kwargs)
        
        # PartCrafter processors usually accept everything in kwargs
        return self.processor(self, x, **kwargs)

class QuantPartCrafterBlock(nn.Module):
    def __init__(self, original_block, args):
        super().__init__()
        self.args = deepcopy(args)
        self.attn1 = QuantAttentionPartCrafter(original_block.attn1, self.args)
        self.attn2 = QuantAttentionPartCrafter(original_block.attn2, self.args)
        
        self.editing = getattr(original_block, "editing", "none")
        if self.editing == "text_cross_attn":
            self.attn_text = QuantAttentionPartCrafter(original_block.attn_text, self.args)
            self.norm_text = original_block.norm_text

        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.norm3 = original_block.norm3
        self.ff = QuantMlpPartCrafter(original_block.ff, self.args)
        
        self.skip_linear = original_block.skip_linear
        self.skip_norm = getattr(original_block, "skip_norm", None)
        self.use_self_attention = original_block.use_self_attention
        self.use_cross_attention = original_block.use_cross_attention
        self.skip_concat_front = original_block.skip_concat_front
        self.skip_norm_last = original_block.skip_norm_last

    def forward(self, hidden_states, encoder_hidden_states=None, temb=None, 
                image_rotary_emb=None, skip=None, attention_kwargs=None, **kwargs):
        
        # Catch extra text_encoder from kwargs
        text_encoder_hidden_states = kwargs.get("text_encoder_hidden_states", None)
        attn_kwargs = attention_kwargs or {}
        
        if self.skip_linear is not None:
            cat = torch.cat([skip, hidden_states] if self.skip_concat_front else [hidden_states, skip], dim=-1)
            hidden_states = self.skip_norm(self.skip_linear(cat)) if self.skip_norm_last else self.skip_linear(self.skip_norm(cat))

        if self.use_self_attention:
            hidden_states = hidden_states + self.attn1(self.norm1(hidden_states), image_rotary_emb=image_rotary_emb, **attn_kwargs)

        if self.use_cross_attention:
            hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states=encoder_hidden_states, image_rotary_emb=image_rotary_emb, **attn_kwargs)
            if self.editing == "text_cross_attn" and text_encoder_hidden_states is not None:
                hidden_states = hidden_states + self.attn_text(self.norm_text(hidden_states), encoder_hidden_states=text_encoder_hidden_states, **attn_kwargs)
        
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states

def apply_qdit_to_model(model, args, device="cuda"):
    model.eval()

    defaults = {
        'quantize_bmm_input': False,
        'tiling': 0,
        'exponential': False,
        'a_clip_ratio': 1.0,
        'w_clip_ratio': 1.0,
        'kv_clip_ratio': 1.0,
        'quant_type': 'int',
        'static': False,
        'w_sym': True,
        'a_sym': False,
        'weight_channel_group': 1,
        'percdamp': 0.01,
        'use_gptq': False
    }

    for key, val in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, val)

    if hasattr(args, "qdit_method"):
        args.quant_method = args.qdit_method

    w_group_size = args.weight_group_size

    print("\n[DEBUG] ===== APPLYING QDIT =====")

    # 🔁 Replace blocks
    for i in range(len(model.blocks)):
        if isinstance(model.blocks[i], PartCrafterBlock):
            print(f"[DEBUG] Replacing block {i}")

            block_args = deepcopy(args)
            block_args.weight_group_size = w_group_size
            block_args.act_group_size = getattr(args, "act_group_size", 64)

            model.blocks[i] = QuantPartCrafterBlock(
                model.blocks[i], block_args
            ).to(device)

    print("\n[DEBUG] ===== QUANTIZING WEIGHTS =====")

    # 🔥 Quantization phase
    for block_idx, block in enumerate(model.blocks):
        if isinstance(block, QuantPartCrafterBlock):
            print(f"[DEBUG] Quantizing block {block_idx}")

            # Attention
            heads = [block.attn1, block.attn2]
            if hasattr(block, "attn_text"):
                heads.append(block.attn_text)

            for h in heads:
                for l in [h.to_q, h.to_k, h.to_v]:
                    l.args.weight_group_size = w_group_size
                    l.quant()

                for ol in h.to_out:
                    if isinstance(ol, QLinearLayer):
                        ol.args.weight_group_size = w_group_size
                        ol.quant()

            # 🔥 MLP FIX
            if block.ff.is_geglu:
                print("[DEBUG] Quantizing GEGLU proj")

                block.ff.act_module.proj.args.weight_group_size = w_group_size
                block.ff.act_module.proj.quant()

            else:
                print("[DEBUG] Quantizing fc1")

                block.ff.fc1.args.weight_group_size = w_group_size
                block.ff.fc1.quant()

            print("[DEBUG] Quantizing fc2")

            block.ff.fc2.args.weight_group_size = w_group_size
            block.ff.fc2.quant()

    print("\n[DEBUG] ===== DONE =====")

    return model