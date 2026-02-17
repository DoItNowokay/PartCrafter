from typing import Callable, List, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph
from einops import rearrange
from torch import nn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def scaled_dot_product_attention_with_entropy(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    compute_entropy: bool = False,
    entropy_list: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    head_dim = query.shape[-1]
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_probs = torch.softmax(attn_weights, dim=-1)
    if compute_entropy and entropy_list is not None:
        # print(attn_probs)
        # print(torch.isnan(attn_probs).any())
        entropy = -(attn_probs * torch.log(attn_probs + 1e-06)).sum(dim=-1)
        # print(torch.isnan(entropy).any())
        avg_entropy = entropy.mean(dim=-1).mean(dim=-1)
        entropy_list.append(avg_entropy)
        # print(avg_entropy)
    return torch.matmul(attn_probs, value)

class FlashTripo2AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Tripo2DiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self, topk=True):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.topk = topk

    def qkv(self, attn, q, k, v, attn_mask, dropout_p, is_causal):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = F.scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                q1 = q_chunk[:, :, ::50, :]
                sim = q1 @ k.transpose(-1, -2)
                sim = torch.mean(sim, -2)
                topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
                topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
                v0 = torch.gather(v, dim=-2, index=topk_ind)
                k0 = torch.gather(k, dim=-2, index=topk_ind)
                out = F.scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that tripo2 split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # flashvdm topk
        hidden_states = self.qkv(attn, query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)   

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class TripoSGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the TripoSG model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that pre-trained models split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FusedTripoSGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0) with fused
    projection layers. This is used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on
    query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedTripoSGAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # NOTE that pre-trained split heads first, then split qkv
        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            query = attn.to_q(hidden_states)

            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# Modified from https://github.com/VAST-AI-Research/MIDI-3D/blob/main/midi/models/attention_processor.py#L264
class PartCrafterAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the PartCrafter model. It applies a normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        num_parts: Optional[Union[int, torch.Tensor]] = None,
        compute_entropy: bool = False,
        entropy_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that pre-trained models split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        entropy_flag = compute_entropy and not attn.is_cross_attention

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        if num_parts is None:
            attn_output = scaled_dot_product_attention_with_entropy(
                query,
                key,
                value,
                attention_mask=attention_mask,
                compute_entropy=entropy_flag,
                entropy_list=entropy_list,
            )
            hidden_states = attn_output.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        elif isinstance(num_parts, torch.Tensor):
            idx = 0
            hidden_states_list = []
            for n_p in num_parts:
                n_p = int(n_p)
                k = key[idx : idx + n_p]
                v = value[idx : idx + n_p]
                q = query[idx : idx + n_p]
                idx += n_p
                if k.shape[2] == q.shape[2]:
                    k = rearrange(
                        k, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
                    )
                    v = rearrange(
                        v, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
                    )
                else:
                    k = k[::n_p]
                    v = v[::n_p]
                q = rearrange(
                    q, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
                )
                h_s = scaled_dot_product_attention_with_entropy(
                    q,
                    k,
                    v,
                    compute_entropy=entropy_flag,
                    entropy_list=entropy_list,
                )
                h_s = h_s.transpose(1, 2).reshape(
                    n_p, -1, attn.heads * head_dim
                )
                h_s = h_s.to(query.dtype)
                hidden_states_list.append(h_s)
            hidden_states = torch.cat(hidden_states_list, dim=0)

        elif isinstance(num_parts, int):
            if key.shape[2] == query.shape[2]:
                key = rearrange(
                    key, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
                )
                value = rearrange(
                    value, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
                )
            else:
                key = key[::num_parts]
                value = value[::num_parts]
            query = rearrange(
                query, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
            )
            attn_output = scaled_dot_product_attention_with_entropy(
                query,
                key,
                value,
                compute_entropy=entropy_flag,
                entropy_list=entropy_list,
            )
            hidden_states = attn_output.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        else:
            raise ValueError(
                "num_parts must be a torch.Tensor or int, but got {}".format(type(num_parts))
            )
        
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PartCrafterEditAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the PartCrafter model. It applies a normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )


    # def __call__(
    #     self,
    #     attn: Attention,
    #     hidden_states: torch.Tensor,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     temb: Optional[torch.Tensor] = None,
    #     image_rotary_emb: Optional[torch.Tensor] = None,
    #     num_parts: Optional[Union[int, torch.Tensor]] = None,
    # ) -> torch.Tensor:
    #     from diffusers.models.embeddings import apply_rotary_emb

    #     residual = hidden_states
    #     if attn.spatial_norm is not None:
    #         hidden_states = attn.spatial_norm(hidden_states, temb)

    #     input_ndim = hidden_states.ndim

    #     if input_ndim == 4:
    #         batch_size, channel, height, width = hidden_states.shape
    #         hidden_states = hidden_states.view(
    #             batch_size, channel, height * width
    #         ).transpose(1, 2)

    #     batch_size, sequence_length, _ = (
    #         hidden_states.shape
    #         if encoder_hidden_states is None
    #         else encoder_hidden_states.shape
    #     )

    #     if attention_mask is not None:
    #         attention_mask = attn.prepare_attention_mask(
    #             attention_mask, sequence_length, batch_size
    #         )
    #         # scaled_dot_product_attention expects attention_mask shape to be
    #         # (batch, heads, source_length, target_length)
    #         attention_mask = attention_mask.view(
    #             batch_size, attn.heads, -1, attention_mask.shape[-1]
    #         )

    #     if attn.group_norm is not None:
    #         hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
    #             1, 2
    #         )

    #     query = attn.to_q(hidden_states)

    #     if encoder_hidden_states is None:
    #         encoder_hidden_states = hidden_states
    #     elif attn.norm_cross:
    #         encoder_hidden_states = attn.norm_encoder_hidden_states(
    #             encoder_hidden_states
    #         )
    #     key = attn.to_k(encoder_hidden_states)
    #     value = attn.to_v(encoder_hidden_states)

    #     # NOTE that pre-trained models split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
    #     # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
    #     if not attn.is_cross_attention:
    #         qkv = torch.cat((query, key, value), dim=-1)
    #         split_size = qkv.shape[-1] // attn.heads // 3
    #         qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
    #         query, key, value = torch.split(qkv, split_size, dim=-1)
    #     else:
    #         kv = torch.cat((key, value), dim=-1)
    #         split_size = kv.shape[-1] // attn.heads // 2
    #         kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
    #         key, value = torch.split(kv, split_size, dim=-1)

    #     head_dim = key.shape[-1]

    #     query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    #     key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    #     value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    #     if attn.norm_q is not None:
    #         query = attn.norm_q(query)
    #     if attn.norm_k is not None:
    #         key = attn.norm_k(key)

    #     # Apply RoPE if needed
    #     if image_rotary_emb is not None:
    #         query = apply_rotary_emb(query, image_rotary_emb)
    #         if not attn.is_cross_attention:
    #             key = apply_rotary_emb(key, image_rotary_emb)

    #     if isinstance(num_parts, torch.Tensor):
    #         # Assume list in training, do not consider classifier-free guidance
    #         idx = 0
    #         hidden_states_list = []
    #         for n_p in num_parts:
    #             k = key[idx : idx + n_p]
    #             v = value[idx : idx + n_p]
    #             q = query[idx : idx + n_p]
    #             idx += n_p
    #             if k.shape[2] == q.shape[2]:
    #                 # Assuming self-attention
    #                 # Here 'b' is always 1
    #                 k = rearrange(
    #                     k, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
    #                 ) # [b, h, ni*nt, c]
    #                 v = rearrange(
    #                     v, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
    #                 ) # [b, h, ni*nt, c]
    #             else:
    #                 # Assuming cross-attention
    #                 # Here 'b' is always 1
    #                 k = k[::n_p]     # [b, h, nt, c]
    #                 v = v[::n_p]     # [b, h, nt, c]
    #             # Here 'b' is always 1
    #             q = rearrange(
    #                 q, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
    #             ) # [b, h, ni*nt, c]
    #             # the output of sdp = (batch, num_heads, seq_len, head_dim)
    #             h_s = F.scaled_dot_product_attention(
    #                 q, k, v,
    #                 dropout_p=0.0,
    #                 is_causal=False,
    #             )
    #             h_s = h_s.transpose(1, 2).reshape(
    #                 n_p, -1, attn.heads * head_dim
    #             )
    #             h_s = h_s.to(query.dtype)
    #             hidden_states_list.append(h_s)
    #         hidden_states = torch.cat(hidden_states_list, dim=0)

    #     elif isinstance(num_parts, int):
    #         # Assume single instance
    #         if key.shape[2] == query.shape[2]:
    #             # Assuming self-attention
    #             # Here we need 'b' when using classifier-free guidance
    #             key = rearrange(
    #                 key, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
    #             ) # [b, h, ni*nt, c]
    #             value = rearrange(
    #                 value, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
    #             ) # [b, h, ni*nt, c]
    #         else:
    #             # Assuming cross-attention
    #             # Here we need 'b' when using classifier-free guidance
    #             # Control signal is repeated ni times within each (b, ni)
    #             # We select only the first instance per group
    #             key = key[::num_parts]     # [b, h, nt, c]
    #             value = value[::num_parts] # [b, h, nt, c]
    #         query = rearrange(
    #             query, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
    #         ) # [b, h, ni*nt, c]

    #         # the output of sdp = (batch, num_heads, seq_len, head_dim)
    #         hidden_states = F.scaled_dot_product_attention(
    #             query,
    #             key,
    #             value,
    #             dropout_p=0.0,
    #             is_causal=False,
    #         )
    #         hidden_states = hidden_states.transpose(1, 2).reshape(
    #             batch_size, -1, attn.heads * head_dim
    #         )
    #         hidden_states = hidden_states.to(query.dtype)

    #     else:
    #         raise ValueError(
    #             "num_parts must be a torch.Tensor or int, but got {}".format(type(num_parts))
    #         )
        
    #     # linear proj
    #     hidden_states = attn.to_out[0](hidden_states)
    #     # dropout
    #     hidden_states = attn.to_out[1](hidden_states)

    #     if input_ndim == 4:
    #         hidden_states = hidden_states.transpose(-1, -2).reshape(
    #             batch_size, channel, height, width
    #         )

    #     if attn.residual_connection:
    #         hidden_states = hidden_states + residual

    #     hidden_states = hidden_states / attn.rescale_output_factor

    #     return hidden_states
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_parts: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        compute_entropy: bool = False,
        entropy_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        import torch.nn.functional as F

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        batch_size_enc, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            if attn.group_norm is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        heads = attn.heads
        head_dim = query.shape[-1] // heads
        query = query.view(-1, query.shape[1], heads, head_dim).transpose(1, 2)
        key = key.view(-1, key.shape[1], heads, head_dim).transpose(1, 2)
        value = value.view(-1, value.shape[1], heads, head_dim).transpose(1, 2)

        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, "norm_k") and attn.norm_k is not None:
            key = attn.norm_k(key)

        entropy_flag = compute_entropy and not attn.is_cross_attention
        attn_output_local = scaled_dot_product_attention_with_entropy(
            query,
            key,
            value,
            compute_entropy=entropy_flag,
            entropy_list=entropy_list,
        )

        if num_parts is None:
            attn_output = attn_output_local
        else:
            if not isinstance(num_parts, torch.Tensor):
                raise ValueError(f"num_parts must be a torch.Tensor of shape [B], got {type(num_parts)}")

            num_objects = int(num_parts.shape[0])
            total_parts_in_batch = query.shape[0]
            if total_parts_in_batch % num_objects != 0:
                raise ValueError(
                    f"Total parts ({total_parts_in_batch}) not divisible by num_objects ({num_objects})"
                )
            max_parts = total_parts_in_batch // num_objects

            if part_mask is not None:
                if part_mask.shape[0] != num_objects or part_mask.shape[1] != max_parts:
                    raise ValueError(
                        f"part_mask shape mismatch. Got {part_mask.shape} but expected [{num_objects}, {max_parts}]"
                    )
                mask_flat = part_mask.view(-1)
            else:
                indices = torch.arange(max_parts, device=num_parts.device).unsqueeze(0)
                mask = indices < num_parts.unsqueeze(1)
                mask_flat = mask.view(-1)

            obj_indices = torch.arange(num_objects, device=query.device).repeat_interleave(max_parts)
            obj_indices = torch.where(mask_flat.to(obj_indices.dtype), obj_indices, torch.full_like(obj_indices, -1))

            seq_per_part = query.shape[2]
            q_global = torch.zeros((num_objects, heads, seq_per_part, head_dim), device=query.device, dtype=query.dtype)
            k_global = torch.zeros_like(q_global)
            v_global = torch.zeros_like(q_global)

            valid_mask = obj_indices >= 0
            tgt_idx = None
            if valid_mask.any():
                src_q = query[valid_mask]
                src_k = key[valid_mask]
                src_v = value[valid_mask]
                tgt_idx = obj_indices[valid_mask].long()
                idx_expanded = tgt_idx.view(-1, 1, 1, 1).expand(-1, heads, seq_per_part, head_dim)
                q_global.scatter_add_(0, idx_expanded, src_q)
                k_global.scatter_add_(0, idx_expanded, src_k)
                v_global.scatter_add_(0, idx_expanded, src_v)

            num_parts_float = num_parts.to(dtype=q_global.dtype).view(num_objects, 1, 1, 1).clamp(min=1.0)
            q_global = q_global / num_parts_float
            k_global = k_global / num_parts_float
            v_global = v_global / num_parts_float

            attn_output_global = F.scaled_dot_product_attention(
                q_global, k_global, v_global, dropout_p=0.0, is_causal=False
            )

            attn_output_part = torch.zeros_like(attn_output_local)
            if valid_mask.any():
                selected_global = attn_output_global[tgt_idx]
                attn_output_part[valid_mask] = selected_global

            attn_output = attn_output_local + attn_output_part

        attn_output = attn_output.transpose(1, 2).reshape(-1, sequence_length, heads * head_dim)
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)

        if input_ndim == 4:
            attn_output = attn_output.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            attn_output = attn_output + residual

        attn_output = attn_output / attn.rescale_output_factor

        return attn_output

