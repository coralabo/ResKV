# SPDX-License-Identifier: Apache-2.0
"""Llama integration using the supported Transformers model/generation stack."""

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, apply_rotary_pos_emb

from .._attention import attention
from ..cache import ResKVCache
from ..config import ResKVKernelConfig, _model_settings


class ResKVMLP(LlamaMLP):
    """Reuse the gate activation for inference-only elementwise multiplication."""

    def forward(self, hidden_states):
        if self.training or torch.is_grad_enabled():
            return super().forward(hidden_states)
        gate = self.act_fn(self.gate_proj(hidden_states))
        gate.mul_(self.up_proj(hidden_states))
        return self.down_proj(gate)


class ResKVAttention(LlamaAttention):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(head_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(head_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        cos, sin = position_embeddings
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
        is_decode = layer is not None and layer.length > 0
        if is_decode:
            if shape[1] != 1:
                raise ValueError("only single-token continuation is supported")
            rk, rv = layer.update(kr, v, {"raw_keys": k})
            if layer.compressed is not None:
                output = attention(
                    q,
                    layer.compressed,
                    position=layer.length - 1,
                    rope_theta=self.config.rope_theta,
                    inv_freq=self.rope_frequencies if layer.config.rope_mode == "model" else None,
                    scaling=self.scaling,
                    recent=(qr, rk, rv),
                )
            else:
                output = F.scaled_dot_product_attention(qr, rk, rv, enable_gqa=True, scale=self.scaling)
        else:
            # Attend to dense K/V before compressing the cache.
            output = F.scaled_dot_product_attention(
                qr,
                kr,
                v,
                attn_mask=attention_mask,
                is_causal=attention_mask is None and shape[1] > 1,
                enable_gqa=True,
                scale=self.scaling,
            )
            if layer is not None:
                layer.update(kr, v, {"raw_keys": k})
        output = output.transpose(1, 2).reshape(*shape, -1).contiguous()
        return self.o_proj(output), None


class ResKVForCausalLM(LlamaForCausalLM):
    """Drop-in Llama checkpoint loading with explicit residual-cache settings.

    The supported release targets Transformers 4.57.x and FP16 NVIDIA CUDA
    inference with 128D heads and GQA ratios 1/4/8. Equal-length, unpadded
    batches are supported; padded generation is rejected explicitly.
    """

    def __init__(self, config, kernel_config: ResKVKernelConfig | None = None):
        if config.model_type != "llama":
            raise ValueError("ResKV CUDA inference requires a Llama configuration")
        settings = _model_settings(config, kernel_config, ResKVKernelConfig, "reskv_kernel_config")
        if getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) != 128:
            raise ValueError("ResKV requires head_dim=128")
        if config.num_key_value_heads < 1 or config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("query heads must be a positive integer multiple of KV heads")
        ratio = config.num_attention_heads // config.num_key_value_heads
        if ratio not in (1, 4, 8):
            raise ValueError("ResKV supports GQA ratios 1, 4, and 8")
        rope = (getattr(config, "rope_scaling", None) or {}).get("rope_type", "default")
        if settings.rope_mode == "model" and rope not in ("default", "llama3", "linear"):
            raise ValueError("this release supports static default, llama3, and linear RoPE")
        # SDPA uses PyTorch FlashAttention where supported and handles native GQA.
        config._attn_implementation = "sdpa"
        super().__init__(config)
        self.kernel_config = settings
        for idx, layer in enumerate(self.model.layers):
            layer.self_attn = ResKVAttention(config, idx)
            layer.mlp = ResKVMLP(config)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=1,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if labels is not None or self.training:
            raise ValueError("ResKV is an inference implementation; call model.eval()")
        if past_key_values is not None and not isinstance(past_key_values, ResKVCache):
            raise ValueError("past_key_values must be a ResKVCache")
        if use_cache and past_key_values is None:
            past_key_values = ResKVCache(self.kernel_config, len(self.model.layers))
        if attention_mask is not None and not bool(attention_mask.eq(1).all()):
            raise ValueError("ResKV requires equal-length unpadded batches")
        if position_ids is not None or cache_position is not None:
            length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            seen = 0 if past_key_values is None else past_key_values.get_seq_length()
            expected = torch.arange(seen, seen + length, device=device)
            for positions in (position_ids, cache_position):
                if positions is not None and (
                    positions.device != device or not bool(positions.eq(expected).all())
                ):
                    raise ValueError("ResKV positions must be contiguous and start at zero")
        frequency = self.model.rotary_emb.inv_freq
        if self.kernel_config.rope_mode == "model" and frequency.dtype != torch.float32:
            # Module.half() also casts buffers; recompute frequencies rather
            # than promoting rounded FP16 values, keeping save/load identical.
            rotary = self.model.rotary_emb
            frequency, rotary.attention_scaling = rotary.rope_init_fn(self.config, device=torch.device("cpu"))
            frequency = frequency.to(rotary.inv_freq.device)
            rotary.register_buffer("inv_freq", frequency, persistent=False)
            rotary.original_inv_freq = frequency
        for layer in self.model.layers:
            layer.self_attn.rope_frequencies = frequency
        # Default to last-token logits; explicit 0/Tensor requests retain the
        # standard Transformers meaning for likelihood/research workflows.
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
