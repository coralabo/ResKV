# SPDX-License-Identifier: Apache-2.0
"""Llama and Qwen3 accuracy models using dense reconstructed, pre-RoPE KV."""

import torch
from transformers import LlamaForCausalLM, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaAttention, eager_attention_forward
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from ..accuracy import ResKVAccuracyCache
from ..config import ResKVConfig, _model_settings


def _rotate(tensor, cos, sin):
    half = tensor.shape[-1] // 2
    rotated = torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
    return tensor * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def _attention(module, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs):
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, module.head_dim)
    q = module.q_proj(hidden_states).view(head_shape)
    k = module.k_proj(hidden_states).view(head_shape)
    if isinstance(module, Qwen3Attention):
        # Compression follows QK normalization and precedes positional encoding.
        q, k = module.q_norm(q), module.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    v = module.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q = _rotate(q, cos, sin)
    if past_key_values is not None:
        prefill = past_key_values.layers[module.layer_idx].get_seq_length() == 0
        k, v = past_key_values.update(k, v, module.layer_idx)
        if not prefill:
            positions = torch.arange(k.shape[-2], device=k.device).unsqueeze(0)
            cos, sin = module.reskv_rotary(v, positions)
    k = _rotate(k, cos, sin)
    interface = (
        eager_attention_forward
        if module.config._attn_implementation == "eager"
        else ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
    )
    output, weights = interface(
        module, q, k, v, attention_mask, dropout=0.0, scaling=module.scaling, **kwargs
    )
    return module.o_proj(output.reshape(*shape, -1).contiguous()), weights


class ResKVLlamaAccuracyAttention(LlamaAttention):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        return _attention(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)


class ResKVQwen3AccuracyAttention(Qwen3Attention):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        return _attention(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)


class _ResKVAccuracyModelMixin:
    def __init__(self, config, reskv_config: ResKVConfig | None = None):
        if config.model_type != self.config_class.model_type:
            raise ValueError(f"{type(self).__name__} requires a {self.config_class.model_type} configuration")
        settings = _model_settings(config, reskv_config, ResKVConfig, "reskv_config")
        rope = (getattr(config, "rope_scaling", None) or {}).get("rope_type", "default")
        if rope not in ("default", "llama3", "linear"):
            raise ValueError("accuracy evaluation supports static default, llama3 and linear RoPE")
        if "sliding_attention" in (getattr(config, "layer_types", None) or []):
            raise ValueError("accuracy evaluation requires full-attention layers")
        if getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if (
            settings.layout == "structured"
            and getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) % 16
        ):
            raise ValueError("4:16 sparsity requires head dimensions divisible by 16")
        super().__init__(config)
        self.reskv_config = settings
        cls = ResKVQwen3AccuracyAttention if config.model_type == "qwen3" else ResKVLlamaAccuracyAttention
        for layer in self.model.layers:
            # Change behavior while preserving projection/norm parameter objects.
            layer.self_attn.__class__ = cls
            # Share model RoPE without registering a second copy of its buffers.
            object.__setattr__(layer.self_attn, "reskv_rotary", self.model.rotary_emb)

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
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if labels is not None or self.training:
            raise ValueError("accuracy compression is for inference accuracy; call model.eval()")
        if use_cache is False:
            raise ValueError("accuracy evaluation requires use_cache=True")
        if past_key_values is not None and not isinstance(past_key_values, ResKVAccuracyCache):
            raise ValueError("past_key_values must be a ResKVAccuracyCache")
        if attention_mask is not None and not bool(attention_mask.eq(1).all()):
            raise ValueError("accuracy evaluation requires unpadded batches")
        length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        seen = 0 if past_key_values is None else past_key_values.get_seq_length()
        if seen and length != 1:
            raise ValueError("accuracy evaluation supports single-token continuation only")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        expected = torch.arange(seen, seen + length, device=device)
        for positions in (position_ids, cache_position):
            if positions is not None and (
                positions.device != device or not bool(positions.eq(expected).all())
            ):
                raise ValueError("accuracy positions must be contiguous and start at zero")
        if past_key_values is None:
            past_key_values = ResKVAccuracyCache(self.reskv_config, len(self.model.layers))
        # Keep frequencies accurate if callers cast the whole model with .half().
        rotary = self.model.rotary_emb
        if rotary.inv_freq.dtype != torch.float32:
            frequency, rotary.attention_scaling = rotary.rope_init_fn(
                self.config, device=rotary.inv_freq.device
            )
            rotary.register_buffer("inv_freq", frequency, persistent=False)
            rotary.original_inv_freq = frequency
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


class ResKVLlamaAccuracyForCausalLM(_ResKVAccuracyModelMixin, LlamaForCausalLM):
    """Evaluate Llama compression error with dense reconstructed, pre-RoPE KV."""


class ResKVQwen3AccuracyForCausalLM(_ResKVAccuracyModelMixin, Qwen3ForCausalLM):
    """Load ordinary Qwen3 weights with native Q/K normalization preserved."""
