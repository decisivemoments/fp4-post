"""Inference-only Qwen2 decoder-layer wrapper for RMSNorm/NVFP4 groups."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .native_nvfp4 import NativeFullNVFP4Linear, PackedMeanActivation


class FusedQwen2DecoderLayerNVFP4(nn.Module):
    """Fuse each Qwen2 RMSNorm directly into its native-linear consumers.

    The wrapped layer retains Qwen's attention, RoPE, residual, SiLU, and
    output projection behavior.  Only Q/K/V and Gate/Up consume a packed
    activation directly, eliminating their normalized BF16 input tensor.
    Unsupported shapes fall back to the original decoder-layer forward.
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    @staticmethod
    def _all_native(*modules: nn.Module) -> bool:
        return all(isinstance(module, NativeFullNVFP4Linear) for module in modules)

    def _can_fuse(self, hidden_states: torch.Tensor) -> bool:
        rows = hidden_states.numel() // hidden_states.shape[-1]
        hidden = hidden_states.shape[-1]
        attn = self.layer.self_attn
        mlp = self.layer.mlp
        return (
            hidden_states.dtype == torch.bfloat16
            and rows % 32 == 0
            and hidden % 32 == 0
            and hidden <= 4096
            and self._all_native(attn.q_proj, attn.k_proj, attn.v_proj)
            and self._all_native(mlp.gate_proj, mlp.up_proj)
        )

    @staticmethod
    def _pack_rmsnorm(
        hidden_states: torch.Tensor, norm: nn.Module
    ) -> PackedMeanActivation:
        return PackedMeanActivation.from_rmsnorm_tensor(
            hidden_states,
            norm.weight,
            float(norm.variance_epsilon),
        )

    @staticmethod
    def _native_projection(
        projection: NativeFullNVFP4Linear,
        shape_carrier: torch.Tensor,
        packed: PackedMeanActivation,
    ) -> torch.Tensor:
        return projection.forward_from_packed_activation(shape_carrier, packed)

    def _attention_from_packed(
        self,
        shape_carrier: torch.Tensor,
        packed: PackedMeanActivation,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Keep this in lockstep with the installed Qwen2Attention.forward.
        # Import lazily so the native linear package remains usable without
        # Transformers for isolated kernel tests.
        from transformers.models.qwen2 import modeling_qwen2 as qwen2

        attn = self.layer.self_attn
        input_shape = shape_carrier.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query_states = self._native_projection(attn.q_proj, shape_carrier, packed)
        key_states = self._native_projection(attn.k_proj, shape_carrier, packed)
        value_states = self._native_projection(attn.v_proj, shape_carrier, packed)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qwen2.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx
            )
        attention_interface = qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, qwen2.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output), attn_weights

    def _mlp_from_packed(
        self,
        shape_carrier: torch.Tensor,
        packed: PackedMeanActivation,
    ) -> torch.Tensor:
        mlp = self.layer.mlp
        gate = self._native_projection(mlp.gate_proj, shape_carrier, packed)
        up = self._native_projection(mlp.up_proj, shape_carrier, packed)
        return mlp.down_proj(mlp.act_fn(gate) * up)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self._can_fuse(hidden_states) or position_embeddings is None:
            return self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        residual = hidden_states
        input_packed = self._pack_rmsnorm(
            hidden_states, self.layer.input_layernorm
        )
        hidden_states, _ = self._attention_from_packed(
            residual,
            input_packed,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        mlp_packed = self._pack_rmsnorm(
            hidden_states, self.layer.post_attention_layernorm
        )
        hidden_states = self._mlp_from_packed(residual, mlp_packed)
        return residual + hidden_states


def fuse_qwen2_decoder_layer_norms(layer: nn.Module) -> nn.Module:
    """Wrap a converted Qwen2 decoder layer with the RMSNorm fusion path."""
    return FusedQwen2DecoderLayerNVFP4(layer)
