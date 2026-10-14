"""Analytic FLOP estimates for BERT/RoBERTa-style encoders.

``linear`` reproduces what torch.profiler attributes to aten::addmm (all dense layers, padding
tokens included); ``attention`` is the QK^T and AV matmul cost that the profiler does not see
when attention runs through fused scaled_dot_product_attention kernels. Both use the
2 FLOPs per multiply-accumulate convention used by torch.profiler.
"""

from __future__ import annotations


def encoder_flops(config, batch_widths: list[tuple[int, int]]) -> dict:
    """batch_widths: list of (batch_size, padded_width). Returns GFLOPs for ONE forward pass."""
    d = config.hidden_size
    dff = config.intermediate_size
    layers = config.num_hidden_layers
    per_token_linear = layers * (8 * d * d + 4 * d * dff)
    head = 2 * d * d + 2 * d * config.num_labels  # RoBERTa head on the <s> token only
    linear = attention = 0
    for bs, width in batch_widths:
        linear += bs * width * per_token_linear + bs * head
        attention += bs * layers * 4 * width * width * d
    return {"linear_gflops": linear / 1e9, "attention_gflops": attention / 1e9,
            "total_gflops": (linear + attention) / 1e9}
