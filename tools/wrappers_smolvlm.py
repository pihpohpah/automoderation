"""ONNX-friendly torch-обёртки над SmolVLM2-256M (Idefics3 vision + connector, VLlama3 language model).

Архитектура стандартная:
  * vision — обычный SigLIP ViT с learned position embedding (без rope, без динамической
    интерполяции): train_vlm.py обучал с `do_image_splitting=False`, поэтому каждая картинка
    ресайзится ровно в image_size x image_size и patch_attention_mask всегда полностью true —
    positional embedding сводится к сложению с таблицей в растровом порядке;
  * язык: VLlama3, то есть обычный Llama с GQA-attention и одномерным RoPE. Нужен только prefill
    целиком (hidden state последнего токена), без KV-кэша.
Поэтому формулы переносятся из transformers напрямую.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * out


class TextWrap(nn.Module):
    """inputs_embeds [1,T,D] + cos/sin [T,head_dim] -> hidden [1,D] последнего токена (после final norm).

    Причинное внимание задаётся явно (`is_causal=True`), без материализации T×T-маски. Сервис
    всегда подаёт один непадденный пример за раз (см. README: расчёт строго по одному запросу),
    поэтому attention_mask не нужна и позиции — обычный arange(T) (без M-RoPE).
    """

    def __init__(self, text_model):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.eps = text_model.config.rms_norm_eps
        n_heads = text_model.config.num_attention_heads
        n_kv_heads = text_model.config.num_key_value_heads
        self.num_kv_groups = n_heads // n_kv_heads
        self.head_dim = getattr(text_model.config, "head_dim", text_model.config.hidden_size // n_heads)

    def forward(self, inputs_embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = inputs_embeds
        c, s = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)  # [1,1,T,hd]
        for layer in self.layers:
            r = h
            x = rms_norm(h, layer.input_layernorm.weight, self.eps)
            T = x.shape[1]
            a = layer.self_attn
            q = a.q_proj(x).view(1, T, -1, self.head_dim).transpose(1, 2)
            k = a.k_proj(x).view(1, T, -1, self.head_dim).transpose(1, 2)
            v = a.v_proj(x).view(1, T, -1, self.head_dim).transpose(1, 2)
            q = q * c + rotate_half(q) * s
            k = k * c + rotate_half(k) * s
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=a.scaling)
            o = o.transpose(1, 2).reshape(1, T, -1)
            h = r + a.o_proj(o)
            r = h
            x = rms_norm(h, layer.post_attention_layernorm.weight, self.eps)
            m = layer.mlp
            h = r + m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))
        h = rms_norm(h, self.norm.weight, self.eps)
        return h[:, -1, :]


class VisionWrap(nn.Module):
    """pixel_values [N,3,image_size,image_size] -> image_embeds [N*tokens_per_image, text_hidden].

    Без do_image_splitting каждая картинка занимает ровно весь патч-грид (image_size/patch_size)^2,
    поэтому patch_attention_mask всегда полностью true: позиционный эмбеддинг — прямое сложение
    с таблицей в растровом порядке (как и даёт `patch_embedding(...).flatten(2).transpose(1,2)`),
    без bucketize/интерполяции. Внимание — полное (bidirectional, без маски: все патчи валидны).
    """

    def __init__(self, vision_model, connector):
        super().__init__()
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.register_buffer("pos_embed", vision_model.embeddings.position_embedding.weight.detach().clone())
        self.layers = vision_model.encoder.layers
        self.post_layernorm = vision_model.post_layernorm
        self.heads = vision_model.config.num_attention_heads
        self.scale = (vision_model.config.hidden_size // vision_model.config.num_attention_heads) ** -0.5
        self.connector = connector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        N = pixel_values.shape[0]
        h = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)  # [N,P,D]
        h = h + self.pos_embed.unsqueeze(0)
        for layer in self.layers:
            r = h
            x = layer.layer_norm1(h)
            a = layer.self_attn
            P = x.shape[1]
            q = a.q_proj(x).view(N, P, self.heads, -1).transpose(1, 2)
            k = a.k_proj(x).view(N, P, self.heads, -1).transpose(1, 2)
            v = a.v_proj(x).view(N, P, self.heads, -1).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            o = o.transpose(1, 2).reshape(N, P, -1)
            h = r + a.out_proj(o)
            r = h
            x = layer.layer_norm2(h)
            h = r + layer.mlp(x)
        h = self.post_layernorm(h)
        feats = self.connector(h)  # [N, tokens_per_image, text_hidden]
        return feats.reshape(-1, feats.shape[-1])
