"""ONNX-friendly torch-обёртки над Qwen3.5-VL (vision tower и language model).

Задача — выразить forward-pass без data-dependent control flow и без KV-кэша:
  * языковая часть: prefill всего промпта + hidden state последнего токена (генерации нет);
  * Gated DeltaNet (linear attention) реализован как chunked-алгоритм: внутри чанка — матрицы,
    между чанками — `torch.scan` (экспортируется в ONNX Loop/Scan с динамической длиной);
  * обратная матрица (I - A)^-1 для строго нижнетреугольной A считается разложением
    (I+A)(I+A^2)(I+A^4)... — это точно для нильпотентной A и даёт 6 матмулов вместо 63 шагов.

Всё, что можно посчитать вне графа (rope cos/sin, position ids, интерполяция pos-embedding,
embedding lookup), считается в numpy на стороне сервиса.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._higher_order_ops.scan import scan

USE_SDPA = True  # True -> F.scaled_dot_product_attention (ONNX Attention op, opset>=23)
USE_SCAN = True  # False -> python-цикл (для проверки в eager)
CHUNK = 32  # длина чанка; промпт паддится вправо до кратного CHUNK


def rms_norm_zc(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return out * (1.0 + weight)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


# --------------------------------------------------------------------------- Gated DeltaNet

def chunk_gated_delta_rule(q, k, v, g, beta, num_chunks: int | torch.SymInt):
    """q,k: [1,H,T,Dk]; v: [1,H,T,Dv]; g,beta: [1,H,T]. T = num_chunks * CHUNK. Возвращает [1,H,T,Dv].

    Математика идентична `torch_chunk_gated_delta_rule` из transformers (без initial_state).
    """
    _, H, T, Dk = q.shape
    Dv = v.shape[-1]
    q = l2norm(q) * (Dk ** -0.5)
    k = l2norm(k)

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    q, k, k_beta, v_beta = [x.reshape(1, H, num_chunks, CHUNK, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    g = g.reshape(1, H, num_chunks, CHUNK)

    cum = g.cumsum(dim=3)                                   # [1,H,C,CHUNK]
    diff = cum.unsqueeze(4) - cum.unsqueeze(3)              # [1,H,C,CHUNK,CHUNK] : cum_i - cum_j
    tri = torch.ones(CHUNK, CHUNK, dtype=torch.bool).tril(0)  # i >= j
    # ВАЖНО: маска до exp, иначе exp(+большое) -> inf, а inf*0 = nan.
    decay = torch.where(tri, diff, torch.full_like(diff, float("-inf"))).exp()

    strict = torch.ones(CHUNK, CHUNK, dtype=torch.bool).tril(-1)
    ut = (k_beta @ k.transpose(-1, -2)) * decay
    A = torch.where(strict, -ut, torch.zeros_like(ut))      # строго нижнетреугольная
    A = A.double()
    eye = torch.eye(CHUNK, dtype=A.dtype)
    inv = eye + A
    p = A
    for _ in range(CHUNK.bit_length() - 2):                  # A^2, A^4, ..., A^(CHUNK/2)
        p = p @ p
        inv = inv + inv @ p                                  # (I+A)(I+A^2)(I+A^4)... = sum_{i<CHUNK} A^i
    # sum_{i<CHUNK} A^i == (I - A)^-1 для нильпотентной A (A^CHUNK = 0).

    inv = inv.to(q.dtype)
    intra = (q @ k.transpose(-1, -2)) * decay
    intra = torch.where(strict.logical_or(torch.eye(CHUNK, dtype=torch.bool)), intra, torch.zeros_like(intra))
    new_values = inv @ v_beta
    k_cumdecay = inv @ (k_beta * cum.exp().unsqueeze(-1))

    q_dec = q * cum.exp().unsqueeze(-1)
    k_dec = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]       # [1,H,C,1,1]

    def to_scan(x):  # [1,H,C,...] -> [C,1,H,...]
        return x.movedim(2, 0)

    xs = [to_scan(t) for t in (new_values, k_cumdecay, q_dec, intra, k_dec, chunk_decay)]
    init = torch.zeros(1, H, Dk, Dv, dtype=q.dtype)

    def step(state, xs_t):
        nv, kc, qd, it, kd, cd = xs_t
        v_new = nv - kc @ state
        out = qd @ state + it @ v_new
        state = state * cd + kd.transpose(-1, -2) @ v_new
        return state, out

    if USE_SCAN:
        _, outs = scan(step, init, xs)                       # outs: [C,1,H,CHUNK,Dv]
    else:                                                    # eager-режим для отладки/паритета
        state, ys = init, []
        for i in range(num_chunks):
            state, o = step(state, [x[i] for x in xs])
            ys.append(o)
        outs = torch.stack(ys)
    outs = outs.movedim(0, 2)                                # [1,H,C,CHUNK,Dv]
    return outs.reshape(1, H, T, Dv)


class LinearAttnWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor, num_chunks) -> torch.Tensor:  # x [1,T,D]
        m = self.m
        T = x.shape[1]
        mixed = m.in_proj_qkv(x).transpose(1, 2)             # [1,C,T]
        z = m.in_proj_z(x).reshape(1, T, m.num_v_heads, m.head_v_dim)
        b = m.in_proj_b(x)
        a = m.in_proj_a(x)

        w = m.conv1d.weight                                   # [C,1,K]
        K = w.shape[-1]
        mixed = F.conv1d(mixed, w, None, padding=K - 1, groups=mixed.shape[1])[:, :, :T]
        mixed = F.silu(mixed).transpose(1, 2)                 # [1,T,C]

        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(1, T, m.num_k_heads, m.head_k_dim)
        k = k.reshape(1, T, m.num_k_heads, m.head_k_dim)
        v = v.reshape(1, T, m.num_v_heads, m.head_v_dim)
        if m.num_k_heads != m.num_v_heads:
            r = m.num_v_heads // m.num_k_heads
            q = q.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        beta = b.sigmoid()
        g = -m.A_log.exp() * F.softplus(a + m.dt_bias)        # [1,T,H]

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))      # [1,H,T,D]
        beta = beta.transpose(1, 2)
        g = g.transpose(1, 2)
        o = chunk_gated_delta_rule(q, k, v, g, beta, num_chunks)  # [1,H,T,Dv]
        o = o.transpose(1, 2)                                 # [1,T,H,Dv]

        # RMSNormGated: norm(o) * w * silu(z)
        eps = m.norm.variance_epsilon
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
        o = m.norm.weight * o
        o = o * F.silu(z)
        return m.out_proj(o.reshape(1, T, -1))


class FullAttnWrap(nn.Module):
    def __init__(self, m, eps: float):
        super().__init__()
        self.m = m
        self.eps = eps

    def forward(self, x, cos, sin):  # x [1,T,D]; cos/sin [1,T,R]
        m = self.m
        T = x.shape[1]
        hd = m.head_dim
        qg = m.q_proj(x).view(1, T, -1, hd * 2)
        q, gate = qg[..., :hd], qg[..., hd:]
        gate = gate.reshape(1, T, -1)
        q = rms_norm_zc(q, m.q_norm.weight, self.eps).transpose(1, 2)   # [1,H,T,hd]
        k = rms_norm_zc(m.k_proj(x).view(1, T, -1, hd), m.k_norm.weight, self.eps).transpose(1, 2)
        v = m.v_proj(x).view(1, T, -1, hd).transpose(1, 2)

        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        R = c.shape[-1]
        q = torch.cat([q[..., :R] * c + rotate_half(q[..., :R]) * s, q[..., R:]], dim=-1)
        k = torch.cat([k[..., :R] * c + rotate_half(k[..., :R]) * s, k[..., R:]], dim=-1)

        rep = m.num_key_value_groups
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        if USE_SDPA:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=m.scaling)
        else:
            scores = (q @ k.transpose(-1, -2)) * m.scaling                  # [1,H,T,T]
            idx = torch.arange(T)
            mask = idx.unsqueeze(0) > idx.unsqueeze(1)                       # j > i  -> masked
            scores = scores.masked_fill(mask, float("-inf"))
            o = torch.softmax(scores, dim=-1) @ v                            # [1,H,T,hd]
        o = o.transpose(1, 2).reshape(1, T, -1)
        o = o * torch.sigmoid(gate)
        return m.o_proj(o)


class LanguageModelWrap(nn.Module):
    """inputs_embeds [1,T,D] (T = CHUNK*C) + cos/sin [T,R] + last_index [1] -> hidden [1,D] (после final norm)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        eps = lm.config.rms_norm_eps
        self.eps = eps
        self.blocks = nn.ModuleList()
        for layer in lm.layers:
            if layer.block_type == "linear_attention":
                self.blocks.append(LinearAttnWrap(layer.linear_attn))
            else:
                self.blocks.append(FullAttnWrap(layer.self_attn, eps))

    def forward(self, inputs_embeds, cos, sin, last_index):
        T = inputs_embeds.shape[1]
        num_chunks = T // CHUNK
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        h = inputs_embeds
        for layer, blk in zip(self.lm.layers, self.blocks):
            r = h
            x = rms_norm_zc(h, layer.input_layernorm.weight, self.eps)
            if layer.block_type == "linear_attention":
                x = blk(x, num_chunks)
            else:
                x = blk(x, cos, sin)
            h = r + x
            r = h
            x = rms_norm_zc(h, layer.post_attention_layernorm.weight, self.eps)
            h = r + layer.mlp(x)
        h = rms_norm_zc(h, self.lm.norm.weight, self.eps)
        return h.index_select(1, last_index).squeeze(1)       # [1,D]


# --------------------------------------------------------------------------- Vision

class VisionWrap(nn.Module):
    """Одна картинка за раз (без cu_seqlens): attention по всем патчам.

    pixel_values [N, 1536], interp_indices [N,4] (int64), interp_weights [N,4],
    rot_cos/rot_sin [N, head_dim] -> image_embeds [N/4, out_hidden].
    """

    def __init__(self, vis):
        super().__init__()
        self.v = vis
        pe = vis.patch_embed
        self.register_buffer("pe_w", pe.proj.weight.detach().reshape(pe.embed_dim, -1).clone())
        self.register_buffer("pe_b", pe.proj.bias.detach().clone())
        self.heads = vis.config.num_heads

    def forward(self, pixel_values, interp_indices, interp_weights, rot_cos, rot_sin):
        v = self.v
        h = pixel_values @ self.pe_w.T + self.pe_b                       # == Conv3d patch embed
        pos = (v.pos_embed(interp_indices) * interp_weights.unsqueeze(-1)).sum(1)
        h = h + pos
        N = h.shape[0]
        cos = rot_cos.unsqueeze(1)                                        # [N,1,hd]
        sin = rot_sin.unsqueeze(1)
        for blk in v.blocks:
            x = blk.norm1(h)
            a = blk.attn
            qkv = a.qkv(x).reshape(N, 3, self.heads, -1).permute(1, 0, 2, 3)
            q, k, val = qkv[0], qkv[1], qkv[2]                            # [N,H,hd]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            q, k, val = (t.transpose(0, 1) for t in (q, k, val))          # [H,N,hd]
            att = torch.softmax((q @ k.transpose(-1, -2)) * a.scaling, dim=-1) @ val
            att = att.transpose(0, 1).reshape(N, -1)
            h = h + a.proj(att)
            h = h + blk.mlp(blk.norm2(h))
        m = v.merger
        x = m.norm(h).reshape(-1, m.hidden_size)
        return m.linear_fc2(m.act_fn(m.linear_fc1(x)))
