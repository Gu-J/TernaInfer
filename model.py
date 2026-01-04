# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# 
# Modified by TernaInfer Authors, 2026


from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import gc
from torch import nn
from torch.nn import functional as F
import ctypes
from xformers.ops import RMSNorm, fmha, rope_padded
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

TernaSpMM_lib = ctypes.CDLL('./kernels/libternaspmm.so')

@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = False
    max_matrix_elements= 13824 * 2560


_GLOBAL_WORKSPACE = None
def get_global_workspace(device='cuda'):
    global _GLOBAL_WORKSPACE
    if _GLOBAL_WORKSPACE is None:
        _GLOBAL_WORKSPACE = torch.empty(ModelArgs().max_matrix_elements, dtype=torch.int8, device=device)
    return _GLOBAL_WORKSPACE


def TernaInfer_linear(  input, 
                        compressed_val, TileOffsets_global,TileOffsets_median, bitmap,
                        workspace,
                        s, ws, ws_num,
                        out_features):
    out_shape = list(input.shape)
    out_shape[-1] = out_features

    stream = torch.cuda.current_stream()

    M = input.shape[0]
    if len(out_shape) == 3: 
        M *= input.shape[1]
    N = out_features
    K = input.shape[-1]

    SPLIT_K=7

    # --- Padding ---
    M_padded = ((M + 7) // 8) * 8  
    pad_rows = M_padded - M
    if pad_rows > 0:
        input_padded = torch.cat([
            input.reshape(M, K),
            torch.empty((pad_rows, K), dtype=input.dtype, device=input.device)
        ], dim=0)
    else:
        input_padded = input.reshape(M, K)

    ret_padded = torch.empty((M_padded,N), dtype=torch.bfloat16, device=input.device)

    TernaSpMM_lib.bitlinear_TernaSpMM(
        ctypes.c_void_p(input_padded.data_ptr()),
        ctypes.c_void_p(compressed_val.data_ptr()),
        ctypes.c_void_p(TileOffsets_global.data_ptr()),
        ctypes.c_void_p(TileOffsets_median.data_ptr()),
        ctypes.c_void_p(bitmap.data_ptr()),
        ctypes.c_void_p(ret_padded.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_int(ws_num),
        ctypes.c_int(M_padded), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_int(SPLIT_K),ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(stream.cuda_stream)
    )
    
    ret=ret_padded[:M,:].reshape(*out_shape)
    return ret

LayerCache = Tuple[torch.Tensor, torch.Tensor]

class SpBitLinear(nn.Module):
    in_features: int
    out_features: int
    weight_compressed_val: torch.Tensor
    weight_TileOffsets_global: torch.Tensor
    weight_TileOffsets_median: torch.Tensor
    weight_bitmap: torch.Tensor
    weight_scale: torch.Tensor

    def __init__(self, in_features:int, out_features:int, dic: dict, key: str,bias: bool = False, ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_compressed_val = torch.nn.Parameter(dic[key+".weight_compressed_val"].cuda(), requires_grad=False)
        self.weight_TileOffsets_global = torch.nn.Parameter(dic[key+".weight_TileOffsets_global"].cuda(), requires_grad=False)
        self.weight_TileOffsets_median = torch.nn.Parameter(dic[key+".weight_TileOffsets_median"].cuda(), requires_grad=False)
        self.weight_bitmap = torch.nn.Parameter(dic[key+".weight_bitmap"].cuda(), requires_grad=False)
        ws = dic[key+".weight_scale"]
        self.weight_scale = torch.nn.Parameter(ws[ws != 0].clone().cuda(), requires_grad=False)
        self.ws_num = len(self.weight_scale)

    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        return TernaInfer_linear( 
            input, 
            self.weight_compressed_val,self.weight_TileOffsets_global,self.weight_TileOffsets_median,self.weight_bitmap,
            get_global_workspace(),
            s, self.weight_scale, self.ws_num,
            self.out_features
        )

class BF16BitLinear(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features:int, out_features:int, dic: dict, key: str,bias: bool = False, ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(dic[key+".weight"].cuda(), requires_grad=False)

    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        input = self.quant_input(input)
        return F.linear(input, self.weight)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
        use_kernel: bool,
        dic: dict,
        layer_prefix: str
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads

        Linear = SpBitLinear if use_kernel else BF16BitLinear

        self.wqkv = Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            dic,
            layer_prefix+".wqkv",
            bias=False,
        )
        self.wo = Linear(
            self.n_local_heads * head_dim,
            dim,
            dic,
            layer_prefix+".wo",
            bias=False,
        )

        self.attn_sub_norm = RMSNorm(dim, norm_eps)
        with torch.no_grad():
            self.attn_sub_norm.weight.copy_(dic[layer_prefix+".attn_sub_norm.weight"])


    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:

        xqkv = self.wqkv(x)
        xq = xqkv[:, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, 1)

        output_shape = xq.shape
        heads_per_group = self.n_local_heads // self.n_local_kv_heads
        xq = xq.view(
            1, xq.shape[0], self.n_local_kv_heads, heads_per_group, self.head_dim
        )
        xk = xk.view(1, xk.shape[0], self.n_local_kv_heads, 1, self.head_dim)
        # xq = rearrange(xq, 'b (g h l d) -> 1 b h g (d l)', g=heads_per_group, h=self.n_local_kv_heads, d=self.head_dim // 2, l=2)
        # xk = rearrange(xk, 'b (g l d) -> 1 b g 1 (d l)', g=self.n_local_kv_heads, d=self.head_dim // 2)
        xv = xv.view(1, xv.shape[0], self.n_local_kv_heads, 1, self.head_dim)
        cache_k, cache_v = cache

        xq = rope_padded(
            xq=xq,
            xk=xk,
            xv=xv,
            cache_k=cache_k,
            cache_v=cache_v,
            attn_bias=attn_bias,
            theta=self.rope_theta,
        )

        output = fmha.memory_efficient_attention_forward(
            xq, cache_k, cache_v, attn_bias, op = fmha.flash.FwOp
        )

        output = output.reshape(output_shape)
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output

@torch.compile
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        use_kernel: bool,
        dic: dict,
        layer_prefix: str
    ):
        super().__init__()

        Linear = SpBitLinear if use_kernel else BF16BitLinear

        self.w13 = Linear(
            dim,
            2 * hidden_dim,
            dic,
            layer_prefix+".w13",
            bias=False,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            dic,
            layer_prefix+".w2",
            bias=False,
        )
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)
        with torch.no_grad():
            self.ffn_sub_norm.weight.copy_(dic[layer_prefix+".ffn_sub_norm.weight"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)
        output = self.w2(inner)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, dic: dict, layer_prefix: str):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        assert args.n_heads % n_kv_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
            dic=dic,
            layer_prefix=layer_prefix+".attention"
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
            dic=dic,
            layer_prefix=layer_prefix+".feed_forward"
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        with torch.no_grad():
            self.attention_norm.weight.copy_(dic[layer_prefix+".attention_norm.weight"])
            self.ffn_norm.weight.copy_(dic[layer_prefix+".ffn_norm.weight"])

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:
        h = x + self.attention.forward(
            self.attention_norm(x),
            cache,
            attn_bias,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, dic: dict):
        super().__init__()
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
        )

        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            self.layers.append(TransformerBlock(args,dic,"layers."+str(i)))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        with torch.no_grad():
            self.norm.weight.copy_(dic["norm.weight"])

        from torch.nn.utils import skip_init
        self.output = skip_init(
            nn.Linear,
            args.dim,
            args.vocab_size,
            bias=False,
        )

        with torch.no_grad():
            self.tok_embeddings.weight.copy_(dic["tok_embeddings.weight"])
        
        self.output.weight=self.tok_embeddings.weight

        gc.collect()
        torch.cuda.empty_cache()


    @torch.no_grad()
    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias: AttnBias,
        cache: list[LayerCache],
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], attn_bias)

        logits = self.output(self.norm(h))
        return logits.float()

    def forward(
        self,
        token_values: torch.Tensor,
        token_lengths: torch.Tensor,
        start_pos: torch.Tensor,
        cache: list[LayerCache],
        kv_padding: int,
    ) -> torch.Tensor:
        attn_bias = AttnBias.from_seqlens(
            q_seqlen=token_lengths.tolist(),
            kv_seqlen=(start_pos + token_lengths).tolist(),
            kv_padding=kv_padding,
        )
        return self.forward_with_attn_bias(token_values, attn_bias, cache)


def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[LayerCache]:
    """
    Allocate a cache to be used with the Transformer module.

    Args:
        args (ModelArgs): the model configuration.
        length (int): per layer cache size.
            It is usually budgeted as ``max_batch * max_seq``
        device (torch.device, optional): the device on which
            the cache should be allocated.
        n_layers (int, optional): the number of layers to
            allocate a cache for (defaults to the model
            settings).
        dtype (torch.dtype, optional): the dtype to use for
            cache entries (defaults to the default dtype).

    Returns:
        The cache object to pass to ``Tranformer.forward``.
    """

    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = args.n_heads
    n_local_kv_heads = n_kv_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_local_kv_heads, 1, head_dim)
    heads_per_group = args.n_heads // n_kv_heads
    expansion = (-1, -1, -1, heads_per_group, -1)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: list[LayerCache], length: int) -> list[LayerCache]:
    """
    Take a prefix view of a larger cache.

    The original cache object remains of identical size and valid
    after the shrinked alias has been used. This function is useful
    when a cache was allocated for a larger batch size than what is
    necessary.

    Args:
        cache: the cache to take a view in.
        length (int): the desired length

    Returns:
        A view in the input cache object.
    """

    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length

    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]