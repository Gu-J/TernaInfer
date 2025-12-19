import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
import os
import argparse
from multiprocessing import Pool, cpu_count
from convert_utils import compress_weight
from dataclasses import dataclass
from safetensors.torch import load_file
from einops import rearrange
from typing import Optional

transformer_configs = {
    "2B": dict(n_layer=30, n_head=20, dim=2560, vocab_size=128256, n_local_heads=5, intermediate_size=6912),
}

@dataclass
class ModelArgs:
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = n_hidden + (256 - n_hidden % 256) if n_hidden % 256 else n_hidden
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        config = [k for k in transformer_configs if k in name.upper() or k in name]
        assert len(config) == 1, f"Unknown model name: {name}"
        return cls(**transformer_configs[config[0]])

def invert_convert_q(w: torch.Tensor, config: ModelArgs) -> torch.Tensor:
    return rearrange(w, '(h l d) i -> (h d l) i', h=config.n_head, l=2)

def invert_convert_k(w: torch.Tensor, config: ModelArgs) -> torch.Tensor:
    return rearrange(w, '(h l d) i -> (h d l) i', h=config.n_local_heads, l=2)

def convert_back(
    safetensors_path: str,
    model_name: Optional[str] = None,
):
    st_dict = load_file(safetensors_path)

    cfg = ModelArgs.from_name(model_name)
    print(f"Using model configurations: {cfg}")

    recovered: dict = {}

    for layer in range(cfg.n_layer):
        base = f"model.layers.{layer}."

        wq = st_dict[f"{base}self_attn.q_proj.weight"]
        wk = st_dict[f"{base}self_attn.k_proj.weight"]
        wv = st_dict[f"{base}self_attn.v_proj.weight"]

        wq = invert_convert_q(wq, cfg)
        wk = invert_convert_k(wk, cfg)

        wqkv = torch.cat([wq, wk, wv], dim=0)
        recovered[f"layers.{layer}.attention.wqkv.weight"] = wqkv

        recovered[f"layers.{layer}.attention.wo.weight"] = st_dict[f"{base}self_attn.o_proj.weight"]

        recovered[f"layers.{layer}.attention_norm.weight"] = st_dict[f"{base}input_layernorm.weight"]
        recovered[f"layers.{layer}.ffn_norm.weight"] = st_dict[f"{base}post_attention_layernorm.weight"]
        recovered[f"layers.{layer}.attention.attn_sub_norm.weight"] = st_dict[f"{base}self_attn.attn_sub_norm.weight"]
        recovered[f"layers.{layer}.feed_forward.ffn_sub_norm.weight"] = st_dict[f"{base}mlp.ffn_sub_norm.weight"]

        gate = st_dict[f"{base}mlp.gate_proj.weight"]
        up   = st_dict[f"{base}mlp.up_proj.weight"]
        w13  = torch.cat([gate, up], dim=0)
        recovered[f"layers.{layer}.feed_forward.w13.weight"] = w13

        recovered[f"layers.{layer}.feed_forward.w2.weight"] = st_dict[f"{base}mlp.down_proj.weight"]

    recovered["tok_embeddings.weight"] = st_dict["model.embed_tokens.weight"]
    recovered["output.weight"]         = st_dict["model.embed_tokens.weight"]
    recovered["norm.weight"]           = st_dict["model.norm.weight"]

    return recovered


@dataclass
class bitnet_ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = False

@torch.inference_mode()
def convert_bitnet_checkpoint_to_bf16(tmp_model):
    def quant_weight_bf16(weight):
        s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        return (weight * s).round().clamp(-1, 1) / s

    bf16_result = {}
    config = bitnet_ModelArgs()
    for key, value in tmp_model.items():
            if 'wqkv' in key:
                wq = value[:config.dim]
                wk = value[
                    config.dim :
                    config.dim // config.n_heads * config.n_kv_heads + config.dim
                ]
                wv = value[
                    config.dim // config.n_heads * config.n_kv_heads + config.dim :
                ]

                wq = quant_weight_bf16(wq)
                wk = quant_weight_bf16(wk)
                wv = quant_weight_bf16(wv)

                bf16_result[key] = torch.cat([wq, wk, wv], dim=0)

            elif 'w13' in key:
                w1 = value[:config.ffn_dim]
                w3 = value[config.ffn_dim:]

                w1 = quant_weight_bf16(w1)
                w3 = quant_weight_bf16(w3)

                bf16_result[key] = torch.cat([w1, w3], dim=0)

            elif 'w2' in key or 'wo' in key:
                bf16_result[key] = quant_weight_bf16(value)

            else:
                bf16_result[key] = value.clone()

    return bf16_result


@torch.inference_mode()
def convert_bitnet_checkpoint_to_int8(tmp_model):

    config = bitnet_ModelArgs()
    print(f"Model config {config.__dict__}")

    def quant_weight_int8(weight):
        s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        new_weight = (weight * s).round().clamp(-1, 1).to(torch.int8)
        new_scale = (1.0 / s).to(torch.bfloat16)
        return new_weight, new_scale.reshape(1)

    int8_result = {}
    zero = torch.zeros(1).to(torch.bfloat16)
    for key, value in tmp_model.items():
        if 'wqkv' in key:
            wq = value[:config.dim]
            wk = value[config.dim:config.dim // config.n_heads * config.n_kv_heads + config.dim]
            wv = value[config.dim // config.n_heads * config.n_kv_heads + config.dim:]
            wq_weight, wa_scale = quant_weight_int8(wq)
            wk_weight, wb_scale = quant_weight_int8(wk)
            wv_weight, wc_scale = quant_weight_int8(wv)
            wqkv_weight = torch.cat([wq_weight, wk_weight, wv_weight], dim=0)
            wqkv_scale = torch.cat([wa_scale, wb_scale, wc_scale, zero], dim=0)
            int8_result[key] = wqkv_weight
            int8_result[key.replace('weight', 'weight_scale')] = wqkv_scale

        elif 'w13' in key:
            w1 = value[:config.ffn_dim]
            w3 = value[config.ffn_dim:]
            w1_weight, w1_scale = quant_weight_int8(w1)
            w3_weight, w3_scale = quant_weight_int8(w3)
            w13_weight = torch.cat([w1_weight, w3_weight], dim=0)
            w13_scale = torch.cat([w1_scale, w3_scale, zero, zero], dim=0)
            int8_result[key] = w13_weight
            int8_result[key.replace('weight', 'weight_scale')] = w13_scale

        elif 'w2' in key or 'wo' in key:
            weight, scale = quant_weight_int8(value)
            scale = torch.cat([scale, zero, zero, zero], dim=0)
            int8_result[key] = weight
            int8_result[key.replace('weight', 'weight_scale')] = scale

        elif 'output.weight' not in key:
            int8_result[key] = value.clone()

    return int8_result

def convert_one_weight(args):
    key, weight = args
    compressed_val_h, TileOffsets_global_h, TileOffsets_median_h, bitmap_h = compress_weight(weight)
    return {
        key + '_compressed_val': compressed_val_h,
        key + '_TileOffsets_global': TileOffsets_global_h,
        key + '_TileOffsets_median': TileOffsets_median_h,
        key + '_bitmap': bitmap_h,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--safetensors_file", type=str, required=True,
        help="Path to input .safetensors file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/",
        help="Path to output .pt file"
    )
    parser.add_argument(
        "--model_name", type=str, default="2B",
        help="Model configuration name to use (e.g. 2B)"
    )
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2))
    args = parser.parse_args()
    save_path = f"{args.output_dir}"

    tmp_model=convert_back(
        safetensors_path=args.safetensors_file,
        model_name=args.model_name,
    )

    bf16_model=convert_bitnet_checkpoint_to_bf16(tmp_model)

    print(f"Saving bf16 checkpoint")
    torch.save(bf16_model, save_path+"/model_bf16.pt")
    
    int8_model=convert_bitnet_checkpoint_to_int8(tmp_model)

    tasks = []
    ternary_result = {}
    non_ternary={}

    for key, value in int8_model.items():
        if value.dim() == 2 and 'layer' in key:
            tasks.append((key, value))
        elif 'output.weight' not in key:
            non_ternary[key] = value.clone()

    print(f"Total weights to convert: {len(tasks)}")
    print(f"Using {args.workers} workers")

    # for key,value in non_ternary.items():
    #     print(key, value.shape,value.dtype)

    with Pool(processes=args.workers) as pool:
        for i, result_dict in enumerate(pool.imap_unordered(convert_one_weight, tasks), start=1):
            ternary_result.update(result_dict)
            key_name = list(result_dict.keys())[0].replace('_compressed_val', '')
            shape = list(dict(tasks)[key_name].shape)
            print(f"[{i}/{len(tasks)}] Converted {key_name} {shape}")


    print(f"Saving checkpoint to {save_path}")
    torch.save(ternary_result, save_path+"/model_TernaInfer_TernaryWeights.pt")
    torch.save(non_ternary, save_path+"/model_embeddings_and_norms_non_ternary.pt")
