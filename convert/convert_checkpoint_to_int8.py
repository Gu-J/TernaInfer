import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import torch
from einops import rearrange
from safetensors.torch import save_file

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
def convert_ts_checkpoint(
    *,
    input_path: str = "",
) -> None:

    config = bitnet_ModelArgs()
    print(f"Model config {config.__dict__}")

    def quant_weight_int8(weight):
        s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        new_weight = (weight * s).round().clamp(-1, 1).to(torch.int8)
        new_scale = (1.0 / s).to(torch.bfloat16)
        return new_weight, new_scale.reshape(1)


    merged_result = torch.load(input_path, map_location="cpu", mmap=True)
    int8_result = {}
    zero = torch.zeros(1).to(torch.bfloat16)
    for key, value in merged_result.items():
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

    output_dir = os.path.dirname(input_path)
    print(f"Saving checkpoint to {output_dir}/model_state_int8.pt")
    torch.save(int8_result, f"{output_dir}/model_state_int8.pt")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert TorchScale checkpoint.')
    parser.add_argument('--input', type=str)

    args = parser.parse_args()
    convert_ts_checkpoint(
        input_path=args.input,
    )
