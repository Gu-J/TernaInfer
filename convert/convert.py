import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
from torch import nn
import numpy as np
from typing import Tuple, List
import os
import argparse
from multiprocessing import Pool, cpu_count

from convert_utils import compress_weight


def convert_one_weight(args):
    key, weight = args
    compressed_val_h, TileOffsets_global_h, TileOffsets_median_h, bitmap_h = compress_weight(weight)
    return {
        key + '_compressed_val': compressed_val_h,
        key + '_TileOffsets_global': TileOffsets_global_h,
        key + '_TileOffsets_median': TileOffsets_median_h,
        key + '_bitmap': bitmap_h,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TorchScale checkpoint.')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2))
    args = parser.parse_args()

    input_path = args.input
    int8_model = torch.load(input_path, map_location="cpu", mmap=True)
    output_dir = os.path.dirname(input_path)
    print(f"Loaded model from {input_path}")

    # 收集需要并行处理的权重
    tasks = []
    sp_result = {}

    for key, value in int8_model.items():
        if value.dim() == 2 and 'layer' in key:
            tasks.append((key, value))
        elif 'output.weight' not in key:
            sp_result[key] = value.clone()

    print(f"Total weights to convert: {len(tasks)}")
    print(f"Using {args.workers} workers")

    # === 并行处理 ===
    with Pool(processes=args.workers) as pool:
        for i, result_dict in enumerate(pool.imap_unordered(convert_one_weight, tasks), start=1):
            sp_result.update(result_dict)
            key_name = list(result_dict.keys())[0].replace('_compressed_val', '')
            shape = list(dict(tasks)[key_name].shape)
            print(f"[{i}/{len(tasks)}] Converted {key_name} {shape}")

    # === 保存结果 ===
    save_path = f"{output_dir}/model_state_sp.pt"
    print(f"Saving checkpoint to {save_path}")
    torch.save(sp_result, save_path)
