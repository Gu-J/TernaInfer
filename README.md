# TernaInfer

This repository is tailored for use with the [BitNet-b1.58-2B-4T](https://arxiv.org/abs/2504.12285) model.

## Usage

Build the kernel:

```bash
# Build libternaspmm.so
# Note: Tested on NVIDIA RTX A6000 GPU (sm_86), uses CUTLASS 2.x Epilogue Visitor
cd kernels
git clone -b v4.3.4 --depth 1 https://github.com/NVIDIA/cutlass.git
bash compile.sh
cd ..

# Run a simple test
./tests/test_kernel 8 2560 2560 45 7

```

End-to-end inference:

```bash
# (Recommended) Create a new conda environment
conda create --name TernaInfer "python<3.13"
conda activate TernaInfer

# Install dependencies
pip install -r requirements.txt

# Download and convert the BitNet-b1.58-2B model
mkdir checkpoints
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./checkpoints/bitnet-b1.58-2B-4T-bf16
python ./convert/convert_TernaInfer.py --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors --output_dir ./checkpoints --model_name 2B

# Inference
python ./generate.py ./checkpoints/ --interactive --chat_format

# Compare output with the original BF16 model
python ./generate.py ./checkpoints/ --interactive --chat_format --BF16

```

## Accuracy

Due to the use of custom GPU kernels, TernaSpMM may introduce minor rounding differences compared to BF16 GEMM. As a result, TernaInfer can produce slightly different outputs from BF16 inference in some cases. These differences are small and lead to only marginal variations in benchmark scores on the following datasets. Evaluation scripts are provided in the `/tests` directory. 

| Method      | WikiText (PPL)   | ARC-Challenge          | HellaSwag         | MMLU         |
|-------------|------------------|------------------------|-------------------|--------------|
| BF16        | 16.1948          | 51.02                  | 50.71             | 53.38        |
| TernaInfer  | 16.1959          | 50.77                  | 50.69             | 53.27        |
| Δ (%)       | 0.0006%          | 0.5096%                | 0.0394%           | 0.2061%      |

## Acknowledgements

This project includes code modified from the following projects:

- https://github.com/microsoft/BitNet/tree/main/gpu  
- https://github.com/HPMLL/SpInfer_EuroSys25
- https://github.com/NVIDIA/cutlass

We thank the authors for their excellent work.