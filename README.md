# TernaInfer

This repository is tailored for use with the [BitNet-b1.58-2B-4T](https://arxiv.org/abs/2504.12285) model.

## Usage

Build the kernel:

```bash
# Build libternaspmm.so
# Note: The implementation has been tested on an NVIDIA RTX A6000 GPU and compiled with -arch=sm_86
cd kernels
bash compile.sh
cd ..

# Run a simple test
./kernels/test_kernel 8 2560 2560 45 7

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
python ./convert/convert_safetensors.py --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors --output checkpoints/model_bf16.pt --model_name 2B
python convert/convert_TernaInfer.py --input checkpoints/model_bf16.pt

# Inference
python ./generate.py ./checkpoints/ --interactive --chat_format
```