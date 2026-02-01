import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

import model as fast
from tokenizer import Tokenizer


def load_fast_model(ckpt_dir, device="cuda", BF16=False):
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    tokenizer = Tokenizer("./tokenizer.model")

    if BF16:
        model_args = fast.ModelArgs(use_kernel=False)
        checkpoint = torch.load(f"{ckpt_dir}/model_bf16.pt", map_location="cpu")
    else:
        model_args = fast.ModelArgs(use_kernel=True)
        ternary = torch.load(f"{ckpt_dir}/model_TernaInfer_TernaryWeights.pt", map_location="cpu")
        non_ternary = torch.load(f"{ckpt_dir}/model_embeddings_and_norms_non_ternary.pt", map_location="cpu")
        checkpoint = ternary | non_ternary

    model = fast.Transformer(model_args, checkpoint).to(device).eval()
    return model, tokenizer, model_args


@torch.inference_mode()
def evaluate_ppl(model, tokenizer, model_args, seqlen=2048):
    device = "cuda"

    print("Loading WikiText2...")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])

    # print("Loading Penn Treebank...")
    # with open("data/ptb/test.txt", 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    # text = "\n\n".join([line.strip() for line in lines if line.strip()])  

    # print("Loading LAMBADA...")
    # with open("data/lambada/test.txt", 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    # text = "\n\n".join([line.strip() for line in lines if line.strip()])

    tokens = tokenizer.encode(text, bos=False, eos=False)
    tokens = torch.tensor(tokens, dtype=torch.int32, device=device)

    nsamples = tokens.numel() // seqlen
    tokens = tokens[: nsamples * seqlen].view(nsamples, seqlen)

    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0
    total_tokens = 0

    print(f"Evaluating {nsamples} segments...")

    for i in tqdm(range(nsamples)):
        segment = tokens[i]  # [T]

        cache = fast.make_cache(
            args=model_args,
            length=seqlen,
            device=device,
        )

        bias = AttnBias.from_seqlens(
            q_seqlen=[seqlen],
            kv_seqlen=[seqlen],
            kv_padding=seqlen,
        )
        bias.q_seqinfo.to(device)
        bias.k_seqinfo.to(device)

        logits = model.forward_with_attn_bias(
            token_values=segment,
            attn_bias=bias,
            cache=cache,
        )

        # logits: [T, vocab]
        shift_logits = logits[:-1].float()
        shift_labels = segment[1:].long()

        loss = loss_fct(shift_logits, shift_labels)
        total_loss += loss
        total_tokens += seqlen - 1

    ppl = torch.exp(total_loss / total_tokens)
    return ppl.item()


if __name__ == "__main__":
    ckpt = "./checkpoints"

    model, tokenizer, model_args = load_fast_model(ckpt, BF16=False)
    ppl = evaluate_ppl(model, tokenizer, model_args)

    print(f"PPL: {ppl:.2f}")
