"""Perplexity evaluation via lm-eval harness (wikitext, lambada, etc.)."""

# Usage: python test_ppl.py --ckpt_dir ./checkpoints --both --tasks wikitext

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import model as fast
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 4096

PPL_TASKS = ["wikitext"]


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class TernaInferLM:
    def __init__(self, ckpt_dir: str, bf16: bool = False):
        from lm_eval.api.model import LM
        self.__class__ = type("TernaInferLM", (TernaInferLM, LM), {})
        LM.__init__(self)

        self._max_seq_len = MAX_SEQ_LEN

        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)

        ckpt = Path(ckpt_dir)
        if bf16:
            model_args = fast.ModelArgs(use_kernel=False)
            checkpoint = torch.load(ckpt / "model_bf16.pt", map_location="cpu")
        else:
            model_args = fast.ModelArgs(use_kernel=True)
            checkpoint = (
                torch.load(ckpt / "model_TernaInfer_TernaryWeights.pt", map_location="cpu")
                | torch.load(ckpt / "model_embeddings_and_norms_non_ternary.pt", map_location="cpu")
            )

        self.model_args = model_args
        self.model = fast.Transformer(model_args, checkpoint).eval()
        self.tokenizer = Tokenizer(str(Path(__file__).resolve().parent.parent / "tokenizer.model"))
        self._cache = fast.make_cache(args=self.model_args, length=MAX_SEQ_LEN, device="cuda")

    # ------------------------------------------------------------------
    # Internal forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _prefill(self, token_ids: List[int]) -> torch.Tensor:
        seq_len = len(token_ids)
        bias = AttnBias.from_seqlens(
            q_seqlen=[seq_len],
            kv_seqlen=[seq_len],
            kv_padding=MAX_SEQ_LEN,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")
        tokens = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
        return self.model.forward_with_attn_bias(
            token_values=tokens, attn_bias=bias, cache=self._cache
        )

    # ------------------------------------------------------------------
    # lm-eval API
    # ------------------------------------------------------------------

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results = []
        for req in tqdm(requests, desc="loglikelihood"):
            context, continuation = req.args
            ctx_ids  = self.tokenizer.encode(context,      bos=False, eos=False)
            cont_ids = self.tokenizer.encode(continuation, bos=False, eos=False)

            max_ctx = self._max_seq_len - len(cont_ids) - 1
            if len(ctx_ids) > max_ctx:
                ctx_ids = ctx_ids[-max_ctx:]

            full_ids    = ctx_ids + cont_ids
            ctx_len     = len(ctx_ids)
            cont_len    = len(cont_ids)
            cont_tensor = torch.tensor(cont_ids, dtype=torch.long, device="cuda")

            logits      = self._prefill(full_ids)
            pred_logits = logits[ctx_len - 1 : ctx_len + cont_len - 1]

            log_probs       = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, cont_tensor.unsqueeze(-1)).squeeze(-1)
            total_log_prob  = token_log_probs.sum().item()
            is_greedy       = (pred_logits.argmax(dim=-1) == cont_tensor).all().item()

            results.append((total_log_prob, bool(is_greedy)))
        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        """Compute rolling log-likelihood for PPL tasks (e.g. wikitext)."""
        results = []
        for req in tqdm(requests, desc="loglikelihood_rolling"):
            text = req.args[0]
            token_ids = self.tokenizer.encode(text, bos=False, eos=False)

            total_log_prob = 0.0
            # Slide over the token sequence in non-overlapping MAX_SEQ_LEN windows
            for start in range(0, len(token_ids), self._max_seq_len):
                chunk = token_ids[start : start + self._max_seq_len]
                if len(chunk) < 2:
                    break

                logits      = self._prefill(chunk)          # [T, V]
                shift_logits = logits[:-1]                  # predict positions 1..T
                shift_labels = torch.tensor(
                    chunk[1:], dtype=torch.long, device="cuda"
                )

                log_probs       = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                total_log_prob += token_log_probs.sum().item()

            results.append(total_log_prob)
        return results

    def generate_until(self, requests) -> List[str]:
        raise NotImplementedError

    @property
    def eot_token_id(self):  return self.tokenizer.eos_id
    @property
    def max_length(self):    return self._max_seq_len
    @property
    def max_gen_toks(self):  return 512
    @property
    def batch_size(self):    return 1
    @property
    def device(self):        return "cuda"


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(ckpt_dir: str, bf16: bool, tasks: List[str], limit: int = None):
    from lm_eval.evaluator import simple_evaluate

    label = "BF16" if bf16 else "TernaInfer"
    print(f"\n{'='*60}\n  mode: {label}   tasks: {', '.join(tasks)}\n{'='*60}\n")

    lm      = TernaInferLM(ckpt_dir=ckpt_dir, bf16=bf16)
    summary = {}

    for task in tasks:
        print(f"\n>>> {task}")
        results  = simple_evaluate(model=lm, tasks=[task], num_fewshot=0,
                                   limit=limit, log_samples=False)
        task_res = results["results"].get(task, {})

        # PPL tasks report word_perplexity or byte_perplexity
        ppl = (task_res.get("word_perplexity,none")
               or task_res.get("word_perplexity")
               or task_res.get("perplexity,none")
               or task_res.get("perplexity"))
        acc = (task_res.get("acc,none") or task_res.get("acc"))

        if ppl is not None:
            summary[task] = ("ppl", ppl)
            print(f"  {task:<25} ppl = {ppl:.4f}")
        elif acc is not None:
            summary[task] = ("acc", acc)
            print(f"  {task:<25} acc = {acc*100:.2f}%")

    print(f"\n{'='*60}\n  {label} summary\n{'='*60}")
    for task, (metric, val) in summary.items():
        if metric == "ppl":
            print(f"  {task:<25} ppl = {val:.4f}")
        else:
            print(f"  {task:<25} acc = {val*100:.2f}%")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--bf16",  action="store_true")
    parser.add_argument("--both",  action="store_true")
    parser.add_argument("--tasks", default=",".join(PPL_TASKS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.both:
        summary_t = evaluate(args.ckpt_dir, bf16=False, tasks=tasks, limit=args.limit)
        summary_b = evaluate(args.ckpt_dir, bf16=True,  tasks=tasks, limit=args.limit)

        print(f"\n{'='*60}\n  comparison\n{'='*60}")
        print(f"  {'task':<25} {'TernaInfer':>14} {'BF16':>10} {'delta':>8}")
        print(f"  {'-'*60}")
        for task in tasks:
            t_entry = summary_t.get(task)
            b_entry = summary_b.get(task)
            if t_entry and b_entry:
                metric, t_val = t_entry
                _,      b_val = b_entry
                if metric == "ppl":
                    print(f"  {task:<25} {t_val:>13.4f}  {b_val:>9.4f}  {(t_val-b_val):>+8.4f}")
                else:
                    print(f"  {task:<25} {t_val*100:>13.2f}% {b_val*100:>9.2f}% {(t_val-b_val)*100:>+7.2f}%")
            else:
                print(f"  {task:<25} {'N/A':>14} {'N/A':>10}")
    else:
        evaluate(args.ckpt_dir, bf16=args.bf16, tasks=tasks, limit=args.limit)