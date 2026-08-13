"""Focused pass@k evaluator for the Qwen laptop systems.

Reuses the canonical scoring/generation primitives from eval.py (GSM8K) and
eval_math500.py (MATH-500 via math-verify). A model is either a full HF repo
(the pushed GRPO baseline) or a 4-bit QLoRA adapter on the profile base
(q0 snapshots, MOPD opsd.pt).

  # GSM8K pass@8 on the HF baseline
  python core/eval_passk.py --dataset gsm8k --k 8 \
      --model hf:Pradheep1647/q0-gsm8k-1_5b-grpo --name baseline_grpo \
      --output experiments/q0_mopd/results/passk_baseline_gsm8k.json

  # MATH-500 pass@4 on a 4-bit adapter
  python core/eval_passk.py --dataset math500 --k 4 --profile qwen1_5b_laptop \
      --model adapter:runs/mopd_qwen1_5b_laptop/opsd.pt --name mopd_t1 \
      --output experiments/q0_mopd/results/passk_mopd_t1_math500.json
"""
import argparse
import json
import os
import tempfile
import time

import torch

import eval as ev
import eval_math500 as em
import train_grpo as tg


def atomic_json_dump(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_model_and_tokenizer(args, device):
    kind, _, ref = args.model.partition(":")
    if kind == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            ref, torch_dtype=torch.bfloat16,
            attn_implementation=tg.resolve_attn_implementation(),
        ).to(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(ref)
        except AttributeError:
            # Some fine-tuned repos ship a tokenizer_config.json whose
            # extra_special_tokens is a list, which the installed Transformers
            # rejects. The baseline was fine-tuned from the base Qwen model with
            # an unchanged vocab, so the base tokenizer is token-identical.
            profile = tg.PROFILES[args.profile]
            tokenizer = tg.load_tokenizer(profile["model_name"], profile["revision"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer, {"hf_repo": ref}
    if kind == "adapter":
        profile = tg.PROFILES[args.profile]
        model = tg.load_model_from_checkpoint(
            profile["model_name"], profile["revision"], ref, device,
            "4bit", args.lora_rank, args.lora_alpha, args.lora_dropout,
        )
        tokenizer = tg.load_tokenizer(profile["model_name"], profile["revision"])
        model.eval()
        return model, tokenizer, {
            "adapter": ref, "base": profile["model_name"],
            "revision": profile["revision"], "quantization": "4bit",
        }
    raise ValueError("--model must be 'hf:<repo>' or 'adapter:<path>'")


@torch.no_grad()
def math500_pass_at_k(model, tokenizer, examples, sample_count, max_new_tokens,
                      batch_size, device, source):
    # mirrors eval.evaluate_pass_at_k but with the MATH-500 prompt and the
    # math-verify symbolic scorer; reports pass@1..sample_count.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tg.set_seed(tg.GSM8K_SEED)
    k_values = tuple(k for k in (1, 4, 8) if k <= sample_count)
    pass_totals = {k: 0.0 for k in k_values}
    parse_totals = {k: 0.0 for k in k_values}
    t0 = time.time()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [em.build_prompt(example["problem"]) for example in batch]
        enc = ev.tokenize_left_padded(tokenizer, prompts, device)
        out = model.generate(
            **enc, do_sample=True, temperature=tg.TEMPERATURE, top_p=tg.TOP_P,
            max_new_tokens=max_new_tokens, num_return_sequences=sample_count,
            pad_token_id=tokenizer.pad_token_id, stop_strings=[tg.ANSWER_STOP_STRING],
            tokenizer=tokenizer, use_cache=True,
        )
        completion_ids = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for index, example in enumerate(batch):
            samples = texts[index * sample_count:(index + 1) * sample_count]
            scores = [em.score_answer(text, example["answer"]) for text in samples]
            correct_count = sum(int(correct) for correct, _, _ in scores)
            parsed_count = sum(int(parsed) for _, parsed, _ in scores)
            for k in k_values:
                pass_totals[k] += ev.pass_at_k_estimate(sample_count, correct_count, k)
                parse_totals[k] += ev.pass_at_k_estimate(sample_count, parsed_count, k)
        print(f"{source}: {min(start + batch_size, len(examples))}/{len(examples)}")
    count = len(examples)
    result = {f"pass_at_{k}": pass_totals[k] / count for k in k_values}
    result.update({f"parse_at_{k}": parse_totals[k] / count for k in k_values})
    result.update({
        "samples_per_problem": sample_count,
        "timing_seconds": time.time() - t0,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device == "cuda" else 0
        ),
        "seed": tg.GSM8K_SEED,
        "source": source,
    })
    return result


def load_gsm8k_val_slice(count):
    # the exact deterministic slice MOPD selects candidates on: gsm8k train,
    # shuffle(seed=42), offset past the held-out eval block and the q0 train block.
    from datasets import load_dataset

    dataset = load_dataset(tg.GSM8K_DATASET, "main", split="train").shuffle(seed=tg.GSM8K_SEED)
    start = tg.GSM8K_HELD_OUT + tg.GSM8K_TRAIN_TARGET
    return list(dataset.select(range(start, start + count)))


def sample_top_p(probs, top_p):
    # nucleus sampling over an already-normalized probability tensor
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    drop = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def ensemble_generate_samples(models, weights, tokenizer, prompts, max_new_tokens,
                              device, sample_count, temperature, top_p):
    # weighted-mixture sampling decode: N adapters share a base but keep separate
    # KV caches; per step we mix softmax(logits/temp) by the learned weights and
    # nucleus-sample. sample_count sequences per prompt via row expansion.
    enc = ev.tokenize_left_padded(tokenizer, prompts, device)
    input_ids = enc["input_ids"].repeat_interleave(sample_count, dim=0)
    gen_mask = enc["attention_mask"].repeat_interleave(sample_count, dim=0)
    rows = input_ids.shape[0]
    eos_id = tokenizer.eos_token_id
    stop_token_ids = torch.tensor(
        tokenizer(tg.ANSWER_STOP_STRING, add_special_tokens=False)["input_ids"], device=device,
    )
    finished = torch.zeros(rows, dtype=torch.bool, device=device)
    generated = input_ids
    past_key_values = [None] * len(models)
    prompt_len = input_ids.shape[1]
    for _ in range(max_new_tokens):
        logits_list = []
        new_pasts = []
        for model, past in zip(models, past_key_values):
            if past is None:
                position_ids = ev.position_ids_from_attention_mask(gen_mask)
                out = model(
                    input_ids=generated, attention_mask=gen_mask,
                    position_ids=position_ids, use_cache=True, logits_to_keep=1,
                )
            else:
                position_ids = ev.position_ids_from_attention_mask(gen_mask)[:, -1:]
                out = model(
                    input_ids=generated[:, -1:], attention_mask=gen_mask,
                    position_ids=position_ids, past_key_values=past, use_cache=True,
                )
            logits_list.append(out.logits[:, -1, :] / temperature)
            new_pasts.append(out.past_key_values)
        past_key_values = new_pasts
        mixed_probs = ev.mix_probabilities(logits_list, weights)
        next_token = sample_top_p(mixed_probs, top_p)
        active = ~finished
        next_token = torch.where(active, next_token, torch.full_like(next_token, tokenizer.pad_token_id))
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        gen_mask = torch.cat([gen_mask, active.long().unsqueeze(1)], dim=1)
        finished = finished | ((next_token == eos_id) & active) | (ev.has_token_suffix(generated, stop_token_ids) & active)
        if finished.all():
            break
    return tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def mixture_pass_at_k(models, weights, tokenizer, items, dataset, sample_count,
                      max_new_tokens, batch_size, device, source):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tg.set_seed(tg.GSM8K_SEED)
    k_values = tuple(k for k in (1, 4, 8) if k <= sample_count)
    pass_totals = {k: 0.0 for k in k_values}
    parse_totals = {k: 0.0 for k in k_values}
    if dataset == "math500":
        prompt_fn = lambda ex: em.build_prompt(ex["problem"])
        score_fn = lambda text, ex: em.score_answer(text, ex["answer"])[:2]
    else:
        prompt_fn = lambda ex: tg.build_prompt(ex["question"])
        score_fn = lambda text, ex: ev.score_completion(text, tg.gsm8k_gold_answer(ex["answer"]))
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [prompt_fn(ex) for ex in batch]
        texts = ensemble_generate_samples(
            models, weights, tokenizer, prompts, max_new_tokens,
            device, sample_count, tg.TEMPERATURE, tg.TOP_P,
        )
        for index, example in enumerate(batch):
            samples = texts[index * sample_count:(index + 1) * sample_count]
            scores = [score_fn(text, example) for text in samples]
            correct_count = sum(int(correct) for correct, _ in scores)
            parsed_count = sum(int(parsed) for _, parsed in scores)
            for k in k_values:
                pass_totals[k] += ev.pass_at_k_estimate(sample_count, correct_count, k)
                parse_totals[k] += ev.pass_at_k_estimate(sample_count, parsed_count, k)
        print(f"{source}: {min(start + batch_size, len(items))}/{len(items)}")
    count = len(items)
    result = {f"pass_at_{k}": pass_totals[k] / count for k in k_values}
    result.update({f"parse_at_{k}": parse_totals[k] / count for k in k_values})
    result.update({
        "samples_per_problem": sample_count,
        "timing_seconds": time.time() - t0,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device == "cuda" else 0,
        "seed": tg.GSM8K_SEED,
        "source": source,
    })
    return result


def load_mixture(spec, device, args):
    run_dir, _, k_text = spec.partition("@")
    if not k_text:
        raise ValueError("mixture spec must be 'mixture:<run_dir>@<k>'")
    k = int(k_text)
    with open(os.path.join(run_dir, "mixture_weights.json")) as handle:
        manifest = json.load(handle)
    entry = manifest["top_k"][str(k)]
    paths, weights = entry["snapshot_paths"], entry["weights"]
    profile = tg.PROFILES[args.profile]
    tokenizer = tg.load_tokenizer(profile["model_name"], profile["revision"])
    models = []
    for rel in paths:
        model = tg.load_model_from_checkpoint(
            profile["model_name"], profile["revision"], os.path.join(run_dir, rel),
            device, "4bit", args.lora_rank, args.lora_alpha, args.lora_dropout,
        )
        model.eval()
        models.append(model)
    meta = {
        "mixture_run_dir": run_dir, "k": k, "snapshot_paths": paths,
        "weights": weights, "base": profile["model_name"], "quantization": "4bit",
    }
    return models, weights, tokenizer, meta


def run(args):
    for required in ("dataset", "model", "name", "output"):
        if not getattr(args, required):
            raise ValueError(f"--{required} is required")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("pass@k evaluation requires cuda")
    if args.model.startswith("mixture:"):
        models, weights, tokenizer, model_meta = load_mixture(
            args.model.partition(":")[2], args.device, args,
        )
        if args.dataset == "math500":
            items = em.load_math500()
            dataset_meta = {"dataset": em.MATH_DATASET, "revision": em.MATH_REVISION,
                            "split": em.MATH_SPLIT, "count": len(items)}
        elif args.dataset == "gsm8k_val":
            items = load_gsm8k_val_slice(args.val_count)
            dataset_meta = {"dataset": tg.GSM8K_DATASET, "split": "train_val_slice", "count": len(items)}
        else:
            items = ev.load_test_set()
            dataset_meta = {"dataset": tg.GSM8K_DATASET, "split": "test", "count": len(items)}
        result = mixture_pass_at_k(
            models, weights, tokenizer, items, args.dataset, args.k,
            args.max_new_tokens, args.batch_size, args.device, args.name,
        )
        payload = {
            "name": args.name, "model": model_meta, "dataset_meta": dataset_meta,
            "requested_k": args.k, "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json_dump(payload, args.output)
        passes = {key: round(value, 4) for key, value in result.items() if key.startswith("pass_at_") and key[8:].isdigit()}
        print(f"{args.name} [{args.dataset}] -> {passes}")
        print(f"wrote {args.output}")
        return
    model, tokenizer, model_meta = load_model_and_tokenizer(args, args.device)
    if args.dataset in ("gsm8k", "gsm8k_val"):
        if args.dataset == "gsm8k_val":
            test_set = load_gsm8k_val_slice(args.val_count)
            dataset_meta = {"dataset": tg.GSM8K_DATASET, "split": "train_val_slice", "count": len(test_set)}
        else:
            test_set = ev.load_test_set()
            dataset_meta = {"dataset": tg.GSM8K_DATASET, "split": "test", "count": len(test_set)}
        # eval.evaluate_pass_at_k samples max(PASS_K_VALUES)=8 and reports 1/4/8
        result = ev.evaluate_pass_at_k(
            model, tokenizer, test_set, args.batch_size,
            args.max_new_tokens, args.device, args.name,
        )
    else:
        examples = em.load_math500()
        result = math500_pass_at_k(
            model, tokenizer, examples, args.k, args.max_new_tokens,
            args.batch_size, args.device, args.name,
        )
        dataset_meta = {
            "dataset": em.MATH_DATASET, "revision": em.MATH_REVISION,
            "split": em.MATH_SPLIT, "count": len(examples),
        }
    payload = {
        "name": args.name,
        "model": model_meta,
        "dataset_meta": dataset_meta,
        "requested_k": args.k,
        "result": result,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json_dump(payload, args.output)
    passes = {key: round(value, 4) for key, value in result.items() if key.startswith("pass_at_") and key[8:].isdigit()}
    print(f"{args.name} [{args.dataset}] -> {passes}")
    print(f"wrote {args.output}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="pass@k eval for a single system")
    parser.add_argument("--dataset", choices=["gsm8k", "gsm8k_val", "math500"])
    parser.add_argument("--model", help="hf:<repo>, adapter:<path>, or mixture:<run_dir>@<k>")
    parser.add_argument("--name", help="system label in the output")
    parser.add_argument("--profile", choices=sorted(tg.PROFILES), default="qwen1_5b_laptop")
    parser.add_argument("--k", type=int, default=8, help="samples per problem (max pass@k reported)")
    parser.add_argument("--val-count", type=int, default=256, help="gsm8k_val slice size")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test():
    # estimator + scorer wiring, no GPU
    assert ev.pass_at_k_estimate(4, 0, 4) == 0.0
    assert ev.pass_at_k_estimate(4, 1, 4) == 1.0
    assert abs(ev.pass_at_k_estimate(4, 1, 1) - 0.25) < 1e-12
    assert em.score_answer("<answer>$\\boxed{14/3}$</answer>", "\\frac{14}{3}")[:2] == (True, True)
    # nucleus sampler: a one-hot distribution must return that token deterministically
    onehot = torch.zeros(1, 5)
    onehot[0, 3] = 1.0
    assert int(sample_top_p(onehot, 0.95).item()) == 3
    # mixture spec parsing
    rd, _, kt = "runs/q0_qwen1_5b_laptop@2".partition("@")
    assert rd == "runs/q0_qwen1_5b_laptop" and int(kt) == 2
    for spec in ("bogus:x", "adapterpath"):
        try:
            argparse.Namespace(model=spec)
            kind = spec.partition(":")[0]
            assert kind not in ("hf", "adapter")
        except Exception:
            pass
    print("eval_passk self-test passed")


if __name__ == "__main__":
    arguments = build_arg_parser().parse_args()
    if arguments.self_test:
        self_test()
    else:
        run(arguments)
