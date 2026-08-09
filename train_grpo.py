import argparse
import json
import math
import os
import random
import re
import tempfile
import time

import torch
import torch.nn.functional as F

# baseline kl-free grpo training on gsm8k, and the shared helpers imported by
# train_q0_grpo.py and eval.py. minimal pip packages for a real run:
#   pip install torch transformers datasets accelerate

# full-run model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
# full-run model_revision = "a10cc1512eabd3dde888204e902eca88bddb4951"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
GSM8K_DATASET = "openai/gsm8k"

GSM8K_SEED = 42
GSM8K_HELD_OUT = 512
# full-run gsm8k_train_target = 6961
GSM8K_TRAIN_TARGET = 2048

GROUP_SIZE = 4
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 128
PROMPT_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 16
PEAK_LR = 5e-6
WEIGHT_DECAY = 0.1
CLIP_EPSILON = 0.2
# full-run num_passes = 16
NUM_PASSES = 4

EXPECTED_PROMPTS = NUM_PASSES * GSM8K_TRAIN_TARGET
EXPECTED_COMPLETIONS = EXPECTED_PROMPTS * GROUP_SIZE

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMERIC_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")
ANSWER_STOP_STRING = "</answer>"

PROMPT_TEMPLATE = (
    "Solve the grade school math problem below. Give a short chain of "
    "reasoning, then finish with the final numeric answer wrapped exactly "
    "as <answer>number</answer>.\n\n"
    "Question: What is 2 + 3?\nReasoning: 2 + 3 = 5. <answer>5</answer>\n\n"
    "Question: {question}\nReasoning:"
)


def build_prompt(question):
    # the one prompt template shared by training generation and eval
    return PROMPT_TEMPLATE.format(question=question.strip())


def normalize_number(s):
    # strip thousands separators and a trailing period, then parse as float
    if s is None:
        return None
    s = s.strip().replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_answer(text):
    # validate the last <answer>...</answer> tag as one numeric value
    matches = ANSWER_TAG_RE.findall(text)
    if not matches:
        return None, False
    content = matches[-1].strip()
    m = NUMERIC_RE.fullmatch(content)
    if not m:
        return None, False
    return normalize_number(content), True


def gsm8k_gold_answer(answer_field):
    # gsm8k solutions end with a line like "#### 18"
    tail = answer_field.split("####")[-1]
    return normalize_number(tail)


def compute_reward(completion_text, gold_value):
    # 1.0 for an exact normalized numeric match, plus 0.1 for a valid answer tag
    pred_value, has_tag = parse_answer(completion_text)
    reward = 0.0
    if has_tag:
        reward += 0.1
    if pred_value is not None and gold_value is not None and abs(pred_value - gold_value) < 1e-6:
        reward += 1.0
    return reward


def group_normalize_advantages(rewards, group_size):
    # rewards is 1d, laid out as contiguous groups of group_size completions
    # per prompt. zero-variance groups get zero advantage instead of nan.
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    advantages = torch.where(
        std > 1e-8, (rewards - mean) / std.clamp_min(1e-8), torch.zeros_like(rewards)
    )
    return advantages.view(-1)


def clipped_grpo_loss(new_logprobs, old_logprobs, advantages, completion_mask, epsilon=CLIP_EPSILON):
    # standard ppo-style clipped surrogate, no reference-model kl term
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages.unsqueeze(1)
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages.unsqueeze(1)
    per_token_loss = -torch.min(unclipped, clipped)
    completion_mask = completion_mask.float()
    return (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)


def cosine_lr(step, total_steps, peak_lr):
    # cosine decay from peak_lr at step 0 down to 0 at the final step
    if total_steps <= 1:
        return peak_lr
    progress = min(step / (total_steps - 1), 1.0)
    return 0.5 * peak_lr * (1.0 + math.cos(math.pi * progress))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batched(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def capture_rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, rng_state, counters, contract):
    # everything needed to resume exactly at a pass or cycle boundary
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": rng_state,
            "counters": counters,
            "contract": contract,
        },
        path,
    )


def atomic_torch_save(value, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_json_dump(value, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def immutable_training_contract(args):
    return {
        "model_name": args.model_name,
        "revision": args.revision,
        "dataset": GSM8K_DATASET,
        "dataset_seed": GSM8K_SEED,
        "held_out": GSM8K_HELD_OUT,
        "train_target": GSM8K_TRAIN_TARGET,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "prompt_batch_size": args.prompt_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "clip_epsilon": args.clip_epsilon,
        "device": args.device,
    }


def baseline_contract(args):
    contract = immutable_training_contract(args)
    contract["num_passes"] = args.num_passes
    return contract


def validate_immutable_training_contract(args, cuda_available):
    fixed_values = {
        "model_name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "group_size": GROUP_SIZE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "clip_epsilon": CLIP_EPSILON,
    }
    for name, expected in fixed_values.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{name} must be {expected!r}")
    if args.device != "cuda":
        raise ValueError("device must be cuda")
    if not cuda_available:
        raise ValueError("cuda is required")


def validate_baseline_contract(args, cuda_available):
    validate_immutable_training_contract(args, cuda_available)
    if args.num_passes != NUM_PASSES:
        raise ValueError(f"num_passes must be {NUM_PASSES}")


def require_matching_contract(saved_contract, current_contract):
    if saved_contract != current_contract:
        raise ValueError("checkpoint contract mismatch")


def assert_counts(num_prompts, num_completions, expected_prompts=EXPECTED_PROMPTS,
                   expected_completions=EXPECTED_COMPLETIONS):
    if num_prompts != expected_prompts:
        raise RuntimeError(f"prompt count mismatch: {num_prompts} != {expected_prompts}")
    if num_completions != expected_completions:
        raise RuntimeError(f"completion count mismatch: {num_completions} != {expected_completions}")


def load_gsm8k_splits(seed=GSM8K_SEED, held_out=GSM8K_HELD_OUT):
    # only the official train split is touched here; the official test split
    # is loaded exclusively by eval.py
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET, "main", split="train")
    ds = ds.shuffle(seed=seed)
    held_out_examples = ds.select(range(held_out))
    train_examples = ds.select(range(held_out, held_out + GSM8K_TRAIN_TARGET))
    if len(train_examples) != GSM8K_TRAIN_TARGET:
        raise ValueError(f"expected {GSM8K_TRAIN_TARGET} train examples, got {len(train_examples)}")
    return list(train_examples), list(held_out_examples)


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "transformers is required for this operation. minimal pip packages:\n"
            "  pip install torch transformers datasets accelerate\n"
            f"original import error: {e}"
        )


def resolve_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_tokenizer(model_name=MODEL_NAME, revision=MODEL_REVISION):
    _require_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name=MODEL_NAME, revision=MODEL_REVISION, device="cuda"):
    _require_transformers()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_attn_implementation(),
    ).to(device)
    return model


def build_model_and_tokenizer(model_name=MODEL_NAME, revision=MODEL_REVISION, device="cuda"):
    tokenizer = load_tokenizer(model_name, revision)
    model = load_base_model(model_name, revision, device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


def load_model_from_checkpoint(model_name, revision, ckpt_path, device):
    model = load_base_model(model_name, revision, device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict({k: v.to(model.dtype) for k, v in state_dict.items()})
    model.eval()
    return model


def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available()
    )


def build_completion_mask(completion_ids, eos_token_id):
    # 1 for real completion tokens up to and including the first eos, 0 after
    is_eos = completion_ids == eos_token_id
    seq_len = completion_ids.shape[1]
    has_eos = is_eos.any(dim=1)
    first_eos = torch.where(
        has_eos, is_eos.float().argmax(dim=1),
        torch.full_like(has_eos, seq_len - 1, dtype=torch.long),
    )
    positions = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0)
    return (positions <= first_eos.unsqueeze(1)).long()


@torch.no_grad()
def generate_group(model, tokenizer, prompts, group_size, temperature, top_p, max_new_tokens, device):
    # sample group_size completions per prompt. the generation scores here are
    # warped by temperature/top_p and must never be used as training logprobs.
    previous_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    finally:
        tokenizer.padding_side = previous_padding_side
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=[ANSWER_STOP_STRING],
        tokenizer=tokenizer,
        use_cache=True,
    )
    completion_ids = out[:, prompt_len:]
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    completion_mask_tail = build_completion_mask(completion_ids, tokenizer.eos_token_id)
    prompt_mask = enc["attention_mask"].repeat_interleave(group_size, dim=0)
    full_mask = torch.cat([prompt_mask, completion_mask_tail], dim=1)
    completion_mask = torch.cat(
        [torch.zeros_like(prompt_mask), completion_mask_tail], dim=1
    )
    return out, full_mask, completion_mask, texts


def completion_token_logprobs(model, input_ids, attention_mask, completion_mask):
    # teacher-forced forward pass; raw causal token logprobs and shifted completion mask
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:].float()
    return token_logprobs, mask


@torch.no_grad()
def old_policy_completion_token_logprobs(model, input_ids, attention_mask, completion_mask, micro_batch_size):
    # keeps the frozen old-policy forward within the training microbatch memory budget
    logprobs = []
    masks = []
    for lo in range(0, input_ids.shape[0], micro_batch_size):
        hi = min(lo + micro_batch_size, input_ids.shape[0])
        token_logprobs, token_mask = completion_token_logprobs(
            model, input_ids[lo:hi], attention_mask[lo:hi], completion_mask[lo:hi],
        )
        logprobs.append(token_logprobs)
        masks.append(token_mask)
    return torch.cat(logprobs), torch.cat(masks)


def sequence_logprobs(model, input_ids, attention_mask, completion_mask):
    # retained for callers that need summed completion logprobs
    token_logprobs, mask = completion_token_logprobs(model, input_ids, attention_mask, completion_mask)
    return (token_logprobs * mask).sum(dim=-1)


def add_common_args(parser):
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--prompt-batch-size", type=int, default=PROMPT_BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PEAK_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--clip-epsilon", type=float, default=CLIP_EPSILON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--self-test", action="store_true",
        help="run offline self-tests only, no network, model download, or gpu needed",
    )
    return parser


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="baseline kl-free grpo training on gsm8k with smollm2-360m-instruct",
        epilog="minimal pip packages for a real run: pip install torch transformers datasets accelerate",
    )
    add_common_args(parser)
    parser.add_argument("--run-id", default="grpo_baseline")
    parser.add_argument("--num-passes", type=int, default=NUM_PASSES)
    return parser


def run_training(args):
    validate_baseline_contract(args, torch.cuda.is_available())
    set_seed(GSM8K_SEED)
    device = args.device
    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    train_examples, _held_out = load_gsm8k_splits(seed=GSM8K_SEED, held_out=GSM8K_HELD_OUT)
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.revision, device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    prompts_per_pass = len(train_examples)
    prompt_batches_per_pass = math.ceil(prompts_per_pass / args.prompt_batch_size)
    total_steps = args.num_passes * prompt_batches_per_pass

    start_pass = 0
    global_step = 0
    prompts_seen = 0
    completions_seen = 0

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path)
        require_matching_contract(ckpt.get("contract"), baseline_contract(args))
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng_state(ckpt["rng_state"])
        counters = ckpt["counters"]
        start_pass = counters["pass"]
        global_step = counters["global_step"]
        prompts_seen = counters["prompts_seen"]
        completions_seen = counters["completions_seen"]
        print(f"resumed from pass {start_pass}, global_step {global_step}")

    for pass_idx in range(start_pass, args.num_passes):
        rng = random.Random(GSM8K_SEED + pass_idx)
        order = list(range(prompts_per_pass))
        rng.shuffle(order)
        shuffled = [train_examples[i] for i in order]

        for batch in batched(shuffled, args.prompt_batch_size):
            questions = [ex["question"] for ex in batch]
            golds = [gsm8k_gold_answer(ex["answer"]) for ex in batch]
            prompts = [build_prompt(q) for q in questions]
            batch_size = len(batch)

            model.eval()
            full_ids, full_mask, completion_mask, texts = generate_group(
                model, tokenizer, prompts, args.group_size,
                args.temperature, args.top_p, args.max_new_tokens, device,
            )
            golds_expanded = []
            for g in golds:
                golds_expanded.extend([g] * args.group_size)
            rewards = torch.tensor(
                [compute_reward(t, g) for t, g in zip(texts, golds_expanded)], dtype=torch.float32,
            )
            advantages = group_normalize_advantages(rewards, args.group_size).to(device)

            old_logprobs, token_mask = old_policy_completion_token_logprobs(
                model, full_ids, full_mask, completion_mask, args.micro_batch_size,
            )
            total_valid_tokens = token_mask.sum().clamp_min(1.0)

            lr = cosine_lr(global_step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            optimizer.zero_grad(set_to_none=True)
            num_rows = full_ids.shape[0]
            num_micro = math.ceil(num_rows / args.micro_batch_size)
            for m in range(num_micro):
                lo = m * args.micro_batch_size
                hi = min(lo + args.micro_batch_size, num_rows)
                new_lp, micro_mask = completion_token_logprobs(
                    model, full_ids[lo:hi], full_mask[lo:hi], completion_mask[lo:hi]
                )
                loss = clipped_grpo_loss(
                    new_lp, old_logprobs[lo:hi], advantages[lo:hi], micro_mask, args.clip_epsilon
                )
                valid_fraction = micro_mask.sum() / total_valid_tokens
                (loss * valid_fraction).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            prompts_seen += batch_size
            completions_seen += num_rows
            print(
                f"pass {pass_idx + 1}/{args.num_passes} step {global_step}/{total_steps} "
                f"lr {lr:.3e} reward {rewards.mean().item():.3f}"
            )

        counters = {
            "pass": pass_idx + 1, "global_step": global_step,
            "prompts_seen": prompts_seen, "completions_seen": completions_seen,
        }
        save_checkpoint(ckpt_path, model, optimizer, capture_rng_state(), counters, baseline_contract(args))
        pass_ckpt_path = os.path.join(run_dir, f"pass_{pass_idx + 1:02d}.pt")
        atomic_torch_save(
            {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()},
            pass_ckpt_path,
        )
        print(f"saved pass {pass_idx + 1} checkpoint to {pass_ckpt_path}")

    assert_counts(prompts_seen, completions_seen)
    print(f"training complete: {prompts_seen} prompts, {completions_seen} completions")


def self_test():
    print("running train_grpo self-test...")

    p = build_prompt("  what is 2+2?  ")
    assert "what is 2+2?" in p
    assert "<answer>number</answer>" in p
    assert "What is 2 + 3?\nReasoning: 2 + 3 = 5. <answer>5</answer>" in p
    assert p.endswith("Question: what is 2+2?\nReasoning:")

    val, has_tag = parse_answer("some reasoning <answer>42</answer>")
    assert has_tag and val == 42.0
    val, has_tag = parse_answer("no tag here")
    assert not has_tag and val is None
    val, has_tag = parse_answer("<answer>junk 42 junk</answer>")
    assert not has_tag and val is None
    val, has_tag = parse_answer("<answer>+42</answer>")
    assert has_tag and val == 42.0
    val, has_tag = parse_answer("<answer>  42  </answer>")
    assert has_tag and val == 42.0
    val, has_tag = parse_answer("<answer>1,234</answer>")
    assert has_tag and val == 1234.0
    val, has_tag = parse_answer("<answer>1,234.5</answer>")
    assert has_tag and val == 1234.5

    gold = gsm8k_gold_answer("some reasoning\n#### 18")
    assert gold == 18.0

    r = compute_reward("reasoning <answer>18</answer>", 18.0)
    assert abs(r - 1.1) < 1e-9
    r = compute_reward("reasoning <answer>19</answer>", 18.0)
    assert abs(r - 0.1) < 1e-9
    r = compute_reward("reasoning <answer>junk 18 junk</answer>", 18.0)
    assert abs(r - 0.0) < 1e-9
    r = compute_reward("reasoning no tag", 18.0)
    assert abs(r - 0.0) < 1e-9

    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv = group_normalize_advantages(rewards, group_size=4)
    assert adv.shape == (8,)
    assert torch.allclose(adv[4:], torch.zeros(4)), "zero-variance group must give zero advantage"
    assert adv[0] > 0 and adv[1] < 0

    new_lp = torch.log(torch.tensor([[2.0, 0.5, 1.0], [1.5, 3.0, 1.0]]))
    old_lp = torch.zeros_like(new_lp)
    adv2 = torch.tensor([1.0, -1.0])
    token_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    loss = clipped_grpo_loss(new_lp, old_lp, adv2, token_mask, epsilon=0.2)
    assert torch.allclose(loss, torch.tensor((-1.2 - 0.5 + 1.5) / 3.0))

    micro_losses = torch.tensor([2.0, 5.0])
    micro_tokens = torch.tensor([3.0, 1.0])
    accumulated = (micro_losses * micro_tokens / micro_tokens.sum()).sum()
    full_batch_loss = (2.0 * 3.0 + 5.0) / 4.0
    assert torch.allclose(accumulated, torch.tensor(full_batch_loss))

    class TinyModel:
        def __call__(self, input_ids, attention_mask, use_cache):
            logits = F.one_hot(input_ids % 16, num_classes=16).float()
            return type("Output", (), {"logits": logits})()

    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    attention = torch.ones_like(ids)
    completion = torch.ones_like(ids)
    full_logprobs, full_mask = completion_token_logprobs(TinyModel(), ids, attention, completion)
    micro_logprobs, micro_mask = old_policy_completion_token_logprobs(
        TinyModel(), ids, attention, completion, micro_batch_size=2,
    )
    assert torch.allclose(micro_logprobs, full_logprobs)
    assert torch.equal(micro_mask, full_mask)
    assert not micro_logprobs.requires_grad

    contract_args = argparse.Namespace(
        model_name=MODEL_NAME, revision=MODEL_REVISION, group_size=GROUP_SIZE,
        temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS,
        prompt_batch_size=PROMPT_BATCH_SIZE, micro_batch_size=MICRO_BATCH_SIZE,
        lr=PEAK_LR, weight_decay=WEIGHT_DECAY, clip_epsilon=CLIP_EPSILON,
        device="cuda", num_passes=NUM_PASSES,
    )
    validate_baseline_contract(contract_args, cuda_available=True)
    bad_contract_args = argparse.Namespace(**vars(contract_args))
    bad_contract_args.group_size = GROUP_SIZE + 1
    try:
        validate_baseline_contract(bad_contract_args, cuda_available=True)
        raise RuntimeError("expected contract rejection")
    except ValueError:
        pass
    try:
        validate_baseline_contract(contract_args, cuda_available=False)
        raise RuntimeError("expected cuda rejection")
    except ValueError:
        pass
    changed_contract_args = argparse.Namespace(**vars(contract_args))
    changed_contract_args.micro_batch_size += 1
    try:
        require_matching_contract(baseline_contract(contract_args), baseline_contract(changed_contract_args))
        raise RuntimeError("expected checkpoint contract mismatch")
    except ValueError:
        pass

    assert abs(cosine_lr(0, 100, 5e-6) - 5e-6) < 1e-12
    assert cosine_lr(99, 100, 5e-6) < 1e-9

    completion_ids = torch.tensor([[5, 5, 2, 9, 9], [5, 5, 5, 5, 5]])
    mask = build_completion_mask(completion_ids, eos_token_id=2)
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    assert EXPECTED_PROMPTS == 8192
    assert EXPECTED_COMPLETIONS == 32768
    assert_counts(EXPECTED_PROMPTS, EXPECTED_COMPLETIONS)
    try:
        assert_counts(1, 1)
        raise RuntimeError("expected count error")
    except RuntimeError:
        pass

    print(f"train_grpo self-test passed (expects {EXPECTED_PROMPTS} prompts, {EXPECTED_COMPLETIONS} completions)")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run_training(args)


if __name__ == "__main__":
    main()
