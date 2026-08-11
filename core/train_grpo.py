import argparse
import functools
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
ANSWER_STOP_STRING = "</answer>"

# profile-scoped config. smol135m is exactly today's module constants, so the
# 135m contract and the a/b reproducibility guarantee stay byte-identical; the
# argparse defaults resolve from whichever profile --profile picks (see
# parse_args_with_profile). qwen1_5b is the scaled-up full fine-tune on a100-40gb.
PROFILES = {
    "smol135m": {
        "model_name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "group_size": GROUP_SIZE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "lr": PEAK_LR,
        "weight_decay": WEIGHT_DECAY,
        "clip_epsilon": CLIP_EPSILON,
        "num_passes": NUM_PASSES,
        "train_target": GSM8K_TRAIN_TARGET,
    },
    "qwen1_5b": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        # big groups kill zero-variance collapse; long gens fit full chain-of-thought
        "group_size": 16,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 384,
        "prompt_batch_size": 48,
        # 1.5b activations are heavy; grad-checkpointing is already on
        "micro_batch_size": 8,
        "lr": 2e-6,
        "weight_decay": 0.1,
        "clip_epsilon": 0.2,
        "num_passes": 4,
        "train_target": 3072,
        # only read when --rollout vllm; kept out of the argparse defaults so these
        # never leak into the training contract.
        "vllm": {
            "gpu_memory_utilization": 0.35,
            "max_model_len": 1024,
            # 32 (was 64) caps how many sequences vllm keeps resident at once, so the
            # rollout memory high-water mark is lower. the 768 completions/step just
            # schedule in more waves; the completions themselves are unchanged. this is
            # to stop the fragmentation-driven cuda illegal-access we hit mid-run.
            "max_num_seqs": 32,
            "enable_sleep_mode": False,
        },
    },
}

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
    # tag-only extraction: return the raw contents of the last well-formed
    # <answer>...</answer> tag, plus whether such a tag was present. numeric /
    # expression validity is decided downstream by math_verify, not here, so
    # that "$18", "18.", "14/3", "18 dollars" etc. are not thrown away.
    matches = ANSWER_TAG_RE.findall(text)
    if not matches:
        return None, False
    content = matches[-1].strip()
    if not content:
        return None, False
    return content, True


@functools.lru_cache(maxsize=1)
def _math_verify():
    # imported lazily and cached: training's minimal install may skip math_verify
    # until a reward is actually scored (mirrors eval_math500.score_answer).
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    return parse, verify, LatexExtractionConfig, ExprExtractionConfig


@functools.lru_cache(maxsize=4096)
def _parse_gold(gold_str):
    parse, _verify, Latex, _Expr = _math_verify()
    return parse(f"$\\boxed{{{gold_str}}}$", extraction_config=[Latex()])


def answer_is_correct(answer_str, gold_value):
    # robust correctness check via math_verify, same configs as the math-500
    # eval path. handles currency symbols, units, trailing periods, thousands
    # separators, and fractions that the old numeric fullmatch rejected.
    if answer_str is None or gold_value is None:
        return False
    # math_verify's expression parser rejects a bare leading unary '+'; strip it.
    cleaned = answer_str.strip().lstrip("+")
    if not cleaned:
        return False
    gold_str = str(int(gold_value)) if float(gold_value).is_integer() else repr(gold_value)
    parse, verify, Latex, Expr = _math_verify()
    gold_parsed = _parse_gold(gold_str)
    answer_parsed = parse(cleaned, extraction_config=[Latex(), Expr()])
    return bool(gold_parsed and answer_parsed and verify(gold_parsed, answer_parsed))


def gsm8k_gold_answer(answer_field):
    # gsm8k solutions end with a line like "#### 18"
    tail = answer_field.split("####")[-1]
    return normalize_number(tail)


def compute_reward(completion_text, gold_value):
    # 0.1 for a well-formed <answer> tag, plus 1.0 when its contents match the
    # gold answer under math_verify. the format bonus is decoupled from numeric
    # validity so a correctly-tagged answer is never scored as no-tag.
    answer_str, has_tag = parse_answer(completion_text)
    reward = 0.0
    if has_tag:
        reward += 0.1
    if answer_is_correct(answer_str, gold_value):
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
        "train_target": args.train_target,
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
    # the frozen knobs come from the selected profile; smol135m's values are
    # exactly today's globals so the existing 135m contract is unchanged.
    profile = PROFILES[args.profile]
    fixed_values = {
        "model_name": profile["model_name"],
        "revision": profile["revision"],
        "group_size": profile["group_size"],
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "max_new_tokens": profile["max_new_tokens"],
        "clip_epsilon": profile["clip_epsilon"],
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
    expected_passes = PROFILES[args.profile]["num_passes"]
    if args.num_passes != expected_passes:
        raise ValueError(f"num_passes must be {expected_passes}")


def require_matching_contract(saved_contract, current_contract):
    if saved_contract != current_contract:
        raise ValueError("checkpoint contract mismatch")


def assert_counts(num_prompts, num_completions, expected_prompts=EXPECTED_PROMPTS,
                   expected_completions=EXPECTED_COMPLETIONS):
    if num_prompts != expected_prompts:
        raise RuntimeError(f"prompt count mismatch: {num_prompts} != {expected_prompts}")
    if num_completions != expected_completions:
        raise RuntimeError(f"completion count mismatch: {num_completions} != {expected_completions}")


def load_gsm8k_splits(seed=GSM8K_SEED, held_out=GSM8K_HELD_OUT, train_target=GSM8K_TRAIN_TARGET):
    # only the official train split is touched here; the official test split
    # is loaded exclusively by eval.py
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET, "main", split="train")
    ds = ds.shuffle(seed=seed)
    held_out_examples = ds.select(range(held_out))
    train_examples = ds.select(range(held_out, held_out + train_target))
    if len(train_examples) != train_target:
        raise ValueError(f"expected {train_target} train examples, got {len(train_examples)}")
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


def build_vllm_engine(args):
    # v0 engine keeps the model in-process (tp=1 UniProcExecutor) so we can hot-swap
    # weights each step with a zero-copy load_weights; the v1 engine runs the model
    # in a child process and breaks that path, hence VLLM_USE_V1=0 before the import.
    # enforce_eager drops cuda-graph memory so vllm's footprint stays predictable.
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM

    vcfg = PROFILES[args.profile].get("vllm", {})
    return LLM(
        model=args.model_name,
        revision=args.revision,
        dtype="bfloat16",
        gpu_memory_utilization=vcfg.get("gpu_memory_utilization", 0.35),
        max_model_len=vcfg.get("max_model_len", 1024),
        max_num_seqs=vcfg.get("max_num_seqs", 64),
        enforce_eager=True,
        enable_sleep_mode=vcfg.get("enable_sleep_mode", False),
    )


def _vllm_outputs_to_tensors(outputs, pad_id, device):
    # rebuild the exact (full_ids, full_mask, completion_mask, texts) contract that
    # generate_group returns, from a list of vllm RequestOutputs. rows are laid out
    # prompt-major and group-contiguous so advantages line up with golds_expanded,
    # and left-padded to match generate_group's padding_side="left".
    rows, texts = [], []
    for out in outputs:
        prompt_ids = list(out.prompt_token_ids)
        for comp in out.outputs:
            rows.append((prompt_ids, list(comp.token_ids)))
            texts.append(comp.text)

    seq_len = max(len(p) + len(c) for p, c in rows)
    full_ids = torch.full((len(rows), seq_len), pad_id, dtype=torch.long)
    full_mask = torch.zeros((len(rows), seq_len), dtype=torch.long)
    completion_mask = torch.zeros((len(rows), seq_len), dtype=torch.long)
    for i, (p, c) in enumerate(rows):
        start = seq_len - (len(p) + len(c))
        full_ids[i, start:] = torch.tensor(p + c, dtype=torch.long)
        full_mask[i, start:] = 1
        # ones only over the completion tokens. vllm already truncates at the stop
        # string / eos, so there are no post-eos tokens to zero out here.
        completion_mask[i, seq_len - len(c):] = 1
    return full_ids.to(device), full_mask.to(device), completion_mask.to(device), texts


def generate_group_vllm(engine, tokenizer, prompts, group_size, temperature, top_p, max_new_tokens, device):
    # vllm samples group_size completions per prompt; we keep only token-ids+text and
    # recompute logprobs on the hf model, so the sampler's own logprobs (warped by
    # temperature/top_p) never touch training. include_stop_str keeps the closing
    # </answer> so compute_reward can still see the tag.
    from vllm import SamplingParams

    sampling = SamplingParams(
        n=group_size,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=[ANSWER_STOP_STRING],
        include_stop_str_in_output=True,
    )
    outputs = engine.generate(prompts, sampling)
    return _vllm_outputs_to_tensors(outputs, tokenizer.pad_token_id, device)


@torch.no_grad()
def sync_weights_to_vllm(engine, hf_model):
    # push the freshly-stepped hf weights into the in-process vllm model so the next
    # rollout is on-policy. we hand over the unfused hf state_dict; vllm's
    # Qwen2ForCausalLM.load_weights fuses q/k/v->qkv_proj and gate/up->gate_up_proj
    # itself. qwen's state_dict materializes lm_head.weight even though it's tied to
    # embed_tokens; vllm accepts it (spike-confirmed on vllm 0.8.5), so we pass it as-is.
    model = engine.llm_engine.model_executor.driver_worker.model_runner.model
    model.load_weights((name, param.detach()) for name, param in hf_model.state_dict().items())


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
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smol135m")
    parser.add_argument("--rollout", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--train-target", type=int, default=GSM8K_TRAIN_TARGET)
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
    # save the full resume state every N steps mid-pass (not just at pass boundaries) so a
    # preempted pod loses minutes of work instead of a whole ~1.8h pass. 0 keeps the old
    # pass-only cadence, so smol135m stays byte-identical.
    parser.add_argument("--ckpt-every-steps", type=int, default=0)
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


def parse_args_with_profile(argv=None):
    # peek at --profile first, then make that profile's values the argparse defaults
    # so every knob the user didn't pass comes from the profile (explicit cli flags
    # still win). smol135m's values equal the module globals, so the default path is
    # byte-identical to before. the nested "vllm" block is not an arg, so skip it.
    parser = build_arg_parser()
    pre, _ = parser.parse_known_args(argv)
    overrides = {k: v for k, v in PROFILES[pre.profile].items() if k != "vllm"}
    parser.set_defaults(**overrides)
    return parser.parse_args(argv)


def run_training(args):
    validate_baseline_contract(args, torch.cuda.is_available())
    set_seed(GSM8K_SEED)
    device = args.device
    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    train_examples, _held_out = load_gsm8k_splits(
        seed=GSM8K_SEED, held_out=GSM8K_HELD_OUT, train_target=args.train_target,
    )
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.revision, device)
    # build the vllm engine after the hf model is on gpu but before the optimizer, so
    # vllm profiles its kv cache against a near-empty device and reserves its footprint
    # before adamw state grows into the rest.
    engine = build_vllm_engine(args) if args.rollout == "vllm" else None
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    prompts_per_pass = len(train_examples)
    prompt_batches_per_pass = math.ceil(prompts_per_pass / args.prompt_batch_size)
    total_steps = args.num_passes * prompt_batches_per_pass

    start_pass = 0
    start_batch = 0
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
        # batch_in_pass is the next unfinished batch within start_pass; older pass-only
        # checkpoints don't carry it, so default to 0 (start of the pass)
        start_batch = counters.get("batch_in_pass", 0)
        global_step = counters["global_step"]
        prompts_seen = counters["prompts_seen"]
        completions_seen = counters["completions_seen"]
        # the fresh vllm engine holds base weights; mirror the restored (trained) weights
        # in so the very first rollout after resume is on-policy, not off the base model
        if engine is not None:
            sync_weights_to_vllm(engine, model)
        print(f"resumed at pass {start_pass}, batch {start_batch}, global_step {global_step}")

    for pass_idx in range(start_pass, args.num_passes):
        rng = random.Random(GSM8K_SEED + pass_idx)
        order = list(range(prompts_per_pass))
        rng.shuffle(order)
        shuffled = [train_examples[i] for i in order]

        # the per-pass shuffle is re-derived from the same seed, so skipping the batches
        # already done before the checkpoint replays the exact same ordering
        skip = start_batch if pass_idx == start_pass else 0
        for batch_idx, batch in enumerate(batched(shuffled, args.prompt_batch_size)):
            if batch_idx < skip:
                continue
            questions = [ex["question"] for ex in batch]
            golds = [gsm8k_gold_answer(ex["answer"]) for ex in batch]
            prompts = [build_prompt(q) for q in questions]
            batch_size = len(batch)

            model.eval()
            if args.rollout == "vllm":
                full_ids, full_mask, completion_mask, texts = generate_group_vllm(
                    engine, tokenizer, prompts, args.group_size,
                    args.temperature, args.top_p, args.max_new_tokens, device,
                )
            else:
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
            # keep the next rollout on-policy by mirroring the updated weights into vllm
            if engine is not None:
                sync_weights_to_vllm(engine, model)

            global_step += 1
            prompts_seen += batch_size
            completions_seen += num_rows
            print(
                f"pass {pass_idx + 1}/{args.num_passes} step {global_step}/{total_steps} "
                f"lr {lr:.3e} reward {rewards.mean().item():.3f}"
            )

            # mid-pass resume point. counters name the NEXT batch to run so a resume
            # skips exactly what's finished. modal's volume background-commits this within
            # a few seconds, so a preemption right after costs at most ckpt_every_steps.
            if args.ckpt_every_steps and global_step % args.ckpt_every_steps == 0:
                counters = {
                    "pass": pass_idx, "batch_in_pass": batch_idx + 1,
                    "global_step": global_step,
                    "prompts_seen": prompts_seen, "completions_seen": completions_seen,
                }
                save_checkpoint(ckpt_path, model, optimizer, capture_rng_state(), counters, baseline_contract(args))

        counters = {
            "pass": pass_idx + 1, "batch_in_pass": 0, "global_step": global_step,
            "prompts_seen": prompts_seen, "completions_seen": completions_seen,
        }
        save_checkpoint(ckpt_path, model, optimizer, capture_rng_state(), counters, baseline_contract(args))
        pass_ckpt_path = os.path.join(run_dir, f"pass_{pass_idx + 1:02d}.pt")
        atomic_torch_save(
            {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()},
            pass_ckpt_path,
        )
        print(f"saved pass {pass_idx + 1} checkpoint to {pass_ckpt_path}")

    # expected totals are profile-derived, not the smol135m globals
    expected_prompts = args.num_passes * args.train_target
    assert_counts(prompts_seen, completions_seen, expected_prompts, expected_prompts * args.group_size)
    print(f"training complete: {prompts_seen} prompts, {completions_seen} completions")


def self_test():
    print("running train_grpo self-test...")

    p = build_prompt("  what is 2+2?  ")
    assert "what is 2+2?" in p
    assert "<answer>number</answer>" in p
    assert "What is 2 + 3?\nReasoning: 2 + 3 = 5. <answer>5</answer>" in p
    assert p.endswith("Question: what is 2+2?\nReasoning:")

    # parse_answer is now tag-only string extraction; validity is math_verify's job
    val, has_tag = parse_answer("some reasoning <answer>42</answer>")
    assert has_tag and val == "42"
    val, has_tag = parse_answer("no tag here")
    assert not has_tag and val is None
    val, has_tag = parse_answer("<answer>  42  </answer>")
    assert has_tag and val == "42"
    val, has_tag = parse_answer("<answer></answer>")
    assert not has_tag and val is None

    gold = gsm8k_gold_answer("some reasoning\n#### 18")
    assert gold == 18.0

    # a well-formed tag whose contents match gold under math_verify -> 1.1
    assert abs(compute_reward("reasoning <answer>18</answer>", 18.0) - 1.1) < 1e-9
    # wrong value but valid tag -> format bonus only
    assert abs(compute_reward("reasoning <answer>19</answer>", 18.0) - 0.1) < 1e-9
    # no tag -> nothing
    assert abs(compute_reward("reasoning no tag", 18.0) - 0.0) < 1e-9
    # formatting the old numeric matcher rejected but are genuinely correct
    for messy in ("$18", "18.", "18 dollars", "= 18", "+18", "1,800"):
        gold = 1800.0 if messy == "1,800" else 18.0
        assert abs(compute_reward(f"r <answer>{messy}</answer>", gold) - 1.1) < 1e-9

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
        profile="smol135m", train_target=GSM8K_TRAIN_TARGET,
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

    # vllm tensor reconstruction: prompt-major group-contiguous order, left-pad, and
    # completion-only masking, checked against fake RequestOutputs (no vllm/gpu).
    class _FakeComp:
        def __init__(self, token_ids, text):
            self.token_ids, self.text = token_ids, text

    class _FakeOut:
        def __init__(self, prompt_token_ids, outputs):
            self.prompt_token_ids, self.outputs = prompt_token_ids, outputs

    fake_outputs = [
        _FakeOut([1, 1, 1], [_FakeComp([7, 8], "a</answer>"), _FakeComp([9], "b</answer>")]),
        _FakeOut([2, 2], [_FakeComp([3, 4, 5], "c</answer>"), _FakeComp([6], "d</answer>")]),
    ]
    fi, fm, cm, ft = _vllm_outputs_to_tensors(fake_outputs, pad_id=0, device="cpu")
    assert fi.shape == fm.shape == cm.shape == (4, 5)  # 2 prompts * group 2, seq_len 5
    assert ft == ["a</answer>", "b</answer>", "c</answer>", "d</answer>"]
    assert fi[0].tolist() == [1, 1, 1, 7, 8] and cm[0].tolist() == [0, 0, 0, 1, 1]
    assert fi[1].tolist() == [0, 1, 1, 1, 9] and fm[1].tolist() == [0, 1, 1, 1, 1]
    assert cm[1].tolist() == [0, 0, 0, 0, 1]
    assert fi[2].tolist() == [2, 2, 3, 4, 5] and cm[2].tolist() == [0, 0, 1, 1, 1]

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
    args = parse_args_with_profile()
    if args.self_test:
        self_test()
        return
    run_training(args)


if __name__ == "__main__":
    main()
