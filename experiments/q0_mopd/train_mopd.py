import argparse
import gc
import json
import math
import os
import random
import tempfile

import torch

import train_grpo as tg


# final-stage on-policy distillation from the top-ranked q0 snapshots
TRAIN_TARGET = 512
PROMPT_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 4
GROUP_SIZE = 1
PEAK_LR = 1e-6
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
SNAPSHOT_COUNT = 5
TEACHER_COUNT = 4
VALIDATION_TARGET = 256
VALIDATION_BATCH_SIZE = 8
CANDIDATE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
Q0_TRAIN_TARGET = tg.GSM8K_TRAIN_TARGET


def candidate_steps(train_target, prompt_batch_size):
    expected_steps = math.ceil(train_target / prompt_batch_size)
    steps = tuple(sorted({max(1, math.ceil(expected_steps * fraction)) for fraction in CANDIDATE_FRACTIONS}))
    return expected_steps, steps


def validate_mixture_manifest(q0_run_dir, teacher_count=TEACHER_COUNT):
    mixture_path = os.path.join(q0_run_dir, "mixture_weights.json")
    if not os.path.isfile(mixture_path):
        raise ValueError(f"missing required mixture weights file: {mixture_path}")
    with open(mixture_path) as handle:
        payload = json.load(handle)

    names = payload.get("snapshot_paths")
    if not isinstance(names, list) or not 1 <= len(names) <= SNAPSHOT_COUNT:
        raise ValueError(f"mixture_weights.json must contain 1-{SNAPSHOT_COUNT} snapshots")
    if len(set(names)) != len(names) or any(not isinstance(name, str) for name in names):
        raise ValueError("mixture snapshot names must be unique strings")
    if any(os.path.basename(name) != name for name in names):
        raise ValueError("mixture snapshot paths must be file names")
    top_k = payload.get("top_k")
    if teacher_count <= 0:
        raise ValueError("teacher count must be positive")
    selected_count = min(teacher_count, len(names))
    teacher_key = str(selected_count)
    if not isinstance(top_k, dict) or not isinstance(top_k.get(teacher_key), dict):
        raise ValueError(f"mixture_weights.json must contain a top_k[{teacher_key}] object")
    selected_entry = top_k[teacher_key]
    selected_names = selected_entry.get("snapshot_paths")
    if (not isinstance(selected_names, list) or len(selected_names) != selected_count
            or len(set(selected_names)) != selected_count
            or any(not isinstance(name, str) or os.path.basename(name) != name for name in selected_names)):
        raise ValueError(f"top_k[{teacher_key}] must contain {selected_count} unique snapshot file names")
    if any(name not in names for name in selected_names):
        raise ValueError(f"top_k[{teacher_key}] snapshot names must exist in the root snapshot list")
    selected_paths = [os.path.join(q0_run_dir, name) for name in selected_names]
    missing = [path for path in selected_paths if not os.path.isfile(path)]
    if missing:
        raise ValueError(f"mixture references missing snapshots: {missing}")
    return mixture_path, selected_names, selected_paths


def validate_config(args, cuda_available):
    # the frozen knobs come from the selected profile, mirroring q0's
    # validate_immutable_training_contract; smol135m's profile equals the module globals.
    profile = tg.PROFILES[args.profile]
    fixed = {
        "model_name": profile["model_name"],
        "revision": profile["revision"],
        "group_size": profile["group_size"],
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "max_new_tokens": profile["max_new_tokens"],
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{name} must be {expected!r}")
    if args.device != "cuda":
        raise ValueError("device must be cuda")
    if not cuda_available:
        raise ValueError("cuda is required")
    if args.prompt_batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.validation_batch_size <= 0 or args.teacher_count <= 0:
        raise ValueError("validation batch size and teacher count must be positive")
    if args.teacher_count > SNAPSHOT_COUNT:
        raise ValueError("teacher count cannot exceed retained snapshot count")
    if getattr(args, "resume", False):
        raise ValueError("resume is not supported for mopd")
    if args.train_target <= 0 or args.validation_target <= 0:
        raise ValueError("train and validation targets must be positive")
    expected_steps, _candidate_steps = candidate_steps(args.train_target, args.prompt_batch_size)
    if expected_steps < 1:
        raise ValueError("training target must produce at least one step")


def uniform_forward_kl(teacher_logits, student_logits, mask):
    if not teacher_logits:
        raise ValueError("at least one teacher is required")
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    per_token = torch.zeros(student_logits.shape[:-1], device=student_logits.device)
    for logits in teacher_logits:
        teacher_log_probs = torch.log_softmax(logits.detach().float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        per_token = per_token + (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    per_token = per_token / len(teacher_logits)
    mask = mask.float()
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def teachers_forward_kl(teachers, input_ids, attention_mask, student_logits, mask):
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    total = torch.zeros(student_logits.shape[:-1], device=student_logits.device)
    for teacher in teachers:
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits[:, :-1, :].float()
            teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()
        total = total + (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    mask = mask.float()
    return (total / len(teachers) * mask).sum() / mask.sum().clamp_min(1.0)


def atomic_jsonl_dump(rows, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-", suffix=".jsonl", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_validation_examples(validation_target):
    from datasets import load_dataset

    dataset = load_dataset(tg.GSM8K_DATASET, "main", split="train").shuffle(seed=tg.GSM8K_SEED)
    start = tg.GSM8K_HELD_OUT + Q0_TRAIN_TARGET
    examples = dataset.select(range(start, start + validation_target))
    if len(examples) != validation_target:
        raise ValueError(f"expected {validation_target} validation examples, got {len(examples)}")
    return list(examples)


def checkpoint_state(model):
    return {
        key: value.detach().to(device="cpu", dtype=torch.bfloat16)
        if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def select_best_candidate(rows):
    if not rows:
        raise ValueError("at least one validation result is required")
    return max(rows, key=lambda row: (
        row["pass_at_8"], row["pass_at_4"], row["pass_at_1"], row["parse_at_8"], row["step"],
    ))


def evaluate_candidates(candidate_paths, tokenizer, validation_examples, args, device):
    import eval as evaluation

    rows = []
    for step, path in candidate_paths:
        model = tg.load_model_from_checkpoint(
            args.model_name, args.revision, path, device,
            args.quantization, args.lora_rank, args.lora_alpha, args.lora_dropout,
        )
        result = evaluation.evaluate_pass_at_k(
            model, tokenizer, validation_examples, args.validation_batch_size,
            args.max_new_tokens, device, source=path,
        )
        result["step"] = step
        result["checkpoint"] = os.path.basename(path)
        rows.append(result)
        print(
            f"validation step {step}: pass@1 {result['pass_at_1']:.4f} "
            f"pass@4 {result['pass_at_4']:.4f} pass@8 {result['pass_at_8']:.4f}"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def save_selected_candidate(run_dir, validation_rows, validation_count):
    best = select_best_candidate(validation_rows)
    best_path = os.path.join(run_dir, best["checkpoint"])
    tg.atomic_torch_save(
        torch.load(best_path, map_location="cpu", weights_only=True),
        os.path.join(run_dir, "opsd.pt"),
    )
    tg.atomic_json_dump(
        {
            "validation_source": "fresh deterministic slice from the unused gsm8k train remainder",
            "validation_dataset": tg.GSM8K_DATASET,
            "validation_seed": tg.GSM8K_SEED,
            "validation_start": tg.GSM8K_HELD_OUT + Q0_TRAIN_TARGET,
            "validation_count": validation_count,
            "selection_order": ["pass_at_8", "pass_at_4", "pass_at_1", "parse_at_8", "later_step"],
            "selected_step": best["step"],
            "selected_checkpoint": best["checkpoint"],
            "candidates": validation_rows,
        },
        os.path.join(run_dir, "validation_results.json"),
    )
    return best


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="one-pass multi-teacher on-policy distillation on gsm8k"
    )
    parser.add_argument("--profile", choices=sorted(tg.PROFILES), default="smol135m")
    parser.add_argument("--model-name", default=tg.MODEL_NAME)
    parser.add_argument("--revision", default=tg.MODEL_REVISION)
    parser.add_argument("--q0-run-dir", default="runs/q0_grpo")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-id", default="mopd")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--temperature", type=float, default=tg.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=tg.TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=tg.MAX_NEW_TOKENS)
    parser.add_argument("--prompt-batch-size", type=int, default=PROMPT_BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--train-target", type=int, default=TRAIN_TARGET)
    parser.add_argument("--validation-target", type=int, default=VALIDATION_TARGET)
    parser.add_argument("--validation-batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    parser.add_argument("--teacher-count", type=int, default=TEACHER_COUNT)
    parser.add_argument("--lr", type=float, default=PEAK_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--phase", choices=["training", "validation"], default="training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_training(args):
    validate_config(args, torch.cuda.is_available())
    tg.set_seed(tg.GSM8K_SEED)
    _mixture_path, _teacher_names, teacher_paths = validate_mixture_manifest(
        args.q0_run_dir, args.teacher_count,
    )
    device = args.device
    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    train_examples, _held_out = tg.load_gsm8k_splits(
        seed=tg.GSM8K_SEED, held_out=tg.GSM8K_HELD_OUT, train_target=Q0_TRAIN_TARGET,
    )
    train_examples = train_examples[:args.train_target]
    validation_examples = load_validation_examples(args.validation_target)
    expected_steps, candidate_steps_for_run = candidate_steps(args.train_target, args.prompt_batch_size)
    if len(train_examples) < args.train_target:
        raise ValueError(f"expected {args.train_target} training examples, got {len(train_examples)}")
    student, tokenizer = tg.build_model_and_tokenizer(
        args.model_name, args.revision, device, args.quantization,
        args.lora_rank, args.lora_alpha, args.lora_dropout,
    )
    teachers = []
    teacher = None
    try:
        for path in teacher_paths:
            teacher = tg.load_model_from_checkpoint(
                args.model_name, args.revision, path, device,
                args.quantization, args.lora_rank, args.lora_alpha, args.lora_dropout,
            )
            teacher.requires_grad_(False)
            teacher.eval()
            teachers.append(teacher)
        optimizer = tg.build_optimizer(student, args.lr, args.weight_decay)
        order = list(range(args.train_target))
        random.Random(tg.GSM8K_SEED).shuffle(order)
        shuffled = [train_examples[index] for index in order]
        metrics = []
        prompts_seen = 0
        completion_tokens_seen = 0
        candidate_paths = []

        for step, batch in enumerate(tg.batched(shuffled, args.prompt_batch_size), start=1):
            prompts = [tg.build_prompt(example["question"]) for example in batch]
            student.eval()
            full_ids, full_mask, completion_mask, _texts = tg.generate_group(
                student, tokenizer, prompts, args.group_size, args.temperature, args.top_p,
                args.max_new_tokens, device,
            )
            lr = tg.cosine_lr(step - 1, expected_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            student.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            completion_token_mask = completion_mask[:, 1:]
            valid_tokens = completion_token_mask.sum().clamp_min(1.0)
            for lo in range(0, full_ids.shape[0], args.micro_batch_size):
                hi = min(lo + args.micro_batch_size, full_ids.shape[0])
                student_logits = student(
                    input_ids=full_ids[lo:hi], attention_mask=full_mask[lo:hi], use_cache=False
                ).logits[:, :-1, :]
                micro_mask = completion_token_mask[lo:hi]
                micro_loss = teachers_forward_kl(
                    teachers, full_ids[lo:hi], full_mask[lo:hi], student_logits, micro_mask
                )
                fraction = micro_mask.sum() / valid_tokens
                (micro_loss * fraction).backward()
                step_loss = step_loss + micro_loss.detach() * fraction
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            batch_tokens = int(completion_token_mask.sum().item())
            prompts_seen += len(batch)
            completion_tokens_seen += batch_tokens
            metrics.append({
                "step": step,
                "lr": lr,
                "loss": float(step_loss.item()),
                "forward_kl": float(step_loss.item()),
                "avg_response_length": float(completion_mask[:, 1:].sum(dim=1).float().mean().item()),
                "prompts_seen": prompts_seen,
                "completion_tokens": completion_tokens_seen,
            })
            print(f"step {step}/{expected_steps} lr {lr:.3e} loss {step_loss.item():.4f}")
            if step in candidate_steps_for_run:
                candidate_path = os.path.join(run_dir, f"candidate_step{step:04d}.pt")
                if args.quantization == "4bit":
                    tg.save_adapter(student, candidate_path)
                else:
                    tg.atomic_torch_save(checkpoint_state(student), candidate_path)
                candidate_paths.append((step, candidate_path))
                print(f"saved validation candidate at step {step}")

        if prompts_seen != args.train_target or len(metrics) != expected_steps:
            raise RuntimeError("mopd training count mismatch")
        atomic_jsonl_dump(metrics, os.path.join(run_dir, "training_metrics.jsonl"))
        del optimizer, student
        teachers.clear()
        teacher = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del _texts, full_ids, full_mask, completion_mask
        gc.collect()
        validation_rows = evaluate_candidates(
            candidate_paths, tokenizer, validation_examples, args, device,
        )
        best = save_selected_candidate(run_dir, validation_rows, len(validation_examples))
        print(
            f"mopd complete: selected step {best['step']} after {prompts_seen} prompts "
            f"and {completion_tokens_seen} completion tokens"
        )
    finally:
        teachers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_selection(args):
    validate_config(args, torch.cuda.is_available())
    tg.set_seed(tg.GSM8K_SEED)
    run_dir = os.path.join(args.output_dir, args.run_id)
    _expected_steps, candidate_steps_for_run = candidate_steps(args.train_target, args.prompt_batch_size)
    candidate_paths = [
        (step, os.path.join(run_dir, f"candidate_step{step:04d}.pt"))
        for step in candidate_steps_for_run
    ]
    missing = [path for _, path in candidate_paths if not os.path.isfile(path)]
    if missing:
        raise ValueError(f"missing validation candidates: {missing}")
    tokenizer = tg.load_tokenizer(args.model_name, args.revision)
    validation_examples = load_validation_examples(args.validation_target)
    rows = evaluate_candidates(candidate_paths, tokenizer, validation_examples, args, args.device)
    best = save_selected_candidate(run_dir, rows, len(validation_examples))
    print(f"pass@k selection complete: selected step {best['step']}")


def self_test():
    assert candidate_steps(512, 16) == (32, (8, 16, 24, 32))
    candidates = [
        {"step": 8, "pass_at_8": 0.4, "pass_at_4": 0.3, "pass_at_1": 0.1, "parse_at_8": 0.8},
        {"step": 16, "pass_at_8": 0.5, "pass_at_4": 0.2, "pass_at_1": 0.1, "parse_at_8": 0.7},
        {"step": 24, "pass_at_8": 0.5, "pass_at_4": 0.3, "pass_at_1": 0.1, "parse_at_8": 0.8},
        {"step": 32, "pass_at_8": 0.5, "pass_at_4": 0.3, "pass_at_1": 0.1, "parse_at_8": 0.8},
    ]
    assert select_best_candidate(candidates)["step"] == 32

    teacher_a = torch.tensor([[[1.0, -1.0], [0.0, 2.0]]])
    teacher_b = teacher_a.clone()
    student = teacher_a.clone().requires_grad_(True)
    all_mask = torch.ones(1, 2)
    zero = uniform_forward_kl([teacher_a, teacher_b], student, all_mask)
    assert zero.item() < 1e-7

    changed_teacher = teacher_a.clone()
    changed_teacher[:, :, 0] += 2.0
    different = uniform_forward_kl([changed_teacher, teacher_b], student, all_mask)
    assert different.item() > 0.0
    second_changed = teacher_a.clone()
    second_changed[:, 1, 0] += 2.0
    masked = uniform_forward_kl([second_changed, teacher_b], student, torch.tensor([[1.0, 0.0]]))
    assert masked.item() < 1e-7

    teacher_with_grad = teacher_a.clone().requires_grad_(True)
    student_with_grad = teacher_a.clone().requires_grad_(True)
    loss = uniform_forward_kl([teacher_with_grad], student_with_grad, all_mask)
    loss.backward()
    assert teacher_with_grad.grad is None
    assert student_with_grad.grad is not None

    with tempfile.TemporaryDirectory() as directory:
        names = [f"snapshot_{index}.pt" for index in range(SNAPSHOT_COUNT)]
        selected = [names[3], names[2], names[4], names[1]]
        top_k = {"4": {"snapshot_paths": selected, "weights": [99, -3, 0, 12]}}
        with open(os.path.join(directory, "mixture_weights.json"), "w") as handle:
            json.dump({"snapshot_paths": names, "weights": "ignored", "top_k": top_k}, handle)
        for name in names:
            open(os.path.join(directory, name), "wb").close()
        _path, selected_names, selected_paths = validate_mixture_manifest(directory)
        assert selected_names == selected
        assert all(os.path.isfile(path) for path in selected_paths)
        top_k["4"]["snapshot_paths"][1] = selected[0]
        with open(os.path.join(directory, "mixture_weights.json"), "w") as handle:
            json.dump({"snapshot_paths": names, "weights": "ignored", "top_k": top_k}, handle)
        try:
            validate_mixture_manifest(directory)
            raise RuntimeError("expected manifest rejection")
        except ValueError:
            pass

    args = build_arg_parser().parse_args([])
    args.device = "cuda"
    validate_config(args, cuda_available=True)
    args.resume = True
    try:
        validate_config(args, cuda_available=True)
        raise RuntimeError("expected resume rejection")
    except ValueError:
        pass
    print("train_mopd self-test passed")


def main():
    args = tg.parse_args_with_profile(parser=build_arg_parser())
    if args.self_test:
        self_test()
    elif args.select_only or args.phase == "validation":
        run_selection(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
