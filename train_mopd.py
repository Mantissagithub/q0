import argparse
import json
import os
import random
import tempfile

import torch

import train_grpo as tg


# final-stage on-policy distillation from the top-ranked q0 snapshots
TRAIN_TARGET = 2048
PROMPT_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 16
GROUP_SIZE = 1
PEAK_LR = 1e-6
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
SNAPSHOT_COUNT = 8
TEACHER_COUNT = 4
EXPECTED_STEPS = TRAIN_TARGET // PROMPT_BATCH_SIZE


def validate_mixture_manifest(q0_run_dir):
    mixture_path = os.path.join(q0_run_dir, "mixture_weights.json")
    if not os.path.isfile(mixture_path):
        raise ValueError(f"missing required mixture weights file: {mixture_path}")
    with open(mixture_path) as handle:
        payload = json.load(handle)

    names = payload.get("snapshot_paths")
    if not isinstance(names, list) or len(names) != SNAPSHOT_COUNT:
        raise ValueError(f"mixture_weights.json must contain exactly {SNAPSHOT_COUNT} snapshots")
    if len(set(names)) != SNAPSHOT_COUNT or any(not isinstance(name, str) for name in names):
        raise ValueError("mixture snapshot names must be unique strings")
    if any(os.path.basename(name) != name for name in names):
        raise ValueError("mixture snapshot paths must be file names")
    top_k = payload.get("top_k")
    if not isinstance(top_k, dict) or not isinstance(top_k.get("4"), dict):
        raise ValueError("mixture_weights.json must contain a top_k[4] object")
    selected_names = top_k["4"].get("snapshot_paths")
    if (not isinstance(selected_names, list) or len(selected_names) != TEACHER_COUNT
            or len(set(selected_names)) != TEACHER_COUNT
            or any(not isinstance(name, str) or os.path.basename(name) != name for name in selected_names)):
        raise ValueError(f"top_k[4] must contain {TEACHER_COUNT} unique snapshot file names")
    if any(name not in names for name in selected_names):
        raise ValueError("top_k[4] snapshot names must exist in the root snapshot list")
    selected_paths = [os.path.join(q0_run_dir, name) for name in selected_names]
    missing = [path for path in selected_paths if not os.path.isfile(path)]
    if missing:
        raise ValueError(f"mixture references missing snapshots: {missing}")
    return mixture_path, selected_names, selected_paths


def validate_config(args, cuda_available):
    fixed = {
        "model_name": tg.MODEL_NAME,
        "revision": tg.MODEL_REVISION,
        "group_size": GROUP_SIZE,
        "temperature": tg.TEMPERATURE,
        "top_p": tg.TOP_P,
        "max_new_tokens": tg.MAX_NEW_TOKENS,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "lr": PEAK_LR,
        "weight_decay": WEIGHT_DECAY,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{name} must be {expected!r}")
    if args.device != "cuda":
        raise ValueError("device must be cuda")
    if not cuda_available:
        raise ValueError("cuda is required")
    if getattr(args, "resume", False):
        raise ValueError("resume is not supported for mopd")
    if TRAIN_TARGET % PROMPT_BATCH_SIZE:
        raise ValueError("training target must divide evenly into prompt batches")


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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="one-pass multi-teacher on-policy distillation on gsm8k"
    )
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
    parser.add_argument("--lr", type=float, default=PEAK_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_training(args):
    validate_config(args, torch.cuda.is_available())
    tg.set_seed(tg.GSM8K_SEED)
    _mixture_path, _teacher_names, teacher_paths = validate_mixture_manifest(args.q0_run_dir)
    device = args.device
    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    train_examples, _held_out = tg.load_gsm8k_splits(seed=tg.GSM8K_SEED, held_out=tg.GSM8K_HELD_OUT)
    if len(train_examples) != TRAIN_TARGET:
        raise ValueError(f"expected {TRAIN_TARGET} training examples, got {len(train_examples)}")
    student, tokenizer = tg.build_model_and_tokenizer(args.model_name, args.revision, device)
    teachers = []
    try:
        for path in teacher_paths:
            teacher = tg.load_model_from_checkpoint(args.model_name, args.revision, path, device)
            teacher.requires_grad_(False)
            teacher.eval()
            teachers.append(teacher)
        optimizer = tg.build_optimizer(student, PEAK_LR, WEIGHT_DECAY)
        order = list(range(TRAIN_TARGET))
        random.Random(tg.GSM8K_SEED).shuffle(order)
        shuffled = [train_examples[index] for index in order]
        metrics = []
        prompts_seen = 0
        completion_tokens_seen = 0

        for step, batch in enumerate(tg.batched(shuffled, PROMPT_BATCH_SIZE), start=1):
            prompts = [tg.build_prompt(example["question"]) for example in batch]
            student.eval()
            full_ids, full_mask, completion_mask, _texts = tg.generate_group(
                student, tokenizer, prompts, GROUP_SIZE, tg.TEMPERATURE, tg.TOP_P,
                tg.MAX_NEW_TOKENS, device,
            )
            lr = tg.cosine_lr(step - 1, EXPECTED_STEPS, PEAK_LR)
            for group in optimizer.param_groups:
                group["lr"] = lr

            student.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            completion_token_mask = completion_mask[:, 1:]
            valid_tokens = completion_token_mask.sum().clamp_min(1.0)
            for lo in range(0, full_ids.shape[0], MICRO_BATCH_SIZE):
                hi = min(lo + MICRO_BATCH_SIZE, full_ids.shape[0])
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
            print(f"step {step}/{EXPECTED_STEPS} lr {lr:.3e} loss {step_loss.item():.4f}")

        if prompts_seen != TRAIN_TARGET or len(metrics) != EXPECTED_STEPS:
            raise RuntimeError("mopd training count mismatch")
        tg.atomic_torch_save(
            {
                key: value.detach().to(device="cpu", dtype=torch.bfloat16)
                if value.is_floating_point() else value.detach().cpu()
                for key, value in student.state_dict().items()
            },
            os.path.join(run_dir, "opsd.pt"),
        )
        atomic_jsonl_dump(metrics, os.path.join(run_dir, "training_metrics.jsonl"))
        print(f"mopd complete: {prompts_seen} prompts, {completion_tokens_seen} completion tokens")
    finally:
        teachers.clear()


def self_test():
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
        selected = [names[3], names[6], names[4], names[5]]
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
    args = build_arg_parser().parse_args()
    if args.self_test:
        self_test()
    else:
        run_training(args)


if __name__ == "__main__":
    main()
