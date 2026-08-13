import argparse
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time

import torch
import torch.nn.functional as F

import train_grpo as tg

# q0 hyper-epoch grpo training on gsm8k: two sequential trajectories of cyclic
# lr/wd schedules, chain distillation against a frozen previous snapshot, a
# deterministic cycle-boundary weight perturbation, and a learned softmax
# mixture prior fit over the 8 snapshots at the end. minimal pip packages
# for a real run: pip install torch transformers datasets accelerate

TRAJECTORY_SEEDS = [42, 4242]
# local laptop profile: one initialization and at most five release snapshots.
LOCAL_TRAJECTORY_SEEDS = [42]
MAX_SNAPSHOTS_PER_TRAJECTORY = 5
# full-run cycles_per_trajectory = 8
CYCLES_PER_TRAJECTORY = 4

DISTILL_WARMUP_CYCLES = 2
DISTILL_ALPHA = 0.45
DISTILL_TEMPERATURE = 1.2

LR_MULT_START = 1.0
LR_MULT_END = 0.08
WD_MULT_START = 0.7
WD_MULT_END = 1.0

PERTURB_SCALE_MAX = 0.01
PERTURB_SCALE_MIN = 0.002

MIXTURE_OPT_STEPS = 300
MIXTURE_LR = 0.5
MIXTURE_WEIGHT_DECAY = 0.0

EXPECTED_PROMPTS = len(TRAJECTORY_SEEDS) * CYCLES_PER_TRAJECTORY * tg.GSM8K_TRAIN_TARGET
EXPECTED_COMPLETIONS = EXPECTED_PROMPTS * tg.GROUP_SIZE


def top_k_mixtures(snapshot_paths, weights):
    if not isinstance(snapshot_paths, list) or not isinstance(weights, list):
        raise ValueError("snapshot_paths and weights must be lists")
    if not snapshot_paths or len(snapshot_paths) != len(weights):
        raise ValueError("snapshot_paths and weights must have the same nonzero length")
    if any(not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight)
           for weight in weights):
        raise ValueError("weights must be finite numbers")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be nonnegative")
    if sum(weights) <= 0:
        raise ValueError("weights must have a positive total")

    ranked_indices = sorted(range(len(weights)), key=lambda index: (-weights[index], index))
    mixtures = {}
    for k in (1, 2, 3, 4, 8):
        selected_indices = ranked_indices[:min(k, len(ranked_indices))]
        selected_weights = [weights[index] for index in selected_indices]
        selected_total = sum(selected_weights)
        mixtures[str(k)] = {
            "snapshot_paths": [snapshot_paths[index] for index in selected_indices],
            "weights": [weight / selected_total for weight in selected_weights],
        }
    return mixtures


def enrich_mixture_file(path):
    with open(path) as f:
        payload = json.load(f)

    snapshot_paths = payload.get("snapshot_paths")
    alpha = payload.get("alpha")
    weights = payload.get("weights")
    if not isinstance(snapshot_paths, list) or not isinstance(alpha, list) or len(alpha) != len(snapshot_paths):
        raise ValueError("alpha must have the same length as snapshot_paths")

    payload["top_k"] = top_k_mixtures(snapshot_paths, weights)
    tg.atomic_json_dump(payload, path)
    return payload


def lr_multiplier(step_in_cycle, cycle_length):
    # linear decay from 1.0 down to 0.08 across one cycle
    if cycle_length <= 1:
        return LR_MULT_START
    progress = step_in_cycle / (cycle_length - 1)
    return LR_MULT_START + (LR_MULT_END - LR_MULT_START) * progress


def wd_multiplier(step_in_cycle, cycle_length):
    # anti-correlated with lr: linear from 0.7 up to 1.0 across one cycle
    if cycle_length <= 1:
        return WD_MULT_START
    progress = step_in_cycle / (cycle_length - 1)
    return WD_MULT_START + (WD_MULT_END - WD_MULT_START) * progress


def should_distill(cycle_num):
    return cycle_num > DISTILL_WARMUP_CYCLES


def perturb_scale_for_cycle(cycle_num, num_cycles):
    # cosine decay of perturbation scale across the nonfinal cycle boundaries
    num_boundaries = num_cycles - 1
    if num_boundaries <= 1:
        return PERTURB_SCALE_MAX
    progress = (cycle_num - 1) / (num_boundaries - 1)
    return PERTURB_SCALE_MIN + (PERTURB_SCALE_MAX - PERTURB_SCALE_MIN) * 0.5 * (1.0 + math.cos(math.pi * progress))


def perturb_weights(model, scale, seed):
    # deterministic, per-parameter, param-std-scaled gaussian noise
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            # only perturb trainable params. under QLoRA the frozen base is packed
            # 4-bit (Params4bit); adding gaussian noise to it corrupts the base
            # weights, so restrict perturbation to the LoRA adapter.
            if p.numel() > 1 and p.requires_grad:
                std = p.detach().float().std().cpu()
                noise = torch.randn(p.shape, generator=rng, dtype=torch.float32) * (std * scale)
                p.add_(noise.to(device=p.device, dtype=p.dtype))


def forward_kl_teacher_student(student_logits, teacher_logits, mask, temperature):
    # kl(teacher || student) over the full vocab, restricted to masked positions
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    teacher_p = teacher_logp.exp()
    per_token_kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_token_kl * mask).sum() / denom


def forward_kl_on_completions(student_model, teacher_model, input_ids, attention_mask, completion_mask, temperature):
    student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
    mask = completion_mask[:, 1:]
    return forward_kl_teacher_student(student_logits, teacher_logits, mask, temperature)


def build_teacher(args, device, snapshot_path):
    model = tg.load_model_from_checkpoint(
        args.model_name, args.revision, snapshot_path, device,
        args.quantization, args.lora_rank, args.lora_alpha, args.lora_dropout,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def build_fitness_target(example):
    # gsm8k official reasoning plus our answer tag, the gold continuation used
    # to teacher-force each snapshot for fitness scoring
    reasoning, _, tail = example["answer"].partition("####")
    return f"{reasoning.strip()}\n<answer>{tail.strip()}</answer>"


@torch.no_grad()
def cache_pgt_for_model(model, tokenizer, held_out, device):
    # teacher-forces the gold reasoning/answer continuation for every held-out
    # example and returns a flat 1d tensor of per-token p(ground-truth-token)
    model.eval()
    all_probs = []
    for example in held_out:
        prompt = tg.build_prompt(example["question"])
        target = build_fitness_target(example)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        target_ids = tokenizer(
            target, return_tensors="pt", truncation=False, add_special_tokens=False,
        ).input_ids.to(device)
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        max_positions = getattr(model.config, "max_position_embeddings", None)
        if max_positions is None:
            raise ValueError("model config must define max_position_embeddings")
        if full_ids.shape[1] > max_positions:
            raise ValueError("gold continuation exceeds model context")
        logits = model(input_ids=full_ids, use_cache=False).logits[:, :-1, :].float()
        targets = full_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.shape[1]
        target_logprobs = token_logprobs[:, prompt_len - 1:]
        all_probs.append(target_logprobs.exp().squeeze(0).cpu())
    return torch.cat(all_probs)


def optimize_mixture_alpha(P_fit, opt_steps=MIXTURE_OPT_STEPS, lr=MIXTURE_LR,
                            weight_decay=MIXTURE_WEIGHT_DECAY, grad_clip=1.0, eps=1e-12, seed=0):
    # softmax mixture fit "inside the log": minimize mean -log(w @ P_fit)
    device = P_fit.device
    M = P_fit.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    alpha = torch.zeros(M, device=device, dtype=torch.float32)
    alpha = alpha + 1e-4 * torch.randn(M, device=device, generator=g)
    alpha.requires_grad_(True)
    opt = torch.optim.AdamW([alpha], lr=lr, weight_decay=weight_decay)
    loss = None
    for _ in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        w = torch.softmax(alpha, dim=0)
        q = (w @ P_fit).clamp_min(eps)
        loss = -torch.log(q).mean()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([alpha], max_norm=grad_clip)
        opt.step()
    return alpha.detach(), float(loss.detach().item())


def q0_contract(args):
    contract = tg.immutable_training_contract(args)
    contract.update({
        "cycles": args.cycles,
        "trajectory_seeds": list(args.trajectory_seeds),
        "num_initializations": args.num_initializations,
        "max_snapshots_per_trajectory": args.max_snapshots,
        "distill_warmup_cycles": DISTILL_WARMUP_CYCLES,
        "distill_alpha": DISTILL_ALPHA,
        "distill_temperature": DISTILL_TEMPERATURE,
        "lr_mult_start": LR_MULT_START,
        "lr_mult_end": LR_MULT_END,
        "wd_mult_start": WD_MULT_START,
        "wd_mult_end": WD_MULT_END,
        "perturb_scale_max": PERTURB_SCALE_MAX,
        "perturb_scale_min": PERTURB_SCALE_MIN,
        "mixture_opt_steps": MIXTURE_OPT_STEPS,
        "mixture_lr": MIXTURE_LR,
        "mixture_weight_decay": MIXTURE_WEIGHT_DECAY,
    })
    return contract


def validate_q0_contract(args, cuda_available):
    tg.validate_immutable_training_contract(args, cuda_available)
    if args.cycles != CYCLES_PER_TRAJECTORY:
        raise ValueError(f"cycles must be {CYCLES_PER_TRAJECTORY}")
    if args.num_initializations != len(args.trajectory_seeds):
        raise ValueError("num_initializations must match the number of trajectory seeds")
    if args.num_initializations != 1:
        raise ValueError("the local q0 workflow supports exactly one initialization")
    if args.max_snapshots > MAX_SNAPSHOTS_PER_TRAJECTORY:
        raise ValueError(f"max_snapshots must be <= {MAX_SNAPSHOTS_PER_TRAJECTORY}")
    if args.max_snapshots < args.cycles:
        raise ValueError("max_snapshots must cover every completed cycle")


def run_trajectory(traj_idx, seed, args, train_examples, run_dir, device):
    tg.set_seed(seed)
    model, tokenizer = tg.build_model_and_tokenizer(
        args.model_name, args.revision, device, args.quantization,
        args.lora_rank, args.lora_alpha, args.lora_dropout,
    )
    optimizer = tg.build_optimizer(model, args.lr, args.weight_decay)

    prompts_per_cycle = len(train_examples)
    cycle_length = math.ceil(prompts_per_cycle / args.prompt_batch_size)

    start_cycle = 0
    prompts_seen = 0
    completions_seen = 0

    ckpt_path = os.path.join(run_dir, f"traj{traj_idx}_checkpoint.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = tg.load_checkpoint(ckpt_path)
        tg.require_matching_contract(ckpt.get("contract"), q0_contract(args))
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        tg.restore_rng_state(ckpt["rng_state"])
        counters = ckpt["counters"]
        start_cycle = counters["cycle"]
        prompts_seen = counters["prompts_seen"]
        completions_seen = counters["completions_seen"]
        print(f"trajectory {traj_idx}: resumed from cycle {start_cycle}")

    teacher_model = None
    if start_cycle >= DISTILL_WARMUP_CYCLES and start_cycle > 0 and args.quantization != "4bit":
        teacher_path = os.path.join(run_dir, f"traj{traj_idx}_cycle{start_cycle:02d}.pt")
        teacher_model = build_teacher(args, device, teacher_path)

    snapshot_paths = [
        os.path.join(run_dir, f"traj{traj_idx}_cycle{c:02d}.pt") for c in range(1, start_cycle + 1)
    ]
    if len(snapshot_paths) > args.max_snapshots:
        raise ValueError("resume state references more snapshots than the configured retention limit")

    for cycle_idx in range(start_cycle, args.cycles):
        cycle_num = cycle_idx + 1
        rng = random.Random(seed + 1000 * cycle_num)
        order = list(range(prompts_per_cycle))
        rng.shuffle(order)
        shuffled = [train_examples[i] for i in order]

        distilling = should_distill(cycle_num) and teacher_model is not None

        for step_in_cycle, batch in enumerate(tg.batched(shuffled, args.prompt_batch_size)):
            questions = [ex["question"] for ex in batch]
            golds = [tg.gsm8k_gold_answer(ex["answer"]) for ex in batch]
            prompts = [tg.build_prompt(q) for q in questions]
            batch_size = len(batch)

            model.eval()
            full_ids, full_mask, completion_mask, texts = tg.generate_group(
                model, tokenizer, prompts, args.group_size,
                args.temperature, args.top_p, args.max_new_tokens, device,
            )
            golds_expanded = []
            for g in golds:
                golds_expanded.extend([g] * args.group_size)
            rewards = torch.tensor(
                [tg.compute_reward(t, g) for t, g in zip(texts, golds_expanded)], dtype=torch.float32,
            )
            advantages = tg.group_normalize_advantages(rewards, args.group_size).to(device)

            old_logprobs, token_mask = tg.old_policy_completion_token_logprobs(
                model, full_ids, full_mask, completion_mask, args.micro_batch_size,
            )
            total_valid_tokens = token_mask.sum().clamp_min(1.0)

            lrm = lr_multiplier(step_in_cycle, cycle_length)
            wdm = wd_multiplier(step_in_cycle, cycle_length)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lrm
                group["weight_decay"] = args.weight_decay * wdm

            model.train()
            optimizer.zero_grad(set_to_none=True)
            num_rows = full_ids.shape[0]
            num_micro = math.ceil(num_rows / args.micro_batch_size)
            for m in range(num_micro):
                lo = m * args.micro_batch_size
                hi = min(lo + args.micro_batch_size, num_rows)
                new_lp, micro_mask = tg.completion_token_logprobs(
                    model, full_ids[lo:hi], full_mask[lo:hi], completion_mask[lo:hi]
                )
                grpo_loss = tg.clipped_grpo_loss(
                    new_lp, old_logprobs[lo:hi], advantages[lo:hi], micro_mask, args.clip_epsilon
                )
                if distilling:
                    kl = forward_kl_on_completions(
                        model, teacher_model, full_ids[lo:hi], full_mask[lo:hi], completion_mask[lo:hi],
                        DISTILL_TEMPERATURE,
                    )
                    loss = (1 - DISTILL_ALPHA) * grpo_loss + DISTILL_ALPHA * (DISTILL_TEMPERATURE ** 2) * kl
                else:
                    loss = grpo_loss
                valid_fraction = micro_mask.sum() / total_valid_tokens
                (loss * valid_fraction).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            prompts_seen += batch_size
            completions_seen += num_rows
            print(
                f"[traj {traj_idx}] cycle {cycle_num}/{args.cycles} step {step_in_cycle + 1}/{cycle_length} "
                f"lrm {lrm:.3f} wdm {wdm:.3f} distill {int(distilling)} reward {rewards.mean().item():.3f}"
            )

        model.eval()
        snap_path = os.path.join(run_dir, f"traj{traj_idx}_cycle{cycle_num:02d}.pt")
        if args.quantization == "4bit":
            tg.save_adapter(model, snap_path)
        else:
            tg.atomic_torch_save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, snap_path)
        snapshot_paths.append(snap_path)
        if len(snapshot_paths) > args.max_snapshots:
            stale_path = snapshot_paths.pop(0)
            if stale_path != snap_path and os.path.exists(stale_path):
                os.unlink(stale_path)
                print(f"[traj {traj_idx}] pruned stale snapshot -> {stale_path}")
        print(f"[traj {traj_idx}] saved snapshot cycle {cycle_num} -> {snap_path}")

        # chain distillation needs a second resident model; on 8 GB VRAM a full-precision
        # teacher will not co-reside with the quantized student, so the laptop 4-bit
        # profile runs the cyclic trajectory + mixture prior without the distillation KL.
        if cycle_num >= DISTILL_WARMUP_CYCLES and args.quantization != "4bit":
            if teacher_model is None:
                teacher_model = tg.load_base_model(args.model_name, args.revision, device)
                teacher_model.eval()
                teacher_model.requires_grad_(False)
            teacher_model.load_state_dict(
                {k: v.to(device=device, dtype=teacher_model.dtype) for k, v in model.state_dict().items()}
            )

        if cycle_num < args.cycles:
            scale = perturb_scale_for_cycle(cycle_num, args.cycles)
            perturb_seed = seed * 100003 + cycle_num * 31337
            perturb_weights(model, scale, perturb_seed)
            print(f"[traj {traj_idx}] perturbed weights scale={scale:.5f} before cycle {cycle_num + 1}")

        counters = {"cycle": cycle_num, "prompts_seen": prompts_seen, "completions_seen": completions_seen}
        tg.save_checkpoint(ckpt_path, model, optimizer, tg.capture_rng_state(), counters, q0_contract(args))
        model.train()

    del model, optimizer, teacher_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompts_seen, completions_seen, snapshot_paths


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_mixture_prior(args, snapshot_paths, held_out, run_dir, device):
    tokenizer = tg.load_tokenizer(args.model_name, args.revision)
    all_probs = []
    for path in snapshot_paths:
        model = tg.load_model_from_checkpoint(
            args.model_name, args.revision, path, device,
            args.quantization, args.lora_rank, args.lora_alpha, args.lora_dropout,
        )
        probs = cache_pgt_for_model(model, tokenizer, held_out, device)
        all_probs.append(probs)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    P_fit = torch.stack(all_probs).to(device)
    alpha, fit_loss = optimize_mixture_alpha(P_fit, seed=0)
    weights = torch.softmax(alpha, dim=0).tolist()

    payload = {
        "snapshot_paths": [os.path.basename(p) for p in snapshot_paths],
        "snapshot_sha256": {os.path.basename(p): sha256_file(p) for p in snapshot_paths},
        "fitness_split": {
            "dataset": tg.GSM8K_DATASET,
            "seed": tg.GSM8K_SEED,
            "start": 0,
            "count": len(held_out),
            "purpose": "mixture fitting only",
        },
        "alpha": alpha.tolist(),
        "weights": weights,
        "fit_loss": fit_loss,
    }
    payload["top_k"] = top_k_mixtures(payload["snapshot_paths"], payload["weights"])
    out_path = os.path.join(run_dir, "mixture_weights.json")
    tg.atomic_json_dump(payload, out_path)
    print(f"saved learned mixture weights to {out_path} (fit_loss={fit_loss:.4f})")


def snapshot_paths_for_run(args, run_dir):
    paths = []
    for traj_idx in range(1, args.num_initializations + 1):
        paths.extend(sorted(
            os.path.join(run_dir, name)
            for name in os.listdir(run_dir)
            if name.startswith(f"traj{traj_idx}_cycle") and name.endswith(".pt")
        ))
    if not paths:
        raise ValueError(f"no q0 snapshots found in {run_dir}")
    if len(paths) > args.num_initializations * args.max_snapshots:
        raise ValueError("q0 snapshot retention limit exceeded")
    return paths


def run_fitness(args):
    validate_q0_contract(args, torch.cuda.is_available())
    run_dir = os.path.join(args.output_dir, args.run_id)
    if not os.path.isdir(run_dir):
        raise ValueError(f"missing q0 run directory: {run_dir}")
    _train_examples, held_out = tg.load_gsm8k_splits(
        seed=tg.GSM8K_SEED, held_out=tg.GSM8K_HELD_OUT, train_target=args.train_target,
    )
    snapshot_paths = snapshot_paths_for_run(args, run_dir)
    fit_mixture_prior(args, snapshot_paths, held_out, run_dir, args.device)


def run_training(args):
    validate_q0_contract(args, torch.cuda.is_available())
    device = args.device
    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    train_examples, held_out = tg.load_gsm8k_splits(
        seed=tg.GSM8K_SEED, held_out=tg.GSM8K_HELD_OUT, train_target=args.train_target,
    )

    total_prompts = 0
    total_completions = 0
    all_snapshot_paths = []

    for traj_idx, seed in enumerate(args.trajectory_seeds, start=1):
        prompts, completions, snapshot_paths = run_trajectory(
            traj_idx, seed, args, train_examples, run_dir, device,
        )
        total_prompts += prompts
        total_completions += completions
        all_snapshot_paths.extend(snapshot_paths)

    expected_prompts = len(args.trajectory_seeds) * args.cycles * args.train_target
    expected_completions = expected_prompts * args.group_size
    tg.assert_counts(total_prompts, total_completions, expected_prompts, expected_completions)
    print(f"q0 training complete: {total_prompts} prompts, {total_completions} completions")

    if args.phase == "fitness":
        fit_mixture_prior(args, all_snapshot_paths, held_out, run_dir, device)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="q0 hyper-epoch grpo training on gsm8k: two trajectories, chain distillation, learned mixture prior",
        epilog="minimal pip packages for a real run: pip install torch transformers datasets accelerate",
    )
    tg.add_common_args(parser)
    parser.add_argument("--run-id", default="q0_grpo")
    parser.add_argument("--cycles", type=int, default=CYCLES_PER_TRAJECTORY)
    parser.add_argument("--num-initializations", type=int, default=1)
    parser.add_argument("--trajectory-seeds", type=int, nargs="+", default=LOCAL_TRAJECTORY_SEEDS)
    parser.add_argument("--max-snapshots", type=int, default=MAX_SNAPSHOTS_PER_TRAJECTORY)
    parser.add_argument("--phase", choices=["training", "fitness"], default="training")
    return parser


def self_test():
    print("running train_q0_grpo self-test...")

    assert abs(lr_multiplier(0, 10) - LR_MULT_START) < 1e-9
    assert abs(lr_multiplier(9, 10) - LR_MULT_END) < 1e-9
    assert abs(wd_multiplier(0, 10) - WD_MULT_START) < 1e-9
    assert abs(wd_multiplier(9, 10) - WD_MULT_END) < 1e-9

    assert not should_distill(1)
    assert not should_distill(2)
    assert should_distill(3)
    assert should_distill(4)

    first_scale = perturb_scale_for_cycle(1, CYCLES_PER_TRAJECTORY)
    last_scale = perturb_scale_for_cycle(CYCLES_PER_TRAJECTORY - 1, CYCLES_PER_TRAJECTORY)
    assert abs(first_scale - PERTURB_SCALE_MAX) < 1e-9
    assert abs(last_scale - PERTURB_SCALE_MIN) < 1e-9

    torch.manual_seed(0)
    linear = torch.nn.Linear(8, 8)
    before = linear.weight.detach().clone()
    perturb_weights(linear, scale=0.01, seed=42)
    after_a = linear.weight.detach().clone()
    assert not torch.allclose(before, after_a)

    linear2 = torch.nn.Linear(8, 8)
    with torch.no_grad():
        linear2.weight.copy_(before)
    perturb_weights(linear2, scale=0.01, seed=42)
    after_b = linear2.weight.detach().clone()
    assert torch.allclose(after_a, after_b), "perturbation must be deterministic for a fixed seed"

    assert TRAJECTORY_SEEDS == [42, 4242]
    assert LOCAL_TRAJECTORY_SEEDS == [42]
    assert MAX_SNAPSHOTS_PER_TRAJECTORY == 5

    long_reasoning = "reasoning " * 10000
    long_target = build_fitness_target({"answer": f"{long_reasoning}#### 42"})
    assert long_reasoning.strip() in long_target
    assert long_target.endswith("<answer>42</answer>")

    contract_args = argparse.Namespace(
        profile="smol135m", train_target=tg.GSM8K_TRAIN_TARGET,
        model_name=tg.MODEL_NAME, revision=tg.MODEL_REVISION, group_size=tg.GROUP_SIZE,
        temperature=tg.TEMPERATURE, top_p=tg.TOP_P, max_new_tokens=tg.MAX_NEW_TOKENS,
        prompt_batch_size=tg.PROMPT_BATCH_SIZE, micro_batch_size=tg.MICRO_BATCH_SIZE,
        lr=tg.PEAK_LR, weight_decay=tg.WEIGHT_DECAY, clip_epsilon=tg.CLIP_EPSILON,
        device="cuda", num_passes=tg.NUM_PASSES, cycles=CYCLES_PER_TRAJECTORY,
        num_initializations=1, trajectory_seeds=list(LOCAL_TRAJECTORY_SEEDS),
        max_snapshots=MAX_SNAPSHOTS_PER_TRAJECTORY, phase="training",
    )
    validate_q0_contract(contract_args, cuda_available=True)
    bad_contract_args = argparse.Namespace(**vars(contract_args))
    bad_contract_args.cycles += 1
    try:
        validate_q0_contract(bad_contract_args, cuda_available=True)
        raise RuntimeError("expected contract rejection")
    except ValueError:
        pass
    changed_contract_args = argparse.Namespace(**vars(contract_args))
    changed_contract_args.prompt_batch_size += 1
    try:
        tg.require_matching_contract(q0_contract(contract_args), q0_contract(changed_contract_args))
        raise RuntimeError("expected checkpoint contract mismatch")
    except ValueError:
        pass

    student_logits = torch.randn(2, 5, 20)
    kl_self = forward_kl_teacher_student(student_logits, student_logits, torch.ones(2, 5), 1.0)
    assert kl_self.item() < 1e-4
    teacher_logits = torch.randn(2, 5, 20) * 5.0
    kl_diff = forward_kl_teacher_student(student_logits, teacher_logits, torch.ones(2, 5), 1.0)
    assert kl_diff.item() > kl_self.item()

    temperature = 2.0
    student_logits = torch.tensor([[[1.0, -1.0]]])
    teacher_logits = torch.tensor([[[0.0, 2.0]]])
    mask = torch.ones(1, 1)
    manual_teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    manual_student_logp = F.log_softmax(student_logits / temperature, dim=-1)
    manual_kl = (manual_teacher_logp.exp() * (manual_teacher_logp - manual_student_logp)).sum()
    computed_kl = forward_kl_teacher_student(student_logits, teacher_logits, mask, temperature)
    assert torch.allclose(computed_kl, manual_kl)

    P_fit = torch.tensor([[0.9, 0.8, 0.85], [0.1, 0.2, 0.15], [0.5, 0.5, 0.5]])
    alpha, loss_after = optimize_mixture_alpha(P_fit, opt_steps=20, seed=0)
    w = torch.softmax(alpha, dim=0)
    assert abs(w.sum().item() - 1.0) < 1e-5
    assert w[0].item() > w[1].item(), "mixture should favor the higher p(gt) model"

    mixture_paths = ["a.pt", "b.pt", "c.pt", "d.pt", "e.pt"]
    mixture_weights = [0.2, 0.4, 0.4, 0.0, 0.1]
    top_k = top_k_mixtures(mixture_paths, mixture_weights)
    assert top_k["1"]["snapshot_paths"] == ["b.pt"]
    assert top_k["2"]["snapshot_paths"] == ["b.pt", "c.pt"]
    assert top_k["3"]["snapshot_paths"] == ["b.pt", "c.pt", "a.pt"]
    assert top_k["4"]["snapshot_paths"] == ["b.pt", "c.pt", "a.pt", "e.pt"]
    assert top_k["8"]["snapshot_paths"] == ["b.pt", "c.pt", "a.pt", "e.pt", "d.pt"]
    assert top_k["2"]["weights"] == [0.5, 0.5]
    for mixture in top_k.values():
        assert abs(sum(mixture["weights"]) - 1.0) < 1e-12

    with tempfile.TemporaryDirectory() as tmp_dir:
        mixture_path = os.path.join(tmp_dir, "mixture_weights.json")
        old_payload = {
            "snapshot_paths": mixture_paths,
            "alpha": [0.1, 0.2, 0.2, -1.0, -0.3],
            "weights": mixture_weights,
            "fit_loss": 0.25,
        }
        tg.atomic_json_dump(old_payload, mixture_path)
        enriched_payload = enrich_mixture_file(mixture_path)
        with open(mixture_path) as f:
            persisted_payload = json.load(f)
        assert enriched_payload == persisted_payload
        assert persisted_payload["snapshot_paths"] == mixture_paths
        assert persisted_payload["alpha"] == old_payload["alpha"]
        assert persisted_payload["weights"] == mixture_weights
        assert persisted_payload["fit_loss"] == old_payload["fit_loss"]
        assert persisted_payload["top_k"] == top_k

    assert EXPECTED_PROMPTS == 16384
    assert EXPECTED_COMPLETIONS == 65536

    print(f"train_q0_grpo self-test passed (expects {EXPECTED_PROMPTS} prompts, {EXPECTED_COMPLETIONS} completions)")


def main():
    args = tg.parse_args_with_profile(parser=build_arg_parser())
    if args.self_test:
        self_test()
        return
    if args.phase == "fitness":
        run_fitness(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
