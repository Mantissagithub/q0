import argparse
import glob
import hashlib
import json
import math
import os
import tempfile
import time

import torch

import train_grpo as tg

# evaluates the untouched base model, all baseline pass checkpoints, all q0
# snapshots, the distilled checkpoint, and q0 ensembles on the
# official gsm8k test split. this is the only file allowed to load that
# split. minimal pip packages for a real run:
#   pip install torch transformers datasets accelerate

# full-run learned_k_values = [1, 2, 4, 8, 16]
LEARNED_K_VALUES = [1, 2, 4, 8]
# full-run uniform_k_values = [2, 4, 8, 16]
UNIFORM_K_VALUES = [2, 4, 8]

# full-run baseline_checkpoint_count = 16
BASELINE_CHECKPOINT_COUNT = 4
# full-run q0_snapshot_count = 16
Q0_SNAPSHOT_COUNT = 5
PASS_K_VALUES = (1, 4, 8)
FINAL_MANIFEST_SCHEMA = 1
FINAL_SYSTEM_NAMES = ("baseline_grpo", "q0_best_single", "q0_learned_top4", "mopd")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(manifest_dir, value):
    if not isinstance(value, str) or not value:
        raise ValueError("manifest checkpoint paths must be nonempty strings")
    path = os.path.abspath(os.path.join(manifest_dir, value))
    if not os.path.isfile(path):
        raise ValueError(f"manifest checkpoint is missing: {path}")
    return path


def load_final_manifest(path, model_name, revision):
    path = os.path.abspath(path)
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != FINAL_MANIFEST_SCHEMA:
        raise ValueError("unsupported final manifest schema")
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("name") != model_name or model.get("revision") != revision:
        raise ValueError("final manifest model does not match evaluator arguments")
    systems = payload.get("systems")
    if not isinstance(systems, dict) or set(systems) != set(FINAL_SYSTEM_NAMES):
        raise ValueError("final manifest must declare exactly the four final systems")
    root = os.path.dirname(path)
    resolved = {}
    for name in FINAL_SYSTEM_NAMES:
        entry = systems[name]
        kind = entry.get("type")
        if kind == "single":
            checkpoint = entry.get("checkpoint")
            if not isinstance(checkpoint, dict):
                raise ValueError(f"manifest system {name} needs a checkpoint")
            checkpoint_path = _manifest_path(root, checkpoint.get("path"))
            if checkpoint.get("sha256") != sha256_file(checkpoint_path):
                raise ValueError(f"manifest hash mismatch for {name}")
            resolved[name] = {"type": kind, "path": checkpoint_path}
        elif kind == "ensemble":
            members = entry.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError(f"manifest system {name} needs ensemble members")
            paths, weights = [], []
            for member in members:
                member_path = _manifest_path(root, member.get("path"))
                if member.get("sha256") != sha256_file(member_path):
                    raise ValueError(f"manifest hash mismatch for {name}")
                paths.append(member_path)
                weights.append(float(member.get("weight")))
            if len(set(paths)) != len(paths) or any(w < 0 or not math.isfinite(w) for w in weights):
                raise ValueError(f"invalid ensemble members for {name}")
            if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-9):
                raise ValueError(f"ensemble weights for {name} must sum to one")
            resolved[name] = {"type": kind, "paths": paths, "weights": weights}
        else:
            raise ValueError(f"unknown manifest system type for {name}")
    return payload, resolved, sha256_file(path)


def _manifest_checkpoint(path, manifest_dir):
    return {"path": os.path.relpath(path, manifest_dir), "sha256": sha256_file(path)}


def freeze_final_manifest(path, baseline_run_dir, q0_run_dir, mopd_run_dir, model_name, revision):
    baseline = os.path.join(baseline_run_dir, "pass_04.pt")
    if not os.path.isfile(baseline):
        raise ValueError(f"missing baseline terminal checkpoint: {baseline}")
    with open(os.path.join(q0_run_dir, "mixture_weights.json")) as handle:
        mixture = json.load(handle)
    names = mixture["snapshot_paths"]
    weights = mixture["weights"]
    top1 = mixture["top_k"]["1"]["snapshot_paths"][0]
    top4 = mixture["top_k"]["4"]
    manifest_dir = os.path.dirname(os.path.abspath(path))
    systems = {
        "baseline_grpo": {"type": "single", "checkpoint": _manifest_checkpoint(baseline, manifest_dir), "selection": "terminal_pass_fixed_a_priori"},
        "q0_best_single": {"type": "single", "checkpoint": _manifest_checkpoint(os.path.join(q0_run_dir, top1), manifest_dir), "selection": "fitness_top_k_1"},
        "q0_learned_top4": {"type": "ensemble", "members": []},
        "mopd": {"type": "single", "checkpoint": _manifest_checkpoint(os.path.join(mopd_run_dir, "opsd.pt"), manifest_dir), "selection": "validation_results.json"},
    }
    for name, weight in zip(top4["snapshot_paths"], top4["weights"]):
        systems["q0_learned_top4"]["members"].append({"weight": weight, **_manifest_checkpoint(os.path.join(q0_run_dir, name), manifest_dir)})
    payload = {"schema_version": FINAL_MANIFEST_SCHEMA, "model": {"name": model_name, "revision": revision}, "validation": {"source": "GSM8K train split; frozen before official test evaluation"}, "systems": systems}
    tg.atomic_json_dump(payload, path)
    return path


def load_test_set():
    from datasets import load_dataset

    ds = load_dataset(tg.GSM8K_DATASET, "main", split="test")
    return list(ds)


def score_completion(text, gold_value):
    answer_str, has_tag = tg.parse_answer(text)
    correct = tg.answer_is_correct(answer_str, gold_value)
    return correct, has_tag


def pass_at_k_estimate(sample_count, correct_count, k):
    if not 1 <= k <= sample_count:
        raise ValueError("k must be between 1 and the sample count")
    if correct_count <= 0:
        return 0.0
    if sample_count - correct_count < k:
        return 1.0
    miss = 1.0
    for index in range(k):
        miss *= (sample_count - correct_count - index) / (sample_count - index)
    return 1.0 - miss


def mix_probabilities(logits_list, weights):
    # mixes next-token probability distributions, never raw logits
    mixed = None
    for logits, w in zip(logits_list, weights):
        probs = torch.softmax(logits.float(), dim=-1)
        mixed = probs * w if mixed is None else mixed + probs * w
    return mixed


def select_top_k(weights, snapshot_names, k):
    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    top = order[:k]
    sel_names = [snapshot_names[i] for i in top]
    sel_weights = [weights[i] for i in top]
    total = sum(sel_weights)
    learned_w = [w / total for w in sel_weights]
    return sel_names, learned_w


def validate_top_k_manifest(top_k, weights, snapshot_names):
    required_keys = {str(k) for k in LEARNED_K_VALUES}
    if not isinstance(top_k, dict) or not required_keys.issubset(top_k):
        raise ValueError("mixture_weights.json top_k is missing required mixture sizes")

    tolerance = 1e-12
    for k in LEARNED_K_VALUES:
        entry = top_k[str(k)]
        if not isinstance(entry, dict):
            raise ValueError(f"mixture_weights.json top_k[{k}] must be an object")
        entry_names = entry.get("snapshot_paths")
        entry_weights = entry.get("weights")
        expected_names, expected_weights = select_top_k(
            weights, snapshot_names, min(k, len(snapshot_names))
        )
        if entry_names != expected_names:
            raise ValueError(f"mixture_weights.json top_k[{k}] snapshot_paths do not match learned ranking")
        if not isinstance(entry_weights, list) or len(entry_weights) != len(expected_weights):
            raise ValueError(f"mixture_weights.json top_k[{k}] weights must match selected snapshots")
        if any(not isinstance(weight, (int, float)) or isinstance(weight, bool) for weight in entry_weights):
            raise ValueError(f"mixture_weights.json top_k[{k}] weights must be numeric")
        if any(not math.isfinite(weight) or weight < 0 for weight in entry_weights):
            raise ValueError(f"mixture_weights.json top_k[{k}] weights must be finite and nonnegative")
        if not math.isclose(math.fsum(entry_weights), 1.0, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"mixture_weights.json top_k[{k}] weights must sum to 1")
        if any(not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
               for actual, expected in zip(entry_weights, expected_weights)):
            raise ValueError(f"mixture_weights.json top_k[{k}] weights do not match learned weights")


def select_first_k(snapshot_names, k):
    names = snapshot_names[:k]
    return names, [1.0 / k for _ in names]


def tokenize_left_padded(tokenizer, prompts, device):
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    finally:
        tokenizer.padding_side = previous_padding_side


def position_ids_from_attention_mask(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 1)


def has_token_suffix(token_ids, suffix):
    if suffix.numel() == 0 or token_ids.shape[1] < suffix.numel():
        return torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
    return (token_ids[:, -suffix.numel():] == suffix).all(dim=1)


@torch.no_grad()
def ensemble_greedy_generate(models, weights, tokenizer, prompts, max_new_tokens, device):
    enc = tokenize_left_padded(tokenizer, prompts, device)
    input_ids = enc["input_ids"]
    gen_mask = enc["attention_mask"]
    batch_size = input_ids.shape[0]
    eos_id = tokenizer.eos_token_id
    stop_token_ids = torch.tensor(
        tokenizer(tg.ANSWER_STOP_STRING, add_special_tokens=False)["input_ids"], device=device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    generated = input_ids
    past_key_values = [None] * len(models)
    for _ in range(max_new_tokens):
        logits_list = []
        new_pasts = []
        for model, past in zip(models, past_key_values):
            if past is None:
                position_ids = position_ids_from_attention_mask(gen_mask)
                out = model(
                    input_ids=generated, attention_mask=gen_mask, position_ids=position_ids, use_cache=True,
                )
            else:
                position_ids = position_ids_from_attention_mask(gen_mask)[:, -1:]
                out = model(
                    input_ids=generated[:, -1:], attention_mask=gen_mask, position_ids=position_ids,
                    past_key_values=past, use_cache=True,
                )
            logits_list.append(out.logits[:, -1, :])
            new_pasts.append(out.past_key_values)
        past_key_values = new_pasts

        mixed_probs = mix_probabilities(logits_list, weights)
        next_token = torch.argmax(mixed_probs, dim=-1)
        active = ~finished
        next_token = torch.where(active, next_token, torch.full_like(next_token, tokenizer.pad_token_id))
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        gen_mask = torch.cat([gen_mask, active.long().unsqueeze(1)], dim=1)
        finished = finished | ((next_token == eos_id) & active) | (has_token_suffix(generated, stop_token_ids) & active)
        if finished.all():
            break

    prompt_len = input_ids.shape[1]
    completion_ids = generated[:, prompt_len:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)


@torch.no_grad()
def evaluate_single_model(model, tokenizer, test_set, batch_size, max_new_tokens, device, source):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    correct = 0
    tagged = 0
    total_len = 0
    n = len(test_set)
    for start in range(0, n, batch_size):
        batch = test_set[start:start + batch_size]
        prompts = [tg.build_prompt(ex["question"]) for ex in batch]
        golds = [tg.gsm8k_gold_answer(ex["answer"]) for ex in batch]
        enc = tokenize_left_padded(tokenizer, prompts, device)
        out = model.generate(
            **enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
            stop_strings=[tg.ANSWER_STOP_STRING], tokenizer=tokenizer,
        )
        completion_ids = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for text, gold in zip(texts, golds):
            is_correct, has_tag = score_completion(text, gold)
            correct += int(is_correct)
            tagged += int(has_tag)
            total_len += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) if device == "cuda" else 0
    return {
        "exact_match": correct / n,
        "parse_rate": tagged / n,
        "avg_response_length": total_len / n,
        "timing_seconds": elapsed,
        "peak_memory_bytes": peak_mem,
        "source": source,
    }


@torch.no_grad()
def evaluate_pass_at_k(model, tokenizer, test_set, batch_size, max_new_tokens, device, source):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tg.set_seed(tg.GSM8K_SEED)
    sample_count = max(PASS_K_VALUES)
    pass_totals = {k: 0.0 for k in PASS_K_VALUES}
    parse_totals = {k: 0.0 for k in PASS_K_VALUES}
    total_len = 0
    t0 = time.time()

    for start in range(0, len(test_set), batch_size):
        batch = test_set[start:start + batch_size]
        prompts = [tg.build_prompt(example["question"]) for example in batch]
        golds = [tg.gsm8k_gold_answer(example["answer"]) for example in batch]
        enc = tokenize_left_padded(tokenizer, prompts, device)
        out = model.generate(
            **enc, do_sample=True, temperature=tg.TEMPERATURE, top_p=tg.TOP_P,
            max_new_tokens=max_new_tokens, num_return_sequences=sample_count,
            pad_token_id=tokenizer.pad_token_id, stop_strings=[tg.ANSWER_STOP_STRING],
            tokenizer=tokenizer, use_cache=True,
        )
        completion_ids = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for index, gold in enumerate(golds):
            samples = texts[index * sample_count:(index + 1) * sample_count]
            scores = [score_completion(text, gold) for text in samples]
            correct_count = sum(int(correct) for correct, _ in scores)
            parsed_count = sum(int(parsed) for _, parsed in scores)
            for k in PASS_K_VALUES:
                pass_totals[k] += pass_at_k_estimate(sample_count, correct_count, k)
                parse_totals[k] += pass_at_k_estimate(sample_count, parsed_count, k)
            total_len += sum(
                len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in samples
            )

    elapsed = time.time() - t0
    count = len(test_set)
    result = {
        f"pass_at_{k}": pass_totals[k] / count for k in PASS_K_VALUES
    }
    result.update({f"parse_at_{k}": parse_totals[k] / count for k in PASS_K_VALUES})
    result.update({
        "samples_per_problem": sample_count,
        "avg_sample_response_length": total_len / (count * sample_count),
        "pass_at_k_timing_seconds": elapsed,
        "pass_at_k_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device == "cuda" else 0
        ),
        "pass_at_k_seed": tg.GSM8K_SEED,
        "pass_at_k_source": source,
    })
    return result


@torch.no_grad()
def evaluate_ensemble(paths, weights, args, tokenizer, test_set, device, label):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    models = [tg.load_model_from_checkpoint(args.model_name, args.revision, p, device) for p in paths]
    for model in models:
        model.eval()
    model = None

    t0 = time.time()
    correct = 0
    tagged = 0
    total_len = 0
    n = len(test_set)
    for start in range(0, n, args.batch_size):
        batch = test_set[start:start + args.batch_size]
        prompts = [tg.build_prompt(ex["question"]) for ex in batch]
        golds = [tg.gsm8k_gold_answer(ex["answer"]) for ex in batch]
        texts = ensemble_greedy_generate(models, weights, tokenizer, prompts, args.max_new_tokens, device)
        for text, gold in zip(texts, golds):
            is_correct, has_tag = score_completion(text, gold)
            correct += int(is_correct)
            tagged += int(has_tag)
            total_len += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) if device == "cuda" else 0

    models.clear()
    models = None
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "exact_match": correct / n,
        "parse_rate": tagged / n,
        "avg_response_length": total_len / n,
        "timing_seconds": elapsed,
        "peak_memory_bytes": peak_mem,
        "source": {"paths": paths, "weights": weights, "label": label},
    }


def validate_checkpoint_inputs(args):
    baseline_paths = sorted(glob.glob(os.path.join(args.baseline_run_dir, "pass_*.pt")))
    q0_paths = sorted(glob.glob(os.path.join(args.q0_run_dir, "traj*_cycle*.pt")))
    expected_counts = {
        "baseline pass": BASELINE_CHECKPOINT_COUNT,
    }
    for label, paths in (("baseline pass", baseline_paths),):
        expected_count = expected_counts[label]
        if len(paths) != expected_count:
            raise ValueError(f"expected exactly {expected_count} {label} paths, found {len(paths)}")
    if not 1 <= len(q0_paths) <= Q0_SNAPSHOT_COUNT:
        raise ValueError(f"expected 1-{Q0_SNAPSHOT_COUNT} q0 snapshot paths, found {len(q0_paths)}")
    missing = [path for path in baseline_paths if not os.path.isfile(path)]
    if missing:
        raise ValueError(f"baseline pass paths must all exist: {missing}")

    mixture_path = os.path.join(args.q0_run_dir, "mixture_weights.json")
    if not os.path.isfile(mixture_path):
        raise ValueError(f"missing required mixture weights file: {mixture_path}")
    with open(mixture_path) as f:
        mixture = json.load(f)
    snapshot_names = mixture.get("snapshot_paths")
    weights = mixture.get("weights")
    if not isinstance(snapshot_names, list) or not isinstance(weights, list):
        raise ValueError("mixture_weights.json must contain snapshot_paths and weights lists")
    if not 1 <= len(snapshot_names) <= Q0_SNAPSHOT_COUNT or len(weights) != len(snapshot_names):
        raise ValueError(
            f"mixture_weights.json must contain 1-{Q0_SNAPSHOT_COUNT} snapshot paths and matching weights"
        )
    if len(set(snapshot_names)) != len(snapshot_names) or any(not isinstance(name, str) for name in snapshot_names):
        raise ValueError(f"mixture_weights.json snapshot paths must be 1-{Q0_SNAPSHOT_COUNT} unique names")
    if any(os.path.basename(name) != name for name in snapshot_names):
        raise ValueError("mixture_weights.json snapshot paths must be file names, not directories")
    if set(snapshot_names) != {os.path.basename(path) for path in q0_paths}:
        raise ValueError("mixture_weights.json snapshot paths must match the q0 snapshots")
    try:
        weights = [float(weight) for weight in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError("mixture_weights.json weights must be numeric") from exc
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("mixture_weights.json weights must be nonnegative with a positive total")
    validate_top_k_manifest(mixture.get("top_k"), weights, snapshot_names)
    mixture_paths = [os.path.join(args.q0_run_dir, name) for name in snapshot_names]
    missing = [path for path in mixture_paths if not os.path.isfile(path)]
    if missing:
        raise ValueError(f"mixture_weights.json references missing snapshots: {missing}")
    return baseline_paths, q0_paths, mixture_path, snapshot_names, weights


def validate_mopd_checkpoint(args):
    if not args.mopd_run_dir:
        return None
    path = os.path.join(args.mopd_run_dir, "opsd.pt")
    if not os.path.isfile(path):
        raise ValueError(f"missing required distilled checkpoint: {path}")
    return path


def fingerprint_file(path, base_dir):
    stat = os.stat(path)
    return {
        "relative_path": os.path.relpath(path, base_dir),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def fingerprint_file_with_sha256(path, base_dir):
    fingerprint = fingerprint_file(path, base_dir)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    fingerprint["sha256"] = digest.hexdigest()
    return fingerprint


def make_source_fingerprint(baseline_paths, q0_paths, mixture_path, snapshot_names, weights, args,
                            mopd_path=None):
    with open(mixture_path, "rb") as f:
        mixture_sha256 = hashlib.sha256(f.read()).hexdigest()
    return {
        "baseline_files": [fingerprint_file(path, args.baseline_run_dir) for path in baseline_paths],
        "q0_files": [fingerprint_file(path, args.q0_run_dir) for path in q0_paths],
        "mixture_sha256": mixture_sha256,
        "mixture_snapshot_paths": snapshot_names,
        "mixture_weights": weights,
        "mopd_file": fingerprint_file(mopd_path, args.mopd_run_dir) if mopd_path else None,
    }


def make_metadata(args, test_count, source_fingerprint, timestamp=None):
    return {
        "model_name": args.model_name,
        "model_revision": args.revision,
        "test_count": test_count,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "source_fingerprint": source_fingerprint,
        "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_results_atomic(output_path, metadata, results):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".eval_results.", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_resume_results(args, test_count, source_fingerprint):
    if not os.path.isfile(args.output):
        raise FileNotFoundError(f"cannot resume evaluation: output file does not exist: {args.output}")
    with open(args.output) as f:
        saved = json.load(f)
    if not isinstance(saved, dict) or not isinstance(saved.get("metadata"), dict) or not isinstance(saved.get("results"), dict):
        raise ValueError("cannot resume evaluation: output must contain metadata and results objects")
    expected = make_metadata(
        args, test_count, source_fingerprint, timestamp=saved["metadata"].get("timestamp"),
    )
    for key, value in expected.items():
        if key != "timestamp" and saved["metadata"].get(key) != value:
            raise ValueError(f"cannot resume evaluation: incompatible metadata field {key}")
    if not saved["metadata"].get("timestamp"):
        raise ValueError("cannot resume evaluation: metadata timestamp is missing")
    return saved["metadata"], saved["results"]


def save_result(output_path, metadata, results, key, value):
    results[key] = value
    write_results_atomic(output_path, metadata, results)


def load_mopd_merge_results(args, test_count, mopd_path):
    if not os.path.isfile(args.output):
        raise FileNotFoundError(f"mopd-only evaluation requires existing results: {args.output}")
    with open(args.output) as handle:
        saved = json.load(handle)
    metadata = saved.get("metadata") if isinstance(saved, dict) else None
    results = saved.get("results") if isinstance(saved, dict) else None
    if not isinstance(metadata, dict) or not isinstance(results, dict):
        raise ValueError("existing evaluation output must contain metadata and results objects")
    expected = {
        "model_name": args.model_name,
        "model_revision": args.revision,
        "test_count": test_count,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"mopd-only evaluation has incompatible metadata field {key}")
    if not isinstance(metadata.get("source_fingerprint"), dict):
        raise ValueError("existing evaluation metadata is missing its source fingerprint")
    fingerprint = fingerprint_file_with_sha256(mopd_path, args.mopd_run_dir)
    return metadata, results, fingerprint


def best_existing_checkpoint(results, prefix):
    candidates = [
        (key, value) for key, value in results.items()
        if key.startswith(prefix) and isinstance(value, dict) and "exact_match" in value
    ]
    if not candidates:
        raise ValueError(f"existing evaluation results contain no {prefix} checkpoints")
    return max(candidates, key=lambda item: item[1]["exact_match"])[0]


def evaluate_winner_pass_at_k(args, metadata, results, tokenizer, test_set, device):
    winners = [
        (best_existing_checkpoint(results, "baseline_pass_"), args.baseline_run_dir, "baseline_"),
        (best_existing_checkpoint(results, "q0_traj"), args.q0_run_dir, "q0_"),
    ]
    fingerprints = metadata["source_fingerprint"].setdefault("pass_at_k_winner_files", {})
    for key, run_dir, key_prefix in winners:
        filename = key[len(key_prefix):]
        path = os.path.join(run_dir, filename)
        if not os.path.isfile(path):
            raise ValueError(f"missing best-checkpoint file for pass@k evaluation: {path}")
        fingerprint = fingerprint_file_with_sha256(path, run_dir)
        if results[key].get("pass_at_8") is not None and fingerprints.get(key) == fingerprint:
            print(f"pass@k result already matches {path}; skipping")
            continue
        model = tg.load_model_from_checkpoint(args.model_name, args.revision, path, device)
        results[key].update(evaluate_pass_at_k(
            model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source=path,
        ))
        results[key]["pass_at_k_checkpoint_sha256"] = fingerprint["sha256"]
        fingerprints[key] = fingerprint
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
        write_results_atomic(args.output, metadata, results)
        print(f"merged pass@k result for {key}")


def run_mopd_only_eval(args):
    mopd_path = validate_mopd_checkpoint(args)
    if mopd_path is None:
        raise ValueError("--mopd-only requires --mopd-run-dir")
    device = args.device
    tokenizer = tg.load_tokenizer(args.model_name, args.revision)
    test_set = load_test_set()
    metadata, results, fingerprint = load_mopd_merge_results(args, len(test_set), mopd_path)
    previous = metadata["source_fingerprint"].get("mopd_file")
    if "mopd" in results and previous == fingerprint:
        print(f"mopd result already matches {mopd_path}; nothing to evaluate")
    else:
        model = tg.load_model_from_checkpoint(args.model_name, args.revision, mopd_path, device)
        value = evaluate_single_model(
            model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source=mopd_path,
        )
        value.update(evaluate_pass_at_k(
            model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source=mopd_path,
        ))
        value["checkpoint_sha256"] = fingerprint["sha256"]
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
        metadata["source_fingerprint"]["mopd_file"] = fingerprint
        metadata["mopd_evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_result(args.output, metadata, results, "mopd", value)
        print(f"merged mopd result into {args.output}; kept {len(results) - 1} existing results")
    if args.compare_winners:
        evaluate_winner_pass_at_k(args, metadata, results, tokenizer, test_set, device)


def run_manifest_eval(args):
    if args.compare_winners:
        raise ValueError("--compare-winners is not allowed in manifest mode")
    payload, systems, manifest_sha256 = load_final_manifest(args.manifest, args.model_name, args.revision)
    device = args.device
    tokenizer = tg.load_tokenizer(args.model_name, args.revision)
    test_set = load_test_set()
    metadata = make_metadata(args, len(test_set), {"manifest_sha256": manifest_sha256})
    metadata["manifest_schema_version"] = payload["schema_version"]
    results = {}
    for name in FINAL_SYSTEM_NAMES:
        if name in results:
            continue
        spec = systems[name]
        if spec["type"] == "single":
            model = tg.load_model_from_checkpoint(args.model_name, args.revision, spec["path"], device)
            value = evaluate_single_model(model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source=spec["path"])
        else:
            value = evaluate_ensemble(spec["paths"], spec["weights"], args, tokenizer, test_set, device, label=name)
        results[name] = value
        save_result(args.output, metadata, results, name, value)
    print(f"wrote manifest-driven eval results to {args.output}")


def run_eval(args):
    if args.manifest:
        run_manifest_eval(args)
        return
    if args.freeze_manifest:
        if not args.mopd_run_dir:
            raise ValueError("--freeze-manifest requires --mopd-run-dir")
        freeze_final_manifest(args.output, args.baseline_run_dir, args.q0_run_dir, args.mopd_run_dir, args.model_name, args.revision)
        print(f"wrote frozen final manifest to {args.output}")
        return
    if args.mopd_only:
        run_mopd_only_eval(args)
        return
    baseline_paths, q0_paths, mixture_path, snapshot_names, weights = validate_checkpoint_inputs(args)
    mopd_path = validate_mopd_checkpoint(args)
    source_fingerprint = make_source_fingerprint(
        baseline_paths, q0_paths, mixture_path, snapshot_names, weights, args, mopd_path,
    )
    device = args.device
    tokenizer = tg.load_tokenizer(args.model_name, args.revision)
    test_set = load_test_set()
    if args.resume:
        metadata, results = load_resume_results(args, len(test_set), source_fingerprint)
    else:
        metadata = make_metadata(args, len(test_set), source_fingerprint)
        results = {}

    if "base" not in results:
        base_model = tg.load_base_model(args.model_name, args.revision, device)
        base_model.eval()
        value = evaluate_single_model(
            base_model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source="base (untouched)",
        )
        base_model = None
        if device == "cuda":
            torch.cuda.empty_cache()
        save_result(args.output, metadata, results, "base", value)

    for path in baseline_paths:
        name = os.path.basename(path)
        key = f"baseline_{name}"
        if key in results:
            continue
        model = tg.load_model_from_checkpoint(args.model_name, args.revision, path, device)
        value = evaluate_single_model(
            model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source=path,
        )
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
        save_result(args.output, metadata, results, key, value)

    for path in q0_paths:
        name = os.path.basename(path)
        key = f"q0_{name}"
        if key in results:
            continue
        model = tg.load_model_from_checkpoint(args.model_name, args.revision, path, device)
        value = evaluate_single_model(
            model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source=path,
        )
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
        save_result(args.output, metadata, results, key, value)

    if mopd_path and "mopd" not in results:
        model = tg.load_model_from_checkpoint(args.model_name, args.revision, mopd_path, device)
        value = evaluate_single_model(
            model, tokenizer, test_set, args.batch_size, args.max_new_tokens, device, source=mopd_path,
        )
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
        save_result(args.output, metadata, results, "mopd", value)

    q0_individual = {k: v for k, v in results.items() if k.startswith("q0_traj")}
    if q0_individual and "q0_best_member" not in results:
        best_key = max(q0_individual, key=lambda k: q0_individual[k]["exact_match"])
        value = dict(results[best_key])
        value["source"] = f"posthoc diagnostic: best individual snapshot: {best_key}"
        save_result(args.output, metadata, results, "q0_best_member", value)

    for k in LEARNED_K_VALUES:
        learned_key = f"q0_learned_prior_k{k}"
        if learned_key not in results:
            names, learned_w = select_top_k(weights, snapshot_names, min(k, len(snapshot_names)))
            paths = [os.path.join(args.q0_run_dir, name) for name in names]
            value = evaluate_ensemble(
                paths, learned_w, args, tokenizer, test_set, device, label=f"learned_prior_k{k}",
            )
            save_result(args.output, metadata, results, learned_key, value)
        if k in UNIFORM_K_VALUES:
            uniform_key = f"q0_uniform_k{k}"
            if uniform_key in results:
                continue
            names, uniform_w = select_first_k(snapshot_names, min(k, len(snapshot_names)))
            paths = [os.path.join(args.q0_run_dir, name) for name in names]
            value = evaluate_ensemble(
                paths, uniform_w, args, tokenizer, test_set, device, label=f"uniform_k{k}",
            )
            save_result(args.output, metadata, results, uniform_key, value)

    print(f"wrote eval results to {args.output}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="evaluate baseline, q0, and distilled gsm8k checkpoints on the official test split",
        epilog="minimal pip packages for a real run: pip install torch transformers datasets accelerate",
    )
    parser.add_argument("--model-name", default=tg.MODEL_NAME)
    parser.add_argument("--revision", default=tg.MODEL_REVISION)
    parser.add_argument("--baseline-run-dir", default=os.path.join("runs", "grpo_baseline"))
    parser.add_argument("--q0-run-dir", default=os.path.join("runs", "q0_grpo"))
    parser.add_argument("--mopd-run-dir", default=None)
    parser.add_argument("--manifest", default=None, help="frozen final-system manifest")
    parser.add_argument("--freeze-manifest", action="store_true")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=tg.MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mopd-only", action="store_true")
    parser.add_argument("--compare-winners", action="store_true")
    parser.add_argument(
        "--self-test", action="store_true",
        help="run offline self-tests only, no network, model download, or gpu needed",
    )
    return parser


def self_test():
    print("running eval self-test...")

    assert PASS_K_VALUES == (1, 4, 8)
    assert pass_at_k_estimate(8, 0, 4) == 0.0
    assert abs(pass_at_k_estimate(8, 1, 1) - 0.125) < 1e-12
    assert pass_at_k_estimate(8, 1, 8) == 1.0
    assert pass_at_k_estimate(8, 8, 4) == 1.0
    existing = {
        "baseline_pass_01.pt": {"exact_match": 0.1},
        "baseline_pass_02.pt": {"exact_match": 0.2},
        "q0_traj1_cycle01.pt": {"exact_match": 0.3},
    }
    assert best_existing_checkpoint(existing, "baseline_pass_") == "baseline_pass_02.pt"
    assert best_existing_checkpoint(existing, "q0_traj") == "q0_traj1_cycle01.pt"
    try:
        pass_at_k_estimate(8, 1, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("pass@k must reject k above the sample count")

    correct, has_tag = score_completion("reasoning <answer>18</answer>", 18.0)
    assert correct and has_tag
    correct, has_tag = score_completion("reasoning <answer>19</answer>", 18.0)
    assert not correct and has_tag
    correct, has_tag = score_completion("no tag here", 18.0)
    assert not correct and not has_tag

    logits_a = torch.tensor([[3.0, 0.0, 0.0]])
    logits_b = torch.tensor([[0.0, 3.0, 3.05]])
    weights = [0.5, 0.5]
    mixed_probs = mix_probabilities([logits_a, logits_b], weights)
    prob_argmax = mixed_probs.argmax(dim=-1).item()
    mixed_logits = weights[0] * logits_a + weights[1] * logits_b
    logit_argmax = mixed_logits.argmax(dim=-1).item()
    assert prob_argmax == 0
    assert logit_argmax == 2
    assert prob_argmax != logit_argmax, "probability mixing must not reduce to logit mixing"

    weights5 = [0.1, 0.4, 0.05, 0.3, 0.15]
    names5 = ["a", "b", "c", "d", "e"]
    sel_names, learned_w = select_top_k(weights5, names5, 2)
    assert sel_names == ["b", "d"]
    assert abs(sum(learned_w) - 1.0) < 1e-9
    first_names, uniform_w = select_first_k(names5, 2)
    assert first_names == ["a", "b"]
    assert uniform_w == [0.5, 0.5]
    assert LEARNED_K_VALUES == [1, 2, 4, 8]
    assert UNIFORM_K_VALUES == [2, 4, 8]
    assert BASELINE_CHECKPOINT_COUNT == 4
    assert Q0_SNAPSHOT_COUNT == 5

    prompt_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    assert position_ids_from_attention_mask(prompt_mask).tolist() == [[1, 1, 0, 1], [1, 0, 1, 2]]
    cached_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    assert position_ids_from_attention_mask(cached_mask)[:, -1:].tolist() == [[2], [3]]
    assert has_token_suffix(torch.tensor([[1, 2, 3], [1, 2, 4]]), torch.tensor([2, 3])).tolist() == [True, False]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "results.json")
        source_dir = os.path.join(temp_dir, "sources")
        os.mkdir(source_dir)
        baseline_path = os.path.join(source_dir, "baseline.pt")
        q0_path = os.path.join(source_dir, "q0.pt")
        mixture_path = os.path.join(source_dir, "mixture_weights.json")
        with open(baseline_path, "wb") as f:
            f.write(b"baseline")
        with open(q0_path, "wb") as f:
            f.write(b"q0")
        with open(mixture_path, "w") as f:
            json.dump({
                "snapshot_paths": ["q0.pt"],
                "weights": [1.0],
                "top_k": {
                    str(k): {"snapshot_paths": ["q0.pt"], "weights": [1.0]}
                    for k in LEARNED_K_VALUES
                },
            }, f)
        test_args = argparse.Namespace(
            model_name="test-model", revision="test-revision", max_new_tokens=12,
            batch_size=16, output=output_path, baseline_run_dir=source_dir, q0_run_dir=source_dir,
        )
        fingerprint = make_source_fingerprint(
            [baseline_path], [q0_path], mixture_path, ["q0.pt"], [1.0], test_args,
        )
        metadata = make_metadata(
            test_args, 3, fingerprint, timestamp="2026-08-09T00:00:00Z",
        )
        write_results_atomic(output_path, metadata, {"base": {"exact_match": 0.0}})
        mopd_path = os.path.join(source_dir, "opsd.pt")
        with open(mopd_path, "wb") as f:
            f.write(b"mopd")
        test_args.mopd_run_dir = source_dir
        merged_metadata, merged_results, mopd_fingerprint = load_mopd_merge_results(
            test_args, 3, mopd_path,
        )
        assert merged_metadata == metadata
        assert set(merged_results) == {"base"}
        assert len(mopd_fingerprint["sha256"]) == 64
        with open(mopd_path, "wb") as f:
            f.write(b"changed-mopd")
        changed_mopd_fingerprint = fingerprint_file_with_sha256(mopd_path, source_dir)
        assert changed_mopd_fingerprint["sha256"] != mopd_fingerprint["sha256"]
        resumed_metadata, resumed_results = load_resume_results(test_args, 3, fingerprint)
        assert resumed_metadata == metadata
        assert "base" in resumed_results
        save_result(output_path, resumed_metadata, resumed_results, "baseline_pass_01.pt", {"exact_match": 1.0})
        with open(output_path) as f:
            saved = json.load(f)
        assert set(saved["results"]) == {"base", "baseline_pass_01.pt"}
        assert set(os.listdir(temp_dir)) == {"results.json", "sources"}
        incompatible_args = argparse.Namespace(**vars(test_args))
        incompatible_args.batch_size = 8
        try:
            load_resume_results(incompatible_args, 3, fingerprint)
        except ValueError as exc:
            assert "batch_size" in str(exc)
        else:
            raise AssertionError("incompatible resume metadata must fail")
        with open(q0_path, "wb") as f:
            f.write(b"changed-q0")
        changed_fingerprint = make_source_fingerprint(
            [baseline_path], [q0_path], mixture_path, ["q0.pt"], [1.0], test_args,
        )
        try:
            load_resume_results(test_args, 3, changed_fingerprint)
        except ValueError as exc:
            assert "source_fingerprint" in str(exc)
        else:
            raise AssertionError("changed evaluation sources must not resume")

        baseline_dir = os.path.join(temp_dir, "baseline")
        q0_dir = os.path.join(temp_dir, "q0")
        mopd_dir = os.path.join(temp_dir, "mopd")
        os.mkdir(baseline_dir)
        os.mkdir(q0_dir)
        os.mkdir(mopd_dir)
        for pass_num in range(1, BASELINE_CHECKPOINT_COUNT + 1):
            with open(os.path.join(baseline_dir, f"pass_{pass_num:02d}.pt"), "wb") as f:
                f.write(b"baseline")
        snapshot_names = []
        for cycle in range(1, Q0_SNAPSHOT_COUNT + 1):
            name = f"traj1_cycle{cycle:02d}.pt"
            snapshot_names.append(name)
            with open(os.path.join(q0_dir, name), "wb") as f:
                f.write(b"q0")
        learned_weights = [1.0] * Q0_SNAPSHOT_COUNT
        mixture = {
            "snapshot_paths": snapshot_names,
            "weights": learned_weights,
            "top_k": {
                str(k): {
                    "snapshot_paths": select_top_k(
                        learned_weights, snapshot_names, min(k, Q0_SNAPSHOT_COUNT)
                    )[0],
                    "weights": select_top_k(
                        learned_weights, snapshot_names, min(k, Q0_SNAPSHOT_COUNT)
                    )[1],
                }
                for k in LEARNED_K_VALUES
            },
        }
        with open(os.path.join(q0_dir, "mixture_weights.json"), "w") as f:
            json.dump(mixture, f)
        mopd_path = os.path.join(mopd_dir, "opsd.pt")
        with open(mopd_path, "wb") as f:
            f.write(b"mopd")
        checkpoint_args = argparse.Namespace(
            baseline_run_dir=baseline_dir, q0_run_dir=q0_dir, mopd_run_dir=mopd_dir,
        )
        baseline_paths, q0_paths, _, loaded_names, loaded_weights = validate_checkpoint_inputs(checkpoint_args)
        assert len(baseline_paths) == BASELINE_CHECKPOINT_COUNT
        assert len(q0_paths) == Q0_SNAPSHOT_COUNT
        assert loaded_names == snapshot_names
        assert loaded_weights == [1.0] * Q0_SNAPSHOT_COUNT
        assert validate_mopd_checkpoint(checkpoint_args) == mopd_path

        mixture["top_k"]["2"]["weights"] = [0.75, 0.25]
        with open(os.path.join(q0_dir, "mixture_weights.json"), "w") as f:
            json.dump(mixture, f)
        try:
            validate_checkpoint_inputs(checkpoint_args)
        except ValueError as exc:
            assert "top_k[2] weights do not match learned weights" in str(exc)
        else:
            raise AssertionError("invalid top_k manifest must fail")

    print("eval self-test passed")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run_eval(args)


if __name__ == "__main__":
    main()
