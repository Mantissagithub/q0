import argparse
import collections
import json
import os
import tempfile
import time

import torch

import eval as ev
import train_grpo as tg


# math-500 transfer check for the final baseline, q0, and mopd systems
MATH_DATASET = "HuggingFaceH4/MATH-500"
MATH_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH_SPLIT = "test"
MATH_COUNT = 500
MAX_NEW_TOKENS = 256
BATCH_SIZE = 16

MATH_PROMPT = (
    "Solve the math problem below. Give a short chain of reasoning, then finish "
    "with the final answer wrapped exactly as <answer>$\\boxed{{answer}}$</answer>. "
    "The answer may be a number, expression, tuple, interval, or short piece of text.\n\n"
    "Problem: What is 2 + 3?\n"
    "Reasoning: 2 + 3 = 5. <answer>$\\boxed{{5}}$</answer>\n\n"
    "Problem: {problem}\nReasoning:"
)


def build_prompt(problem):
    return MATH_PROMPT.format(problem=problem.strip())


def extract_answer(text):
    matches = tg.ANSWER_TAG_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def score_answer(text, gold):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    tagged_answer = extract_answer(text)
    answer = tagged_answer or text.strip() or None
    if answer is None:
        return False, False, None
    gold_parsed = parse(
        f"$\\boxed{{{gold}}}$", extraction_config=[LatexExtractionConfig()],
    )
    answer_parsed = parse(
        answer, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )
    correct = bool(gold_parsed and answer_parsed and verify(gold_parsed, answer_parsed))
    return correct, bool(answer_parsed), answer


def load_math500():
    from datasets import load_dataset

    dataset = load_dataset(
        MATH_DATASET, split=MATH_SPLIT, revision=MATH_REVISION,
    )
    if len(dataset) != MATH_COUNT:
        raise ValueError(f"expected {MATH_COUNT} math-500 examples, got {len(dataset)}")
    return list(dataset)


def atomic_json_dump(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_jsonl_dump(rows, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-", suffix=".jsonl", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def summarize_breakdown(counts):
    return {
        str(name): {
            "correct": values[0],
            "count": values[1],
            "accuracy": values[0] / values[1],
        }
        for name, values in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def score_outputs(texts, examples, tokenizer, system_name, source, elapsed, peak_memory):
    correct = 0
    parsed = 0
    total_tokens = 0
    subject_counts = collections.defaultdict(lambda: [0, 0])
    level_counts = collections.defaultdict(lambda: [0, 0])
    predictions = []

    for text, example in zip(texts, examples):
        is_correct, is_parsed, answer = score_answer(text, example["answer"])
        correct += int(is_correct)
        parsed += int(is_parsed)
        total_tokens += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        subject_counts[example["subject"]][0] += int(is_correct)
        subject_counts[example["subject"]][1] += 1
        level_counts[example["level"]][0] += int(is_correct)
        level_counts[example["level"]][1] += 1
        predictions.append({
            "system": system_name,
            "unique_id": example["unique_id"],
            "subject": example["subject"],
            "level": example["level"],
            "gold": example["answer"],
            "prediction": answer,
            "correct": is_correct,
            "parsed": is_parsed,
            "completion": text,
        })

    count = len(examples)
    result = {
        "correct": correct,
        "count": count,
        "accuracy": correct / count,
        "parse_rate": parsed / count,
        "avg_response_length": total_tokens / count,
        "timing_seconds": elapsed,
        "peak_memory_bytes": peak_memory,
        "by_subject": summarize_breakdown(subject_counts),
        "by_level": summarize_breakdown(level_counts),
        "source": source,
    }
    return result, predictions


@torch.no_grad()
def evaluate_single(model, tokenizer, examples, args, system_name, source):
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    texts = []
    started = time.time()
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        prompts = [build_prompt(example["problem"]) for example in batch]
        encoded = ev.tokenize_left_padded(tokenizer, prompts, args.device)
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=[tg.ANSWER_STOP_STRING],
            tokenizer=tokenizer,
            use_cache=True,
        )
        completion_ids = output[:, encoded["input_ids"].shape[1]:]
        texts.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))
        print(f"{system_name}: {min(start + args.batch_size, len(examples))}/{len(examples)}")
    elapsed = time.time() - started
    peak_memory = torch.cuda.max_memory_allocated(args.device) if args.device == "cuda" else 0
    return score_outputs(
        texts, examples, tokenizer, system_name, source, elapsed, peak_memory,
    )


@torch.no_grad()
def evaluate_ensemble(paths, weights, tokenizer, examples, args):
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    models = [
        tg.load_model_from_checkpoint(args.model_name, args.revision, path, args.device)
        for path in paths
    ]
    for model in models:
        model.eval()

    texts = []
    started = time.time()
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        prompts = [build_prompt(example["problem"]) for example in batch]
        texts.extend(ev.ensemble_greedy_generate(
            models, weights, tokenizer, prompts, args.max_new_tokens, args.device,
        ))
        print(f"q0_learned_top4: {min(start + args.batch_size, len(examples))}/{len(examples)}")
    elapsed = time.time() - started
    peak_memory = torch.cuda.max_memory_allocated(args.device) if args.device == "cuda" else 0
    result = score_outputs(
        texts,
        examples,
        tokenizer,
        "q0_learned_top4",
        {"paths": paths, "weights": weights},
        elapsed,
        peak_memory,
    )
    models.clear()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return result


def load_top4(q0_run_dir):
    mixture_path = os.path.join(q0_run_dir, "mixture_weights.json")
    with open(mixture_path) as handle:
        mixture = json.load(handle)
    names = mixture.get("snapshot_paths")
    weights = mixture.get("weights")
    if not isinstance(names, list) or not isinstance(weights, list) or len(names) != len(weights):
        raise ValueError("mixture manifest must contain aligned snapshot_paths and weights")
    selected_names, selected_weights = ev.select_top_k(weights, names, 4)
    top4 = mixture.get("top_k", {}).get("4", {})
    if top4.get("snapshot_paths") != selected_names:
        raise ValueError("top_k[4] snapshot paths do not match the learned ranking")
    paths = [os.path.join(q0_run_dir, name) for name in selected_names]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise ValueError(f"missing q0 top-4 checkpoints: {missing}")
    return paths, selected_weights


def validate_inputs(args):
    if not args.manifest:
        raise ValueError("--manifest is required for final MATH-500 evaluation")
    _payload, systems, manifest_sha256 = ev.load_final_manifest(
        args.manifest, args.model_name, args.revision,
    )
    paths = {name: spec["path"] for name, spec in systems.items() if spec["type"] == "single"}
    return paths, systems["q0_learned_top4"]["paths"], systems["q0_learned_top4"]["weights"], manifest_sha256


def save_progress(args, metadata, results, predictions):
    atomic_json_dump({"metadata": metadata, "results": results}, args.output)
    atomic_jsonl_dump(predictions, args.predictions)


def run_eval(args):
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("the math-500 run requires cuda")
    paths, top4_paths, top4_weights, manifest_sha256 = validate_inputs(args)
    examples = load_math500()
    tokenizer = tg.load_tokenizer(args.model_name, args.revision)
    metadata = {
        "dataset": MATH_DATASET,
        "dataset_revision": MATH_REVISION,
        "split": MATH_SPLIT,
        "count": len(examples),
        "model_name": args.model_name,
        "model_revision": args.revision,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "scorer": "math-verify 0.9.0 symbolic equivalence",
        "manifest_sha256": manifest_sha256,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results = {}
    predictions = []
    if args.resume and os.path.isfile(args.output):
        with open(args.output) as handle:
            saved = json.load(handle)
        results = saved.get("results", {})
        if os.path.isfile(args.predictions):
            with open(args.predictions) as handle:
                predictions = [json.loads(line) for line in handle if line.strip()]

    for system_name in ("baseline_grpo", "q0_best_single"):
        if system_name in results:
            continue
        model = tg.load_model_from_checkpoint(
            args.model_name, args.revision, paths[system_name], args.device,
        )
        model.eval()
        result, rows = evaluate_single(
            model, tokenizer, examples, args, system_name, paths[system_name],
        )
        results[system_name] = result
        predictions.extend(rows)
        model = None
        torch.cuda.empty_cache()
        save_progress(args, metadata, results, predictions)
        print(f"{system_name}: {result['correct']}/{result['count']} = {result['accuracy']:.4f}")

    if "q0_learned_top4" not in results:
        result, rows = evaluate_ensemble(
            top4_paths, top4_weights, tokenizer, examples, args,
        )
        results["q0_learned_top4"] = result
        predictions.extend(rows)
        save_progress(args, metadata, results, predictions)
        print(f"q0_learned_top4: {result['correct']}/{result['count']} = {result['accuracy']:.4f}")

    if "mopd" not in results:
        model = tg.load_model_from_checkpoint(
            args.model_name, args.revision, paths["mopd"], args.device,
        )
        model.eval()
        result, rows = evaluate_single(
            model, tokenizer, examples, args, "mopd", paths["mopd"],
        )
        results["mopd"] = result
        predictions.extend(rows)
        model = None
        torch.cuda.empty_cache()
        save_progress(args, metadata, results, predictions)
        print(f"mopd: {result['correct']}/{result['count']} = {result['accuracy']:.4f}")

    print(f"wrote {args.output} and {args.predictions}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="evaluate the final systems on math-500")
    parser.add_argument("--model-name", default=tg.MODEL_NAME)
    parser.add_argument("--revision", default=tg.MODEL_REVISION)
    parser.add_argument("--manifest", required=False, help="frozen final-system manifest")
    parser.add_argument("--baseline-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--q0-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--q0-run-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mopd-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", default="math500_results.json")
    parser.add_argument("--predictions", default="math500_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test():
    prompt = build_prompt("solve x + 1 = 2")
    assert "<answer>$\\boxed{answer}$</answer>" in prompt
    assert "<answer>$\\boxed{5}$</answer>" in prompt
    assert "solve x + 1 = 2" in prompt
    assert extract_answer("work <answer>$\\boxed{14/3}$</answer>") == "$\\boxed{14/3}$"
    assert extract_answer("no answer") is None
    assert score_answer("<answer>$\\boxed{14/3}$</answer>", "\\frac{14}{3}")[:2] == (True, True)
    assert score_answer("the answer is $\\boxed{14/3}$", "\\frac{14}{3}")[:2] == (True, True)
    assert score_answer("<answer>$\\boxed{p-q}$</answer>", "p - q")[:2] == (True, True)
    assert score_answer("<answer>$\\boxed{7}$</answer>", "8")[:2] == (False, True)
    counts = summarize_breakdown({"algebra": [2, 4]})
    assert counts["algebra"]["accuracy"] == 0.5
    print("eval_math500 self-test passed")


if __name__ == "__main__":
    arguments = build_arg_parser().parse_args()
    if arguments.self_test:
        self_test()
    else:
        run_eval(arguments)
