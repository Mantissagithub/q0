# math-500 transfer check for the qwen1_5b grpo passes — the full 500 problems.
# reuses eval_math500's MATH prompt and the same math_verify symbolic scorer, but
# points at base + whatever pass_*.pt saved on the volume (the gsm8k run trained on
# gsm8k, so this is pure transfer).
#   modal run modal_math500_qwen.py

import modal

app = modal.App("q0-grpo-qwen-math500")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers==4.51.3", "datasets", "accelerate",
        "math_verify", "huggingface_hub",
    )
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
    .add_local_file("core/eval.py", "/root/eval.py")
    .add_local_file("core/eval_math500.py", "/root/eval_math500.py")
)

vol = modal.Volume.from_name("q0-grpo-runs")

RUN_ID = "grpo_qwen1_5b"
N = 500
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256


@app.function(image=image, gpu="A100-40GB", timeout=2 * 3600, volumes={"/outputs": vol})
def evaluate():
    import glob
    import json
    import os
    import re

    import torch
    from datasets import load_dataset

    import eval as ev
    import eval_math500 as em
    import train_grpo as tg

    os.chdir("/root")

    prof = tg.PROFILES["qwen1_5b"]
    model_name, revision = prof["model_name"], prof["revision"]

    # the full math-500 test split
    dataset = load_dataset(em.MATH_DATASET, split=em.MATH_SPLIT, revision=em.MATH_REVISION)
    examples = list(dataset)[:N]
    tokenizer = tg.load_tokenizer(model_name, revision)

    @torch.no_grad()
    def score_model(model, name):
        model.eval()
        texts = []
        for start in range(0, len(examples), BATCH_SIZE):
            batch = examples[start:start + BATCH_SIZE]
            prompts = [em.build_prompt(e["problem"]) for e in batch]
            encoded = ev.tokenize_left_padded(tokenizer, prompts, "cuda")
            out = model.generate(
                **encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=[tg.ANSWER_STOP_STRING], tokenizer=tokenizer, use_cache=True,
            )
            completion = out[:, encoded["input_ids"].shape[1]:]
            texts.extend(tokenizer.batch_decode(completion, skip_special_tokens=True))
        correct = parsed = 0
        for text, e in zip(texts, examples):
            is_correct, is_parsed, _ = em.score_answer(text, e["answer"])
            correct += int(is_correct)
            parsed += int(is_parsed)
        return {"correct": correct, "count": len(examples),
                "accuracy": correct / len(examples), "parse_rate": parsed / len(examples)}

    results = {}
    base = tg.load_base_model(model_name, revision, "cuda")
    results["base"] = score_model(base, "base")
    del base
    torch.cuda.empty_cache()

    # only pass_02 — it was the best pass on gsm8k, so it's the one worth the transfer
    # check; base above stays as the reference point. skip pass_01/pass_03 to save credits.
    paths = sorted(
        glob.glob(f"/outputs/{RUN_ID}/pass_02.pt"),
        key=lambda p: int(re.search(r"pass_(\d+)", p).group(1)),
    )
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        model = tg.load_model_from_checkpoint(model_name, revision, path, "cuda")
        results[name] = score_model(model, name)
        del model
        torch.cuda.empty_cache()

    payload = {"eval": results, "n": len(examples), "dataset": "MATH-500"}
    with open("/outputs/results_math500_qwen.json", "w") as f:
        json.dump(payload, f, indent=2)
    vol.commit()
    return payload


@app.local_entrypoint()
def main():
    import json

    res = evaluate.remote()
    with open("experiments/qwen_1_5b/results/results_math500_qwen.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved results_math500_qwen.json locally")
    print(f"\n===== MATH-500 transfer (math_verify, n={res['n']}) =====")
    for name, r in res["eval"].items():
        print(f"{name:10} accuracy={r['accuracy']:.4f}  parse_rate={r['parse_rate']:.4f}"
              f"  ({r['correct']}/{r['count']})")
