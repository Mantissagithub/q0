# standalone eval for the qwen1_5b run: the full run timed out mid-pass-4 (modal 6h
# wall) before its bundled eval could run, so we eval the checkpoints that did save.
# discovers whatever pass_*.pt exist on the volume rather than assuming all 4 passes,
# scores base + each pass with the SAME math_verify scorer, writes results_qwen.json
# so modal_push_qwen.py can pick the best pass.
#   modal run modal_eval_qwen.py

import modal

app = modal.App("q0-grpo-qwen-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers==4.51.3", "datasets", "accelerate",
        "math_verify", "huggingface_hub",
    )
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
    .add_local_file("core/eval.py", "/root/eval.py")
)

vol = modal.Volume.from_name("q0-grpo-runs")

RUN_ID = "grpo_qwen1_5b"


@app.function(image=image, gpu="A100-40GB", timeout=2 * 3600, volumes={"/outputs": vol})
def evaluate():
    import glob
    import json
    import os
    import re

    import torch

    import eval as ev
    import train_grpo as tg

    os.chdir("/root")

    prof = tg.PROFILES["qwen1_5b"]
    model_name, revision = prof["model_name"], prof["revision"]
    max_new_tokens = prof["max_new_tokens"]

    device = "cuda"
    tokenizer = tg.load_tokenizer(model_name, revision)
    test_set = ev.load_test_set()

    def score(model, source):
        model.eval()
        with torch.no_grad():
            return ev.evaluate_single_model(
                model, tokenizer, test_set, 32, max_new_tokens, device, source
            )

    results = {}
    base = tg.load_base_model(model_name, revision, device)
    results["base"] = score(base, "base")
    del base
    torch.cuda.empty_cache()

    # only the passes that actually saved before the timeout — sort by pass number so
    # the table reads base, pass_01, pass_02, ...
    paths = sorted(
        glob.glob(f"/outputs/{RUN_ID}/pass_*.pt"),
        key=lambda p: int(re.search(r"pass_(\d+)", p).group(1)),
    )
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        model = tg.load_model_from_checkpoint(model_name, revision, path, device)
        results[name] = score(model, name)
        del model
        torch.cuda.empty_cache()

    # reward_curve is left empty here — the per-step lines live in the local run log,
    # not on the volume, so the training reward trajectory is reported from there.
    payload = {"eval": results, "reward_curve": [], "test_n": len(test_set)}
    with open("/outputs/results_qwen.json", "w") as f:
        json.dump(payload, f, indent=2)
    vol.commit()
    return payload


@app.local_entrypoint()
def main():
    import json

    res = evaluate.remote()
    with open("experiments/qwen_1_5b/results/results_qwen.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved results_qwen.json locally")
    print("\n===== EVAL (same math_verify scorer for every model) =====")
    print(f"gsm8k test examples: {res['test_n']}")
    for name, r in res["eval"].items():
        print(f"{name:10} exact_match={r['exact_match']:.4f}  parse_rate={r['parse_rate']:.4f}"
              f"  avg_len={r['avg_response_length']:.1f}")
