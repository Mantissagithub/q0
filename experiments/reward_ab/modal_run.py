"""Rent a GPU on Modal, run new-reward GRPO training, then eval base vs each
pass checkpoint with the same robust math_verify scorer.

Usage:
    modal run modal_run.py

Single run, no old-reward baseline (per request): shows the new-reward GRPO
trains cleanly and reports final GSM8K accuracy under the fixed evaluator, with
the base model as a floor reference. It does NOT by itself prove the reward fix
caused the gain (that needs an old-vs-new A/B under the same scorer).
"""

import modal

app = modal.App("q0-grpo-verify")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "accelerate", "math_verify")
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
    .add_local_file("core/eval.py", "/root/eval.py")
)

vol = modal.Volume.from_name("q0-grpo-runs", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=4 * 3600, volumes={"/outputs": vol})
def train_and_eval():
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root")

    # 1) train — the working-tree (new-reward) train_grpo.py. its contract
    # enforces cuda + the exact 8192-prompt / 32768-completion reduced config.
    proc = subprocess.run(
        [sys.executable, "train_grpo.py", "--device", "cuda", "--output-dir", "/outputs"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout[-6000:])
    # persist the FULL raw training log to the volume (console echo above is
    # truncated; this file is complete, for graphs / auditing).
    with open("/outputs/train_stdout.log", "w") as f:
        f.write(proc.stdout)
    proc.check_returncode()
    vol.commit()

    # 2) eval base vs each pass checkpoint with the SAME robust scorer
    import torch

    import eval as ev
    import train_grpo as tg

    device = "cuda"
    tokenizer = tg.load_tokenizer(tg.MODEL_NAME, tg.MODEL_REVISION)
    test_set = ev.load_test_set()

    def score(model, source):
        model.eval()
        with torch.no_grad():
            return ev.evaluate_single_model(
                model, tokenizer, test_set, 64, tg.MAX_NEW_TOKENS, device, source
            )

    results = {}
    base = tg.load_base_model(tg.MODEL_NAME, tg.MODEL_REVISION, device)
    results["base"] = score(base, "base")
    del base
    torch.cuda.empty_cache()

    for p in range(1, tg.NUM_PASSES + 1):
        path = f"/outputs/grpo_baseline/pass_{p:02d}.pt"
        model = tg.load_model_from_checkpoint(tg.MODEL_NAME, tg.MODEL_REVISION, path, device)
        results[f"pass_{p:02d}"] = score(model, f"pass_{p:02d}")
        del model
        torch.cuda.empty_cache()

    # training reward trajectory from the per-step log lines
    reward_curve = [
        float(line.split("reward")[-1])
        for line in proc.stdout.splitlines()
        if "reward" in line and " step " in line
    ]

    payload = {"eval": results, "reward_curve": reward_curve, "test_n": len(test_set)}
    with open("/outputs/verify_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    vol.commit()
    return payload


@app.local_entrypoint()
def main():
    import json

    res = train_and_eval.remote()
    with open("experiments/reward_ab/results/verify_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved verify_results.json locally")
    print("\n===== EVAL (same robust scorer for every model) =====")
    print(f"gsm8k test examples: {res['test_n']}")
    for name, r in res["eval"].items():
        print(f"{name:10} exact_match={r['exact_match']:.4f}  parse_rate={r['parse_rate']:.4f}"
              f"  avg_len={r['avg_response_length']:.1f}")
    rc = res["reward_curve"]
    if rc:
        print(f"\nreward curve: first={rc[0]:.3f}  last={rc[-1]:.3f}  max={max(rc):.3f}  steps={len(rc)}")
