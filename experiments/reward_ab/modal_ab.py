"""A/B: old-reward (strict NUMERIC_RE) vs new-reward (math_verify) GRPO.

Both arms are scored with the SAME new robust evaluator so the comparison
reflects learning, not measurement. The new-reward arm reuses the checkpoints
already on the volume (grpo_baseline, identical seed/config); this job trains
only the old-reward arm (grpo_oldreward), then evals base + both arms.

    modal run modal_ab.py
"""

import modal

app = modal.App("q0-grpo-ab")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "accelerate", "math_verify")
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
    .add_local_file("core/eval.py", "/root/eval.py")
    .add_local_file("experiments/reward_ab/train_grpo_old.py", "/root/train_grpo_old.py")
)

vol = modal.Volume.from_name("q0-grpo-runs")


@app.function(image=image, gpu="A10G", timeout=4 * 3600, volumes={"/outputs": vol})
def ab():
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root")

    # 1) train the OLD-reward arm (new-reward arm already on the volume)
    proc = subprocess.run(
        [sys.executable, "train_grpo_old.py", "--device", "cuda",
         "--output-dir", "/outputs", "--run-id", "grpo_oldreward"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout[-4000:])
    with open("/outputs/train_oldreward_stdout.log", "w") as f:
        f.write(proc.stdout)
    proc.check_returncode()
    vol.commit()
    old_curve = [
        float(line.split("reward")[-1])
        for line in proc.stdout.splitlines()
        if "reward" in line and " step " in line
    ]

    # 2) eval base + both arms with the SAME new math_verify scorer
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

    for tag, run in [("new", "grpo_baseline"), ("old", "grpo_oldreward")]:
        for p in range(1, tg.NUM_PASSES + 1):
            path = f"/outputs/{run}/pass_{p:02d}.pt"
            model = tg.load_model_from_checkpoint(tg.MODEL_NAME, tg.MODEL_REVISION, path, device)
            results[f"{tag}_pass_{p:02d}"] = score(model, f"{tag}_pass_{p:02d}")
            del model
            torch.cuda.empty_cache()

    payload = {"eval": results, "old_reward_curve": old_curve, "test_n": len(test_set)}
    with open("/outputs/ab_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    vol.commit()
    return payload


@app.local_entrypoint()
def main():
    import json

    res = ab.remote()
    with open("experiments/reward_ab/results/ab_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved ab_results.json locally")
    print(f"\nGSM8K test n={res['test_n']} — same math_verify scorer for every row")
    for k, v in res["eval"].items():
        print(f"{k:14} exact_match={v['exact_match']:.4f}  parse_rate={v['parse_rate']:.4f}")
