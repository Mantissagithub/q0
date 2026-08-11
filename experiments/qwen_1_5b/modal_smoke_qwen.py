# smoke run before the full job: real qwen1_5b config (group 16, 384 tok, mb 8,
# vllm resident + adamw) but only train_target=64 -> 4 passes * 2 batches = 8 steps.
# the point is to catch a100-40gb oom / loop-integration bugs cheaply before the
# ~3-5h full run. group_size/max_new_tokens/num_passes are frozen by the contract,
# so train_target is the only knob we shrink.
#   modal run modal_smoke_qwen.py

import modal

app = modal.App("q0-grpo-qwen-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5", "transformers==4.51.3", "datasets",
        "accelerate", "math_verify", "huggingface_hub",
    )
    .env({"VLLM_USE_V1": "0"})
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
    .add_local_file("core/eval.py", "/root/eval.py")
)

vol = modal.Volume.from_name("q0-grpo-runs", create_if_missing=True)

RUN_ID = "grpo_qwen_smoke"


@app.function(image=image, gpu="A100-40GB", timeout=45 * 60, volumes={"/outputs": vol})
def smoke():
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root")

    proc = subprocess.run(
        [sys.executable, "train_grpo.py",
         "--profile", "qwen1_5b", "--rollout", "vllm", "--train-target", "64",
         "--device", "cuda", "--output-dir", "/outputs", "--run-id", RUN_ID],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout[-6000:])
    # commit the log to the volume BEFORE anything can tear the pod down
    with open(f"/outputs/train_{RUN_ID}_stdout.log", "w") as f:
        f.write(proc.stdout)
    vol.commit()
    proc.check_returncode()

    # tiny eval sanity: base vs final pass on 64 test examples, exercises the
    # checkpoint-load + eval path on a real qwen checkpoint
    import torch

    import eval as ev
    import train_grpo as tg

    prof = tg.PROFILES["qwen1_5b"]
    model_name, revision, max_new_tokens = prof["model_name"], prof["revision"], prof["max_new_tokens"]
    tokenizer = tg.load_tokenizer(model_name, revision)
    test_set = ev.load_test_set()[:64]

    def score(model, src):
        model.eval()
        with torch.no_grad():
            return ev.evaluate_single_model(model, tokenizer, test_set, 16, max_new_tokens, "cuda", src)

    out = {}
    base = tg.load_base_model(model_name, revision, "cuda")
    out["base"] = score(base, "base")
    del base
    torch.cuda.empty_cache()

    final = f"/outputs/{RUN_ID}/pass_{prof['num_passes']:02d}.pt"
    model = tg.load_model_from_checkpoint(model_name, revision, final, "cuda")
    out["final_pass"] = score(model, "final_pass")

    steps = [l for l in proc.stdout.splitlines() if " step " in l and "reward" in l]
    return {"eval": out, "n": len(test_set), "train_steps": len(steps),
            "last_step_line": steps[-1] if steps else None}


@app.local_entrypoint()
def main():
    r = smoke.remote()
    print("\n===== qwen smoke result =====")
    print(f"train steps run: {r['train_steps']}  (expect 8)")
    print(f"last step: {r['last_step_line']}")
    for k, v in r["eval"].items():
        print(f"{k:11} exact_match={v['exact_match']:.4f}  parse_rate={v['parse_rate']:.4f}  (n={r['n']})")
