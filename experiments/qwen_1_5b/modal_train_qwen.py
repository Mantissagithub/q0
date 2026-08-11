# scale-up run: qwen2.5-1.5b full fine-tune on a single a100-40gb with a colocated
# vllm rollout, then eval base vs each pass under the same math_verify scorer.
#   modal run modal_train_qwen.py

import modal

app = modal.App("q0-grpo-qwen")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # vllm pins its own torch/transformers; let it resolve, add the rest on top
    .pip_install(
        "vllm==0.8.5",
        "transformers==4.51.3",
        "datasets",
        "accelerate",
        "math_verify",
        "huggingface_hub",
    )
    # v0 engine keeps the model in-process so per-step weight sync is zero-copy.
    # expandable_segments lets the caching allocator grow/shrink segments instead of
    # fragmenting the arena over a long colocated vllm+training run — this is the fix for
    # the cuda "illegal memory access" that surfaced ~30-50 steps in on the un-tuned run.
    .env({"VLLM_USE_V1": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
    .add_local_file("core/eval.py", "/root/eval.py")
)

vol = modal.Volume.from_name("q0-grpo-runs", create_if_missing=True)

RUN_ID = "grpo_qwen1_5b"


@app.function(image=image, gpu="A100-40GB", timeout=6 * 3600, volumes={"/outputs": vol})
def train_and_eval():
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root")

    # 1) train the qwen1_5b profile with the vllm rollout. the profile supplies
    # group/gen/lr/passes/train_target; qwen is ungated so no hf token needed.
    # --resume + --ckpt-every-steps make preemption cheap: modal auto-restarts this
    # function from the top, and with --resume the child picks up /outputs/<run>/
    # checkpoint.pt (background-committed to the volume every few sec) and skips the
    # batches it already finished, instead of restarting from step 1.
    # stream the child stdout line-by-line so per-step reward shows up live over the
    # multi-hour run (buffering it would leave us blind until the very end) while we
    # still keep the full text for the log file + reward curve.
    proc = subprocess.Popen(
        [sys.executable, "train_grpo.py",
         "--profile", "qwen1_5b", "--rollout", "vllm",
         "--resume", "--ckpt-every-steps", "16",
         "--device", "cuda", "--output-dir", "/outputs", "--run-id", RUN_ID],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    stdout_text = "".join(lines)
    # write + commit the log BEFORE reacting to the return code, so a crash still
    # leaves the full training log on the volume
    with open(f"/outputs/train_{RUN_ID}_stdout.log", "w") as f:
        f.write(stdout_text)
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"training failed with return code {proc.returncode}")

    # 2) eval base vs each pass with the SAME math_verify scorer. eval stays hf
    # greedy; pull model id / revision / gen length from the qwen profile, not the
    # smol135m module globals.
    import torch

    import eval as ev
    import train_grpo as tg

    prof = tg.PROFILES["qwen1_5b"]
    model_name, revision = prof["model_name"], prof["revision"]
    max_new_tokens, num_passes = prof["max_new_tokens"], prof["num_passes"]

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

    for p in range(1, num_passes + 1):
        path = f"/outputs/{RUN_ID}/pass_{p:02d}.pt"
        model = tg.load_model_from_checkpoint(model_name, revision, path, device)
        results[f"pass_{p:02d}"] = score(model, f"pass_{p:02d}")
        del model
        torch.cuda.empty_cache()

    # training reward trajectory from the per-step log lines
    reward_curve = [
        float(line.split("reward")[-1])
        for line in stdout_text.splitlines()
        if "reward" in line and " step " in line
    ]

    payload = {"eval": results, "reward_curve": reward_curve, "test_n": len(test_set)}
    with open("/outputs/results_qwen.json", "w") as f:
        json.dump(payload, f, indent=2)
    vol.commit()
    return payload


@app.local_entrypoint()
def main():
    import json

    res = train_and_eval.remote()
    with open("experiments/qwen_1_5b/results/results_qwen.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved results_qwen.json locally")
    print("\n===== EVAL (same math_verify scorer for every model) =====")
    print(f"gsm8k test examples: {res['test_n']}")
    for name, r in res["eval"].items():
        print(f"{name:10} exact_match={r['exact_match']:.4f}  parse_rate={r['parse_rate']:.4f}"
              f"  avg_len={r['avg_response_length']:.1f}")
    rc = res["reward_curve"]
    if rc:
        print(f"\nreward curve: first={rc[0]:.3f}  last={rc[-1]:.3f}  max={max(rc):.3f}  steps={len(rc)}")
