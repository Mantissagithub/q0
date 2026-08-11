# push the best qwen1_5b grpo checkpoint to the hub. runs on modal (local env's
# huggingface_hub is incompatible). the best pass is read from results_qwen.json on
# the volume, so we never hardcode which pass won.
#   HF_TOKEN=hf_xxx modal run modal_push_qwen.py

import os

import modal

REPO = "Pradheep1647/q0-gsm8k-1_5b-grpo"
RUN_ID = "grpo_qwen1_5b"

app = modal.App("q0-hf-push-qwen")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers")
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
)

vol = modal.Volume.from_name("q0-grpo-runs")


@app.function(
    image=image,
    timeout=1800,
    volumes={"/outputs": vol},
    # token injected only as a secret from the local env var, never written in-file
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})],
)
def push():
    import json
    import os

    import train_grpo as tg

    prof = tg.PROFILES["qwen1_5b"]
    model_name, revision = prof["model_name"], prof["revision"]

    # pick the pass with the highest exact_match from the eval table
    with open("/outputs/results_qwen.json") as f:
        evals = json.load(f)["eval"]
    passes = {k: v for k, v in evals.items() if k.startswith("pass_")}
    best = max(passes, key=lambda k: passes[k]["exact_match"])
    ckpt = f"/outputs/{RUN_ID}/{best}.pt"
    print(f"best checkpoint: {best} (exact_match={passes[best]['exact_match']:.4f}) -> {ckpt}")

    token = os.environ["HF_TOKEN"]
    model = tg.load_model_from_checkpoint(model_name, revision, ckpt, "cpu")
    tokenizer = tg.load_tokenizer(model_name, revision)
    msg = f"qwen2.5-1.5b grpo best checkpoint ({best}, math_verify reward)"
    model.push_to_hub(REPO, token=token, commit_message=msg)
    tokenizer.push_to_hub(REPO, token=token, commit_message=msg)
    return f"https://huggingface.co/{REPO}"


@app.local_entrypoint()
def main():
    print("pushed ->", push.remote())
