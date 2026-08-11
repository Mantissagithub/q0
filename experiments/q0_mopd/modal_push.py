"""Push the best new-reward GRPO checkpoint (pass_03) to the HF Hub.

Runs on Modal (local transformers is incompatible with the installed
huggingface_hub). Reads the checkpoint from the q0-grpo-runs volume, rebuilds a
proper SmolLM2-135M model + tokenizer, and pushes to REPO.

Token is injected as a Modal secret from the local HF_TOKEN env var, so it is
never written into this file:
    HF_TOKEN=hf_xxx modal run modal_push.py
"""

import os

import modal

REPO = "Pradheep1647/q0-gsm8k-135m-grpo"
CKPT = "/outputs/grpo_baseline/pass_03.pt"

app = modal.App("q0-hf-push")

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
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})],
)
def push():
    import os

    import train_grpo as tg

    token = os.environ["HF_TOKEN"]
    model = tg.load_model_from_checkpoint(tg.MODEL_NAME, tg.MODEL_REVISION, CKPT, "cpu")
    tokenizer = tg.load_tokenizer(tg.MODEL_NAME, tg.MODEL_REVISION)
    msg = "new-reward GRPO best checkpoint (pass_03, math_verify reward)"
    model.push_to_hub(REPO, token=token, commit_message=msg)
    tokenizer.push_to_hub(REPO, token=token, commit_message=msg)
    return f"https://huggingface.co/{REPO}"


@app.local_entrypoint()
def main():
    print("pushed ->", push.remote())
