"""Merge the best MOPD adapter (t=2) into a 16-bit Qwen2.5-1.5B model and push.

The MOPD run saves adapter-only LoRA weights (opsd.pt). For a deployable
checkpoint we load the base model in bf16 (unquantized), apply the LoRA adapter,
merge_and_unload to fold the adapter into the base weights, and push the merged
16-bit model + tokenizer to the Hub. CPU-only: no GPU required.

Auth uses the locally stored HF token (huggingface-cli login); no token is
written into this file or any command.
"""

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

import train_grpo as tg

REPO = "Pradheep1647/q0-gsm8k-1_5b-mopd"
ADAPTER = "runs/mopd_qwen1_5b_laptop_t2/opsd.pt"
PROFILE = tg.PROFILES["qwen1_5b_laptop"]
NAME, REV = PROFILE["model_name"], PROFILE["revision"]

MSG = "publish selected checkpoint and documentation"

CARD = f"""---
base_model: {NAME}
library_name: transformers
tags:
- qwen2
- gsm8k
- distillation
- mopd
- q0
---

# q0-gsm8k-1.5b-mopd

Qwen2.5-1.5B-Instruct fine-tuned with **MOPD** (multi-teacher on-policy
distillation) on GSM8K, distilled from a q0 cyclic-trajectory GRPO mixture.

- **Base:** `{NAME}` (revision `{REV}`)
- **Method:** MOPD, teacher_count=2 (top-2 q0 snapshots: cycle02 + cycle03,
  uniformly averaged), student = 4-bit QLoRA (NF4, r=8, alpha=16).
- **Selection:** step 192 by GSM8K validation pass@8 (256-example slice, seed 42).
- **GSM8K validation:** pass@1 49.07%, pass@4 76.29%, pass@8 84.77% (256-example deterministic slice).
- **MATH-500 test:** pass@1 16.40%, pass@4 35.00% (500 problems, four samples each).
- **Format:** LoRA adapter merged into 16-bit base weights (deployable directly).

Trained on an RTX 4060 Laptop (8 GB) with 4-bit QLoRA. This is the 1.5B
checkpoint (the repository's 1B-scale target), not a literal 1.0B model.
"""


def main():
    print(f"loading base {NAME}@{REV[:8]} (bf16, cpu)...")
    base = AutoModelForCausalLM.from_pretrained(NAME, revision=REV, torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(
        r=PROFILE["lora_rank"], lora_alpha=PROFILE["lora_alpha"],
        lora_dropout=PROFILE["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    print(f"loading adapter {ADAPTER}...")
    state = torch.load(ADAPTER, map_location="cpu", weights_only=True)
    result = set_peft_model_state_dict(model, state)
    if getattr(result, "unexpected_keys", None):
        raise SystemExit(f"adapter load had unexpected keys: {result.unexpected_keys[:5]}")
    print("merging adapter into base weights...")
    merged = model.merge_and_unload()
    tokenizer = tg.load_tokenizer(NAME, REV)

    print(f"pushing merged 16-bit model -> {REPO} ...")
    merged.push_to_hub(REPO, commit_message=MSG)
    tokenizer.push_to_hub(REPO, commit_message=MSG)
    from huggingface_hub import upload_file
    import io
    upload_file(path_or_fileobj=io.BytesIO(CARD.encode()), path_in_repo="README.md",
                repo_id=REPO, commit_message="add model card")
    print(f"done -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
