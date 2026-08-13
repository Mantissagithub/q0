"""Publish the q0 top-2 weighted mixture (adapters + weights) to the Hub.

The q0 ensemble mixes adapters in probability space (softmax per adapter,
weighted sum, sample) — it is NOT a weight-space merge, so it cannot be folded
into a single checkpoint. We therefore upload the two top-2 LoRA adapters plus a
manifest describing base model, LoRA config, and mixture weights, so the
ensemble can be reconstructed for generation later.

Auth uses the locally stored HF token; no token is written into this file.
"""

import io
import json

from huggingface_hub import HfApi, upload_file

import train_grpo as tg

REPO = "Pradheep1647/q0-gsm8k-1_5b-mixture-top2"
Q0_DIR = "runs/q0_qwen1_5b_laptop"
PROFILE = tg.PROFILES["qwen1_5b_laptop"]
NAME, REV = PROFILE["model_name"], PROFILE["revision"]

top2 = json.load(open(f"{Q0_DIR}/mixture_weights.json"))["top_k"]["2"]

MANIFEST = {
    "base_model": NAME,
    "base_revision": REV,
    "ensemble_type": "probability_space_weighted",
    "note": ("Per-adapter softmax over next-token logits, weighted sum by the "
             "weights below, then sample. Not a weight-space merge."),
    "lora": {"r": PROFILE["lora_rank"], "alpha": PROFILE["lora_alpha"],
             "dropout": PROFILE["lora_dropout"],
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]},
    "adapters": [
        {"file": "adapter_cycle02.pt", "weight": top2["weights"][0]},
        {"file": "adapter_cycle03.pt", "weight": top2["weights"][1]},
    ],
}

CARD = f"""---
base_model: {NAME}
library_name: peft
tags:
- qwen2
- gsm8k
- grpo
- q0
- ensemble
---

# q0-gsm8k-1.5b-mixture-top2

Top-2 weighted **mixture** from q0 cyclic-trajectory GRPO on GSM8K
(Qwen2.5-1.5B-Instruct). Two LoRA adapters combined in **probability space**
(not a weight merge):

| adapter | weight |
|---|---|
| `adapter_cycle02.pt` | {top2['weights'][0]:.4f} |
| `adapter_cycle03.pt` | {top2['weights'][1]:.4f} |

- **Base:** `{NAME}` (revision `{REV}`)
- **LoRA:** r={PROFILE['lora_rank']}, alpha={PROFILE['lora_alpha']}, NF4 4-bit QLoRA.
- **GSM8K validation (top-2):** pass@1 58.50%, pass@4 79.97%, pass@8 85.55% (256-example deterministic slice).
- **MATH-500 test (top-2):** pass@1 15.55%, pass@4 39.00% (500 problems, four samples each).
- **Selection note:** top-3 won GSM8K validation, but its third adapter had only 0.068% learned weight; top-2 is the practical published ensemble.

## Use

Load the base model twice with each adapter, take `softmax` over next-token
logits per adapter, combine as `w0*p0 + w1*p1`, and sample. See
`mixture_top2.json` for the exact weights and config. This is the repository's
1.5B (1B-scale) target, not a literal 1.0B model.
"""


def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    print(f"uploading adapters + manifest -> {REPO} ...")
    upload_file(path_or_fileobj=f"{Q0_DIR}/traj1_cycle02.pt",
                path_in_repo="adapter_cycle02.pt", repo_id=REPO,
                commit_message="publish selected adapter")
    upload_file(path_or_fileobj=f"{Q0_DIR}/traj1_cycle03.pt",
                path_in_repo="adapter_cycle03.pt", repo_id=REPO,
                commit_message="publish selected adapter")
    upload_file(path_or_fileobj=io.BytesIO(json.dumps(MANIFEST, indent=2).encode()),
                path_in_repo="mixture_top2.json", repo_id=REPO,
                commit_message="publish configuration manifest")
    upload_file(path_or_fileobj=io.BytesIO(CARD.encode()),
                path_in_repo="README.md", repo_id=REPO, commit_message="model card")
    print(f"done -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
