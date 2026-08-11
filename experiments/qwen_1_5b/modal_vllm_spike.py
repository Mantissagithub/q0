# step-0 de-risk: pin the one version-sensitive piece before the full run. loads
# qwen2.5-1.5b into both hf and an in-process v0 vllm engine, confirms the weight
# object is reachable, does one unfused+tied-embeddings state_dict -> load_weights
# sync, and proves the sync actually takes effect (output changes after we perturb
# hf weights and re-sync).
#   modal run modal_vllm_spike.py

import modal

app = modal.App("q0-vllm-spike")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.5", "transformers==4.51.3", "accelerate")
    .env({"VLLM_USE_V1": "0"})
    .add_local_file("core/train_grpo.py", "/root/train_grpo.py")
)


@app.function(image=image, gpu="A10G", timeout=1800)
def spike():
    import os
    os.chdir("/root")
    os.environ["VLLM_USE_V1"] = "0"

    import torch
    import train_grpo as tg

    prof = tg.PROFILES["qwen1_5b"]
    model_name, revision = prof["model_name"], prof["revision"]

    # 1) hf model + tokenizer
    tokenizer = tg.load_tokenizer(model_name, revision)
    hf_model = tg.load_base_model(model_name, revision, "cuda")
    sd = hf_model.state_dict()
    has_lm_head = any(k.endswith("lm_head.weight") for k in sd)
    print(f"state_dict keys: {len(sd)} | lm_head.weight present: {has_lm_head} (tied -> expect False)")

    # 2) in-process v0 vllm engine (small kv budget, this is only a spike)
    from vllm import LLM, SamplingParams
    engine = LLM(
        model=model_name, revision=revision, dtype="bfloat16",
        gpu_memory_utilization=0.35, max_model_len=512, max_num_seqs=8,
        enforce_eager=True,
    )

    # 3) reach the model object via the documented v0 attribute path, else fall back
    try:
        vmodel = engine.llm_engine.model_executor.driver_worker.model_runner.model
        path = "driver_worker.model_runner.model"
    except AttributeError as e:
        vmodel = None
        path = f"ATTR-FAIL: {e}"
    print(f"weight object path: {path}")

    prompt = tg.build_prompt("What is 6 times 7?")
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=32,
                        stop=[tg.ANSWER_STOP_STRING], include_stop_str_in_output=True)

    def gen():
        return engine.generate([prompt], sp)[0].outputs[0].text

    before = gen()
    print("BEFORE sync:", repr(before))

    # 4) sync the (unmodified) hf weights in -> must not raise for a tied model
    sync_ok, sync_err = True, None
    try:
        if vmodel is not None:
            vmodel.load_weights((n, p.detach()) for n, p in hf_model.state_dict().items())
        else:
            raise RuntimeError("no in-process model object")
    except Exception as e:
        sync_ok, sync_err = False, repr(e)
    print(f"identity sync ok: {sync_ok} err={sync_err}")

    # 5) prove the sync path is live: zero out the final norm, re-sync, expect the
    # greedy output to change (a no-op sync path would leave it identical).
    changed = None
    if sync_ok:
        with torch.no_grad():
            for n, p in hf_model.named_parameters():
                if n.endswith("model.norm.weight"):
                    p.mul_(0.0)
        vmodel.load_weights((n, p.detach()) for n, p in hf_model.state_dict().items())
        after = gen()
        changed = (after != before)
        print("AFTER perturb+sync:", repr(after))
        print(f"output changed after re-sync: {changed}")

    return {"has_lm_head": has_lm_head, "weight_path": path,
            "identity_sync_ok": sync_ok, "sync_err": sync_err,
            "sync_takes_effect": changed}


@app.local_entrypoint()
def main():
    r = spike.remote()
    print("\n===== vllm spike result =====")
    for k, v in r.items():
        print(f"{k}: {v}")
    ok = (r["identity_sync_ok"] and r["sync_takes_effect"] and not r["has_lm_head"])
    print("\nSPIKE VERDICT:", "PASS — weight-sync path pinned" if ok else "FAIL — inspect above")
