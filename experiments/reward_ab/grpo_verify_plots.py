import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# run from this script's folder so results/ and figures/ resolve no matter where i call it from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("results/verify_results.json") as f:
    data = json.load(f)

evals = data["eval"]
curve = data["reward_curve"]
test_n = data["test_n"]
models = ["base", "pass_01", "pass_02", "pass_03", "pass_04"]

print(f"{'model':10} {'exact_match%':>13} {'parse_rate%':>12}")
for m in models:
    e = evals[m]
    print(f"{m:10} {e['exact_match']*100:13.2f} {e['parse_rate']*100:12.2f}")

# Figure 1: reward curve
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(curve) + 1), curve, color="#2b6cb0", lw=1.5)
ax.axhline(0.1, ls="--", color="#a0aec0", lw=1, label="format-bonus only (0.1)")
ax.set_xlabel("training step")
ax.set_ylabel("mean batch reward")
ax.set_title("New-reward GRPO: training reward per step")
ax.legend()
fig.tight_layout()
fig.savefig("figures/grpo_reward_curve.png", dpi=150)

# Figure 2: accuracy vs parse-rate bars
em = [evals[m]["exact_match"] * 100 for m in models]
pr = [evals[m]["parse_rate"] * 100 for m in models]
x = range(len(models))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar([i - w / 2 for i in x], em, w, label="exact_match", color="#2f855a")
b2 = ax.bar([i + w / 2 for i in x], pr, w, label="parse_rate", color="#dd6b20")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylabel("percent")
ax.set_title(f"GSM8K test (n={test_n}): exact-match vs parse-rate")
ax.legend()
fig.tight_layout()
fig.savefig("figures/grpo_accuracy_comparison.png", dpi=150)
print("saved grpo_reward_curve.png and grpo_accuracy_comparison.png")
