import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# run from this script's folder so results/, logs/ and figures/ resolve no matter where i call it from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BLUE, GREEN, ORANGE, GREY = "#2b6cb0", "#2f855a", "#dd6b20", "#a0aec0"


def rolling(xs, w=8):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        seg = xs[lo:i + 1]
        out.append(sum(seg) / len(seg))
    return out


with open("results/results_qwen.json") as f:
    qwen = json.load(f)

evals = qwen["eval"]
model_names = ["base"] + sorted(
    [name for name in evals if name.startswith("pass_")]
)
labels = ["base"] + [name.replace("pass_", "p") for name in model_names[1:]]

os.makedirs("figures", exist_ok=True)

# pull the reward per step out of the log
steps = []
rewards = []
passes = []
pattern = re.compile(r"^pass (\d+)/\d+ step (\d+)/\d+ .* reward ([+-]?(?:\d+\.?\d*|\.\d+))")
with open("logs/modal_qwen_full.timeout_step208.log") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            passes.append(int(match.group(1)))
            steps.append(int(match.group(2)))
            rewards.append(float(match.group(3)))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, rewards, color=BLUE, lw=1, alpha=0.4)
ax.plot(steps, rolling(rewards), color=BLUE, lw=2, label="math_verify reward (rolling-8)")
ax.set_xlabel("training step (logging starts at step 49)")
ax.set_ylabel("mean batch reward")
ax.set_title("math_verify reward for the Qwen1.5B GRPO run")
ax.text(0.02, 0.04, "logging starts at step 49", transform=ax.transAxes, color=GREY, fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig("figures/reward_curve.png", dpi=300)
plt.close(fig)

# make the GSM8K bars and show their change from base
base_gsm = evals["base"]["exact_match"] * 100
vals = [evals[name]["exact_match"] * 100 for name in model_names]
colors = [GREY] + [GREEN] * (len(model_names) - 1)
x = list(range(len(model_names)))
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(x, vals, color=colors, width=0.7)
for i, bar in enumerate(bars):
    delta = vals[i] - base_gsm
    text = f"{vals[i]:.2f}%" if i == 0 else f"{vals[i]:.2f}%\n(+{delta:.2f})"
    ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.axhline(base_gsm, ls="--", color=GREY, lw=1, label="base baseline")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("exact match (%)")
ax.set_title(f"GSM8K exact-match (n={qwen['test_n']})")
ax.legend(handles=[Patch(color=GREEN, label="GRPO pass"),
                   Patch(color=GREY, label="base")], fontsize=8)
fig.tight_layout()
fig.savefig("figures/gsm8k_bars.png", dpi=300)
plt.close(fig)

math500 = None
if os.path.exists("results/results_math500_qwen.json"):
    with open("results/results_math500_qwen.json") as f:
        math500 = json.load(f)
    math_evals = math500["eval"]
    math_metric = "accuracy" if "accuracy" in next(iter(math_evals.values())) else "exact_match"
    math_names = [name for name in model_names if name in math_evals]
    math_labels = ["base"] + [name.replace("pass_", "p") for name in math_names[1:]]
    math_base = math_evals["base"][math_metric] * 100
    math_vals = [math_evals[name][math_metric] * 100 for name in math_names]
    math_x = list(range(len(math_names)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    math_bars = ax.bar(math_x, math_vals, color=[GREY] + [ORANGE] * (len(math_names) - 1), width=0.7)
    for i, bar in enumerate(math_bars):
        delta = math_vals[i] - math_base
        text = f"{math_vals[i]:.2f}%" if i == 0 else f"{math_vals[i]:.2f}%\n({delta:+.2f})"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(math_base, ls="--", color=GREY, lw=1, label="base baseline")
    ax.set_xticks(math_x)
    ax.set_xticklabels(math_labels)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"MATH-500 transfer accuracy (n={math500['n']})")
    ax.legend(handles=[Patch(color=ORANGE, label="GRPO pass"),
                       Patch(color=GREY, label="base")], fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/math500_bars.png", dpi=300)
    plt.close(fig)
else:
    print("math-500 results not ready, skipping that figure")

# compare in-domain and transfer scores in one grouped bar chart
metrics = [("GSM8K", evals, "exact_match", GREEN)]
if math500 is not None:
    metrics.append(("MATH-500", math500["eval"], math_metric, ORANGE))
fig, ax = plt.subplots(figsize=(8, 4.5))
width = 0.35 if len(metrics) == 2 else 0.55
for i, (metric_name, source, metric, color) in enumerate(metrics):
    offsets = [(j - (len(metrics) - 1) / 2) * width for j in x]
    metric_names = [name for name in model_names if name in source]
    metric_vals = [source[name][metric] * 100 for name in metric_names]
    metric_x = x[:len(metric_vals)]
    ax.bar([pos + offsets[j] for j, pos in enumerate(metric_x)], metric_vals,
           width=width, color=color, label=metric_name)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("score (%)")
summary_suffix = "" if math500 is not None else " (GSM8K only; MATH-500 not ready)"
ax.set_title(f"Qwen1.5B GRPO results: in-domain vs transfer{summary_suffix}")
ax.legend()
fig.tight_layout()
fig.savefig("figures/summary.png", dpi=300)
plt.close(fig)

print("saved Qwen figures to figures/")
