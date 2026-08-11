import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# run from this script's folder so results/ and figures/ resolve no matter where i call it from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("results/ab_results.json") as f:
    ab = json.load(f)
with open("results/verify_results.json") as f:
    new = json.load(f)

evals = ab["eval"]
test_n = ab["test_n"]
old_curve = ab["old_reward_curve"]
new_curve = new["reward_curve"]

BLUE, GREEN, ORANGE, GREY = "#2b6cb0", "#2f855a", "#dd6b20", "#a0aec0"


def rolling(xs, w=8):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        seg = xs[lo:i + 1]
        out.append(sum(seg) / len(seg))
    return out


# Figure 1: reward curves overlay
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(new_curve) + 1), new_curve, color=BLUE, lw=1, alpha=0.35)
ax.plot(range(1, len(old_curve) + 1), old_curve, color=ORANGE, lw=1, alpha=0.35)
ax.plot(range(1, len(new_curve) + 1), rolling(new_curve), color=BLUE, lw=2,
        label="math_verify reward (rolling-8)")
ax.plot(range(1, len(old_curve) + 1), rolling(old_curve), color=ORANGE, lw=2,
        label="NUMERIC_RE reward (rolling-8)")
ax.set_xlabel("training step")
ax.set_ylabel("mean batch reward")
ax.set_title("GRPO training reward: math_verify vs NUMERIC_RE")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ab_reward_curves.png", dpi=150)

# Bar layout shared by figures 2 and 3
order = ["base"] + [f"new_pass_{p:02d}" for p in range(1, 5)] + \
        [f"old_pass_{p:02d}" for p in range(1, 5)]
labels = ["base"] + [f"new p{p}" for p in range(1, 5)] + [f"old p{p}" for p in range(1, 5)]
colors = [GREY] + [GREEN] * 4 + [ORANGE] * 4
x = range(len(order))


def bar_fig(metric, title, fname, pct=True):
    vals = [evals[m][metric] * (100 if pct else 1) for m in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(x, vals, color=colors, width=0.7)
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(evals["base"][metric] * (100 if pct else 1), ls="--", color=GREY, lw=1,
               label="base")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("percent")
    ax.set_title(title)
    # legend for arm colors
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="new (math_verify)"),
                       Patch(color=ORANGE, label="old (NUMERIC_RE)"),
                       Patch(color=GREY, label="base")], fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


bar_fig("exact_match", f"GSM8K exact-match (n={test_n}), same math_verify scorer",
        "figures/ab_accuracy.png")
bar_fig("parse_rate", f"GSM8K parse-rate (n={test_n}): format learning",
        "figures/ab_parse_rate.png")

print("saved ab_reward_curves.png, ab_accuracy.png, ab_parse_rate.png")
