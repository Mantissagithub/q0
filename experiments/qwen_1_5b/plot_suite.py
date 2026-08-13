from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/q0_mopd/results"
OUT = ROOT / "experiments/qwen_1_5b/figures"

COLORS = {
    "baseline": "#0072B2",
    "q0": "#D55E00",
    "mopd_t1": "#009E73",
    "mopd_t2": "#CC79A7",
    "metric": "#333333",
}


def load_json(name):
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pct(value):
    return np.nan if value is None else 100 * float(value)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def label_bars(ax, values, percent=False, precision=1):
    for bar, value in zip(ax.patches, values):
        if not np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, 4, "unavailable", ha="center", va="bottom", rotation=90, fontsize=8)
            continue
        text = f"{value:.{precision}f}%" if percent else f"{value:.{precision}f}"
        ax.text(bar.get_x() + bar.get_width() / 2, value + (1 if percent else max(abs(value) * 0.02, 0.005)), text, ha="center", va="bottom", fontsize=8)


def label_line_points(ax, x, values, percent=False, every=None):
    for i, (x_value, value) in enumerate(zip(x, values)):
        if not np.isfinite(value) or (every and i % every):
            continue
        text = f"{value:.1f}%" if percent else f"{value:.3g}"
        ax.annotate(text, (x_value, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"))
    ax.spines[["top", "right"]].set_visible(False)


def result_row(dataset, split, count, method, metric, value, source, status="available"):
    return {
        "dataset": dataset,
        "split": split,
        "count": count,
        "method": method,
        "metric": metric,
        "value": "" if value is None or (isinstance(value, float) and math.isnan(value)) else value,
        "source": source,
        "status": status,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []

    q0_metrics = []
    with (RESULTS / "training_metrics.csv").open(newline="") as handle:
        q0_metrics = list(csv.DictReader(handle))
    q0 = [r for r in q0_metrics if r["run"] == "q0"]
    baseline = [r for r in q0_metrics if r["run"] == "baseline"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for rows, label, color in [(q0, "Q0", COLORS["q0"]), (baseline, "Baseline GRPO", COLORS["baseline"])]:
        x = [int(r["step"]) for r in rows]
        axes[0, 0].plot(x, [float(r["reward"]) for r in rows], lw=1.3, label=label, color=color)
        axes[0, 1].plot(x, [float(r["lr"]) for r in rows], lw=1.3, label=label, color=color)
        axes[1, 0].plot(x, [float(r["lr_multiplier"]) for r in rows], lw=1.1, label=label, color=color)
        axes[1, 1].plot(x, [float(r["weight_decay_multiplier"] or "nan") for r in rows], lw=1.1, label=label, color=color)
    style(axes[0, 0], "Q0 and baseline training reward", "step", "reward")
    style(axes[0, 1], "Q0 and baseline learning rate", "step", "learning rate")
    style(axes[1, 0], "Learning-rate multiplier", "step", "multiplier")
    style(axes[1, 1], "Weight-decay multiplier", "step", "multiplier")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_yscale("log")
    axes[0, 1].yaxis.set_major_formatter(ScalarFormatter())
    for ax in axes.flat:
        for line in ax.lines:
            label_line_points(ax, line.get_xdata(), line.get_ydata(), every=max(1, len(line.get_xdata()) // 4))
    save(fig, "01_q0_training_metrics")

    mix = load_json("../runs/q0_qwen1_5b_laptop/mixture_weights.json")
    if mix is None:
        mix = json.loads((ROOT / "runs/q0_qwen1_5b_laptop/mixture_weights.json").read_text())
    labels = [Path(p).stem.replace("_", " ") for p in mix["snapshot_paths"]]
    labels = [label.replace("checkpoint", "step ").replace("step step ", "step ") for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, mix["weights"], color=COLORS["q0"])
    style(ax, "Q0 learned mixture weights", ylabel="weight")
    label_bars(ax, mix["weights"], precision=3)
    ax.tick_params(axis="x", rotation=45)
    save(fig, "02_q0_mixture_weights")
    for k, top in mix.get("top_k", {}).items():
        summary.append(result_row("Q0 mixture", "training", "", f"q0_k{k}", "fit_loss", mix.get("fit_loss"), "runs/q0_qwen1_5b_laptop/mixture_weights.json"))

    mopd_t1 = load_jsonl(ROOT / "runs/mopd_qwen1_5b_laptop/training_metrics.jsonl")
    mopd_t2 = load_jsonl(ROOT / "runs/mopd_qwen1_5b_laptop_t2/training_metrics.jsonl")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for rows, label, color in [(mopd_t1, "MOPD t=1", COLORS["mopd_t1"]), (mopd_t2, "MOPD t=2", COLORS["mopd_t2"])]:
        x = [r["step"] for r in rows]
        for ax, key in zip(axes.flat, ["loss", "lr", "avg_response_length", "completion_tokens"]):
            ax.plot(x, [r[key] for r in rows], lw=1.1, label=label, color=color)
    style(axes[0, 0], "MOPD training loss", "step", "loss")
    style(axes[0, 1], "MOPD learning rate", "step", "learning rate")
    style(axes[1, 0], "Average response length", "step", "tokens")
    style(axes[1, 1], "Cumulative completion tokens", "step", "tokens")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_yscale("log")
    axes[0, 1].yaxis.set_major_formatter(ScalarFormatter())
    for ax in axes.flat:
        for line in ax.lines:
            label_line_points(ax, line.get_xdata(), line.get_ydata(), every=max(1, len(line.get_xdata()) // 4))
    save(fig, "03_mopd_training_metrics")
    for label, rows, source in [("mopd_t1", mopd_t1, "runs/mopd_qwen1_5b_laptop/training_metrics.jsonl"), ("mopd_t2", mopd_t2, "runs/mopd_qwen1_5b_laptop_t2/training_metrics.jsonl")]:
        for metric in ["loss", "lr", "avg_response_length", "completion_tokens"]:
            summary.append(result_row("training", "steps", len(rows), label, metric, rows[-1][metric], source))

    validations = [("MOPD t=1", ROOT / "runs/mopd_qwen1_5b_laptop/validation_results.json", COLORS["mopd_t1"]), ("MOPD t=2", ROOT / "runs/mopd_qwen1_5b_laptop_t2/validation_results.json", COLORS["mopd_t2"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, path, color in validations:
        data = json.loads(path.read_text())
        steps = [x["step"] for x in data["candidates"]]
        for key in ["pass_at_1", "pass_at_4", "pass_at_8"]:
            axes[0].plot(steps, [pct(x[key]) for x in data["candidates"]], marker="o", label=f"{label} {key.replace('_', '@')}", color=color, alpha={"pass_at_1": .55, "pass_at_4": .75, "pass_at_8": 1}[key])
        for key in ["parse_at_1", "parse_at_4", "parse_at_8"]:
            axes[1].plot(steps, [pct(x[key]) for x in data["candidates"]], marker="o", label=f"{label} {key.replace('_', '@')}", color=color, alpha={"parse_at_1": .55, "parse_at_4": .75, "parse_at_8": 1}[key])
        for key in ["pass_at_1", "pass_at_4", "pass_at_8"]:
            label_line_points(axes[0], steps, [pct(x[key]) for x in data["candidates"]], percent=True)
        for key in ["parse_at_1", "parse_at_4", "parse_at_8"]:
            label_line_points(axes[1], steps, [pct(x[key]) for x in data["candidates"]], percent=True)
        for candidate in data["candidates"]:
            for key in ["pass_at_1", "pass_at_4", "pass_at_8", "parse_at_1", "parse_at_4", "parse_at_8"]:
                summary.append(result_row("GSM8K", "train_val_slice", data["validation_count"], label, key, candidate[key], str(path.relative_to(ROOT))))
    style(axes[0], "GSM8K validation pass@k by checkpoint", "training step", "percent")
    style(axes[1], "GSM8K validation parse@k by checkpoint", "training step", "percent")
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[1].legend(fontsize=8, frameon=False, ncol=2)
    for ax in axes:
        ax.set_ylim(0, 105)
    save(fig, "04_mopd_validation_curves")

    passk = {}
    for path in sorted(RESULTS.glob("passk_*_gsm8kval.json")):
        data = load_json(path.name)
        if data:
            passk[data["name"]] = data
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ["baseline_grpo", "q0_k1", "q0_k2", "q0_k3"]
    display_names = {"baseline_grpo": "Baseline GRPO", "q0_k1": "q0 top-1", "q0_k2": "q0 top-2", "q0_k3": "q0 top-3"}
    x = np.arange(3)
    for name in names:
        data = passk.get(name)
        vals = [pct(data["result"].get(k)) if data else np.nan for k in ["pass_at_1", "pass_at_4", "pass_at_8"]]
        axes[0].plot(x, vals, marker="o", label=display_names[name], color=COLORS["baseline"] if name == "baseline_grpo" else COLORS["q0"])
        label_line_points(axes[0], x, vals, percent=True)
        parse = [pct(data["result"].get(k)) if data else np.nan for k in ["parse_at_1", "parse_at_4", "parse_at_8"]]
        axes[1].plot(x, parse, marker="o", label=display_names[name], color=COLORS["baseline"] if name == "baseline_grpo" else COLORS["q0"])
        label_line_points(axes[1], x, parse, percent=True)
        for metric, value in zip(["pass_at_1", "pass_at_4", "pass_at_8", "parse_at_1", "parse_at_4", "parse_at_8"], (vals + parse)):
            summary.append(result_row("GSM8K", "train_val_slice", 256, name, metric, None if np.isnan(value) else value / 100, f"experiments/q0_mopd/results/passk_{name}_gsm8kval.json" if data else "pending", "available" if data else "missing"))
    for ax, title, keys in [(axes[0], "GSM8K validation pass@k Q0 sweep", [1, 4, 8]), (axes[1], "GSM8K validation parse@k Q0 sweep", [1, 4, 8])]:
        style(ax, title, "k", "percent")
        ax.set_xticks(x, [f"pass@{k}" for k in keys])
        ax.set_ylim(0, 105)
        ax.legend(frameon=False)
    save(fig, "05_q0_gsm8k_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, color in [("mopd_t1", COLORS["mopd_t1"]), ("mopd_t2", COLORS["mopd_t2"])]:
        data = json.loads((ROOT / ("runs/mopd_qwen1_5b_laptop/validation_results.json" if name == "mopd_t1" else "runs/mopd_qwen1_5b_laptop_t2/validation_results.json")).read_text())
        best = max(data["candidates"], key=lambda x: x["pass_at_8"])
        axes[0].bar("MOPD t=" + name[-1], pct(best["pass_at_8"]), color=color)
        axes[1].bar("MOPD t=" + name[-1], pct(best["pass_at_4"]), color=color)
        summary.append(result_row("GSM8K", "train_val_slice", data["validation_count"], name, "best_pass_at_8", best["pass_at_8"], "validation_results.json"))
        summary.append(result_row("GSM8K", "train_val_slice", data["validation_count"], name, "best_pass_at_4", best["pass_at_4"], "validation_results.json"))
    style(axes[0], "MOPD t=1 vs t=2 best validation pass@8", ylabel="percent")
    style(axes[1], "MOPD t=1 vs t=2 best validation pass@4", ylabel="percent")
    axes[0].set_ylim(0, 100); axes[1].set_ylim(0, 100)
    label_bars(axes[0], [pct(max(json.loads(path.read_text())["candidates"], key=lambda x: x["pass_at_8"])["pass_at_8"]) for _, path, _ in validations], percent=True)
    label_bars(axes[1], [pct(max(json.loads(path.read_text())["candidates"], key=lambda x: x["pass_at_8"])["pass_at_4"]) for _, path, _ in validations], percent=True)
    save(fig, "06_mopd_t1_vs_t2")

    math500 = load_json("math500_results.json")
    mathk = load_json("passk_mopd_best_math500.json")
    math_baseline = load_json("passk_baseline_math500.json")
    math_q0 = load_json("passk_q0_best_math500.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    historical = math500["results"] if math500 else {}
    historical_names = {
        "baseline_grpo": "Baseline",
        "q0_best_single": "q0 single",
        "q0_learned_top4": "q0 mixture",
        "mopd": "MOPD",
    }
    for name, data in historical.items():
        display_name = historical_names.get(name, name.replace("_", " "))
        color = COLORS["baseline"] if "baseline" in name else COLORS["q0"] if "q0" in name else COLORS["mopd_t1"]
        axes[0, 0].bar(display_name, pct(data.get("accuracy")), color=color)
        axes[0, 1].bar(display_name, pct(data.get("parse_rate")), color=color)
        summary.append(result_row("MATH-500", "test", 500, name, "accuracy_single_pass_historical", data.get("accuracy"), "math500_results.json"))
        summary.append(result_row("MATH-500", "test", 500, name, "parse_rate_single_pass_historical", data.get("parse_rate"), "math500_results.json"))

    current = [
        (math_baseline, "Baseline GRPO", COLORS["baseline"], "passk_baseline_math500.json"),
        (math_q0, "q0 top-2", COLORS["q0"], "passk_q0_best_math500.json"),
        (mathk, "MOPD t=2", COLORS["mopd_t2"], "passk_mopd_best_math500.json"),
    ]
    current_names = [name for _, name, _, _ in current]
    current_x = np.arange(len(current))
    pass_values = [pct(data["result"].get("pass_at_4")) if data else np.nan for data, _, _, _ in current]
    parse_values = [pct(data["result"].get("parse_at_4")) if data else np.nan for data, _, _, _ in current]
    for ax, values in [(axes[1, 0], pass_values), (axes[1, 1], parse_values)]:
        heights = [value if np.isfinite(value) else 0 for value in values]
        colors = [color if np.isfinite(value) else "#D9D9D9" for value, (_, _, color, _) in zip(values, current)]
        bars = ax.bar(current_x, heights, color=colors, width=0.62)
        ax.set_xticks(current_x, current_names)
        for bar, value in zip(bars, values):
            x_center = bar.get_x() + bar.get_width() / 2
            if np.isfinite(value):
                ax.text(x_center, value + 1, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(x_center, 3, "Pending", ha="center", va="bottom", fontsize=8, color="#666666")
    for data, name, _, source in current:
        value = data["result"].get("pass_at_4") if data else None
        summary.append(result_row("MATH-500", "test", 500, name, "pass_at_4", value, source, "available" if data else "missing"))
    if mathk:
        summary.append(result_row("MATH-500", "test", 500, "mopd_t2", "pass_at_4", mathk["result"].get("pass_at_4"), "passk_mopd_best_math500.json"))

    panels = [
        (axes[0, 0], "Historical single-pass accuracy"),
        (axes[0, 1], "Historical single-pass parse rate"),
        (axes[1, 0], "Current Qwen pass@4"),
        (axes[1, 1], "Current Qwen parse@4"),
    ]
    for index, (ax, title) in enumerate(panels):
        style(ax, title, ylabel="percent")
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, 110)
        if index < 2:
            label_bars(ax, [bar.get_height() for bar in ax.patches], percent=True)
    fig.suptitle("MATH-500 comparison (test, n=500)", fontsize=16, weight="bold")
    save(fig, "07_math500_comparisons")

    gsm_full = load_json("passk_baseline_gsm8k.json")
    val_count = json.loads((ROOT / "runs/mopd_qwen1_5b_laptop/validation_results.json").read_text())["validation_count"]
    gsm_val = {"baseline_grpo": passk.get("baseline_grpo"), "q0": passk.get("q0_k3"), "mopd": None}
    gsm_full_value = gsm_full["result"].get("pass_at_1") if gsm_full else None
    final = {"baseline_grpo": {"MATH-500": math_baseline, "GSM8K test": gsm_full_value}, "q0": {"MATH-500": math_q0, "GSM8K test": None}, "mopd": {"MATH-500": mathk, "GSM8K test": None}}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    methods = ["Baseline GRPO", "q0 top-2", "MOPD t=2"]
    colors = [COLORS["baseline"], COLORS["q0"], COLORS["mopd_t2"]]

    gsm_val_methods = ["Baseline GRPO", "q0 top-3"]
    gsm_val_values = [pct(gsm_val["baseline_grpo"]["result"].get("pass_at_1")), pct(gsm_val["q0"]["result"].get("pass_at_1"))]
    axes[0, 0].bar(gsm_val_methods, gsm_val_values, color=colors[:2], width=0.62)
    style(axes[0, 0], "GSM8K validation — pass@1", ylabel="percent")
    axes[0, 0].text(0.01, 0.97, f"train validation slice, n={val_count}", transform=axes[0, 0].transAxes, va="top", fontsize=9, color="#555555")

    gsm_test_value = pct(gsm_full_value)
    axes[0, 1].bar(["Baseline GRPO"], [gsm_test_value], color=[COLORS["baseline"]], width=0.5)
    style(axes[0, 1], "GSM8K test — pass@1", ylabel="percent")
    axes[0, 1].text(0.01, 0.97, "full test set, n=1,319; q0/MOPD not evaluated", transform=axes[0, 1].transAxes, va="top", fontsize=9, color="#555555")

    math_pass1 = [pct(x["result"].get("pass_at_1")) for x in [math_baseline, math_q0, mathk]]
    axes[1, 0].bar(methods, math_pass1, color=colors, width=0.62)
    style(axes[1, 0], "MATH-500 — pass@1", ylabel="percent")
    axes[1, 0].text(0.01, 0.97, "test set, n=500", transform=axes[1, 0].transAxes, va="top", fontsize=9, color="#555555")

    math_pass4 = [pct(x["result"].get("pass_at_4")) for x in [math_baseline, math_q0, mathk]]
    axes[1, 1].bar(methods, math_pass4, color=colors, width=0.62)
    style(axes[1, 1], "MATH-500 — pass@4", ylabel="percent")
    axes[1, 1].text(0.01, 0.97, "test set, n=500", transform=axes[1, 1].transAxes, va="top", fontsize=9, color="#555555")

    for ax in axes.flat:
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=10)
        label_bars(ax, [bar.get_height() for bar in ax.patches], percent=True)
    fig.suptitle("Qwen 1.5B experiment summary", fontsize=16, weight="bold")
    save(fig, "08_final_experiment_comparison")
    for method, data in [("baseline_grpo", math_baseline), ("q0", math_q0), ("mopd", mathk)]:
        for metric in ["pass_at_1", "pass_at_4"]:
            value = data["result"].get(metric) if data else None
            summary.append(result_row("MATH-500", "test", 500, method, metric, value, "passk result", "available" if data else "missing"))
    summary.append(result_row("GSM8K", "test", gsm_full["dataset_meta"].get("count", 1319) if gsm_full else 1319, "baseline_grpo", "pass_at_1", gsm_full_value, "passk_baseline_gsm8k.json", "available" if gsm_full else "missing"))

    with (OUT / "metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "count", "method", "metric", "value", "source", "status"])
        writer.writeheader(); writer.writerows(summary)
    (OUT / "README.md").write_text("""# Qwen 1.5B q0/MOPD graph suite\n\nRun `python experiments/qwen_1_5b/plot_suite.py` from the repository root. The script reads structured logs/results on every run and writes PNG figures plus `metrics_summary.csv` here. Training and sweep figures are independent of pending evaluations; comparison figures update automatically when pending pass-k JSON files appear.\n\n## Caveats\n\n- GSM8K checkpoint validation counts are read from each validation file; current files use their recorded `train_val_slice` count. GSM8K sweep files use their recorded count (currently 256). GSM8K full-test baseline, when available, is plotted in the summary only against itself because q0/MOPD full-test values are not available.\n- MATH-500 is the `test` split, count 500. Existing `math500_results.json` is a historical single-pass run with SmolLM2-135M metadata, so it is plotted separately from current Qwen pass@k results and must not be treated as a like-for-like Qwen comparison.\n- Missing pending values are left blank in the CSV and labeled `missing` in the figures.\n""")


if __name__ == "__main__":
    main()
""