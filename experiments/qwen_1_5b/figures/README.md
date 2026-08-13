# Qwen 1.5B q0/MOPD graph suite

Run `python experiments/qwen_1_5b/plot_suite.py` from the repository root. The script reads structured logs/results on every run and writes PNG figures plus `metrics_summary.csv` here. Training and sweep figures are independent of pending evaluations; comparison figures update automatically when pending pass-k JSON files appear.

## Caveats

- GSM8K checkpoint validation counts are read from each validation file; current files use their recorded `train_val_slice` count. GSM8K sweep files use their recorded count (currently 256). GSM8K full-test baseline, when available, is plotted in the summary only against itself because q0/MOPD full-test values are not available.
- MATH-500 is the `test` split, count 500, current Qwen 1.5B pass@4/parse@4 only. `math500_results.json` (historical SmolLM2-135M single-pass run) is not plotted here - not comparable to the current Qwen pass@k results.
- Missing pending values are left blank in the CSV and labeled `missing` in the figures.
