# q0 on gsm8k

a small, exploratory adaptation of [q0: Primitives for Hyper-Epoch Pretraining](https://arxiv.org/abs/2606.03938) to reasoning with GRPO. the base model is `HuggingFaceTB/SmolLM2-135M-Instruct`, pinned to revision `12fd25f`.

🤗 [baseline checkpoints](https://huggingface.co/Pradheep1647/q0-gsm8k-135m-baseline) · 🤗 [q0 snapshots and weights](https://huggingface.co/Pradheep1647/q0-gsm8k-135m) · 🤗 [distilled checkpoint](https://huggingface.co/Pradheep1647/q0-gsm8k-135m-mopd)

![q0 pipeline](q0.png)

## what this run does

the diagram reads left to right:

1. train two independent GRPO trajectories for four cycles and save every cycle boundary
2. use plain GRPO for cycles 1 and 2, then GRPO plus forward KL to the previous snapshot for cycles 3 and 4
3. freeze all eight snapshots and fit softmax mixture weights with ordinary NLL on 512 held-out gold continuations
4. keep the learned top-K snapshots, renormalize their weights, and mix their next-token probabilities at inference

the baseline gets four passes over 2,048 examples. q0 runs four cycles for each of two trajectories, so it uses twice the total training prompts across the snapshot pool. this is a small systems check, not a compute-matched benchmark.

## result

all numbers below use the complete 1,319-example GSM8K test split.

| system | correct | exact match |
| --- | ---: | ---: |
| untouched base | 10 | 0.758% |
| best baseline, pass 4 | 12 | 0.910% |
| best q0 snapshot | 10 | 0.758% |
| learned ensemble, K=8 | 11 | 0.834% |
| uniform ensemble, K=8 | 12 | 0.910% |
| distilled student, selected at step 8 | 12 | 0.910% |

![checkpoint accuracy comparison](checkpoint_accuracy_comparison.png)

## how i read this

- training reward improved more clearly than exact match. part of that is formatting: parse rate rose from 38.97% for the untouched base to 48.45% for baseline pass 4, while only two more questions became correct.
- the learned prior did not rank snapshots by downstream accuracy. it gave 30.65% weight to trajectory 2 cycle 2, which was the weakest individual snapshot at 7/1,319. the prior was trained on teacher-forced token likelihood, not generated-answer exact match.
- learned weighting never beat uniform weighting here. uniform K=8 tied the best baseline at 12/1,319, while learned K=8 reached 11/1,319.
- the whole run sits close to the floor. with every system solving only 7 to 12 questions, a two-answer gap is too small to treat as evidence of an algorithmic win.

so the useful result is negative: this configuration validates the q0-style snapshot, prior-fitting, top-K, and probability-mixture pipeline, but it does not show a GSM8K accuracy gain. a stronger base model, repeated seeds, and a prior objective closer to generated-answer correctness are the next experiments i would trust.

![training reward comparison](training_reward_comparison.png)

![ensemble comparison](ensemble_comparison.png)

![learned mixture weights](mixture_weights_comparison.png)

## one-model follow-up

`train_mopd.py` adds one final consolidation stage inspired by [Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2606.30406). a fresh student starts from the shared base and generates its own GSM8K response. all four frozen top-ranked q0 snapshots score those exact student prefixes, then the student minimizes their average forward KL

$$
\mathcal{L} = \frac{1}{4}\sum_{i=1}^{4} KL\left(p_{T_i} \parallel p_\theta\right).
$$

`mixture_weights.json` is used only to choose the top-4 snapshot names. their numeric weights do not enter the loss; each teacher contributes equally. this is not an exact MOPD reproduction because the paper routes domains and uses reverse KL. this small variant keeps the shared-origin teachers and student-generated states, but uses direct forward-KL distillation.

the student ran one pass over 2,048 prompts, with candidates saved at steps 8, 16, 24, and 32. selection used a fresh 512-question slice from the unused GSM8K train remainder. each question got eight sampled answers, then candidates were ranked by pass@8, pass@4, and pass@1. step 8 won with 5.47% validation pass@8.

![distillation training curves](mopd_training_curves.png)

the forward KL stayed in a narrow 0.0040 to 0.0046 range. it did not steadily fall, which already suggests the extra steps were not producing a clear distillation gain.

![distilled checkpoint validation](mopd_validation_comparison.png)

| official test system | greedy | pass@1 | pass@4 | pass@8 |
| --- | ---: | ---: | ---: | ---: |
| best baseline, pass 4 | 0.91% | 0.78% | 2.85% | 5.16% |
| best q0 checkpoint | 0.76% | 0.54% | 2.03% | 3.79% |
| distilled student, step 8 | 0.91% | 0.51% | 1.94% | 3.64% |

![final distilled checkpoint comparison](mopd_final_comparison.png)

the single student matched the best baseline and uniform K=8 ensemble under greedy decoding, with 12/1,319 correct. sampled pass@k was weaker than both comparison checkpoints. so this run compressed the four-teacher training signal into one model, but did not improve reasoning accuracy. the validation winner also did not transfer into a pass@k win on the official test split.

## math-500 transfer

the same selected checkpoints were evaluated on all 500 MATH-500 problems with greedy decoding, a 256-token answer limit, and `math-verify` symbolic equivalence. nothing was retrained on MATH-500.

| system | correct | accuracy | parse rate | average response | generation time | peak cuda allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline, pass 4 | 5/500 | 1.0% | 83.0% | 99.7 tokens | 4m 32s | 0.84 GiB |
| q0 best single | 5/500 | 1.0% | 85.6% | 115.6 tokens | 4m 24s | 0.84 GiB |
| q0 learned top-4 | 6/500 | 1.2% | 82.6% | 111.5 tokens | 17m 02s | 8.36 GiB |
| distilled student | 6/500 | 1.2% | 86.6% | 118.6 tokens | 4m 27s | 0.84 GiB |

![math-500 comparison](math500_comparison.png)

this is still floor-level performance. the learned top-4 adds one answer over baseline, but a 1/500 gap is not evidence of a reliable generalization gain. it solves the baseline's five correct problems plus one more. the distilled student keeps the same total in one checkpoint and gets the best parse rate, while running about as quickly as a single model. it is not just copying the ensemble's six wins: only four solved problems overlap.

the failures are broad. every system scores zero on counting and probability and number theory. baseline, q0 single, and q0 top-4 solve nothing at level 4, while only q0 single solves one level-5 problem. the distilled student solves one problem at each of levels 4 and 5, but the total remains too small to read as stronger hard-math reasoning.

so the transfer result agrees with the GSM8K result: q0-style ensembling and distillation work mechanically, and distillation removes the four-model inference cost, but this 135M base is too close to the reasoning floor to separate the methods cleanly. `math500_results.json` keeps every aggregate and breakdown, while `math500_predictions.jsonl` keeps all 2,000 scored generations for inspection.

## files

- `train_grpo.py`: four-pass GRPO baseline
- `train_q0_grpo.py`: cyclic trajectories, KL distillation, snapshots, and learned prior
- `train_mopd.py`: one-pass on-policy distillation into a single checkpoint
- `eval.py`: individual, distilled, pass@k, and probability-space ensemble evaluation
- `eval_math500.py`: deterministic MATH-500 transfer evaluation
- `run_all.sh`: baseline, q0, distillation, then evaluation
- `eval_results.json` and `mixture_weights.json`: complete evaluation and mixture artifacts
- `mopd_training_metrics.jsonl`, `mopd_validation_results.json`, and `mopd_*.log`: distilled run metrics and logs
- `math500_results.json`, `math500_predictions.jsonl`, and `math500_run.log`: complete MATH-500 scores, raw generations, and run log
