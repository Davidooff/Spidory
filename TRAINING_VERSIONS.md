# Spidory training versions

Short summary per version. The full record, with every model, test and problem, is in
`docs/EXPERIMENT_LOG.md`.

Each version below says how it trained and what it showed. All held-out numbers come from
splits that training never saw.

## v0: first build, never trained on real data
- About 4M parameters: 2 input, 3 core and 2 output layers.
- Run neuron trained with the model, with a two-sided penalty on average runs per expert.
- Replaced before any real run, when training moved to the Dory recipe.

## v1: Dory recipe, 4M parameters, CPU smoke tests only
- One run first, then loops sampled from {1, 2, 3, 4} with TBPTL depth 1. No run neuron.
- Per-expert run targets in rounds, each checked against a control branch.

## v1.1: 4M parameters, RTX 4090, early check failed
- Data: TinyStories, DeepMind arithmetic, CodeSearchNet Python, SQuAD.
- Dory's loop formula: s0 = x, u = N(s + x), s = N(F(u)), plus run-count and expert
  signals into the output block.
- Plain model only. Loops gained 0.21% held-out loss, below the 1% bar.
- Maths accuracy was 2.7% at every run count: the task was too hard.

## v2: 24M parameters, easier maths, early check failed
- 4 input, 10 looped and 4 output layers, width 320, 4 experts, two-layer router.
- Generated maths in 5 levels.
- Plain model only. Best run count beat one run by -0.54% loss and +1.2 maths points.
- Story and code did best at 2 runs; maths and QA at 1. QA overfit badly
  (held-out loss rose to 5.3).

## v2-100M: 102M experts model, one seed, finished
- Same design at width 672, with story and code training data grown about 5 and 4 times.
- Rounds never formed a group, so per-expert targets stayed at 2.5.
- Run neuron trained last, on a frozen model, to copy the smallest run count within 0.5% of
  the best. This was not the user's design.
- Test loss: 1 run 2.862, 2 runs 2.871, run neuron 2.857 at 1.58 average runs.
  Maths accuracy 99.2%.
- The neuron chose about 2 runs for story and code, 1 for maths and 1.3 for QA.

## v3 attempt 1: early check failed (24M, data v2)
- Plain only. Two runs beat one on every dataset, but no dataset gained 1%:
  story 0.77%, code 0.69%, QA 0.42%, maths 0.09%, maths accuracy +1.8 points.
- Data v2 fixed QA overfitting (held-out loss 3.0, from 5.3), and maths reached about 67%.

## v3: the user's run-neuron design, on nGPT
- Architecture moved to Dory's: nGPT blocks (unit-length states and weights, learned step
  sizes, no LayerNorm, no weight decay), SwiGLU, RoPE, plus Dory's attention-output gates and
  extra normalizations.
- Data v3: 856M training tokens (story 406M, maths 120M, code 159M, QA 171M).
- Learning rate 6e-3 with 1,000 warm-up steps, from a laptop comparison where nGPT at 6e-3 beat
  the old blocks' best by 8% loss after 1,200 steps.
- Data v2: maths levels 2 and 5-9 (long chains, variable tracking, nested brackets) and QA
  grown to 171M tokens with HotpotQA, Natural Questions and Simple Wikipedia.
- Run neuron trained jointly with the model in stage 3, with a two-sided penalty on each
  expert's average, and group targets raised by activation while the overall average stays 2.5.
- Three ways to train the neuron: sampled run count, the average of all 4 run counts (the
  user's note), and mixing run states (the original idea). A single-layer neuron is compared
  with a small network on the sampled way. All against Plain and Constant, one seed.
- **Result (2 seeds, test):** loops give about 2.2% loss at double compute, almost all from the
  second run. Experts specialise without labels but Constant is 0.5% worse than Plain. The neuron
  beat the best fixed run count in 4 of 14 tries, never on both seeds. Per-loop routing had the
  best maths (87.4% / 85.8%).

## Between block (branch `arch-between`), seed 0
- 6 nGPT layers between centre runs: shared, per gap, or per gap with a learned input merge
  (with or without boundary normalization). 34.9-52.0M parameters.
- Per gap was best (1.906 at 3 runs) but still 0.6% worse than Plain (1.894). Maths 87.4%, the
  same as per-loop routing without the block. The input merge hurt.

## v4: run signal and per-token routing (branch `run-signal-per-token`), seed 0
- **Run signal (the user's idea):** the centre gets `u = N(a_k·s + b_k·x + e_k)` per run number,
  starting as plain Dory. One version drops the normalization.
- **Per-token routing:** each token gets its own expert per run and its own run count. A token
  pays for as many runs as any later token.
- **Data v6:** 7 sets, a seventh of each batch: story, maths, QA, mixed topics (739 templates
  from a cheap-model subagent), maths explained in words, compose a problem, and Orca-Math with
  GSM8K train (GSM8K test as the exam).
- **Result:** run signal helped both models it was added to (maths 61% → 74% and 66% → 82%).
  Per-token + run signal was best (loss 1.696, maths 82.4%). Per-token beat per-loop routing by
  5.4%, but its experts don't follow topics. The neuron didn't beat the best fixed count, and the
  between block was worse again. GSM8K stayed at 0-4%.

