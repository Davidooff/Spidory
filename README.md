# Spidory

A small looped transformer (26M parameters) built on NVIDIA's Dory recipe: an input block, a shared
centre block run `k` times (`u = N(s + x)`, `s = N(F(u))`), and an output block, all with nGPT
layers. Spidory adds four small experts to the centre block and tests whether the router should pick
an expert once or again on every loop. It also tests whether looping beats a non-looped model of the
same size.

**Paper draft:** [`paper/expert_reselection.pdf`](paper/expert_reselection.pdf)
(*Choose Again Each Loop: Expert Re-selection and the One-Loop Tax in a Small Looped Transformer*).

**Weights:** the paper's 30 models (10 variants × 3 seeds) are on Hugging Face at
[Davidow/spidory](https://huggingface.co/Davidow/spidory):
```python
from spidory.hub import load
model, tok = load("token_signal_s0")
```

## Main results (3 seeds, 1.2B tokens per model)
- With per-token routing, choosing the expert again before each loop gives 71.3% multi-step
  arithmetic accuracy, against 59.7% when the expert is fixed after the first loop. Re-selection
  wins on every seed, at equal test loss (1.711).
- Looping does not pay at this scale. The same model trained only at one loop matches the looped
  model at three loops (loss 1.710 vs 1.708) at a third of the compute. Extra loops mostly recover
  the one-loop tax.
- Training on the mean loss over `k = 1..4` shrinks the tax about 2.5-fold.

## Layout
| Path | What it is |
|---|---|
| `spidory/model.py`, `ngpt.py` | Looped nGPT, experts, router, run index, between-loop block |
| `spidory/train.py`, `rounds.py`, `joint.py` | Training stages, rounds, learned loop controller |
| `spidory/data*.py`, `grow_v4.py`, `clean_v5.py`, `mixed_data.py` | Data building and leak removal |
| `spidory/evaluate.py` | Test loss, arithmetic and other exams |
| `spidory/hub.py` | Export to safetensors and load the released weights |
| `spidory/ui/` | Local playground: `python -m spidory.ui.server --root .`, then http://127.0.0.1:8765 |
| `scripts/` | Run and rental scripts (`run_mixed.sh`, `run_jobs.sh`, `build_mixed_then_run.sh`) |
| `tests/` | 319 tests |
| `docs/EXPERIMENT_LOG.md` | Full record: every model, result, problem and cost |
| `TRAINING_VERSIONS.md` | One summary per model version |

## Quick start
```bash
pip install -e '.[dev]'
pytest
# On a GPU machine with the data built (see the log): seed 0, two models
scripts/run_mixed.sh 0 plain,token_signal
```
`scripts/run_mixed.sh` lists every model variant. Data and runs are not in the repository.

## License
Apache-2.0 ([`LICENSE`](LICENSE)), for the code and the released weights.

## Branches
`main` holds the latest code (three seeds, models without loops). `arch-between`,
`run-signal-per-token`, `v4-seed1` and `v4-seeds` hold the earlier cycles, and `playground-*` hold the
playground versions.
