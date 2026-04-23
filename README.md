# MIE1613 Final Project

Portfolio reinforcement-learning experiments in PyTorch: policy-gradient variants, actor–critic, belief-aware actor–critic, hindsight-style training, and leave-one-out policy gradients, with in-sample and out-of-sample evaluation artifacts.

## Requirements

- Python 3.9+ recommended  
- Dependencies are listed in `requirements.txt` (NumPy, Pandas, Matplotlib, PyTorch, yfinance, SciPy).

## Setup

```bash
cd /path/to/MIE1613_final_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Repository layout

| Path | Role |
|------|------|
| `vanilla_pg.py` | Vanilla policy gradient baseline |
| `actor_critic.py` | Actor–critic |
| `belief_aware_actor_critic.py` | Belief-aware actor–critic |
| `hindsight_training.py` | Hindsight-style training |
| `loo_pg.py` | Leave-one-out policy gradient |
| `return_distributions.py` | Return distribution summaries / plots |
| `in_sample_bootstrap_eval.py` | In-sample bootstrap evaluation |
| `out_of_sample_evaluation.py` | Out-of-sample evaluation |
| `analyze_in_sample_paths.py` | In-sample path analysis |
| `compute_training_gradient_metrics.py` | Training / gradient metrics |
| `compute_utility_delta_ci_from_summary.py` | Utility delta CIs from summaries |
| `*_outputs/` | Run outputs (configs, metrics, checkpoints, curves) |
| `evaluation/` | Aggregated tables, figures, and OOS summaries |

## Running training scripts

Each top-level `*.py` training script is intended to be run as a module entry point, for example:

```bash
python vanilla_pg.py
python actor_critic.py
python belief_aware_actor_critic.py
python hindsight_training.py
python loo_pg.py
```

Use `python <script>.py --help` if the script exposes CLI options.

## Large local-only files

`evaluation/oos_2020_2025/oos_bootstrap_paths_long.csv` is **not** tracked in Git (GitHub file-size limits). Regenerate it with your evaluation pipeline if you need it on a new clone; summaries derived from it remain in the smaller CSVs under `evaluation/oos_2020_2025/`.

## License / course use

Course project for MIE1613; adapt reuse to your institution’s academic integrity rules.
