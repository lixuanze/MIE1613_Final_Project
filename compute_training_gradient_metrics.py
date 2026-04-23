"""
Quickly summarize training gradient-norm metrics from the five iteration-metrics CSVs.

Default inputs:
  - vanilla_pg_outputs/vanilla_pg_iteration_metrics.csv
  - pg_loo_outputs/pg_loo_iteration_metrics.csv
  - actor_critic_outputs/actor_critic_iteration_metrics.csv
  - belief_aware_ac_outputs/belief_aware_actor_critic_iteration_metrics.csv
  - hindsight_outputs/hindsight_training_iteration_metrics.csv

Outputs:
  - Prints a compact table to stdout
  - Writes evaluation/in_sample_training_gradient_norm_from_iterations.csv by default
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()


METHOD_FILES: Tuple[Tuple[str, str, str], ...] = (
    ("vanilla_pg", "vanilla_pg_outputs", "vanilla_pg_iteration_metrics.csv"),
    ("pg_loo", "pg_loo_outputs", "pg_loo_iteration_metrics.csv"),
    ("actor_critic", "actor_critic_outputs", "actor_critic_iteration_metrics.csv"),
    (
        "belief_aware_actor_critic",
        "belief_aware_ac_outputs",
        "belief_aware_actor_critic_iteration_metrics.csv",
    ),
    ("hindsight_training", "hindsight_outputs", "hindsight_training_iteration_metrics.csv"),
)


def summarize_one(method: str, csv_path: Path) -> Dict[str, object]:
    if not csv_path.is_file():
        return {
            "method": method,
            "file": str(csv_path),
            "status": "missing",
            "n_iterations_with_gradnorm": 0,
            "mean_gradient_norm_training": float("nan"),
            "std_gradient_norm_training": float("nan"),
            "min_gradient_norm_training": float("nan"),
            "max_gradient_norm_training": float("nan"),
        }

    df = pd.read_csv(csv_path)
    if "gradient_norm" not in df.columns:
        return {
            "method": method,
            "file": str(csv_path),
            "status": "no_gradient_norm_column",
            "n_iterations_with_gradnorm": 0,
            "mean_gradient_norm_training": float("nan"),
            "std_gradient_norm_training": float("nan"),
            "min_gradient_norm_training": float("nan"),
            "max_gradient_norm_training": float("nan"),
        }

    g = pd.to_numeric(df["gradient_norm"], errors="coerce").dropna()
    if g.empty:
        return {
            "method": method,
            "file": str(csv_path),
            "status": "empty_gradient_norm",
            "n_iterations_with_gradnorm": 0,
            "mean_gradient_norm_training": float("nan"),
            "std_gradient_norm_training": float("nan"),
            "min_gradient_norm_training": float("nan"),
            "max_gradient_norm_training": float("nan"),
        }

    return {
        "method": method,
        "file": str(csv_path),
        "status": "ok",
        "n_iterations_with_gradnorm": int(g.shape[0]),
        "mean_gradient_norm_training": float(g.mean()),
        "std_gradient_norm_training": float(g.std(ddof=0)),
        "min_gradient_norm_training": float(g.min()),
        "max_gradient_norm_training": float(g.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize training gradient_norm from 5 iteration-metrics CSVs.")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT,
        help="Project root that contains *_outputs folders.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluation" / "in_sample_training_gradient_norm_from_iterations.csv",
        help="Where to save the summary CSV.",
    )
    args, _unknown = p.parse_known_args()

    rows: List[Dict[str, object]] = []
    for method, out_dir, fname in METHOD_FILES:
        path = args.base_dir / out_dir / fname
        rows.append(summarize_one(method, path))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
