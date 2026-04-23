"""
Read a paths summary CSV (in-sample or OOS bootstrap) and compute paired
terminal-utility differences vs ``vanilla_pg`` with a normal 95% CI.

  Δ(path) = U_method(path) - U_vanilla(path)

CI: mean(Δ) ± 1.96 × SE(Δ), with SE = std(Δ, ddof=0) / sqrt(n).

Expected columns: ``method``, ``path_id``, ``terminal_utility`` (same as
``in_sample_paths_summary.csv`` / ``oos_bootstrap_paths_summary.csv``).

Examples
--------
  python compute_utility_delta_ci_from_summary.py
  python compute_utility_delta_ci_from_summary.py --preset oos_bootstrap
  python compute_utility_delta_ci_from_summary.py --summary path/to/oos_bootstrap_paths_summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

BASELINE = "vanilla_pg"
Z = 1.96
COMPARE: tuple[str, ...] = (
    "pg_loo",
    "actor_critic",
    "belief_aware_actor_critic",
    "hindsight_training",
)


def utility_by_path_id(df: pd.DataFrame, method: str) -> pd.Series:
    sub = df.loc[df["method"] == method]
    return sub.groupby("path_id", sort=False)["terminal_utility"].first().astype(float)


def compute_delta_ci(df: pd.DataFrame) -> pd.DataFrame:
    if BASELINE not in df["method"].values:
        raise ValueError(f"No rows with method={BASELINE!r}")

    u_base = utility_by_path_id(df, BASELINE)
    rows: List[dict] = []
    for m in COMPARE:
        if m not in df["method"].values:
            continue
        u_m = utility_by_path_id(df, m).reindex(u_base.index)
        d = (u_m - u_base).dropna()
        n = int(d.shape[0])
        if n < 2:
            continue
        mean_d = float(d.mean())
        se = float(d.std(ddof=0) / math.sqrt(n))
        lo, hi = mean_d - Z * se, mean_d + Z * se
        rows.append(
            {
                "method": m,
                "baseline": BASELINE,
                "n_paths_paired": n,
                "mean_delta_utility": mean_d,
                "se_delta_utility": se,
                "ci95_delta_utility_low": lo,
                "ci95_delta_utility_high": hi,
                "ci95_contains_zero": int(lo <= 0.0 <= hi),
            }
        )
    return pd.DataFrame(rows)


def default_summary_path(preset: str, oos_dir: Path) -> Path:
    if preset == "in_sample":
        return REPO_ROOT / "evaluation" / "in_sample_paths_summary.csv"
    return oos_dir / "oos_bootstrap_paths_summary.csv"


def default_output_path(summary_path: Path) -> Path:
    name = summary_path.name.lower()
    if "oos_bootstrap" in name:
        out_name = "oos_bootstrap_utility_delta_vs_vanilla_ci95_from_summary.csv"
    else:
        out_name = "in_sample_utility_delta_vs_vanilla_ci95_from_summary.csv"
    return summary_path.parent / out_name


def main() -> None:
    p = argparse.ArgumentParser(
        description="Paired utility Δ vs vanilla_pg from in-sample or OOS bootstrap paths summary CSV."
    )
    p.add_argument(
        "--preset",
        choices=("in_sample", "oos_bootstrap"),
        default="in_sample",
        help="Which default summary to use when --summary is omitted (default: in_sample).",
    )
    p.add_argument(
        "--oos-dir",
        type=Path,
        default=REPO_ROOT / "evaluation" / "oos_2020_2025",
        help="Directory containing oos_bootstrap_paths_summary.csv (--preset oos_bootstrap).",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Override summary CSV (must have method, path_id, terminal_utility).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: next to summary, name depends on in-sample vs OOS file).",
    )
    args, _unknown = p.parse_known_args()

    summary_path = args.summary if args.summary is not None else default_summary_path(args.preset, args.oos_dir)
    if not summary_path.is_file():
        raise SystemExit(f"File not found: {summary_path}")

    df = pd.read_csv(summary_path)
    out = compute_delta_ci(df)
    print(out.to_string(index=False))

    out_path = args.output if args.output is not None else default_output_path(summary_path)
    out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
