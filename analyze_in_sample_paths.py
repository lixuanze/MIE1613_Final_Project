"""
Post-process in-sample evaluation outputs.

Inputs (default under evaluation/):
  - in_sample_paths_long.csv
  - in_sample_paths_summary.csv
  - in_sample_aggregate.csv (optional, only for metadata checks)

Outputs:
  - in_sample_turnover_summary.csv
  - in_sample_concentration_by_time.csv
  - in_sample_asset_concentration_time.csv
  - in_sample_asset_concentration_<method>.png (5 files, one per strategy)
  - in_sample_best_median_worst_path_ids.csv
  - in_sample_best_median_worst_1x3.png

Notes
-----
- Turnover per step is computed as:
    sum_i |trade_i,t| / wealth_{t-1}
  for t >= 1 within each (method, path_id) trajectory.
- Exposure concentration is computed from risky holdings x_i at each time t via
  HHI on absolute risky exposures:
    w_i,t = |x_i,t| / sum_j |x_j,t|
    HHI_t = sum_i w_i,t^2
- Asset-level concentration for plotting is computed as the average of w_i,t
  across simulated paths at each time t (stacked-area per method over t=1..horizon).
- "Best / median / worst" paths are selected separately for each method based
  on terminal wealth in in_sample_paths_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_trade_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("trade_")], key=lambda s: int(s.split("_")[1]))


def _pick_x_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("x_")], key=lambda s: int(s.split("_")[1]))


def _choose_shared_path_ids(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select one common best/median/worst path_id across methods.

    We aggregate each path_id across the five methods using average terminal wealth,
    then choose:
      - best: largest aggregated wealth
      - worst: smallest aggregated wealth
      - median: closest to aggregated median
    """
    path_agg = (
        summary_df.groupby("path_id", as_index=False)["terminal_wealth"]
        .mean()
        .rename(columns={"terminal_wealth": "avg_terminal_wealth_across_methods"})
        .sort_values("avg_terminal_wealth_across_methods", ascending=True)
        .reset_index(drop=True)
    )
    worst = int(path_agg.iloc[0]["path_id"])
    best = int(path_agg.iloc[-1]["path_id"])
    med_val = float(path_agg["avg_terminal_wealth_across_methods"].median())
    med_idx = (path_agg["avg_terminal_wealth_across_methods"] - med_val).abs().idxmin()
    median = int(path_agg.loc[med_idx, "path_id"])

    return pd.DataFrame(
        [
            {"category": "worst", "path_id": worst},
            {"category": "median", "path_id": median},
            {"category": "best", "path_id": best},
        ]
    )


def _load_asset_labels(n_assets: int, repo_root: Path) -> List[str]:
    """
    Try to load ticker names from saved real returns header.
    Fallback to asset_0..asset_{n-1} if unavailable.
    """
    candidates = [
        repo_root / "vanilla_pg_outputs" / "vanilla_pg_real_returns.csv",
        repo_root / "actor_critic_outputs" / "actor_critic_real_returns.csv",
        repo_root / "pg_loo_outputs" / "pg_loo_real_returns.csv",
        repo_root / "belief_aware_ac_outputs" / "belief_aware_actor_critic_real_returns.csv",
        repo_root / "hindsight_outputs" / "hindsight_training_real_returns.csv",
    ]
    for p in candidates:
        if p.is_file():
            cols = pd.read_csv(p, nrows=0).columns.tolist()
            if len(cols) >= n_assets + 1:
                labels = cols[1 : 1 + n_assets]  # skip Date
                if len(labels) == n_assets:
                    return labels
    return [f"asset_{i}" for i in range(n_assets)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze in-sample path-level diagnostics.")
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("evaluation"),
        help="Directory containing in_sample_paths_long.csv and in_sample_paths_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save analysis outputs (default: same as --evaluation-dir)",
    )
    args, _ = parser.parse_known_args()

    eval_dir = args.evaluation_dir
    out_dir = args.output_dir if args.output_dir is not None else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = eval_dir / "in_sample_paths_long.csv"
    summary_path = eval_dir / "in_sample_paths_summary.csv"
    if not long_path.is_file():
        raise FileNotFoundError(f"Missing required file: {long_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing required file: {summary_path}")

    long_df = pd.read_csv(long_path)
    summary_df = pd.read_csv(summary_path)

    trade_cols = _pick_trade_cols(long_df)
    x_cols = _pick_x_cols(long_df)
    if len(trade_cols) == 0 or len(x_cols) == 0:
        raise RuntimeError("Expected trade_* and x_* columns in in_sample_paths_long.csv")

    long_df["path_id"] = long_df["path_id"].astype(int)
    long_df["t"] = long_df["t"].astype(int)
    long_df = long_df.sort_values(["method", "path_id", "t"]).reset_index(drop=True)

    # --- Turnover summary ---
    long_df["pre_wealth"] = long_df.groupby(["method", "path_id"], sort=False)["wealth"].shift(1)
    long_df["turnover_dollars"] = long_df[trade_cols].abs().sum(axis=1)
    long_df["turnover_frac"] = long_df["turnover_dollars"] / long_df["pre_wealth"].clip(lower=1e-12)
    turnover_step_df = long_df[long_df["t"] >= 1].copy()
    turnover_summary = (
        turnover_step_df.groupby("method", as_index=False)["turnover_frac"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_turnover_per_step", "std": "std_turnover_per_step"})
    )
    n_steps = turnover_step_df.groupby("method")["turnover_frac"].count().rename("n_steps").reset_index()
    turnover_summary = turnover_summary.merge(n_steps, on="method", how="left")
    turnover_summary.to_csv(out_dir / "in_sample_turnover_summary.csv", index=False)

    # --- Exposure concentration over time (HHI on |x_i| risky holdings) ---
    abs_x = long_df[x_cols].abs()
    gross_risky = abs_x.sum(axis=1)
    weights = abs_x.div(gross_risky.replace(0.0, np.nan), axis=0)
    long_df["exposure_hhi"] = (weights.pow(2)).sum(axis=1)
    conc_by_time = (
        long_df.groupby(["method", "t"], as_index=False)["exposure_hhi"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_exposure_hhi", "std": "std_exposure_hhi"})
    )
    conc_by_time.to_csv(out_dir / "in_sample_concentration_by_time.csv", index=False)

    # --- Asset-level concentration by method and time (avg over paths) ---
    # Use t>=1 to match "over 50 periods".
    weight_cols: List[str] = []
    for c in x_cols:
        wc = f"w_{c}"
        long_df[wc] = abs_x[c] / gross_risky.replace(0.0, np.nan)
        weight_cols.append(wc)
    w_df = long_df[long_df["t"] >= 1].copy()
    n_assets = len(x_cols)
    asset_labels = _load_asset_labels(n_assets, Path.cwd())
    wt = (
        w_df.groupby(["method", "t"], as_index=False)[weight_cols]
        .mean()
        .sort_values(["method", "t"])
        .reset_index(drop=True)
    )
    # Normalize each (method,t) row to sum to 1 to stabilize stacked areas.
    row_sum = wt[weight_cols].sum(axis=1).replace(0.0, np.nan)
    wt[weight_cols] = wt[weight_cols].div(row_sum, axis=0)

    # Long-format CSV for reporting / reproducibility.
    long_rows = []
    for _, r in wt.iterrows():
        for j, wc in enumerate(weight_cols):
            long_rows.append(
                {
                    "method": r["method"],
                    "t": int(r["t"]),
                    "asset_idx": j,
                    "asset": asset_labels[j],
                    "avg_concentration": float(r[wc]),
                }
            )
    pd.DataFrame(long_rows).to_csv(out_dir / "in_sample_asset_concentration_time.csv", index=False)

    # One stacked-area plot per method: x=time, y=asset concentration layers.
    for method, g in wt.groupby("method", sort=False):
        g = g.sort_values("t")
        x = g["t"].to_numpy()
        y = np.vstack([g[wc].to_numpy(dtype=float) for wc in weight_cols])
        fig_m, ax_m = plt.subplots(figsize=(9, 5))
        ax_m.stackplot(x, y, labels=asset_labels)
        ax_m.set_xlabel("Time t")
        ax_m.set_ylabel("Concentration per asset")
        ax_m.set_ylim(0.0, 1.0)
        ax_m.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax_m.set_title(f"{method}: asset concentration over time")
        ax_m.grid(axis="y", alpha=0.3)
        ax_m.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
        fig_m.tight_layout()
        safe_method = method.replace("/", "_").replace(" ", "_")
        fig_m.savefig(
            out_dir / f"in_sample_asset_concentration_{safe_method}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig_m)

    # --- Best / median / worst path IDs by method ---
    selected_paths = _choose_shared_path_ids(summary_df)
    selected_paths.to_csv(out_dir / "in_sample_best_median_worst_path_ids.csv", index=False)

    # --- 1x3 figure: each panel one common category path; lines are five methods ---
    categories = ["best", "median", "worst"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, cat in zip(axes, categories):
        pid = int(selected_paths.loc[selected_paths["category"] == cat, "path_id"].iloc[0])
        method_order = sorted(long_df["method"].unique().tolist())
        for method in method_order:
            traj = long_df[(long_df["method"] == method) & (long_df["path_id"] == pid)].sort_values("t")
            ax.plot(
                traj["t"].to_numpy(),
                traj["wealth"].to_numpy(),
                linewidth=1.8,
                label=method,
            )
        ax.set_title(f"{cat.capitalize()} path")
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Wealth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle("In-sample wealth trajectories on common best / median / worst paths", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "in_sample_best_median_worst_1x3.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved analysis outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

