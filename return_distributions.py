"""
Weekly return diagnostics from Yahoo Finance.

Returns are **simple percentage changes** from **Close** only (not Adj Close):
``pct_change()`` on the Close price panel, i.e. (P_t - P_{t-1}) / P_{t-1} in
decimal form (0.02 = 2%). ``auto_adjust=False`` so Close is unadjusted close.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import vanilla_pg as vanilla


def download_weekly_returns(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1wk",
) -> pd.DataFrame:
    """
    Download prices and compute **simple weekly percentage returns** from **Close** only.

    Delegates to :func:`vanilla_pg.download_weekly_returns` so diagnostics match training data.
    """
    return vanilla.download_weekly_returns(tickers, start_date, end_date, interval)


def build_summary(returns_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "mean": returns_df.mean(),
            "std": returns_df.std(ddof=0),
            "min": returns_df.min(),
            "q01": returns_df.quantile(0.01),
            "q05": returns_df.quantile(0.05),
            "median": returns_df.quantile(0.50),
            "q95": returns_df.quantile(0.95),
            "q99": returns_df.quantile(0.99),
            "max": returns_df.max(),
            "skew": returns_df.skew(),
            "kurtosis": returns_df.kurtosis(),
        }
    )
    return summary


def normality_tests(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hypothesis tests for Gaussian weekly returns (large samples: tiny p-values are common).

    - **Jarque–Bera**: joint skewness/excess kurtosis vs normal.
    - **D'Agostino–Pearson** (``scipy.stats.normaltest``): omnibus based on skew and kurtosis.

    With thousands of weeks, even small departures from normality often yield p ≈ 0;
    rely on effect size (skew, excess kurtosis) and QQ plots, not p-values alone.
    """
    rows: List[dict] = []
    for col in returns_df.columns:
        x = returns_df[col].dropna().to_numpy(dtype=np.float64)
        n = int(x.size)
        jb_stat, jb_p = stats.jarque_bera(x)
        nt_stat, nt_p = stats.normaltest(x)
        rows.append(
            {
                "ticker": col,
                "n": n,
                "jarque_bera_statistic": float(jb_stat),
                "jarque_bera_pvalue": float(jb_p),
                "dagostino_pearson_statistic": float(nt_stat),
                "dagostino_pearson_pvalue": float(nt_p),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def save_qq_plots(returns_df: pd.DataFrame, output_dir: Path, filename: str = "weekly_return_qq_10_tickers.png") -> None:
    """One QQ plot per asset vs normal (same order as histogram grid)."""
    if returns_df.shape[1] == 0:
        raise RuntimeError("returns_df has no columns; cannot build QQ plots.")
    n_assets = returns_df.shape[1]
    n_cols = 2
    n_rows = int(np.ceil(n_assets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, ticker in enumerate(returns_df.columns):
        ax = axes[i]
        x = returns_df[ticker].dropna().to_numpy(dtype=np.float64)
        stats.probplot(x, dist="norm", plot=ax)
        ax.set_title(f"{ticker} QQ (normal)")
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Normal QQ plots — weekly simple % returns from Close (pct_change)",
        fontsize=14,
    )
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def save_histograms(returns_df: pd.DataFrame, output_dir: Path, bins: int = 80) -> None:
    if returns_df.shape[1] == 0:
        raise RuntimeError("returns_df has no columns; cannot build histograms.")
    n_assets = returns_df.shape[1]
    n_cols = 2
    n_rows = int(np.ceil(n_assets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, ticker in enumerate(returns_df.columns):
        ax = axes[i]
        values = returns_df[ticker].to_numpy()
        ax.hist(values, bins=bins, density=True, alpha=0.8, color="#2C7FB8")
        ax.axvline(np.mean(values), color="#D7191C", linestyle="--", linewidth=1.2, label="mean")
        ax.set_title(ticker)
        ax.set_xlabel("Weekly simple return (Close, pct_change)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Weekly simple % return distributions — Close only, pct_change (10 tickers)",
        fontsize=14,
    )
    fig.savefig(output_dir / "weekly_return_distributions_10_tickers.png", dpi=160)
    plt.close(fig)


def save_boxplot(returns_df: pd.DataFrame, output_dir: Path) -> None:
    if returns_df.shape[1] == 0:
        raise RuntimeError("returns_df has no columns; cannot build boxplot.")
    fig, ax = plt.subplots(figsize=(13, 6))
    series_list = [returns_df[c].to_numpy() for c in returns_df.columns]
    labels = list(returns_df.columns)
    try:
        ax.boxplot(series_list, tick_labels=labels, showfliers=False)
    except TypeError:
        # Older matplotlib: use `labels=` instead of `tick_labels=`.
        ax.boxplot(series_list, labels=labels, showfliers=False)
    ax.set_title("Weekly simple % returns (Close, pct_change) — outliers hidden")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Weekly return")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "weekly_return_boxplot_10_tickers.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download weekly returns and visualize distributions.")
    parser.add_argument(
        "--start-date",
        type=str,
        default="1990-01-01",
        help="Inclusive start date for Yahoo download (default: 1990-01-01).",
    )
    parser.add_argument("--end-date", type=str, default="2019-12-31", help="End date.")
    parser.add_argument("--interval", type=str, default="1wk", help="Price interval (default: 1wk).")
    parser.add_argument("--output-dir", type=str, default="return_distribution_outputs", help="Output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    returns_df = download_weekly_returns(
        tickers=list(vanilla.SELECTED_TICKERS_10),
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
    )
    returns_df.to_csv(output_dir / "weekly_returns_10_tickers.csv")

    summary_df = build_summary(returns_df)
    summary_df.to_csv(output_dir / "weekly_returns_summary_10_tickers.csv")

    norm_df = normality_tests(returns_df)
    norm_df.to_csv(output_dir / "weekly_returns_normality_tests_10_tickers.csv")

    save_histograms(returns_df, output_dir=output_dir, bins=80)
    save_boxplot(returns_df, output_dir=output_dir)
    save_qq_plots(returns_df, output_dir=output_dir)

    print("Done.")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Observations: {len(returns_df)} weekly rows")
    print("Returns: simple percentage (Close only, pct_change); see module docstring.")
    print(f"Saved returns: {output_dir / 'weekly_returns_10_tickers.csv'}")
    print(f"Saved summary: {output_dir / 'weekly_returns_summary_10_tickers.csv'}")
    print(f"Saved normality tests: {output_dir / 'weekly_returns_normality_tests_10_tickers.csv'}")
    print(f"Saved histogram figure: {output_dir / 'weekly_return_distributions_10_tickers.png'}")
    print(f"Saved boxplot figure: {output_dir / 'weekly_return_boxplot_10_tickers.png'}")
    print(f"Saved QQ plot figure: {output_dir / 'weekly_return_qq_10_tickers.png'}")


if __name__ == "__main__":
    main()
