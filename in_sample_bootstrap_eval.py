"""
Load the five trained *best* policy checkpoints and evaluate them in-sample on
many bootstrap-simulated return paths (same empirical row-sampling as training).

Writes under ``evaluation/`` (by default):
  - in_sample_paths_long.csv       — one row per (method, path, time) with wealth, holdings, trades
  - in_sample_paths_summary.csv    — one row per (method, path) with terminal wealth & utility
  - in_sample_aggregate.csv        — per-method Monte Carlo means (expected utility estimate)

The policy state is always [holdings, cash, time_remaining]; belief-aware and hindsight
critics are not used at deployment — only the actor weights matter here.

Examples
--------
  python in_sample_bootstrap_eval.py --n-paths 5000
  python in_sample_bootstrap_eval.py --n-paths 1000 --action-mode sample --eval-seed 999
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import belief_aware_actor_critic as bac
import vanilla_pg as vanilla

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    # Jupyter/Colab cells do not define __file__.
    REPO_ROOT = Path.cwd()
DEBUG_LOG_PATH = REPO_ROOT / ".cursor" / "debug-40abc9.log"
DEBUG_SESSION_ID = "40abc9"

# (label for CSV, output subdirectory, best checkpoint name, fallback final name)
MODEL_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    ("vanilla_pg", "vanilla_pg_outputs", "vanilla_pg_policy_best.pt", "vanilla_pg_policy_final.pt"),
    ("pg_loo", "pg_loo_outputs", "pg_loo_policy_best.pt", "pg_loo_policy_final.pt"),
    ("actor_critic", "actor_critic_outputs", "actor_critic_policy_best.pt", "actor_critic_policy_final.pt"),
    (
        "belief_aware_actor_critic",
        "belief_aware_ac_outputs",
        "belief_aware_actor_critic_policy_best.pt",
        "belief_aware_actor_critic_policy_final.pt",
    ),
    (
        "hindsight_training",
        "hindsight_outputs",
        "hindsight_training_policy_best.pt",
        "hindsight_training_policy_final.pt",
    ),
)


def _dtype_from_saved(s: str) -> torch.dtype:
    s = str(s).lower().replace("torch.", "")
    mapping = {
        "float64": torch.float64,
        "double": torch.float64,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported dtype string in saved config: {s!r}")
    return mapping[s]


def config_from_checkpoint_dict(raw: Dict[str, Any]) -> vanilla.Config:
    """Rebuild ``vanilla.Config`` from ``save_checkpoint`` / JSON payload."""
    d = dict(raw)
    d["dtype"] = _dtype_from_saved(d.pop("dtype", "torch.float64"))
    d["tickers"] = tuple(d["tickers"])
    if "gradient_checkpoints" in d:
        d["gradient_checkpoints"] = tuple(int(x) for x in d["gradient_checkpoints"])
    valid = {f.name for f in fields(vanilla.Config)}
    kwargs = {k: v for k, v in d.items() if k in valid}
    return vanilla.Config(**kwargs)


def resolve_checkpoint(base: Path, out_subdir: str, best_name: str, final_name: str) -> Path:
    out = base / out_subdir
    p_best = out / best_name
    if p_best.is_file():
        return p_best
    p_final = out / final_name
    if p_final.is_file():
        return p_final
    raise FileNotFoundError(f"No checkpoint in {out}: tried {best_name} and {final_name}")


def debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    # region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # endregion


def load_gradient_summary(training_out: Path, method: str) -> Dict[str, float]:
    summary_path = training_out / f"{method}_gradient_checkpoint_summary.csv"
    iter_path = training_out / f"{method}_iteration_metrics.csv"
    train_grad_mean = float("nan")
    train_grad_std = float("nan")
    n_train_iters = 0.0
    if iter_path.is_file():
        iter_df = pd.read_csv(iter_path)
        if (not iter_df.empty) and ("gradient_norm" in iter_df.columns):
            g = pd.to_numeric(iter_df["gradient_norm"], errors="coerce").dropna()
            if not g.empty:
                train_grad_mean = float(g.mean())
                train_grad_std = float(g.std(ddof=0))
                n_train_iters = float(g.shape[0])

    if not summary_path.is_file():
        return {
            "mean_gradient_norm_training": train_grad_mean,
            "std_gradient_norm_training": train_grad_std,
            "n_training_iterations_with_gradnorm": n_train_iters,
            "mean_snr": float("nan"),
            "mean_grad_norm_std": float("nan"),
            "mean_estimator_variance_l2": float("nan"),
            "n_gradient_checkpoints": 0.0,
        }
    df = pd.read_csv(summary_path)
    if df.empty:
        return {
            "mean_gradient_norm_training": train_grad_mean,
            "std_gradient_norm_training": train_grad_std,
            "n_training_iterations_with_gradnorm": n_train_iters,
            "mean_snr": float("nan"),
            "mean_grad_norm_std": float("nan"),
            "mean_estimator_variance_l2": float("nan"),
            "n_gradient_checkpoints": 0.0,
        }
    return {
        "mean_gradient_norm_training": train_grad_mean,
        "std_gradient_norm_training": train_grad_std,
        "n_training_iterations_with_gradnorm": n_train_iters,
        "mean_snr": float(df["snr"].mean()),
        "mean_grad_norm_std": float(df["grad_norm_std"].mean()),
        "mean_estimator_variance_l2": float(df["estimator_variance_l2"].mean()),
        "n_gradient_checkpoints": float(df.shape[0]),
    }


def load_returns_for_cfg(cfg: vanilla.Config, training_out: Path) -> pd.DataFrame:
    csv_name = cfg.returns_used_name
    path = training_out / csv_name
    if path.is_file():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = vanilla.download_weekly_returns(
            cfg.tickers, cfg.start_date, cfg.end_date, cfg.interval
        )
    tickers = list(cfg.tickers)
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise RuntimeError(f"Returns columns missing tickers {missing}; have {list(df.columns)}")
    return df[tickers].copy()


def build_policy_and_env(
    method_label: str,
    returns_df: pd.DataFrame,
    cfg: vanilla.Config,
) -> Tuple[vanilla.PortfolioEnv, vanilla.StandardPolicy]:
    """Bootstrap market + env appropriate for the method (belief uses ``EmpiricalBeliefMarket``)."""
    empirical = vanilla.EmpiricalReturnMarket(returns_df, cfg)

    if method_label == "belief_aware_actor_critic":
        regime = bac.fit_two_regime_proxy_hmm(returns_df, jitter=1e-6)
        market = bac.EmpiricalBeliefMarket(
            empirical,
            regime["mu"],
            regime["cov"],
            float(regime["initial_belief"][0]),
            cfg,
            belief_temperature=1.0,
            belief_next_mode="carry",
        )
        env = bac.BeliefPortfolioEnv(market, cfg)
    else:
        env = vanilla.PortfolioEnv(empirical, cfg)

    state_dim = returns_df.shape[1] + 2
    action_dim = returns_df.shape[1]
    policy = vanilla.StandardPolicy(state_dim, action_dim, cfg.hidden_size, cfg).to(
        cfg.device, dtype=cfg.dtype
    )
    return env, policy


def rollout_path_rows(
    env: vanilla.PortfolioEnv,
    policy: vanilla.StandardPolicy,
    cfg: vanilla.Config,
    *,
    method: str,
    path_id: int,
    action_mode: str,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """One path; returns list of time-indexed dict rows, terminal wealth, raw terminal utility."""
    policy.eval()
    state = env.reset()
    n = env.n_assets
    rows: List[Dict[str, Any]] = []

    def snapshot(t: int, trade_vec: Optional[torch.Tensor]) -> Dict[str, Any]:
        w = float(env.terminal_wealth().item())
        c = float(env.c.item())
        x_np = env.x.detach().cpu().numpy()
        row: Dict[str, Any] = {
            "method": method,
            "path_id": float(path_id),
            "t": float(t),
            "wealth": w,
            "cash": c,
        }
        for i in range(n):
            row[f"x_{i}"] = float(x_np[i])
        if trade_vec is None:
            for i in range(n):
                row[f"trade_{i}"] = float("nan")
        else:
            tv = trade_vec.detach().cpu().numpy()
            for i in range(n):
                row[f"trade_{i}"] = float(tv[i])
        return row

    rows.append(snapshot(0, None))

    done = False
    t = 0
    while not done:
        if action_mode == "mean":
            action = policy.mean_action(state)
        elif action_mode == "sample":
            action, _ = policy.sample_action(state)
        else:
            raise ValueError("action_mode must be 'mean' or 'sample'")

        actual_trade = env._project_trade(action)
        state, _, done = env.step(action)
        t += 1
        rows.append(snapshot(t, actual_trade))

    wT = float(env.terminal_wealth().item())
    uT = float(vanilla.utility(torch.tensor(wT, dtype=cfg.dtype, device=cfg.device), cfg).item())
    return rows, wT, uT


def main() -> None:
    parser = argparse.ArgumentParser(description="In-sample bootstrap evaluation for five RL baselines.")
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT, help="Project root with *_outputs/ folders")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "evaluation", help="Where to save CSVs")
    parser.add_argument("--n-paths", type=int, default=5000, help="Bootstrap paths per method")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=16130,
        help="Base seed; bootstrap RNG uses derived seed per method (independent streams).",
    )
    parser.add_argument(
        "--action-mode",
        choices=("mean", "sample"),
        default="mean",
        help="mean: deterministic tanh(mean) trades (matches training-time evaluate_policy); "
        "sample: stochastic policy rollouts.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated subset of: vanilla_pg,pg_loo,actor_critic,belief_aware_actor_critic,hindsight_training",
    )
    args, _unknown = parser.parse_known_args()
    base: Path = args.base_dir
    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if args.methods.strip().lower() == "all":
        specs = MODEL_SPECS
    else:
        want = {x.strip() for x in args.methods.split(",") if x.strip()}
        specs = tuple(s for s in MODEL_SPECS if s[0] in want)
        missing = want - {s[0] for s in specs}
        if missing:
            raise SystemExit(f"Unknown method(s): {sorted(missing)}")

    long_path = out_root / "in_sample_paths_long.csv"
    summary_path = out_root / "in_sample_paths_summary.csv"
    aggregate_path = out_root / "in_sample_aggregate.csv"

    n_assets_ref: Optional[int] = None
    fieldnames_long: Optional[List[str]] = None

    summary_rows: List[Dict[str, Any]] = []
    aggregate: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    merged_rows: List[Dict[str, Any]] = []

    # Write long CSV incrementally
    long_fp = open(long_path, "w", newline="", encoding="utf-8")
    long_writer: Optional[csv.DictWriter] = None

    try:
        debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="in_sample_bootstrap_eval.py:main",
            message="Starting in-sample evaluation run",
            data={"n_paths": args.n_paths, "methods": args.methods, "action_mode": args.action_mode},
        )
        for spec_idx, (label, out_subdir, best_name, final_name) in enumerate(specs):
            ckpt_path = resolve_checkpoint(base, out_subdir, best_name, final_name)
            payload = torch.load(ckpt_path, map_location="cpu")
            cfg_train = config_from_checkpoint_dict(payload["config"])

            cfg_eval = vanilla.Config(**asdict(cfg_train))
            cfg_eval.seed = int(args.eval_seed) + spec_idx * 9973
            cfg_eval.device = cfg_train.device
            if not torch.cuda.is_available() and "cuda" in str(cfg_eval.device).lower():
                cfg_eval.device = "cpu"

            returns_df = load_returns_for_cfg(cfg_train, base / out_subdir)
            env, policy = build_policy_and_env(label, returns_df, cfg_eval)
            policy.load_state_dict(payload["policy_state_dict"])
            policy.to(cfg_eval.device, dtype=cfg_eval.dtype)

            n = env.n_assets
            if n_assets_ref is None:
                n_assets_ref = n
            elif n != n_assets_ref:
                raise RuntimeError(f"Asset count mismatch: {label} has {n}, expected {n_assets_ref}")

            if fieldnames_long is None:
                fieldnames_long = (
                    ["method", "path_id", "t", "wealth", "cash"]
                    + [f"x_{i}" for i in range(n)]
                    + [f"trade_{i}" for i in range(n)]
                )
                long_writer = csv.DictWriter(long_fp, fieldnames=fieldnames_long)
                long_writer.writeheader()

            utilities: List[float] = []
            wealths: List[float] = []

            for p in range(args.n_paths):
                path_rows, wT, uT = rollout_path_rows(
                    env,
                    policy,
                    cfg_eval,
                    method=label,
                    path_id=p,
                    action_mode=args.action_mode,
                )
                utilities.append(uT)
                wealths.append(wT)
                assert long_writer is not None
                for r in path_rows:
                    long_writer.writerow({k: r[k] for k in fieldnames_long})

            grad_summary = load_gradient_summary(base / out_subdir, label)
            gradient_rows.append({"method": label, **grad_summary})
            aggregate.append(
                {
                    "method": label,
                    "n_paths": args.n_paths,
                    "action_mode": args.action_mode,
                    "eval_seed_base": args.eval_seed,
                    "mean_terminal_wealth": float(np.mean(wealths)),
                    "std_terminal_wealth": float(np.std(wealths, ddof=0)),
                    "mean_terminal_utility": float(np.mean(utilities)),
                    "std_terminal_utility": float(np.std(utilities, ddof=0)),
                    "se_terminal_utility": float(np.std(utilities, ddof=0) / math.sqrt(len(utilities))),
                    "checkpoint": str(ckpt_path.resolve()),
                }
            )
            merged_rows.append({**aggregate[-1], **grad_summary})
            for p in range(args.n_paths):
                summary_rows.append(
                    {
                        "method": label,
                        "path_id": p,
                        "terminal_wealth": wealths[p],
                        "terminal_utility": utilities[p],
                    }
                )

            print(
                f"{label}: mean U = {aggregate[-1]['mean_terminal_utility']:.6f} "
                f"(se = {aggregate[-1]['se_terminal_utility']:.6f}), "
                f"mean W = {aggregate[-1]['mean_terminal_wealth']:.6f}"
            )
            debug_log(
                run_id="pre-fix",
                hypothesis_id="H4",
                location="in_sample_bootstrap_eval.py:main",
                message="Completed method with aggregate and gradient stats",
                data={
                    "method": label,
                    "mean_terminal_utility": aggregate[-1]["mean_terminal_utility"],
                    "mean_snr": grad_summary["mean_snr"],
                },
            )

    finally:
        long_fp.close()

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(aggregate).to_csv(aggregate_path, index=False)
    gradient_path = out_root / "in_sample_training_stability_summary.csv"
    pd.DataFrame(gradient_rows).to_csv(gradient_path, index=False)
    merged_path = out_root / "in_sample_performance_vs_stability_summary.csv"
    pd.DataFrame(merged_rows).to_csv(merged_path, index=False)
    debug_log(
        run_id="pre-fix",
        hypothesis_id="H4",
        location="in_sample_bootstrap_eval.py:main",
        message="Saved in-sample outputs including gradient summaries",
        data={"aggregate_csv": str(aggregate_path), "gradient_csv": str(gradient_path), "merged_csv": str(merged_path)},
    )

    meta = {
        "n_paths_per_method": args.n_paths,
        "action_mode": args.action_mode,
        "eval_seed_base": args.eval_seed,
        "long_csv": str(long_path.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "aggregate_csv": str(aggregate_path.resolve()),
        "gradient_summary_csv": str(gradient_path.resolve()),
        "performance_vs_stability_csv": str(merged_path.resolve()),
    }
    with open(out_root / "in_sample_eval_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\nSaved:\n  {long_path}\n  {summary_path}\n  {aggregate_path}\n"
        f"  {gradient_path}\n  {merged_path}"
    )


if __name__ == "__main__":
    main()