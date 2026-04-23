"""
Out-of-sample evaluation (2020–2025 weekly returns, same 10 tickers as training).

1) **Bootstrap OOS** — Same structure as ``in_sample_bootstrap_eval.py``, but each path
   draws rows **with replacement** from the OOS weekly return matrix (6-year window).

2) **Realized path** — One walk forward on the **actual** weekly return sequence: each
   week the policy observes the current state and acts (default: ``mean_action``), then
   the environment applies that week's realized returns. Saves wealth, holdings, trades,
   risky vs cash dollars, turnover.

3) **1/N baseline** — Equal weight in all 10 risky assets, **rebalanced every week**
   (not buy-and-hold), using the same env constraints and transaction costs.

4) **Metrics** — Sharpe (annualized from weekly excess vs risk-free), max drawdown,
   win rate (fraction of weeks with positive portfolio return), average turnover
   (sum(|trade|)/pre-trade wealth per week).

5) **Plot** — Cumulative gross return index ``W_t / W_0`` for all five policies plus 1/N.

**Note:** Policies were trained with a shorter horizon (e.g. 50 weeks); here the realized
episode length equals the number of OOS weeks, so ``time_remaining`` is computed with
that horizon. This is standard for testing but implies some distribution shift.

Examples
--------
  python out_of_sample_evaluation.py
  python out_of_sample_evaluation.py --n-paths 2000 --output-dir evaluation/oos_2020_2025
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import date, timedelta
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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


def normal_approx_ci(x: List[float], z: float = 1.96) -> Tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    arr = np.asarray(x, dtype=np.float64)
    m = float(np.mean(arr))
    se = float(np.std(arr, ddof=0) / math.sqrt(arr.size))
    return (m - z * se, m + z * se)


def download_oos_returns(
    tickers: Tuple[str, ...],
    start_date: str,
    end_date: str,
    interval: str,
    dtype: torch.dtype,
    device: str,
) -> pd.DataFrame:
    """Weekly returns for OOS window (yfinance ``end`` is typically exclusive)."""
    df = vanilla.download_weekly_returns(list(tickers), start_date, end_date, interval)
    df = df.sort_index()
    if df.empty:
        raise RuntimeError("OOS return window is empty after download.")
    return df[list(tickers)].astype(np.float64)


# ── Sequential markets (realized path) ─────────────────────────────────────────


class SequentialReturnMarket:
    """Returns row ``k`` on the ``k``-th ``sample_returns()`` call (one per env step)."""

    def __init__(self, returns_df: pd.DataFrame, cfg: vanilla.Config):
        self.cfg = cfg
        self.n_assets = returns_df.shape[1]
        self._tensor = torch.tensor(
            returns_df.to_numpy(dtype=np.float64),
            dtype=cfg.dtype,
            device=cfg.device,
        )
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def sample_returns(self) -> torch.Tensor:
        if self._idx >= self._tensor.shape[0]:
            raise RuntimeError(
                f"SequentialReturnMarket: step beyond last row ({self._tensor.shape[0]})."
            )
        r = self._tensor[self._idx].clone()
        self._idx += 1
        return r


class SequentialBeliefMarket:
    """Like ``EmpiricalBeliefMarket`` but observations follow the fixed OOS path in order."""

    def __init__(
        self,
        returns_df: pd.DataFrame,
        mu: np.ndarray,
        cov: np.ndarray,
        initial_belief: float,
        cfg: vanilla.Config,
        *,
        belief_temperature: float = 1.0,
        belief_next_mode: str = "carry",
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.n_assets = returns_df.shape[1]
        self._tensor = torch.tensor(
            returns_df.to_numpy(dtype=np.float64),
            dtype=cfg.dtype,
            device=cfg.device,
        )
        self._idx = 0
        self.mu = torch.tensor(mu, dtype=cfg.dtype, device=cfg.device)
        self.cov = torch.tensor(cov, dtype=cfg.dtype, device=cfg.device)
        self.initial_belief = float(np.clip(initial_belief, 1e-6, 1.0 - 1e-6))
        self.belief_temperature = float(max(belief_temperature, 1e-6))
        self.belief_next_mode = str(belief_next_mode)

        self.chol = torch.linalg.cholesky(self.cov)
        self.inv_cov = torch.cholesky_inverse(self.chol)
        self.logdet = 2.0 * torch.log(torch.diagonal(self.chol, dim1=1, dim2=2)).sum(dim=1)

        self.belief_prior: float = self.initial_belief
        self.belief_posterior: float = self.initial_belief

    def reset(self) -> None:
        self._idx = 0
        self.belief_prior = self.initial_belief
        self.belief_posterior = self.initial_belief

    def _prior_from_posterior(self, post_p1: float) -> float:
        if self.belief_next_mode == "carry":
            return float(np.clip(post_p1, 0.0, 1.0))
        raise RuntimeError(f"Unknown belief_next_mode: {self.belief_next_mode!r}")

    def _log_likelihood_per_regime(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(0) - self.mu
        quad = torch.einsum("ki,kij,kj->k", diff, self.inv_cov, diff)
        d = r.shape[0]
        return -0.5 * (d * math.log(2.0 * math.pi) + self.logdet + quad)

    def sample_returns(self) -> torch.Tensor:
        if self._idx >= self._tensor.shape[0]:
            raise RuntimeError("SequentialBeliefMarket: past end of OOS path.")
        r = self._tensor[self._idx].clone()
        self._idx += 1
        ll = self._log_likelihood_per_regime(r)
        prior_p1 = float(np.clip(self.belief_prior, 1e-12, 1.0 - 1e-12))
        logits = torch.stack(
            [
                ll[0] + math.log(1.0 - prior_p1),
                ll[1] + math.log(prior_p1),
            ]
        ) / self.belief_temperature
        post = torch.softmax(logits, dim=0)
        self.belief_posterior = float(post[1].item())
        self.belief_prior = float(np.clip(self._prior_from_posterior(self.belief_posterior), 0.0, 1.0))
        return r


def build_policy_and_env(
    method_label: str,
    returns_df: pd.DataFrame,
    cfg: vanilla.Config,
    *,
    sequential: bool,
) -> Tuple[vanilla.PortfolioEnv, vanilla.StandardPolicy]:
    n = returns_df.shape[1]
    if sequential:
        if method_label == "belief_aware_actor_critic":
            regime = bac.fit_two_regime_proxy_hmm(returns_df, jitter=1e-6)
            market = SequentialBeliefMarket(
                returns_df,
                regime["mu"],
                regime["cov"],
                float(regime["initial_belief"][0]),
                cfg,
            )
            env = bac.BeliefPortfolioEnv(market, cfg)
        else:
            env = vanilla.PortfolioEnv(SequentialReturnMarket(returns_df, cfg), cfg)
    else:
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

    state_dim = n + 2
    policy = vanilla.StandardPolicy(state_dim, n, cfg.hidden_size, cfg).to(cfg.device, dtype=cfg.dtype)
    return env, policy


def rollout_bootstrap_path_rows(
    env: vanilla.PortfolioEnv,
    policy: vanilla.StandardPolicy,
    cfg: vanilla.Config,
    *,
    method: str,
    path_id: int,
    action_mode: str,
) -> Tuple[List[Dict[str, Any]], float, float]:
    policy.eval()
    state = env.reset()
    n = env.n_assets
    rows: List[Dict[str, Any]] = []

    def snap(t: int, trade_vec: Optional[torch.Tensor]) -> Dict[str, Any]:
        w = float(env.terminal_wealth().item())
        risky = float(env.x.sum().item())
        cash = float(env.c.item())
        x_np = env.x.detach().cpu().numpy()
        row: Dict[str, Any] = {
            "method": method,
            "path_id": path_id,
            "t": t,
            "wealth": w,
            "risky_dollars": risky,
            "riskfree_dollars": cash,
            "cash": cash,
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

    rows.append(snap(0, None))
    done = False
    t = 0
    while not done:
        if action_mode == "mean":
            action = policy.mean_action(state)
        else:
            action, _ = policy.sample_action(state)
        actual_trade = env._project_trade(action)
        state, _, done = env.step(action)
        t += 1
        rows.append(snap(t, actual_trade))

    wT = float(env.terminal_wealth().item())
    uT = float(vanilla.utility(torch.tensor(wT, dtype=cfg.dtype, device=cfg.device), cfg).item())
    return rows, wT, uT


def run_realized_path_records(
    env: vanilla.PortfolioEnv,
    policy: Optional[vanilla.StandardPolicy],
    cfg: vanilla.Config,
    dates: pd.DatetimeIndex,
    *,
    method: str,
    action_mode: str,
    equal_weight: bool,
) -> List[Dict[str, Any]]:
    """One full OOS episode; if ``equal_weight``, ``policy`` is ignored."""
    env.reset()
    if policy is not None:
        policy.eval()

    n = env.n_assets
    records: List[Dict[str, Any]] = []
    weekly_rf = cfg.weekly_rf()

    # t=0 snapshot (start of first week, before any trade)
    w0 = float(env.terminal_wealth().item())
    records.append(
        _record_row(
            dates,
            0,
            method,
            env,
            None,
            float("nan"),
            float("nan"),
            float("nan"),
            weekly_rf,
            n,
            pre_wealth=w0,
            week_date=str(dates[0].date()) if len(dates) else "",
        )
    )

    done = False
    state = env.state()
    t = 0
    while not done:
        wealth_pre = float(env.terminal_wealth().item())
        if equal_weight:
            w = wealth_pre
            x_tgt = torch.full((n,), w / float(n), dtype=cfg.dtype, device=cfg.device)
            action = x_tgt - env.x
        else:
            if action_mode == "mean":
                action = policy.mean_action(state)  # type: ignore[union-attr]
            else:
                action, _ = policy.sample_action(state)  # type: ignore[union-attr]

        actual_trade = env._project_trade(action)
        turnover_dollar = float(torch.sum(torch.abs(actual_trade)).item())
        turnover_frac = turnover_dollar / max(wealth_pre, 1e-12)

        state, _, done = env.step(action)
        t += 1
        wealth_post = float(env.terminal_wealth().item())
        port_ret = wealth_post / max(wealth_pre, 1e-12) - 1.0

        date_str = str(dates[t - 1].date()) if t - 1 < len(dates) else ""

        records.append(
            _record_row(
                dates,
                t,
                method,
                env,
                actual_trade,
                turnover_dollar,
                turnover_frac,
                port_ret,
                weekly_rf,
                n,
                pre_wealth=wealth_pre,
                week_date=date_str,
            )
        )
        if done:
            break

    return records


def _record_row(
    dates: pd.DatetimeIndex,
    t: int,
    method: str,
    env: vanilla.PortfolioEnv,
    trade: Optional[torch.Tensor],
    turnover_dollar: float,
    turnover_frac: float,
    port_ret: float,
    weekly_rf: float,
    n: int,
    *,
    pre_wealth: float,
    week_date: str = "",
) -> Dict[str, Any]:
    risky = float(env.x.sum().item())
    cash = float(env.c.item())
    w = float(env.terminal_wealth().item())
    x_np = env.x.detach().cpu().numpy()
    row: Dict[str, Any] = {
        "method": method,
        "t": t,
        "date": week_date
        if week_date
        else (str(dates[t].date()) if t < len(dates) else ""),
        "wealth": w,
        "risky_dollars": risky,
        "riskfree_dollars": cash,
        "pre_trade_wealth": pre_wealth,
        "portfolio_simple_return": port_ret,
        "weekly_rf": weekly_rf,
        "excess_return": (port_ret - weekly_rf) if np.isfinite(port_ret) else float("nan"),
        "turnover_dollars": turnover_dollar,
        "turnover_frac_wealth": turnover_frac,
    }
    if trade is not None:
        tv = trade.detach().cpu().numpy()
        for i in range(n):
            row[f"trade_{i}"] = float(tv[i])
    else:
        for i in range(n):
            row[f"trade_{i}"] = float("nan")
    for i in range(n):
        row[f"x_{i}"] = float(x_np[i])
    return row


def compute_path_metrics(
    wealth: np.ndarray,
    port_ret: np.ndarray,
    turnover_frac: np.ndarray,
    weekly_rf: float,
) -> Dict[str, float]:
    """``wealth`` includes initial W0 as first element; ``port_ret`` per completed week."""
    w = np.asarray(wealth, dtype=np.float64)
    r = np.asarray(port_ret, dtype=np.float64)
    turn = np.asarray(turnover_frac, dtype=np.float64)
    if r.shape != turn.shape:
        raise ValueError("port_ret and turnover_frac must have the same length.")
    ok = np.isfinite(r) & np.isfinite(turn)
    r = r[ok]
    turn = turn[ok]

    if w.size < 2:
        return {
            "n_weeks": 0.0,
            "total_gross_return": float("nan"),
            "sharpe_annualized": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
            "avg_turnover_frac": float("nan"),
        }

    w0 = w[0]
    cum = w / w0
    peak = np.maximum.accumulate(cum)
    dd = 1.0 - cum / np.maximum(peak, 1e-12)
    max_dd = float(np.max(dd))

    excess = r - weekly_rf
    if len(excess) > 1 and np.std(excess, ddof=1) > 1e-12:
        sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(52.0))
    else:
        sharpe = float("nan")

    win_rate = float(np.mean(r > 0)) if len(r) else float("nan")
    avg_turn = float(np.nanmean(turn)) if len(turn) else float("nan")

    return {
        "n_weeks": float(len(r)),
        "total_gross_return": float(w[-1] / w0 - 1.0),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_turnover_frac": avg_turn,
    }


def run_equal_weight_path(
    returns_df: pd.DataFrame,
    cfg: vanilla.Config,
    dates: pd.DatetimeIndex,
) -> List[Dict[str, Any]]:
    env = vanilla.PortfolioEnv(SequentialReturnMarket(returns_df, cfg), cfg)
    return run_realized_path_records(
        env, None, cfg, dates, method="equal_weight_1N", action_mode="mean", equal_weight=True
    )


def main() -> None:
    p = argparse.ArgumentParser(description="OOS evaluation: bootstrap, realized path, 1/N, metrics, plots.")
    p.add_argument("--base-dir", type=Path, default=REPO_ROOT)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "evaluation" / "oos_2020_2025")
    p.add_argument("--oos-start", type=str, default="2020-01-01")
    p.add_argument("--oos-end", type=str, default="2025-12-31")
    p.add_argument("--n-paths", type=int, default=5000)
    p.add_argument("--eval-seed", type=int, default=20261)
    p.add_argument("--action-mode", choices=("mean", "sample"), default="mean")
    p.add_argument(
        "--bootstrap-horizon-mode",
        choices=("training", "oos"),
        default="training",
        help="Bootstrap OOS path length: training horizon or full OOS length.",
    )
    p.add_argument("--methods", type=str, default="all")
    args, _unknown = p.parse_known_args()

    base = args.base_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.methods.strip().lower() == "all":
        specs = MODEL_SPECS
    else:
        want = {x.strip() for x in args.methods.split(",") if x.strip()}
        specs = tuple(s for s in MODEL_SPECS if s[0] in want)
        if not specs:
            raise SystemExit("No valid methods in --methods.")

    # Load one checkpoint to get tickers / training hyperparameters
    first_ckpt = resolve_checkpoint(base, specs[0][1], specs[0][2], specs[0][3])
    payload0 = torch.load(first_ckpt, map_location="cpu")
    cfg0 = config_from_checkpoint_dict(payload0["config"])

    if not torch.cuda.is_available() and "cuda" in str(cfg0.device).lower():
        device = "cpu"
    else:
        device = str(cfg0.device)

    # yfinance end is exclusive — extend slightly to include late-2025 weeks
    end_download = args.oos_end
    if len(end_download) == 10:
        end_parts = end_download.split("-")
        y, m, d = int(end_parts[0]), int(end_parts[1]), int(end_parts[2])
        end_download = (date(y, m, d) + timedelta(days=7)).isoformat()

    oos_df = download_oos_returns(
        cfg0.tickers,
        args.oos_start,
        end_download,
        cfg0.interval,
        cfg0.dtype,
        device,
    )
    oos_df = oos_df.dropna(how="any")
    if len(oos_df) < 5:
        raise RuntimeError(f"Too few OOS observations: {len(oos_df)}")

    T = len(oos_df)
    dates = pd.DatetimeIndex(oos_df.index)
    debug_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="out_of_sample_evaluation.py:main",
        message="Loaded OOS data and resolved horizon mode",
        data={"n_oos_weeks": T, "bootstrap_horizon_mode": args.bootstrap_horizon_mode},
    )

    # ── 1) Bootstrap OOS (path length = training horizon from checkpoint) ───
    H = T if args.bootstrap_horizon_mode == "oos" else int(cfg0.horizon)
    long_path = out / "oos_bootstrap_paths_long.csv"
    summary_path = out / "oos_bootstrap_paths_summary.csv"
    aggregate_path = out / "oos_bootstrap_aggregate.csv"

    fieldnames_long: Optional[List[str]] = None
    long_fp = open(long_path, "w", newline="", encoding="utf-8")
    long_writer: Optional[csv.DictWriter] = None
    summary_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    try:
        for spec_idx, (label, out_subdir, best_name, final_name) in enumerate(specs):
            ckpt_path = resolve_checkpoint(base, out_subdir, best_name, final_name)
            payload = torch.load(ckpt_path, map_location="cpu")
            cfg_train = config_from_checkpoint_dict(payload["config"])

            cfg_boot = vanilla.Config(**asdict(cfg_train))
            cfg_boot.seed = int(args.eval_seed) + spec_idx * 7919
            cfg_boot.device = device
            cfg_boot.horizon = H

            env, policy = build_policy_and_env(label, oos_df, cfg_boot, sequential=False)
            policy.load_state_dict(payload["policy_state_dict"])
            policy.to(device, dtype=cfg_boot.dtype)

            n = env.n_assets
            if fieldnames_long is None:
                fieldnames_long = (
                    ["method", "path_id", "t", "wealth", "risky_dollars", "riskfree_dollars", "cash"]
                    + [f"x_{i}" for i in range(n)]
                    + [f"trade_{i}" for i in range(n)]
                )
                long_writer = csv.DictWriter(long_fp, fieldnames=fieldnames_long)
                long_writer.writeheader()

            utils: List[float] = []
            wealths: List[float] = []
            for path_id in range(args.n_paths):
                rows, wT, uT = rollout_bootstrap_path_rows(
                    env,
                    policy,
                    cfg_boot,
                    method=label,
                    path_id=path_id,
                    action_mode=args.action_mode,
                )
                utils.append(uT)
                wealths.append(wT)
                assert long_writer is not None
                for r in rows:
                    long_writer.writerow({k: r[k] for k in fieldnames_long})

            aggregate_rows.append(
                {
                    "method": label,
                    "n_paths": args.n_paths,
                    "horizon": H,
                    "horizon_mode": args.bootstrap_horizon_mode,
                    "action_mode": args.action_mode,
                    "mean_terminal_wealth": float(np.mean(wealths)),
                    "std_terminal_wealth": float(np.std(wealths, ddof=0)),
                    "mean_terminal_utility": float(np.mean(utils)),
                    "std_terminal_utility": float(np.std(utils, ddof=0)),
                    "se_terminal_utility": float(np.std(utils, ddof=0) / math.sqrt(len(utils))),
                    "ci95_terminal_utility_low": normal_approx_ci(utils)[0],
                    "ci95_terminal_utility_high": normal_approx_ci(utils)[1],
                    "ci95_terminal_wealth_low": normal_approx_ci(wealths)[0],
                    "ci95_terminal_wealth_high": normal_approx_ci(wealths)[1],
                    "checkpoint": str(ckpt_path.resolve()),
                }
            )
            for path_id in range(args.n_paths):
                summary_rows.append(
                    {
                        "method": label,
                        "path_id": path_id,
                        "terminal_wealth": wealths[path_id],
                        "terminal_utility": utils[path_id],
                    }
                )
            print(f"[bootstrap OOS] {label}: mean U = {aggregate_rows[-1]['mean_terminal_utility']:.6f}")
            debug_log(
                run_id="pre-fix",
                hypothesis_id="H2",
                location="out_of_sample_evaluation.py:main",
                message="Computed bootstrap aggregate with confidence intervals",
                data={
                    "method": label,
                    "mean_u": aggregate_rows[-1]["mean_terminal_utility"],
                    "ci_u_low": aggregate_rows[-1]["ci95_terminal_utility_low"],
                    "ci_u_high": aggregate_rows[-1]["ci95_terminal_utility_high"],
                },
            )
    finally:
        long_fp.close()

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)

    # ── 2) Realized path + 1/N + metrics ──────────────────────────────────────
    real_csv = out / "oos_realized_path_weekly.csv"
    metrics_rows: List[Dict[str, Any]] = []
    cumulative_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    cfg_seq = vanilla.Config(**asdict(cfg0))
    cfg_seq.device = device
    cfg_seq.horizon = T

    all_real_records: List[Dict[str, Any]] = []

    for label, out_subdir, best_name, final_name in specs:
        ckpt_path = resolve_checkpoint(base, out_subdir, best_name, final_name)
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg_train = config_from_checkpoint_dict(payload["config"])
        cfg_m = vanilla.Config(**asdict(cfg_train))
        cfg_m.device = device
        cfg_m.horizon = T

        env, policy = build_policy_and_env(label, oos_df, cfg_m, sequential=True)
        policy.load_state_dict(payload["policy_state_dict"])
        policy.to(device, dtype=cfg_m.dtype)

        recs = run_realized_path_records(
            env, policy, cfg_m, dates, method=label, action_mode=args.action_mode, equal_weight=False
        )
        all_real_records.extend(recs)

        # Metrics from weekly rows (skip t=0 initial row for returns / turnover stats)
        sub = pd.DataFrame(recs)
        w_series = sub["wealth"].to_numpy()
        r_series = sub["portfolio_simple_return"].to_numpy()[1:]  # first row NaN
        turn_series = sub["turnover_frac_wealth"].to_numpy()[1:]
        rf = float(cfg_m.weekly_rf())
        m = compute_path_metrics(w_series, r_series, turn_series, rf)
        m["method"] = label
        metrics_rows.append(m)

        cum = w_series / w_series[0]
        cumulative_series[label] = (np.arange(len(cum)), cum)

    eq_recs = run_equal_weight_path(oos_df, cfg_seq, dates)
    all_real_records.extend(eq_recs)
    sub_eq = pd.DataFrame(eq_recs)
    w_eq = sub_eq["wealth"].to_numpy()
    r_eq = sub_eq["portfolio_simple_return"].to_numpy()[1:]
    turn_eq = sub_eq["turnover_frac_wealth"].to_numpy()[1:]
    m_eq = compute_path_metrics(w_eq, r_eq, turn_eq, float(cfg_seq.weekly_rf()))
    m_eq["method"] = "equal_weight_1N"
    metrics_rows.append(m_eq)
    cumulative_series["equal_weight_1N"] = (np.arange(len(w_eq)), w_eq / w_eq[0])

    pd.DataFrame(all_real_records).to_csv(real_csv, index=False)
    metrics_path = out / "oos_portfolio_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    merged_path = out / "oos_combined_summary.csv"
    oos_boot = pd.DataFrame(aggregate_rows)
    oos_real = pd.DataFrame(metrics_rows)
    oos_merged = oos_real.merge(oos_boot, on="method", how="left", suffixes=("_realized", "_bootstrap"))
    oos_merged.to_csv(merged_path, index=False)
    debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="out_of_sample_evaluation.py:main",
        message="Saved realized metrics and merged OOS summary",
        data={"metrics_csv": str(metrics_path), "merged_csv": str(merged_path)},
    )

    # ── 3) Plot cumulative gross return index ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (name, (xs, ys)) in enumerate(cumulative_series.items()):
        if name == "equal_weight_1N":
            ax.plot(xs, ys - 1.0, label=name, linestyle="--", linewidth=2.0, color="black")
        else:
            ax.plot(xs, ys - 1.0, label=name, color=colors[i % 10], linewidth=1.5)
    ax.set_xlabel("Week index (OOS)")
    ax.set_ylabel("Cumulative gross return  $W_t/W_0 - 1$")
    ax.set_title(f"Out-of-sample ({args.oos_start} – {args.oos_end})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out / "oos_cumulative_returns.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    rl_items = [(k, v) for k, v in cumulative_series.items() if k != "equal_weight_1N"]
    for i, (name, (xs, ys)) in enumerate(rl_items):
        ax_top.plot(xs, ys - 1.0, label=name, color=colors[i % 10], linewidth=1.5)
    ax_top.set_ylabel("Cumulative gross return  $W_t/W_0 - 1$")
    ax_top.set_title(f"Five learned policies — OOS {args.oos_start} to {args.oos_end}")
    ax_top.legend(loc="best", fontsize=7)
    ax_top.grid(True, alpha=0.3)

    xs_eq, ys_eq = cumulative_series["equal_weight_1N"]
    ax_bot.plot(xs_eq, ys_eq - 1.0, color="black", linestyle="--", linewidth=2.0, label="equal_weight_1N")
    ax_bot.set_xlabel("Week index (OOS)")
    ax_bot.set_ylabel("Cumulative gross return  $W_t/W_0 - 1$")
    ax_bot.set_title("Baseline: equal weight 1/N, rebalanced weekly")
    ax_bot.legend(loc="best")
    ax_bot.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2_path = out / "oos_cumulative_returns_split.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    meta = {
        "oos_start": args.oos_start,
        "oos_end": args.oos_end,
        "n_oos_weeks": T,
        "tickers": list(cfg0.tickers),
        "bootstrap_horizon": H,
        "n_bootstrap_paths": args.n_paths,
        "action_mode": args.action_mode,
        "files": {
            "bootstrap_long": str(long_path.resolve()),
            "bootstrap_summary": str(summary_path.resolve()),
            "bootstrap_aggregate": str(aggregate_path.resolve()),
            "realized_weekly": str(real_csv.resolve()),
            "metrics": str(metrics_path.resolve()),
            "combined_summary": str(merged_path.resolve()),
            "plot_all": str(fig_path.resolve()),
            "plot_split": str(fig2_path.resolve()),
        },
    }
    with open(out / "oos_evaluation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    oos_df.to_csv(out / "oos_weekly_returns_used.csv")
    print(f"\nSaved OOS evaluation under:\n  {out.resolve()}")


if __name__ == "__main__":
    main()