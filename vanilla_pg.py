from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.distributions import Normal


TICKERS_40: List[str] = [
    # Information Technology
    "AAPL", "MSFT", "IBM", "ORCL",
    # Communication Services
    "DIS", "VZ", "T", "CMCSA",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE",
    # Consumer Staples
    "KO", "PEP", "PG", "WMT",
    # Health Care
    "JNJ", "PFE", "MRK", "BMY",
    # Financials
    "JPM", "BAC", "WFC", "C",
    # Industrials
    "CAT", "UPS", "HON", "GE",
    # Materials
    "DD", "APD", "SHW",
    # Energy
    "XOM", "CVX", "COP",
    # Utilities
    "NEE", "SO", "DUK",
    # Real Estate
    "SPG", "PLD", "O",
]

SELECTED_TICKERS_10: List[str] = [
    "AAPL", "MSFT",
    "DIS", "CMCSA",
    "AMZN", "HD",
    "KO",
    "JNJ",
    "JPM",
    "XOM",
]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    start_date: str = "1990-01-01"
    end_date: str = "2019-12-31"
    interval: str = "1wk"
    tickers: Tuple[str, ...] = tuple(SELECTED_TICKERS_10)

    seed: int = 1613
    dtype: torch.dtype = torch.float64
    device: str = field(default_factory=default_device)

    horizon: int = 50
    batch_size: int = 64
    n_iterations: int = 500
    eval_every: int = 25
    eval_paths: int = 512

    learning_rate: float = 1e-3
    initial_wealth: float = 1.0
    annual_rf: float = 0.0154
    tc_rate: float = 0.005

    hidden_size: int = 128
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    # Max |trade_i| per asset before projection is this × current wealth W = sum(x)+c.
    # 1.0 matches “±100% of W” per risky leg at the policy head; env still projects/feasibility-clips.
    max_trade_fraction: float = 1.0

    # CRRA utility U(w) = w^gamma / gamma (here gamma = -1 => U(w) = -1/w).
    crra_gamma: float = -1.0
    wealth_floor: float = 1e-5

    projection_bisection_steps: int = 25

    # 130/30-style caps on gross long / short risky notionals **after** a feasible trade,
    # as multiples of post-trade net wealth ``W_post = W_pre − TC`` (same units as ``x``).
    # Set either to ``None`` to disable that side (or both ``None`` to remove gross caps).
    gross_long_cap: Optional[float] = 1.3
    gross_short_cap: Optional[float] = 0.3

    gradient_checkpoints: Tuple[int, ...] = (25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    gradient_repeats: int = 100
    save_full_training_gradients: bool = True

    # Training stability (shared across REINFORCE, LOO, actor–critic, belief-aware, hindsight).
    terminal_utility_clip: float = 100.0
    actor_grad_clip: float = 10.0
    critic_grad_clip: float = 10.0

    plot: bool = True
    output_dir: str = "vanilla_pg_outputs"
    save_best_model: bool = True
    final_model_name: str = "vanilla_pg_policy_final.pt"
    best_model_name: str = "vanilla_pg_policy_best.pt"
    training_plot_name: str = "vanilla_pg_training_curve.png"
    iteration_metrics_name: str = "vanilla_pg_iteration_metrics.csv"
    gradient_summary_name: str = "vanilla_pg_gradient_checkpoint_summary.csv"
    training_gradient_matrix_name: str = "vanilla_pg_training_gradients.npz"
    returns_used_name: str = "vanilla_pg_real_returns.csv"
    config_name: str = "vanilla_pg_config.json"

    def weekly_rf(self) -> float:
        return (1.0 + self.annual_rf) ** (1.0 / 52.0) - 1.0


@dataclass
class Episode:
    log_probs: List[torch.Tensor]
    terminal_wealth: torch.Tensor
    terminal_utility: torch.Tensor


@dataclass
class TrainingResult:
    iteration_metrics: List[Dict[str, float]]
    eval_curve: List[float]
    eval_steps: List[int]
    returns_df: pd.DataFrame
    best_eval_utility: Optional[float] = None
    best_eval_wealth: Optional[float] = None


class EmpiricalReturnMarket:
    """
    Samples percentage return vectors by bootstrap from real downloaded data.
    No log-return transformation is used anywhere. Just purely empirical returns.
    """

    def __init__(self, returns_df: pd.DataFrame, cfg: Config):
        if returns_df.empty:
            raise RuntimeError("Received empty returns DataFrame.")
        self.n_assets = returns_df.shape[1]
        self.returns_tensor = torch.tensor(
            returns_df.to_numpy(dtype=np.float64),
            dtype=cfg.dtype,
            device=cfg.device,
        )
        self._bootstrap_rng = torch.Generator(device=torch.device("cpu"))
        self._bootstrap_rng.manual_seed(derived_torch_seed(cfg.seed, "EmpiricalReturnMarket", "bootstrap"))

    def sample_returns(self) -> torch.Tensor:
        idx = torch.randint(
            low=0,
            high=self.returns_tensor.shape[0],
            size=(1,),
            generator=self._bootstrap_rng,
            device=torch.device("cpu"),
        ).item()
        return self.returns_tensor[idx].clone()


class PortfolioEnv:
    """
    Dynamic portfolio environment without hidden regimes.

    State:
        [x_t, c_t, time_remaining]
        x_t: vector of dollar risky holdings
        c_t: dollar cash

    Action:
        signed dollar trade vector a_t in R^n

    Bookkeeping after a feasible trade (before asset returns):
        Let W_pre = sum(x)+c and TC = tc_rate * sum(|a|). Then post-trade
        holdings satisfy sum(x_post) + c_post = W_pre - TC. Each risky leg
        is capped at |x_post,i| <= W_post = W_pre - TC (±100% of that net mark-to-market).

    Constraints after trading:
        - no borrowing in the cash account: c_post >= 0
        - each risky position is bounded by [-100%, 100%] of post-trade wealth
        - gross caps (130/30): sum(max(x_post,0)) <= gross_long_cap * W_post and
          sum(max(-x_post,0)) <= gross_short_cap * W_post when set on ``Config``
        - proportional transaction costs on turnover

    Returns are percentage returns.
    """

    def __init__(self, market: EmpiricalReturnMarket, cfg: Config):
        self.market = market
        self.cfg = cfg
        self.n_assets = market.n_assets
        self.weekly_rf = cfg.weekly_rf()
        self.reset()

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.x = torch.zeros(self.n_assets, dtype=self.cfg.dtype, device=self.cfg.device)
        self.c = torch.tensor(self.cfg.initial_wealth, dtype=self.cfg.dtype, device=self.cfg.device)
        return self.state()

    def terminal_wealth(self) -> torch.Tensor:
        return self.x.sum() + self.c

    def state(self) -> torch.Tensor:
        time_remaining = torch.tensor(
            [(self.cfg.horizon - self.t) / self.cfg.horizon],
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        return torch.cat([self.x, self.c.view(1), time_remaining])

    def _is_feasible_scaled_trade(self, proposed_trade: torch.Tensor, scale: torch.Tensor) -> bool:
        scaled_trade = scale * proposed_trade
        turnover = torch.sum(torch.abs(scaled_trade))
        wealth_pre = self.terminal_wealth()
        wealth_post = wealth_pre - self.cfg.tc_rate * turnover
        if wealth_post <= self.cfg.wealth_floor:
            return False

        x_post = self.x + scaled_trade
        c_post = self.c - scaled_trade.sum() - self.cfg.tc_rate * turnover
        if c_post < -1e-12:
            return False

        if self.cfg.gross_long_cap is not None:
            long_exp = torch.relu(x_post).sum()
            if long_exp > self.cfg.gross_long_cap * wealth_post + 1e-8:
                return False
        if self.cfg.gross_short_cap is not None:
            short_exp = torch.relu(-x_post).sum()
            if short_exp > self.cfg.gross_short_cap * wealth_post + 1e-8:
                return False

        if torch.any(x_post > wealth_post + 1e-12):
            return False
        if torch.any(x_post < -wealth_post - 1e-12):
            return False
        return True

    def _project_trade(self, proposed_trade: torch.Tensor) -> torch.Tensor:
        if proposed_trade.ndim != 1 or proposed_trade.shape[0] != self.n_assets:
            raise ValueError(f"Expected action shape {(self.n_assets,)}, got {tuple(proposed_trade.shape)}")

        wealth_pre = self.terminal_wealth()
        if wealth_pre <= self.cfg.wealth_floor:
            return torch.zeros_like(proposed_trade)

        lo = torch.zeros((), dtype=self.cfg.dtype, device=self.cfg.device)
        hi = torch.ones((), dtype=self.cfg.dtype, device=self.cfg.device)
        if self._is_feasible_scaled_trade(proposed_trade, hi):
            return proposed_trade

        for _ in range(self.cfg.projection_bisection_steps):
            mid = 0.5 * (lo + hi)
            if self._is_feasible_scaled_trade(proposed_trade, mid):
                lo = mid
            else:
                hi = mid
        return lo * proposed_trade

    def step(self, proposed_trade: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        proposed_trade = proposed_trade.detach().to(dtype=self.cfg.dtype, device=self.cfg.device)
        wealth_pre = self.terminal_wealth()
        actual_trade = self._project_trade(proposed_trade)
        turnover = torch.sum(torch.abs(actual_trade))
        tc_paid = self.cfg.tc_rate * turnover

        x_post = self.x + actual_trade
        c_post = self.c - actual_trade.sum() - tc_paid

        wealth_post_trade = wealth_pre - tc_paid
        book = x_post.sum() + c_post
        if not torch.isclose(book, wealth_post_trade, atol=1e-8, rtol=1e-8):
            raise RuntimeError(
                "Post-trade book does not match wealth minus transaction costs: "
                f"sum(x_post)+c_post={float(book.item())}, "
                f"W_pre-TC={float(wealth_post_trade.item())}"
            )

        c_post = torch.clamp(c_post, min=0.0)

        risky_returns = self.market.sample_returns()
        self.x = x_post * (1.0 + risky_returns)
        self.c = c_post * (1.0 + self.weekly_rf)
        self.t += 1

        done = bool(self.t >= self.cfg.horizon or self.terminal_wealth() <= self.cfg.wealth_floor)
        reward = torch.zeros((), dtype=self.cfg.dtype, device=self.cfg.device)
        if done:
            reward = utility(self.terminal_wealth(), self.cfg).detach()
        return self.state(), reward, done


class StandardPolicy(nn.Module):
    """
    Standard stochastic policy for continuous trade actions.

    The policy outputs the parameters of a diagonal Normal distribution for a
    latent action. The latent action is squashed with tanh and then scaled into
    dollar trade amounts.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, min=self.cfg.min_log_std, max=self.cfg.max_log_std)
        return mean, log_std

    def _trade_scale(self, state: torch.Tensor) -> torch.Tensor:
        x = state[:-2]
        c = state[-2]
        wealth = torch.clamp(x.sum() + c, min=self.cfg.wealth_floor)
        return self.cfg.max_trade_fraction * wealth

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        base_dist = Normal(mean, std)

        latent = base_dist.sample()
        squashed = torch.tanh(latent)
        scale = self._trade_scale(state)
        trade = scale * squashed

        log_prob = base_dist.log_prob(latent) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        return trade, log_prob.sum()

    @torch.no_grad()
    def mean_action(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self(state)
        squashed = torch.tanh(mean)
        scale = self._trade_scale(state)
        return scale * squashed


def set_seed(seed: int) -> None:
    """Seed python, NumPy, and the *global* torch RNG (policy init + policy noise).

    Empirical bootstrap, HMM shocks, and critic initialization use
    ``derived_torch_seed`` + dedicated ``torch.Generator`` instances so the
    global stream stays aligned across trainers that share ``StandardPolicy``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derived_torch_seed(base_seed: int, *tags: str) -> int:
    """
    Deterministic 31-bit seed for auxiliary torch.Generator streams.

    The global torch RNG after ``set_seed`` is reserved for ``StandardPolicy``
    initialization and policy action noise so it can match across trainers;
    bootstrap / HMM / critic init should use separate generators seeded here.
    """
    x = int(base_seed) & 0x7FFFFFFF
    if x == 0:
        x = 1
    for tag in tags:
        for ch in tag:
            x = (x * 0x01000193 ^ ord(ch)) & 0xFFFFFFFF
    x &= 0x7FFFFFFF
    return x if x != 0 else 1


def save_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def clamp_terminal_utility(terminal_utility: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Clamp terminal utility for policy / critic targets (numerical stability)."""
    return torch.clamp(
        terminal_utility,
        min=-float(cfg.terminal_utility_clip),
        max=float(cfg.terminal_utility_clip),
    )


def utility(wealth: torch.Tensor, cfg: Config) -> torch.Tensor:
    """CRRA: U(w) = w^gamma / gamma. Default gamma = -1 gives U(w) = -1/w."""
    wealth = torch.clamp(wealth, min=cfg.wealth_floor)
    gamma = cfg.crra_gamma
    if abs(gamma) < 1e-12:
        raise ValueError("crra_gamma must be non-zero for U(w)=w^gamma/gamma.")
    return wealth.pow(gamma) / gamma


def download_weekly_returns(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    interval: str = "1wk",
) -> pd.DataFrame:
    data = yf.download(
        list(tickers),
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError("No data downloaded. Check tickers/date range/network access.")

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if "Close" in level0:
            prices = data["Close"].copy()
        else:
            raise RuntimeError("Could not find 'Close' in yfinance output.")
    else:
        if len(tickers) > 1:
            raise RuntimeError(
                "yfinance returned a flat (non-MultiIndex) panel for multiple tickers; "
                "expected column MultiIndex. Check yfinance version or download tickers separately."
            )
        if "Close" not in data.columns:
            raise RuntimeError("Could not find 'Close' in yfinance output (flat columns).")
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.sort_index().dropna(axis=0, how="any").dropna(axis=1, how="any")
    missing = sorted(set(tickers) - set(prices.columns.tolist()))
    if missing:
        raise RuntimeError(f"Missing tickers after cleaning: {missing}")

    returns_df = prices.pct_change().dropna(how="any")
    if returns_df.empty:
        raise RuntimeError("Return matrix is empty after cleaning.")
    return returns_df[list(tickers)]


def build_market_env_policy_from_returns(
    returns_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[EmpiricalReturnMarket, PortfolioEnv, StandardPolicy]:
    market = EmpiricalReturnMarket(returns_df, cfg)
    env = PortfolioEnv(market, cfg)
    state_dim = market.n_assets + 2
    action_dim = market.n_assets
    policy = StandardPolicy(state_dim, action_dim, cfg.hidden_size, cfg).to(cfg.device, dtype=cfg.dtype)
    return market, env, policy


def build_market_env_policy(cfg: Config) -> Tuple[pd.DataFrame, EmpiricalReturnMarket, PortfolioEnv, StandardPolicy]:
    returns_df = download_weekly_returns(cfg.tickers, cfg.start_date, cfg.end_date, cfg.interval)
    market, env, policy = build_market_env_policy_from_returns(returns_df, cfg)
    return returns_df, market, env, policy


def collect_reinforce_episode(env: PortfolioEnv, policy: StandardPolicy) -> Episode:
    state = env.reset()
    log_probs: List[torch.Tensor] = []
    done = False

    while not done:
        action, log_prob = policy.sample_action(state)
        next_state, _, done = env.step(action)
        log_probs.append(log_prob)
        state = next_state.detach()

    terminal_wealth = env.terminal_wealth().detach()
    terminal_utility = clamp_terminal_utility(
        utility(terminal_wealth, env.cfg).detach(), env.cfg
    )
    return Episode(
        log_probs=log_probs,
        terminal_wealth=terminal_wealth,
        terminal_utility=terminal_utility,
    )


def flatten_gradients(model: nn.Module) -> np.ndarray:
    grads: List[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.detach().reshape(-1))
    return torch.cat(grads).detach().cpu().numpy().astype(np.float64)


def compute_gradient_estimate_from_episodes(
    episodes: List[Episode],
    model: nn.Module,
    cfg: Config,
) -> Tuple[torch.Tensor, float, float, float]:
    episode_losses: List[torch.Tensor] = []
    terminal_utilities = [float(ep.terminal_utility.item()) for ep in episodes]
    terminal_wealths = [float(ep.terminal_wealth.item()) for ep in episodes]

    for ep in episodes:
        log_prob_sum = torch.stack(ep.log_probs).sum()
        episode_losses.append(-log_prob_sum * ep.terminal_utility)

    loss = torch.stack(episode_losses).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.actor_grad_clip)
    return (
        loss.detach(),
        float(np.mean(terminal_utilities)),
        float(np.std(terminal_utilities, ddof=0)),
        float(np.mean(terminal_wealths)),
    )


def estimate_gradient_replicates(
    policy_state_dict: Dict[str, torch.Tensor],
    returns_df: pd.DataFrame,
    cfg: Config,
    n_repeats: int,
) -> Dict[str, object]:
    checkpoint_cfg = Config(**asdict(cfg))
    checkpoint_cfg.plot = False
    checkpoint_cfg.save_best_model = False

    _, env, policy = build_market_env_policy_from_returns(returns_df, checkpoint_cfg)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    grad_vectors: List[np.ndarray] = []
    losses: List[float] = []
    batch_util_means: List[float] = []
    batch_util_stds: List[float] = []
    batch_wealth_means: List[float] = []

    for _ in range(n_repeats):
        episodes = [collect_reinforce_episode(env, policy) for _ in range(checkpoint_cfg.batch_size)]
        loss, util_mean, util_std, wealth_mean = compute_gradient_estimate_from_episodes(
            episodes, policy, checkpoint_cfg
        )
        grad_vectors.append(flatten_gradients(policy))
        losses.append(float(loss.item()))
        batch_util_means.append(util_mean)
        batch_util_stds.append(util_std)
        batch_wealth_means.append(wealth_mean)

    grad_matrix = np.stack(grad_vectors, axis=0)
    grad_mean = grad_matrix.mean(axis=0)
    centered = grad_matrix - grad_mean[None, :]
    per_component_var_mean = float(grad_matrix.var(axis=0, ddof=1).mean()) if n_repeats > 1 else 0.0
    estimator_variance_l2 = float(np.mean(np.sum(centered ** 2, axis=1))) if n_repeats > 1 else 0.0
    grad_norms = np.linalg.norm(grad_matrix, axis=1)
    grad_mean_norm = float(np.linalg.norm(grad_mean))
    grad_norm_mean = float(np.mean(grad_norms))
    grad_norm_std = float(np.std(grad_norms, ddof=1)) if n_repeats > 1 else 0.0
    snr = grad_mean_norm ** 2 / estimator_variance_l2 if estimator_variance_l2 > 0 else np.nan

    return {
        "grad_matrix": grad_matrix,
        "losses": np.asarray(losses, dtype=np.float64),
        "batch_util_means": np.asarray(batch_util_means, dtype=np.float64),
        "batch_util_stds": np.asarray(batch_util_stds, dtype=np.float64),
        "batch_wealth_means": np.asarray(batch_wealth_means, dtype=np.float64),
        "summary": {
            "per_component_var_mean": per_component_var_mean,
            "estimator_variance_l2": estimator_variance_l2,
            "grad_mean_norm": grad_mean_norm,
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_std": grad_norm_std,
            "snr": float(snr),
            "loss_mean": float(np.mean(losses)),
            "utility_mean": float(np.mean(batch_util_means)),
            "utility_std_mean": float(np.mean(batch_util_stds)),
            "wealth_mean": float(np.mean(batch_wealth_means)),
        },
    }


@torch.no_grad()
def evaluate_policy(
    env: PortfolioEnv,
    policy: StandardPolicy,
    cfg: Config,
    n_paths: Optional[int] = None,
) -> Tuple[float, float]:
    n_eval = n_paths if n_paths is not None else cfg.eval_paths
    wealths: List[float] = []
    utilities: List[float] = []

    for _ in range(n_eval):
        state = env.reset()
        done = False
        while not done:
            action = policy.mean_action(state)
            state, _, done = env.step(action)
        wealth = float(env.terminal_wealth().item())
        util = float(utility(torch.tensor(wealth, dtype=cfg.dtype, device=cfg.device), cfg).item())
        wealths.append(wealth)
        utilities.append(util)

    return float(np.mean(wealths)), float(np.mean(utilities))


def maybe_save_plot(
    iteration_metrics: Sequence[Dict[str, float]],
    eval_steps: Sequence[int],
    eval_curve: Sequence[float],
    title: str,
    filename: str,
    cfg: Config,
) -> None:
    if not cfg.plot:
        return

    import matplotlib.pyplot as plt

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_curve = [row["train_avg_utility"] for row in iteration_metrics]
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(train_curve) + 1), train_curve, label="Train avg utility")
    if len(eval_steps) > 0:
        plt.plot(eval_steps, eval_curve, marker="o", label="Eval avg utility")
    plt.xlabel("Iteration")
    plt.ylabel("Utility")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_dir / filename), dpi=150)
    plt.close()


def config_to_serializable_dict(cfg: Config) -> Dict[str, object]:
    out = asdict(cfg)
    out["dtype"] = str(cfg.dtype)
    out["tickers"] = list(cfg.tickers)
    return out


def save_checkpoint(
    path: Path,
    policy: StandardPolicy,
    cfg: Config,
    result: TrainingResult,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    payload: Dict[str, object] = {
        "policy_state_dict": policy.state_dict(),
        "config": config_to_serializable_dict(cfg),
        "selected_tickers": list(cfg.tickers),
        "iteration_metrics": result.iteration_metrics,
        "eval_curve": result.eval_curve,
        "eval_steps": result.eval_steps,
        "best_eval_utility": result.best_eval_utility,
        "best_eval_wealth": result.best_eval_wealth,
    }
    if extra is not None:
        payload["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def print_run_header(method_name: str, cfg: Config) -> None:
    print(f"Method: {method_name}")
    print(f"Device: {cfg.device}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print(f"Universe size: {len(cfg.tickers)} assets")
    print(f"Selected tickers: {list(cfg.tickers)}")
    print(f"Batch size: {cfg.batch_size} | Iterations: {cfg.n_iterations} | Horizon: {cfg.horizon}")
    print(f"Gradient checkpoints: {list(cfg.gradient_checkpoints)} | Repeats per checkpoint: {cfg.gradient_repeats}")


def train_vanilla_reinforce(cfg: Config) -> Tuple[nn.Module, TrainingResult]:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / cfg.config_name, "w", encoding="utf-8") as f:
        json.dump(config_to_serializable_dict(cfg), f, indent=2)

    returns_df, _, env, policy = build_market_env_policy(cfg)
    returns_df.to_csv(out_dir / cfg.returns_used_name)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    iteration_metrics: List[Dict[str, float]] = []
    eval_curve: List[float] = []
    eval_steps: List[int] = []
    gradient_checkpoint_rows: List[Dict[str, float]] = []
    full_training_gradients: List[np.ndarray] = []
    best_eval_utility: Optional[float] = None
    best_eval_wealth: Optional[float] = None

    result = TrainingResult(
        iteration_metrics=iteration_metrics,
        eval_curve=eval_curve,
        eval_steps=eval_steps,
        returns_df=returns_df,
        best_eval_utility=None,
        best_eval_wealth=None,
    )

    checkpoint_iterations = sorted({i for i in cfg.gradient_checkpoints if 1 <= i <= cfg.n_iterations})

    for iteration in range(1, cfg.n_iterations + 1):
        print(f"Starting iteration {iteration}/{cfg.n_iterations}...", flush=True)

        episodes = [collect_reinforce_episode(env, policy) for _ in range(cfg.batch_size)]

        print(f"Finished collecting episodes for iteration {iteration}. Updating policy...", flush=True)

        loss, avg_train_u, std_train_u, avg_train_w = compute_gradient_estimate_from_episodes(
            episodes, policy, cfg
        )
        grad_vector = flatten_gradients(policy)
        grad_norm = float(np.linalg.norm(grad_vector))
        grad_sq_norm = float(np.dot(grad_vector, grad_vector))
        if cfg.save_full_training_gradients:
            full_training_gradients.append(grad_vector)

        optimizer.step()

        row: Dict[str, float] = {
            "iteration": float(iteration),
            "loss": float(loss.item()),
            "train_avg_utility": avg_train_u,
            "train_std_utility": std_train_u,
            "train_avg_wealth": avg_train_w,
            "gradient_norm": grad_norm,
            "gradient_sq_norm": grad_sq_norm,
            "eval_avg_utility": np.nan,
            "eval_avg_wealth": np.nan,
        }
        iteration_metrics.append(row)

        if iteration % cfg.eval_every == 0:
            print(f"Running evaluation at iteration {iteration}...", flush=True)
            eval_mean_w, eval_mean_u = evaluate_policy(env, policy, cfg)
            eval_curve.append(eval_mean_u)
            eval_steps.append(iteration)
            row["eval_avg_utility"] = eval_mean_u
            row["eval_avg_wealth"] = eval_mean_w

            print(
                f"Iteration {iteration:4d}/{cfg.n_iterations} | "
                f"train avg U = {avg_train_u: .6f} | "
                f"eval avg U = {eval_mean_u: .6f} | "
                f"eval avg W = {eval_mean_w: .6f} | "
                f"grad norm = {grad_norm: .6f}",
                flush=True,
            )

            if best_eval_utility is None or eval_mean_u > best_eval_utility:
                best_eval_utility = eval_mean_u
                best_eval_wealth = eval_mean_w
                result.best_eval_utility = best_eval_utility
                result.best_eval_wealth = best_eval_wealth
                if cfg.save_best_model:
                    save_checkpoint(
                        out_dir / cfg.best_model_name,
                        policy,
                        cfg,
                        result,
                        extra={"best_iteration": iteration},
                    )

        if iteration in checkpoint_iterations:
            print(f"Estimating repeated gradient statistics at iteration {iteration}...", flush=True)
            rng_state = save_rng_state()
            checkpoint_data = estimate_gradient_replicates(
                policy_state_dict={k: v.detach().clone() for k, v in policy.state_dict().items()},
                returns_df=returns_df,
                cfg=cfg,
                n_repeats=cfg.gradient_repeats,
            )
            restore_rng_state(rng_state)

            grad_npz_path = out_dir / f"vanilla_pg_gradient_checkpoint_iter_{iteration:04d}.npz"
            np.savez_compressed(
                grad_npz_path,
                grad_matrix=checkpoint_data["grad_matrix"],
                losses=checkpoint_data["losses"],
                batch_util_means=checkpoint_data["batch_util_means"],
                batch_util_stds=checkpoint_data["batch_util_stds"],
                batch_wealth_means=checkpoint_data["batch_wealth_means"],
            )

            summary = checkpoint_data["summary"]
            gradient_checkpoint_rows.append(
                {
                    "iteration": float(iteration),
                    "per_component_var_mean": summary["per_component_var_mean"],
                    "estimator_variance_l2": summary["estimator_variance_l2"],
                    "grad_mean_norm": summary["grad_mean_norm"],
                    "grad_norm_mean": summary["grad_norm_mean"],
                    "grad_norm_std": summary["grad_norm_std"],
                    "snr": summary["snr"],
                    "loss_mean": summary["loss_mean"],
                    "utility_mean": summary["utility_mean"],
                    "utility_std_mean": summary["utility_std_mean"],
                    "wealth_mean": summary["wealth_mean"],
                    "repeats": float(cfg.gradient_repeats),
                }
            )

        pd.DataFrame(iteration_metrics).to_csv(out_dir / cfg.iteration_metrics_name, index=False)
        if gradient_checkpoint_rows:
            pd.DataFrame(gradient_checkpoint_rows).to_csv(out_dir / cfg.gradient_summary_name, index=False)

    result.best_eval_utility = best_eval_utility
    result.best_eval_wealth = best_eval_wealth

    if cfg.save_full_training_gradients and full_training_gradients:
        np.savez_compressed(
            out_dir / cfg.training_gradient_matrix_name,
            gradient_matrix=np.stack(full_training_gradients, axis=0),
        )

    maybe_save_plot(
        iteration_metrics,
        eval_steps,
        eval_curve,
        "Vanilla REINFORCE (10-stock trade-based benchmark)",
        cfg.training_plot_name,
        cfg,
    )

    save_checkpoint(
        out_dir / cfg.final_model_name,
        policy,
        cfg,
        result,
        extra={"final_iteration": cfg.n_iterations},
    )
    return policy, result


if __name__ == "__main__":
    cfg = Config()
    print_run_header("Vanilla REINFORCE", cfg)
    policy, result = train_vanilla_reinforce(cfg)

    print("\nCompleted training.")
    print(f"Universe size: {len(cfg.tickers)} assets")
    print(f"Selected tickers: {list(cfg.tickers)}")
    print(f"Final training utility: {result.iteration_metrics[-1]['train_avg_utility']: .6f}")
    if result.eval_curve:
        print(f"Final evaluation utility: {result.eval_curve[-1]: .6f}")
    if result.best_eval_utility is not None:
        print(f"Best evaluation utility: {result.best_eval_utility: .6f}")
        print(f"Best evaluation wealth: {result.best_eval_wealth: .6f}")
    print(f"Saved real returns to: {Path(cfg.output_dir) / cfg.returns_used_name}")
    print(f"Saved iteration metrics to: {Path(cfg.output_dir) / cfg.iteration_metrics_name}")
    print(f"Saved gradient summary to: {Path(cfg.output_dir) / cfg.gradient_summary_name}")
    if cfg.save_full_training_gradients:
        print(f"Saved training gradient matrix to: {Path(cfg.output_dir) / cfg.training_gradient_matrix_name}")
    print(f"Saved final model to: {Path(cfg.output_dir) / cfg.final_model_name}")
    if cfg.save_best_model:
        print(f"Saved best model to: {Path(cfg.output_dir) / cfg.best_model_name}")
