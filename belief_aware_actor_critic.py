from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import vanilla_pg as vanilla


class BeliefValueFunction(nn.Module):
    """Critic V_phi(x_t, c_t, time_remaining, belief_t)."""

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        *,
        init_generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.reset_parameters(init_generator)

    def reset_parameters(self, init_generator: Optional[torch.Generator] = None) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_generator is not None:
                    nn.init.xavier_normal_(module.weight, generator=init_generator)
                else:
                    nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class EpisodeWithBeliefStates(vanilla.Episode):
    def __init__(
        self,
        log_probs: List[torch.Tensor],
        terminal_wealth: torch.Tensor,
        terminal_utility: torch.Tensor,
        critic_states: List[torch.Tensor],
    ):
        super().__init__(log_probs=log_probs, terminal_wealth=terminal_wealth, terminal_utility=terminal_utility)
        self.critic_states = critic_states


class EmpiricalBeliefMarket:
    """
    Same return process as ``vanilla.EmpiricalReturnMarket`` (bootstrap rows from history).

    HMM parameters ``mu``, ``cov`` from ``fit_two_regime_proxy_hmm`` are used **only** to
    evaluate Gaussian log-likelihoods and update a causal belief over regime 1 vs 0 (carry prior).
    The transition matrix is **not** used for belief in this class (see ``belief_next_mode``).
    """

    def __init__(
        self,
        empirical: vanilla.EmpiricalReturnMarket,
        mu: np.ndarray,
        cov: np.ndarray,
        initial_belief: float,
        cfg: vanilla.Config,
        *,
        belief_temperature: float = 1.0,
        belief_next_mode: str = "carry",
    ):
        self._empirical = empirical
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.n_assets = empirical.n_assets
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
        r = self._empirical.sample_returns()
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


class HiddenRegimeGaussianMarket:
    """
    Synthetic two-regime HMM: Markov hidden state and Gaussian returns (not bootstrap).

    Main belief-aware training now uses this market so trajectories are sampled by:
      - hidden regime transition dynamics from fitted ``transition``
      - regime-specific Gaussian emissions from fitted ``mu`` / ``cov``.
    Hindsight training uses analogous regime-dependent simulation.

    Belief is filtered causally:
      posterior_t = P(S_t=1 | info_t); carry prior for next step (no transition in filter).
    """

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        transition: np.ndarray,
        initial_belief: float,
        cfg: vanilla.Config,
        *,
        belief_temperature: float = 1.0,
        belief_next_mode: str = "carry",
        static_belief: float = 0.5,
        return_clip: float = 0.95,
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.mu = torch.tensor(mu, dtype=cfg.dtype, device=cfg.device)
        self.cov = torch.tensor(cov, dtype=cfg.dtype, device=cfg.device)
        self.transition = torch.tensor(transition, dtype=cfg.dtype, device=cfg.device)
        self.initial_belief = float(np.clip(initial_belief, 1e-6, 1.0 - 1e-6))
        self.belief_temperature = float(max(belief_temperature, 1e-6))
        self.belief_next_mode = str(belief_next_mode)
        self.static_belief = float(np.clip(static_belief, 0.0, 1.0))
        self.return_clip = float(max(return_clip, 0.0))

        self.n_assets = int(self.mu.shape[1])

        self._market_rng = torch.Generator(device=torch.device("cpu"))
        self._market_rng.manual_seed(
            vanilla.derived_torch_seed(cfg.seed, type(self).__qualname__, "market")
        )

        # Precompute Gaussian terms for fast likelihood evaluation.
        self.chol = torch.linalg.cholesky(self.cov)
        self.inv_cov = torch.cholesky_inverse(self.chol)
        self.logdet = 2.0 * torch.log(torch.diagonal(self.chol, dim1=1, dim2=2)).sum(dim=1)

        self.hidden_state: int = 0
        self.belief_prior: float = self.initial_belief
        self.belief_posterior: float = self.initial_belief

    def _bernoulli1(self, p: float) -> int:
        u = torch.rand((), generator=self._market_rng, dtype=torch.float64, device=torch.device("cpu"))
        return int(float(u.item()) < float(p))

    def _eps_shock(self) -> torch.Tensor:
        eps = torch.randn(
            (self.n_assets,),
            generator=self._market_rng,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        return eps.to(device=self.device, dtype=self.dtype)

    def reset(self) -> None:
        self.belief_prior = self.initial_belief
        self.belief_posterior = self.initial_belief
        self.hidden_state = self._bernoulli1(self.initial_belief)

    def _prior_from_posterior(self, post_p1: float) -> float:
        # Belief tracking uses pure carry (no transition-prediction step).
        return float(np.clip(post_p1, 0.0, 1.0))

    def _log_likelihood_per_regime(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(0) - self.mu  # [2, d]
        quad = torch.einsum("ki,kij,kj->k", diff, self.inv_cov, diff)
        d = r.shape[0]
        return -0.5 * (d * math.log(2.0 * math.pi) + self.logdet + quad)

    def sample_returns(self) -> torch.Tensor:
        # 1) Simulate hidden regime transition.
        p_to_one = float(self.transition[self.hidden_state, 1].item())
        self.hidden_state = self._bernoulli1(p_to_one)

        # 2) Sample returns from the new regime.
        eps = self._eps_shock()
        r = self.mu[self.hidden_state] + self.chol[self.hidden_state] @ eps
        r = torch.clamp(r, min=-self.return_clip, max=self.return_clip)

        # 3) Causal Bayes filter update with current observation.
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


BeliefMarket = Union[EmpiricalBeliefMarket, HiddenRegimeGaussianMarket]


class BeliefPortfolioEnv(vanilla.PortfolioEnv):
    def __init__(self, market: BeliefMarket, cfg: vanilla.Config):
        self.market = market
        self.cfg = cfg
        self.n_assets = market.n_assets
        self.weekly_rf = cfg.weekly_rf()
        self.reset()

    def reset(self) -> torch.Tensor:
        self.market.reset()
        self.t = 0
        self.x = torch.zeros(self.n_assets, dtype=self.cfg.dtype, device=self.cfg.device)
        self.c = torch.tensor(self.cfg.initial_wealth, dtype=self.cfg.dtype, device=self.cfg.device)
        return self.state()

    def state(self) -> torch.Tensor:
        # Policy state unchanged to keep comparison fair with vanilla/LOO/value-baseline.
        time_remaining = torch.tensor(
            [(self.cfg.horizon - self.t) / self.cfg.horizon],
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        return torch.cat([self.x, self.c.view(1), time_remaining])

    def critic_state(self) -> torch.Tensor:
        # Belief-aware critic state: [x_t, c_t, time_remaining, belief_prior_t]
        belief = torch.tensor([self.market.belief_prior], dtype=self.cfg.dtype, device=self.cfg.device)
        return torch.cat([self.state(), belief])


def _regularized_cov(x: np.ndarray, jitter: float) -> np.ndarray:
    cov = np.cov(x, rowvar=False, ddof=0)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    cov += float(jitter) * np.eye(cov.shape[0], dtype=np.float64)
    return cov


def fit_two_regime_proxy_hmm(
    returns_df: pd.DataFrame,
    *,
    jitter: float = 1e-6,
    min_regime_obs: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Lightweight HMM proxy fit:
      - infer 2-state sequence from median split of equal-weight market return
      - estimate transition matrix and Gaussian emissions per state
    """
    x = returns_df.to_numpy(dtype=np.float64)
    mkt = x.mean(axis=1)
    states = (mkt >= np.median(mkt)).astype(np.int64)

    # Ensure both regimes are represented.
    cnt0 = int((states == 0).sum())
    cnt1 = int((states == 1).sum())
    if cnt0 < min_regime_obs or cnt1 < min_regime_obs:
        q = np.quantile(mkt, 0.7)
        states = (mkt >= q).astype(np.int64)
        cnt0 = int((states == 0).sum())
        cnt1 = int((states == 1).sum())
        if cnt0 < min_regime_obs or cnt1 < min_regime_obs:
            half = len(states) // 2
            order = np.argsort(mkt)
            states = np.zeros_like(states)
            states[order[half:]] = 1

    mu = np.zeros((2, x.shape[1]), dtype=np.float64)
    cov = np.zeros((2, x.shape[1], x.shape[1]), dtype=np.float64)
    for k in (0, 1):
        xk = x[states == k]
        if xk.shape[0] == 0:
            xk = x
        mu[k] = xk.mean(axis=0)
        cov[k] = _regularized_cov(xk, jitter=jitter)

    counts = np.ones((2, 2), dtype=np.float64)  # Laplace smoothing.
    for t in range(len(states) - 1):
        counts[states[t], states[t + 1]] += 1.0
    transition = counts / counts.sum(axis=1, keepdims=True)
    initial_belief = float(np.clip(states.mean(), 1e-6, 1.0 - 1e-6))

    return {
        "mu": mu,
        "cov": cov,
        "transition": transition,
        "initial_belief": np.array([initial_belief], dtype=np.float64),
    }


def build_belief_market_env_policy_from_returns(
    returns_df: pd.DataFrame,
    cfg: vanilla.Config,
) -> Tuple[HiddenRegimeGaussianMarket, BeliefPortfolioEnv, vanilla.StandardPolicy, Dict[str, np.ndarray]]:
    regime = fit_two_regime_proxy_hmm(returns_df, jitter=1e-6)
    market = HiddenRegimeGaussianMarket(
        mu=regime["mu"],
        cov=regime["cov"],
        transition=regime["transition"],
        initial_belief=float(regime["initial_belief"][0]),
        cfg=cfg,
        belief_temperature=1.0,
        belief_next_mode="carry",
    )
    env = BeliefPortfolioEnv(market, cfg)
    policy_state_dim = market.n_assets + 2
    action_dim = market.n_assets
    policy = vanilla.StandardPolicy(policy_state_dim, action_dim, cfg.hidden_size, cfg).to(cfg.device, dtype=cfg.dtype)
    return market, env, policy, regime


def build_belief_market_env_policy_from_regime(
    returns_df: pd.DataFrame,
    regime: Dict[str, np.ndarray],
    cfg: vanilla.Config,
) -> Tuple[HiddenRegimeGaussianMarket, BeliefPortfolioEnv, vanilla.StandardPolicy]:
    market = HiddenRegimeGaussianMarket(
        mu=regime["mu"],
        cov=regime["cov"],
        transition=regime["transition"],
        initial_belief=float(regime["initial_belief"][0]),
        cfg=cfg,
        belief_temperature=1.0,
        belief_next_mode="carry",
    )
    env = BeliefPortfolioEnv(market, cfg)
    policy_state_dim = market.n_assets + 2
    action_dim = market.n_assets
    policy = vanilla.StandardPolicy(policy_state_dim, action_dim, cfg.hidden_size, cfg).to(cfg.device, dtype=cfg.dtype)
    return market, env, policy


def build_belief_market_env_policy(
    cfg: vanilla.Config,
) -> Tuple[pd.DataFrame, HiddenRegimeGaussianMarket, BeliefPortfolioEnv, vanilla.StandardPolicy, Dict[str, np.ndarray]]:
    returns_df = vanilla.download_weekly_returns(cfg.tickers, cfg.start_date, cfg.end_date, cfg.interval)
    market, env, policy, regime = build_belief_market_env_policy_from_returns(returns_df, cfg)
    return returns_df, market, env, policy, regime


def make_belief_actor_critic_config(base_cfg: vanilla.Config) -> vanilla.Config:
    cfg = vanilla.Config(**asdict(base_cfg))
    cfg.gross_long_cap = 1.3
    cfg.gross_short_cap = 0.3
    cfg.output_dir = "belief_aware_ac_outputs"
    cfg.final_model_name = "belief_aware_actor_critic_policy_final.pt"
    cfg.best_model_name = "belief_aware_actor_critic_policy_best.pt"
    cfg.training_plot_name = "belief_aware_actor_critic_training_curve.png"
    cfg.iteration_metrics_name = "belief_aware_actor_critic_iteration_metrics.csv"
    cfg.gradient_summary_name = "belief_aware_actor_critic_gradient_checkpoint_summary.csv"
    cfg.training_gradient_matrix_name = "belief_aware_actor_critic_training_gradients.npz"
    cfg.config_name = "belief_aware_actor_critic_config.json"
    cfg.returns_used_name = "belief_aware_actor_critic_real_returns.csv"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def collect_belief_actor_critic_episode(
    env: BeliefPortfolioEnv, policy: vanilla.StandardPolicy
) -> EpisodeWithBeliefStates:
    state = env.reset()
    log_probs: List[torch.Tensor] = []
    critic_states: List[torch.Tensor] = []
    done = False

    while not done:
        critic_states.append(env.critic_state().clone())
        action, log_prob = policy.sample_action(state)
        next_state, _, done = env.step(action)
        log_probs.append(log_prob)
        state = next_state.detach()

    terminal_wealth = env.terminal_wealth().detach()
    terminal_utility = vanilla.clamp_terminal_utility(
        vanilla.utility(terminal_wealth, env.cfg).detach(), env.cfg
    )
    return EpisodeWithBeliefStates(
        log_probs=log_probs,
        terminal_wealth=terminal_wealth,
        terminal_utility=terminal_utility,
        critic_states=critic_states,
    )


def train_belief_value_function(
    episodes: List[EpisodeWithBeliefStates],
    value_net: BeliefValueFunction,
    value_optimizer: optim.Optimizer,
    cfg: vanilla.Config,
) -> float:
    states: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for ep in episodes:
        for s in ep.critic_states:
            states.append(s)
            targets.append(ep.terminal_utility.to(dtype=s.dtype, device=s.device))

    if len(states) == 0:
        raise RuntimeError("No states collected for belief value-function training.")

    state_batch = torch.stack(states)
    target_batch = torch.stack(targets)
    pred = value_net(state_batch)
    value_loss = ((pred - target_batch) ** 2).mean()

    value_optimizer.zero_grad(set_to_none=True)
    value_loss.backward()
    nn.utils.clip_grad_norm_(value_net.parameters(), cfg.critic_grad_clip)
    value_optimizer.step()
    return float(value_loss.detach().item())


def compute_belief_actor_critic_gradient_estimate(
    episodes: List[EpisodeWithBeliefStates],
    policy: nn.Module,
    value_net: BeliefValueFunction,
    cfg: vanilla.Config,
) -> Tuple[torch.Tensor, float, float, float]:
    episode_losses: List[torch.Tensor] = []
    terminal_utilities = [float(ep.terminal_utility.item()) for ep in episodes]
    terminal_wealths = [float(ep.terminal_wealth.item()) for ep in episodes]

    for ep in episodes:
        if len(ep.log_probs) == 0:
            raise RuntimeError("Encountered episode with zero actions. Check horizon/environment setup.")
        if len(ep.critic_states) != len(ep.log_probs):
            raise RuntimeError("Episode critic_states and log_probs lengths do not match.")

        step_losses: List[torch.Tensor] = []
        for t in range(len(ep.log_probs)):
            advantage = ep.terminal_utility - value_net(ep.critic_states[t]).detach()
            step_losses.append(-ep.log_probs[t] * advantage)
        episode_losses.append(torch.stack(step_losses).sum())

    loss = torch.stack(episode_losses).mean()
    policy.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), cfg.actor_grad_clip)
    return (
        loss.detach(),
        float(np.mean(terminal_utilities)),
        float(np.std(terminal_utilities, ddof=0)),
        float(np.mean(terminal_wealths)),
    )


def estimate_belief_actor_critic_gradient_replicates(
    policy_state_dict: Dict[str, torch.Tensor],
    value_state_dict: Dict[str, torch.Tensor],
    returns_df: pd.DataFrame,
    regime: Dict[str, np.ndarray],
    cfg: vanilla.Config,
    n_repeats: int,
) -> Dict[str, object]:
    checkpoint_cfg = vanilla.Config(**asdict(cfg))
    checkpoint_cfg.plot = False
    checkpoint_cfg.save_best_model = False

    _, env, policy = build_belief_market_env_policy_from_regime(returns_df, regime, checkpoint_cfg)
    critic_state_dim = env.n_assets + 3
    critic_gen = torch.Generator(device=torch.device("cpu"))
    critic_gen.manual_seed(
        vanilla.derived_torch_seed(checkpoint_cfg.seed, "belief_aware_actor_critic", "BeliefValueFunction")
    )
    value_net = BeliefValueFunction(
        state_dim=critic_state_dim,
        hidden_size=checkpoint_cfg.hidden_size,
        init_generator=critic_gen,
    ).to(checkpoint_cfg.device, dtype=checkpoint_cfg.dtype)
    value_net.load_state_dict(value_state_dict)
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    value_net.eval()

    grad_vectors: List[np.ndarray] = []
    losses: List[float] = []
    batch_util_means: List[float] = []
    batch_util_stds: List[float] = []
    batch_wealth_means: List[float] = []

    for _ in range(n_repeats):
        episodes = [collect_belief_actor_critic_episode(env, policy) for _ in range(checkpoint_cfg.batch_size)]
        loss, util_mean, util_std, wealth_mean = compute_belief_actor_critic_gradient_estimate(
            episodes=episodes,
            policy=policy,
            value_net=value_net,
            cfg=checkpoint_cfg,
        )
        grad_vectors.append(vanilla.flatten_gradients(policy))
        losses.append(float(loss.item()))
        batch_util_means.append(util_mean)
        batch_util_stds.append(util_std)
        batch_wealth_means.append(wealth_mean)

    grad_matrix = np.stack(grad_vectors, axis=0)
    grad_mean = grad_matrix.mean(axis=0)
    centered = grad_matrix - grad_mean[None, :]
    per_component_var_mean = float(grad_matrix.var(axis=0, ddof=1).mean()) if n_repeats > 1 else 0.0
    estimator_variance_l2 = float(np.mean(np.sum(centered**2, axis=1))) if n_repeats > 1 else 0.0
    grad_norms = np.linalg.norm(grad_matrix, axis=1)
    grad_mean_norm = float(np.linalg.norm(grad_mean))
    grad_norm_mean = float(np.mean(grad_norms))
    grad_norm_std = float(np.std(grad_norms, ddof=1)) if n_repeats > 1 else 0.0
    snr = grad_mean_norm**2 / estimator_variance_l2 if estimator_variance_l2 > 0 else np.nan

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


def train_belief_actor_critic(
    cfg: vanilla.Config,
) -> Tuple[nn.Module, BeliefValueFunction, vanilla.TrainingResult]:
    vanilla.set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / cfg.config_name, "w", encoding="utf-8") as f:
        json.dump(vanilla.config_to_serializable_dict(cfg), f, indent=2)

    returns_df, _, env, policy, regime = build_belief_market_env_policy(cfg)
    returns_df.to_csv(out_dir / cfg.returns_used_name)

    critic_state_dim = env.n_assets + 3  # [x, c, time, belief]
    critic_gen = torch.Generator(device=torch.device("cpu"))
    critic_gen.manual_seed(vanilla.derived_torch_seed(cfg.seed, "belief_aware_actor_critic", "BeliefValueFunction"))
    value_net = BeliefValueFunction(
        state_dim=critic_state_dim, hidden_size=cfg.hidden_size, init_generator=critic_gen
    ).to(cfg.device, dtype=cfg.dtype)

    actor_optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=cfg.learning_rate)

    iteration_metrics: List[Dict[str, float]] = []
    eval_curve: List[float] = []
    eval_steps: List[int] = []
    gradient_checkpoint_rows: List[Dict[str, float]] = []
    full_training_gradients: List[np.ndarray] = []
    best_eval_utility: Optional[float] = None
    best_eval_wealth: Optional[float] = None

    result = vanilla.TrainingResult(
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
        episodes = [collect_belief_actor_critic_episode(env, policy) for _ in range(cfg.batch_size)]
        print(f"Finished collecting episodes for iteration {iteration}. Updating critic and policy...", flush=True)

        value_loss = train_belief_value_function(
            episodes=episodes,
            value_net=value_net,
            value_optimizer=value_optimizer,
            cfg=cfg,
        )
        actor_loss, avg_train_u, std_train_u, avg_train_w = compute_belief_actor_critic_gradient_estimate(
            episodes=episodes,
            policy=policy,
            value_net=value_net,
            cfg=cfg,
        )
        grad_vector = vanilla.flatten_gradients(policy)
        grad_norm = float(np.linalg.norm(grad_vector))
        grad_sq_norm = float(np.dot(grad_vector, grad_vector))
        if cfg.save_full_training_gradients:
            full_training_gradients.append(grad_vector)

        actor_optimizer.step()

        row: Dict[str, float] = {
            "iteration": float(iteration),
            "loss": float(actor_loss.item()),
            "value_loss": float(value_loss),
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
            eval_mean_w, eval_mean_u = vanilla.evaluate_policy(env, policy, cfg)
            eval_curve.append(eval_mean_u)
            eval_steps.append(iteration)
            row["eval_avg_utility"] = eval_mean_u
            row["eval_avg_wealth"] = eval_mean_w

            print(
                f"Iteration {iteration:4d}/{cfg.n_iterations} | "
                f"train avg U = {avg_train_u: .6f} | "
                f"eval avg U = {eval_mean_u: .6f} | "
                f"eval avg W = {eval_mean_w: .6f} | "
                f"grad norm = {grad_norm: .6f} | "
                f"value loss = {value_loss: .6f}",
                flush=True,
            )

            if best_eval_utility is None or eval_mean_u > best_eval_utility:
                best_eval_utility = eval_mean_u
                best_eval_wealth = eval_mean_w
                result.best_eval_utility = best_eval_utility
                result.best_eval_wealth = best_eval_wealth
                if cfg.save_best_model:
                    vanilla.save_checkpoint(
                        out_dir / cfg.best_model_name,
                        policy,
                        cfg,
                        result,
                        extra={
                            "best_iteration": iteration,
                            "value_state_dict": value_net.state_dict(),
                            "regime": {k: v.tolist() for k, v in regime.items()},
                        },
                    )

        if iteration in checkpoint_iterations:
            print(f"Estimating repeated gradient statistics at iteration {iteration}...", flush=True)
            rng_state = vanilla.save_rng_state()
            checkpoint_data = estimate_belief_actor_critic_gradient_replicates(
                policy_state_dict={k: v.detach().clone() for k, v in policy.state_dict().items()},
                value_state_dict={k: v.detach().clone() for k, v in value_net.state_dict().items()},
                returns_df=returns_df,
                regime=regime,
                cfg=cfg,
                n_repeats=cfg.gradient_repeats,
            )
            vanilla.restore_rng_state(rng_state)

            grad_npz_path = out_dir / f"belief_aware_actor_critic_gradient_checkpoint_iter_{iteration:04d}.npz"
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

    vanilla.maybe_save_plot(
        iteration_metrics,
        eval_steps,
        eval_curve,
        "Belief-aware actor–critic (learned value + belief, 10-stock benchmark)",
        cfg.training_plot_name,
        cfg,
    )

    vanilla.save_checkpoint(
        out_dir / cfg.final_model_name,
        policy,
        cfg,
        result,
        extra={
            "final_iteration": cfg.n_iterations,
            "value_state_dict": value_net.state_dict(),
            "regime": {k: v.tolist() for k, v in regime.items()},
        },
    )
    return policy, value_net, result


if __name__ == "__main__":
    base_cfg = vanilla.Config()
    cfg = make_belief_actor_critic_config(base_cfg)
    vanilla.print_run_header("Belief-aware actor–critic (bootstrap returns + filtered belief)", cfg)
    policy, value_net, result = train_belief_actor_critic(cfg)

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


# Backward-compatible aliases (older notebooks / scripts)
make_belief_value_baseline_config = make_belief_actor_critic_config
train_belief_aware_value_baseline = train_belief_actor_critic
collect_episode_with_belief_states = collect_belief_actor_critic_episode
train_value_baseline = train_belief_value_function
compute_actor_gradient_estimate_with_belief_baseline = compute_belief_actor_critic_gradient_estimate
estimate_gradient_replicates_belief_baseline = estimate_belief_actor_critic_gradient_replicates
