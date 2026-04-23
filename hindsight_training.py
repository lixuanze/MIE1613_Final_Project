from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import vanilla_pg as vanilla
import belief_aware_actor_critic as bac


# ── MAML hyperparameter defaults ──────────────────────────────────────────────

MAML_INNER_LR: float = 1e-4
MAML_INNER_STEPS: int = 1

# ── RegimeReplayMarket ────────────────────────────────────────────────────────

class RegimeReplayMarket:
    """
    Simulate a two-regime Gaussian HMM episode:
      1) sample hidden regimes from the fitted transition matrix
      2) sample returns from regime-specific Gaussian moments (mu/cov)
      3) run a causal belief update from observed returns for critic state features

    At each ``reset()``, the full episode path is pre-sampled so the critic can access
    future hidden-regime labels ``h_{t:T}`` during training.
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
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.n_assets = int(mu.shape[1])
        self.mu = torch.tensor(mu, dtype=cfg.dtype, device=cfg.device)
        self.cov = torch.tensor(cov, dtype=cfg.dtype, device=cfg.device)
        self.transition = torch.tensor(transition, dtype=cfg.dtype, device=cfg.device)
        self.initial_belief = float(np.clip(initial_belief, 1e-6, 1.0 - 1e-6))
        self.belief_temperature = float(max(belief_temperature, 1e-6))
        self.belief_next_mode = str(belief_next_mode)

        self.chol = torch.linalg.cholesky(self.cov)
        self.inv_cov = torch.cholesky_inverse(self.chol)
        self.logdet = 2.0 * torch.log(torch.diagonal(self.chol, dim1=1, dim2=2)).sum(dim=1)
        self._market_rng = torch.Generator(device=torch.device("cpu"))
        self._market_rng.manual_seed(
            vanilla.derived_torch_seed(cfg.seed, type(self).__qualname__, "market")
        )

        self._episode_returns: List[torch.Tensor] = []
        self._episode_posterior: List[float] = []
        self._episode_regime_labels: List[int] = []
        self._fixed_episode_path: Optional[
            Tuple[List[torch.Tensor], List[float], List[int]]
        ] = None
        self._step: int = 0
        self.hidden_state: int = 0
        self.belief_prior: float = self.initial_belief
        self.belief_posterior: float = self.initial_belief

    def _prior_from_posterior(self, post_p1: float) -> float:
        if self.belief_next_mode == "carry":
            return float(np.clip(post_p1, 0.0, 1.0))
        raise RuntimeError(f"Unknown belief_next_mode: {self.belief_next_mode!r}")

    def _bernoulli1(self, p: float) -> int:
        u = torch.rand(
            (),
            generator=self._market_rng,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        return int(float(u.item()) < float(p))

    def _eps_shock(self) -> torch.Tensor:
        eps = torch.randn(
            (self.n_assets,),
            generator=self._market_rng,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        return eps.to(device=self.device, dtype=self.dtype)

    def _log_likelihood_per_regime(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(0) - self.mu
        quad = torch.einsum("ki,kij,kj->k", diff, self.inv_cov, diff)
        d = r.shape[0]
        return -0.5 * (d * math.log(2.0 * math.pi) + self.logdet + quad)

    def _update_belief_from_return(self, r: torch.Tensor) -> None:
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
        self.belief_prior = float(
            np.clip(self._prior_from_posterior(self.belief_posterior), 0.0, 1.0)
        )

    def _sample_episode_path(
        self,
    ) -> Tuple[List[torch.Tensor], List[float], List[int]]:
        """Sample one hidden-regime path and corresponding returns/beliefs."""
        ep_returns: List[torch.Tensor] = []
        ep_posterior: List[float] = []
        ep_regimes: List[int] = []

        self.belief_prior = self.initial_belief
        self.belief_posterior = self.initial_belief
        self.hidden_state = self._bernoulli1(self.initial_belief)
        for _ in range(self.cfg.horizon):
            p_to_one = float(self.transition[self.hidden_state, 1].item())
            self.hidden_state = self._bernoulli1(p_to_one)
            r = self.mu[self.hidden_state] + self.chol[self.hidden_state] @ self._eps_shock()
            self._update_belief_from_return(r)
            ep_returns.append(r.clone())
            ep_posterior.append(self.belief_posterior)
            ep_regimes.append(self.hidden_state)
        return ep_returns, ep_posterior, ep_regimes

    def sample_new_episode_path(self) -> Tuple[List[torch.Tensor], List[float], List[int]]:
        """Public helper for outer loop: draw one exogenous sequence z for this iteration."""
        return self._sample_episode_path()

    def set_fixed_episode_path(
        self, episode_path: Tuple[List[torch.Tensor], List[float], List[int]]
    ) -> None:
        """Replay the same exogenous sequence on each reset (paper-style input-repeatability)."""
        rets, post, reg = episode_path
        self._fixed_episode_path = ([r.clone() for r in rets], list(post), list(reg))

    def clear_fixed_episode_path(self) -> None:
        self._fixed_episode_path = None

    def reset(self) -> None:
        """Initialize the current episode from fixed path (if set) or by fresh simulation."""
        self._step = 0
        if self._fixed_episode_path is None:
            rets, post, reg = self._sample_episode_path()
        else:
            fr, fp, fg = self._fixed_episode_path
            rets, post, reg = ([r.clone() for r in fr], list(fp), list(fg))
        self._episode_returns = rets
        self._episode_posterior = post
        self._episode_regime_labels = reg

        self.belief_prior = self.initial_belief
        self.belief_posterior = self.initial_belief
        self.hidden_state = self._episode_regime_labels[0] if self._episode_regime_labels else 0

    def sample_returns(self) -> torch.Tensor:
        i = self._step
        if i >= len(self._episode_returns):
            raise RuntimeError(
                "Episode return buffer exhausted — call reset() before continuing."
            )
        self._step += 1
        r = self._episode_returns[i]
        self.belief_posterior = self._episode_posterior[i]
        self.belief_prior = float(
            np.clip(self._prior_from_posterior(self.belief_posterior), 0.0, 1.0)
        )
        self.hidden_state = self._episode_regime_labels[i]
        return r


# ── RegimeConditionedEnv ──────────────────────────────────────────────────────
class RegimeConditionedEnv(bac.BeliefPortfolioEnv):
    """
    Critic state appends the **future hidden-regime path** from HMM simulation:
    indices ``t..T-1`` filled with ``market._episode_regime_labels[j]``, past indices zero.
    """

    def __init__(self, market: RegimeReplayMarket, cfg: vanilla.Config):
        super().__init__(market, cfg)

    def critic_state(self) -> torch.Tensor:
        T = self.cfg.horizon
        regime_vec = torch.zeros(T, dtype=self.cfg.dtype, device=self.cfg.device)
        path = self.market._episode_regime_labels
        for j in range(self.t, T):
            if j < len(path):
                regime_vec[j] = float(path[j])
        return torch.cat([self.state(), regime_vec])


# ── MetaValueBaselineNet + functional forward ─────────────────────────────────
class MetaValueBaselineNet(nn.Module):
    """Critic ``V_phi(s_t, h_{t:T}, t)`` for the MAML meta-learning baseline."""

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


def functional_forward(
    params: Dict[str, torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """Forward pass through the 3-layer Sequential using explicit *params*."""
    x = F.linear(x, params["net.0.weight"], params["net.0.bias"])
    x = torch.tanh(x)
    x = F.linear(x, params["net.2.weight"], params["net.2.bias"])
    x = torch.tanh(x)
    x = F.linear(x, params["net.4.weight"], params["net.4.bias"])
    return x.squeeze(-1)


# ── Episode data structure ────────────────────────────────────────────────────


class EpisodeWithRegimeStates(vanilla.Episode):
    def __init__(
        self,
        log_probs: List[torch.Tensor],
        terminal_wealth: torch.Tensor,
        terminal_utility: torch.Tensor,
        critic_states: List[torch.Tensor],
    ):
        super().__init__(
            log_probs=log_probs,
            terminal_wealth=terminal_wealth,
            terminal_utility=terminal_utility,
        )
        self.critic_states = critic_states


# ── MAML adaptation ──────────────────────────────────────────────────────────


def _collect_value_targets(
    episodes: List[EpisodeWithRegimeStates],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack all (critic_state, terminal_utility) pairs from *episodes*."""
    states: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for ep in episodes:
        for cs in ep.critic_states:
            states.append(cs)
            targets.append(ep.terminal_utility.to(dtype=cs.dtype, device=cs.device))
    return torch.stack(states), torch.stack(targets)


def maml_adapt(
    meta_params: Dict[str, torch.Tensor],
    episodes: List[EpisodeWithRegimeStates],
    alpha: float,
    n_steps: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    MAML inner loop: starting from *meta_params*, take *n_steps* gradient
    steps on the value MSE loss computed from *episodes*.

    ``create_graph=True`` so that the outer (meta) gradient can flow through.
    """
    adapted: Dict[str, torch.Tensor] = OrderedDict(meta_params)
    states, targets = _collect_value_targets(episodes)

    for _ in range(n_steps):
        pred = functional_forward(adapted, states)
        loss = ((pred - targets) ** 2).mean()
        grads = torch.autograd.grad(loss, list(adapted.values()), create_graph=True)
        adapted = OrderedDict(
            (k, p - alpha * g) for (k, p), g in zip(adapted.items(), grads)
        )
    return adapted


# ── Episode collection ────────────────────────────────────────────────────────


def collect_episode_with_regime_states(
    env: RegimeConditionedEnv,
    policy: vanilla.StandardPolicy,
) -> EpisodeWithRegimeStates:
    """Rollout one episode (``reset()`` pre-builds bootstrap returns + thresholded regime path)."""
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
    return EpisodeWithRegimeStates(
        log_probs=log_probs,
        terminal_wealth=terminal_wealth,
        terminal_utility=terminal_utility,
        critic_states=critic_states,
    )

# ── Builder helpers ───────────────────────────────────────────────────────────

def build_regime_market_env_policy_from_returns(
    returns_df: pd.DataFrame,
    cfg: vanilla.Config,
) -> Tuple[
    RegimeReplayMarket,
    RegimeConditionedEnv,
    vanilla.StandardPolicy,
    Dict[str, np.ndarray],
]:
    regime = bac.fit_two_regime_proxy_hmm(returns_df, jitter=1e-6)
    market = RegimeReplayMarket(
        mu=regime["mu"],
        cov=regime["cov"],
        transition=regime["transition"],
        initial_belief=float(regime["initial_belief"][0]),
        cfg=cfg,
    )
    env = RegimeConditionedEnv(market, cfg)
    policy_state_dim = market.n_assets + 2
    action_dim = market.n_assets
    policy = vanilla.StandardPolicy(
        policy_state_dim, action_dim, cfg.hidden_size, cfg
    ).to(cfg.device, dtype=cfg.dtype)
    return market, env, policy, regime


def build_regime_market_env_policy_from_regime(
    returns_df: pd.DataFrame,
    regime: Dict[str, np.ndarray],
    cfg: vanilla.Config,
) -> Tuple[RegimeReplayMarket, RegimeConditionedEnv, vanilla.StandardPolicy]:
    market = RegimeReplayMarket(
        mu=regime["mu"],
        cov=regime["cov"],
        transition=regime["transition"],
        initial_belief=float(regime["initial_belief"][0]),
        cfg=cfg,
    )
    env = RegimeConditionedEnv(market, cfg)
    policy_state_dim = market.n_assets + 2
    action_dim = market.n_assets
    policy = vanilla.StandardPolicy(
        policy_state_dim, action_dim, cfg.hidden_size, cfg
    ).to(cfg.device, dtype=cfg.dtype)
    return market, env, policy


def build_regime_market_env_policy(
    cfg: vanilla.Config,
) -> Tuple[
    pd.DataFrame,
    RegimeReplayMarket,
    RegimeConditionedEnv,
    vanilla.StandardPolicy,
    Dict[str, np.ndarray],
]:
    returns_df = vanilla.download_weekly_returns(
        cfg.tickers, cfg.start_date, cfg.end_date, cfg.interval
    )
    market, env, policy, regime = build_regime_market_env_policy_from_returns(
        returns_df, cfg
    )
    return returns_df, market, env, policy, regime


# ── Config factory ────────────────────────────────────────────────────────────


def make_hindsight_training_config(base_cfg: vanilla.Config) -> vanilla.Config:
    cfg = vanilla.Config(**asdict(base_cfg))
    cfg.gross_long_cap = 1.3
    cfg.gross_short_cap = 0.3
    cfg.output_dir = "hindsight_outputs"
    cfg.final_model_name = "hindsight_training_policy_final.pt"
    cfg.best_model_name = "hindsight_training_policy_best.pt"
    cfg.training_plot_name = "hindsight_training_curve.png"
    cfg.iteration_metrics_name = "hindsight_training_iteration_metrics.csv"
    cfg.gradient_summary_name = "hindsight_training_gradient_checkpoint_summary.csv"
    cfg.training_gradient_matrix_name = "hindsight_training_gradients.npz"
    cfg.config_name = "hindsight_training_config.json"
    cfg.returns_used_name = "hindsight_training_real_returns.csv"
    return cfg


# ── MAML one-iteration: adapt + compute baselines + losses ────────────────────


def _maml_iteration(
    meta_params: Dict[str, torch.Tensor],
    episodes: List[EpisodeWithRegimeStates],
    maml_inner_lr: float,
    maml_inner_steps: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Perform the full MAML cross-validation procedure on *episodes*.

    Requires ``len(episodes) >= 2`` (needs two halves for adaptation vs baseline).

    Returns
    -------
    meta_loss : torch.Tensor
        Differentiable loss for the meta-value-network update.
    baselines : list[torch.Tensor]
        Pre-computed (detached) baseline value per time-step for every episode,
        in the same order as *episodes*.
    """
    n_ep = len(episodes)
    if n_ep < 2:
        raise RuntimeError(
            "MAML baseline needs at least 2 episodes per batch (got "
            f"{n_ep}). Increase batch_size."
        )
    half = n_ep // 2
    group_a = episodes[:half]
    group_b = episodes[half:]

    adapted_a = maml_adapt(meta_params, group_a, maml_inner_lr, maml_inner_steps)
    adapted_b = maml_adapt(meta_params, group_b, maml_inner_lr, maml_inner_steps)

    states_b, targets_b = _collect_value_targets(group_b)
    states_a, targets_a = _collect_value_targets(group_a)
    meta_loss_b = ((functional_forward(adapted_a, states_b) - targets_b) ** 2).mean()
    meta_loss_a = ((functional_forward(adapted_b, states_a) - targets_a) ** 2).mean()
    meta_loss = meta_loss_b + meta_loss_a

    baselines: List[torch.Tensor] = [torch.empty(0)] * len(episodes)
    with torch.no_grad():
        det_a = OrderedDict((k, v.detach()) for k, v in adapted_a.items())
        det_b = OrderedDict((k, v.detach()) for k, v in adapted_b.items())
        for idx, ep in enumerate(group_a):
            cs = torch.stack(ep.critic_states)
            baselines[idx] = functional_forward(det_b, cs)
        for idx, ep in enumerate(group_b):
            cs = torch.stack(ep.critic_states)
            baselines[half + idx] = functional_forward(det_a, cs)

    return meta_loss, baselines


def _compute_actor_loss(
    episodes: List[EpisodeWithRegimeStates],
    baselines: List[torch.Tensor],
) -> Tuple[torch.Tensor, float, float, float]:
    """Policy gradient loss using pre-computed baselines (detached)."""
    episode_losses: List[torch.Tensor] = []
    terminal_utilities: List[float] = []
    terminal_wealths: List[float] = []

    for idx, ep in enumerate(episodes):
        if len(ep.log_probs) == 0:
            raise RuntimeError(
                "Encountered episode with zero actions. Check horizon / env setup."
            )
        terminal_utilities.append(float(ep.terminal_utility.item()))
        terminal_wealths.append(float(ep.terminal_wealth.item()))

        step_losses: List[torch.Tensor] = []
        bl = baselines[idx]
        for t in range(len(ep.log_probs)):
            advantage = ep.terminal_utility - bl[t]
            step_losses.append(-ep.log_probs[t] * advantage)
        episode_losses.append(torch.stack(step_losses).sum())

    loss = torch.stack(episode_losses).mean()
    return (
        loss,
        float(np.mean(terminal_utilities)),
        float(np.std(terminal_utilities, ddof=0)),
        float(np.mean(terminal_wealths)),
    )


# ── Gradient checkpoint replicates ────────────────────────────────────────────


def estimate_gradient_replicates_hindsight_training(
    policy_state_dict: Dict[str, torch.Tensor],
    meta_value_state_dict: Dict[str, torch.Tensor],
    returns_df: pd.DataFrame,
    regime: Dict[str, np.ndarray],
    cfg: vanilla.Config,
    n_repeats: int,
    maml_inner_lr: float,
    maml_inner_steps: int,
) -> Dict[str, object]:
    cp_cfg = vanilla.Config(**asdict(cfg))
    cp_cfg.plot = False
    cp_cfg.save_best_model = False

    market, env, policy = build_regime_market_env_policy_from_regime(
        returns_df, regime, cp_cfg
    )

    critic_dim = market.n_assets + 2 + cp_cfg.horizon
    critic_gen = torch.Generator(device=torch.device("cpu"))
    critic_gen.manual_seed(
        vanilla.derived_torch_seed(cp_cfg.seed, "hindsight_training", "MetaValueBaselineNet")
    )
    meta_vnet = MetaValueBaselineNet(
        state_dim=critic_dim, hidden_size=cp_cfg.hidden_size, init_generator=critic_gen
    ).to(cp_cfg.device, dtype=cp_cfg.dtype)
    meta_vnet.load_state_dict(meta_value_state_dict)
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    meta_vnet.eval()

    grad_vectors: List[np.ndarray] = []
    losses: List[float] = []
    batch_util_means: List[float] = []
    batch_util_stds: List[float] = []
    batch_wealth_means: List[float] = []

    for _ in range(n_repeats):
        shared_path = market.sample_new_episode_path()
        market.set_fixed_episode_path(shared_path)
        episodes = [
            collect_episode_with_regime_states(env, policy)
            for _ in range(cp_cfg.batch_size)
        ]
        market.clear_fixed_episode_path()

        meta_params = OrderedDict(
            (n, p.detach().clone().requires_grad_(True))
            for n, p in meta_vnet.named_parameters()
        )
        _, baselines = _maml_iteration(
            meta_params, episodes, maml_inner_lr, maml_inner_steps
        )

        actor_loss, util_mean, util_std, wealth_mean = _compute_actor_loss(
            episodes, baselines
        )
        policy.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cp_cfg.actor_grad_clip)
        grad_vectors.append(vanilla.flatten_gradients(policy))
        losses.append(float(actor_loss.detach().item()))
        batch_util_means.append(util_mean)
        batch_util_stds.append(util_std)
        batch_wealth_means.append(wealth_mean)

    grad_matrix = np.stack(grad_vectors, axis=0)
    grad_mean = grad_matrix.mean(axis=0)
    centered = grad_matrix - grad_mean[None, :]
    per_component_var_mean = (
        float(grad_matrix.var(axis=0, ddof=1).mean()) if n_repeats > 1 else 0.0
    )
    estimator_variance_l2 = (
        float(np.mean(np.sum(centered**2, axis=1))) if n_repeats > 1 else 0.0
    )
    grad_norms = np.linalg.norm(grad_matrix, axis=1)
    grad_mean_norm = float(np.linalg.norm(grad_mean))
    grad_norm_mean = float(np.mean(grad_norms))
    grad_norm_std = float(np.std(grad_norms, ddof=1)) if n_repeats > 1 else 0.0
    snr = (
        grad_mean_norm**2 / estimator_variance_l2
        if estimator_variance_l2 > 0
        else np.nan
    )

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


# ── Main training loop (Mao et al. 2019-style meta baseline; see note in train loop) ──


def train_hindsight_training(
    cfg: vanilla.Config,
    *,
    maml_inner_lr: float = MAML_INNER_LR,
    maml_inner_steps: int = MAML_INNER_STEPS,
    actor_grad_clip: Optional[float] = None,
    meta_grad_clip: Optional[float] = None,
) -> Tuple[nn.Module, MetaValueBaselineNet, vanilla.TrainingResult]:
    if cfg.batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 for MAML cross-validation (got {cfg.batch_size})."
        )
    actor_gc = float(cfg.actor_grad_clip if actor_grad_clip is None else actor_grad_clip)
    meta_gc = float(cfg.critic_grad_clip if meta_grad_clip is None else meta_grad_clip)
    vanilla.set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = vanilla.config_to_serializable_dict(cfg)
    serializable["maml_inner_lr"] = maml_inner_lr
    serializable["maml_inner_steps"] = maml_inner_steps
    serializable["actor_grad_clip"] = actor_gc
    serializable["meta_grad_clip"] = meta_gc
    with open(out_dir / cfg.config_name, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    returns_df, market, env, policy, regime = build_regime_market_env_policy(cfg)
    returns_df.to_csv(out_dir / cfg.returns_used_name)

    critic_dim = market.n_assets + 2 + cfg.horizon
    critic_gen = torch.Generator(device=torch.device("cpu"))
    critic_gen.manual_seed(
        vanilla.derived_torch_seed(cfg.seed, "hindsight_training", "MetaValueBaselineNet")
    )
    meta_value_net = MetaValueBaselineNet(
        state_dim=critic_dim, hidden_size=cfg.hidden_size, init_generator=critic_gen
    ).to(cfg.device, dtype=cfg.dtype)

    actor_optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    meta_optimizer = optim.Adam(meta_value_net.parameters(), lr=cfg.learning_rate)

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

    checkpoint_iterations = sorted(
        {i for i in cfg.gradient_checkpoints if 1 <= i <= cfg.n_iterations}
    )

    for iteration in range(1, cfg.n_iterations + 1):
        print(
            f"Starting iteration {iteration}/{cfg.n_iterations}...", flush=True
        )

        # ── 1. Collect k episodes under one shared exogenous sequence z (Mao et al.) ──
        shared_path = market.sample_new_episode_path()
        market.set_fixed_episode_path(shared_path)
        episodes = [
            collect_episode_with_regime_states(env, policy)
            for _ in range(cfg.batch_size)
        ]
        market.clear_fixed_episode_path()
        print(
            f"Collected {len(episodes)} episodes for iteration {iteration}.",
            flush=True,
        )

        # ── 2. MAML cross-validated adaptation + baselines ───────────────
        meta_params = OrderedDict(
            (n, p) for n, p in meta_value_net.named_parameters()
        )
        meta_loss, baselines = _maml_iteration(
            meta_params, episodes, maml_inner_lr, maml_inner_steps
        )

        # ── 3–4. Policy update then meta-value update (Algorithm 1 order: policy then θ_V)
        actor_loss, avg_train_u, std_train_u, avg_train_w = _compute_actor_loss(
            episodes, baselines
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), actor_gc)
        grad_vector = vanilla.flatten_gradients(policy)
        grad_norm = float(np.linalg.norm(grad_vector))
        grad_sq_norm = float(np.dot(grad_vector, grad_vector))
        if cfg.save_full_training_gradients:
            full_training_gradients.append(grad_vector)
        actor_optimizer.step()

        meta_optimizer.zero_grad()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(meta_value_net.parameters(), meta_gc)
        meta_optimizer.step()

        # ── Iteration metrics ─────────────────────────────────────────────
        row: Dict[str, float] = {
            "iteration": float(iteration),
            "loss": float(actor_loss.detach().item()),
            "meta_value_loss": float(meta_loss.detach().item()),
            "train_avg_utility": avg_train_u,
            "train_std_utility": std_train_u,
            "train_avg_wealth": avg_train_w,
            "gradient_norm": grad_norm,
            "gradient_sq_norm": grad_sq_norm,
            "eval_avg_utility": np.nan,
            "eval_avg_wealth": np.nan,
        }
        iteration_metrics.append(row)

        # ── Evaluation ────────────────────────────────────────────────────
        if iteration % cfg.eval_every == 0:
            print(
                f"Running evaluation at iteration {iteration}...", flush=True
            )
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
                f"meta V loss = {meta_loss.item(): .6f}",
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
                            "meta_value_state_dict": meta_value_net.state_dict(),
                            "regime": {k: v.tolist() for k, v in regime.items()},
                            "maml_inner_lr": maml_inner_lr,
                            "maml_inner_steps": maml_inner_steps,
                        },
                    )

        # ── Gradient checkpoint ───────────────────────────────────────────
        if iteration in checkpoint_iterations:
            print(
                f"Estimating repeated gradient statistics at iteration "
                f"{iteration}...",
                flush=True,
            )
            rng_state = vanilla.save_rng_state()
            checkpoint_data = estimate_gradient_replicates_hindsight_training(
                policy_state_dict={
                    k: v.detach().clone() for k, v in policy.state_dict().items()
                },
                meta_value_state_dict={
                    k: v.detach().clone()
                    for k, v in meta_value_net.state_dict().items()
                },
                returns_df=returns_df,
                regime=regime,
                cfg=cfg,
                n_repeats=cfg.gradient_repeats,
                maml_inner_lr=maml_inner_lr,
                maml_inner_steps=maml_inner_steps,
            )
            vanilla.restore_rng_state(rng_state)

            npz_path = (
                out_dir
                / f"hindsight_training_gradient_checkpoint_iter_{iteration:04d}.npz"
            )
            np.savez_compressed(
                npz_path,
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

        # ── Incremental CSV saves ─────────────────────────────────────────
        pd.DataFrame(iteration_metrics).to_csv(
            out_dir / cfg.iteration_metrics_name, index=False
        )
        if gradient_checkpoint_rows:
            pd.DataFrame(gradient_checkpoint_rows).to_csv(
                out_dir / cfg.gradient_summary_name, index=False
            )

    # ── Finalize ──────────────────────────────────────────────────────────────
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
        "REINFORCE + Hindsight Training MAML Baseline (10-stock benchmark)",
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
            "meta_value_state_dict": meta_value_net.state_dict(),
            "regime": {k: v.tolist() for k, v in regime.items()},
            "maml_inner_lr": maml_inner_lr,
            "maml_inner_steps": maml_inner_steps,
        },
    )
    return policy, meta_value_net, result

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_cfg = vanilla.Config()
    cfg = make_hindsight_training_config(base_cfg)
    vanilla.print_run_header(
        "REINFORCE + hindsight training MAML baseline (Mao et al.)", cfg
    )
    print(f"MAML inner lr: {MAML_INNER_LR}")
    print(f"MAML inner steps: {MAML_INNER_STEPS}")
    print(f"Terminal utility clip: {cfg.terminal_utility_clip}")
    print(f"Actor grad clip: {cfg.actor_grad_clip}")
    print(f"Critic / meta-value grad clip: {cfg.critic_grad_clip}")

    policy, meta_vnet, result = train_hindsight_training(
        cfg,
        maml_inner_lr=MAML_INNER_LR,
        maml_inner_steps=MAML_INNER_STEPS,
    )

    print("\nCompleted training.")
    print(f"Universe size: {len(cfg.tickers)} assets")
    print(f"Selected tickers: {list(cfg.tickers)}")
    print(
        f"Final training utility: "
        f"{result.iteration_metrics[-1]['train_avg_utility']: .6f}"
    )
    if result.eval_curve:
        print(f"Final evaluation utility: {result.eval_curve[-1]: .6f}")
    if result.best_eval_utility is not None:
        print(f"Best evaluation utility: {result.best_eval_utility: .6f}")
        print(f"Best evaluation wealth: {result.best_eval_wealth: .6f}")
    print(f"Saved real returns to: {Path(cfg.output_dir) / cfg.returns_used_name}")
    print(
        f"Saved iteration metrics to: "
        f"{Path(cfg.output_dir) / cfg.iteration_metrics_name}"
    )
    if cfg.save_full_training_gradients:
        print(
            f"Saved training gradient matrix to: "
            f"{Path(cfg.output_dir) / cfg.training_gradient_matrix_name}"
        )
    print(f"Saved final model to: {Path(cfg.output_dir) / cfg.final_model_name}")
    if cfg.save_best_model:
        print(f"Saved best model to: {Path(cfg.output_dir) / cfg.best_model_name}")