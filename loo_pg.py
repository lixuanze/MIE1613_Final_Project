from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import vanilla_pg as vanilla

def make_pg_loo_config(base_cfg: vanilla.Config) -> vanilla.Config:
    cfg = vanilla.Config(**asdict(base_cfg))
    cfg.gross_long_cap = 1.3
    cfg.gross_short_cap = 0.3
    cfg.output_dir = "pg_loo_outputs"
    cfg.final_model_name = "pg_loo_policy_final.pt"
    cfg.best_model_name = "pg_loo_policy_best.pt"
    cfg.training_plot_name = "pg_loo_training_curve.png"
    cfg.iteration_metrics_name = "pg_loo_iteration_metrics.csv"
    cfg.gradient_summary_name = "pg_loo_gradient_checkpoint_summary.csv"
    cfg.training_gradient_matrix_name = "pg_loo_training_gradients.npz"
    cfg.returns_used_name = "pg_loo_real_returns.csv"
    cfg.config_name = "pg_loo_config.json"
    return cfg


def compute_gradient_estimate_from_episodes_pg_loo(
    episodes: List[vanilla.Episode],
    model: nn.Module,
    cfg: vanilla.Config,
) -> Tuple[torch.Tensor, float, float, float]:
    if len(episodes) == 0:
        raise RuntimeError("No episodes provided for gradient estimate.")

    terminal_utilities = torch.tensor(
        [float(ep.terminal_utility.item()) for ep in episodes],
        dtype=torch.float64,
    )
    terminal_wealths = [float(ep.terminal_wealth.item()) for ep in episodes]
    n_episodes = len(episodes)

    if n_episodes > 1:
        util_sum = terminal_utilities.sum()
        baselines = (util_sum - terminal_utilities) / (n_episodes - 1)
    else:
        # With one episode, leave-one-out is undefined; default to zero baseline.
        baselines = torch.zeros_like(terminal_utilities)

    episode_losses: List[torch.Tensor] = []
    for idx, ep in enumerate(episodes):
        if len(ep.log_probs) == 0:
            raise RuntimeError("Encountered an episode with zero actions. Check horizon/environment setup.")
        log_prob_sum = torch.stack(ep.log_probs).sum()
        advantage = terminal_utilities[idx].to(dtype=log_prob_sum.dtype, device=log_prob_sum.device) - baselines[
            idx
        ].to(dtype=log_prob_sum.dtype, device=log_prob_sum.device)
        episode_losses.append(-log_prob_sum * advantage)

    loss = torch.stack(episode_losses).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.actor_grad_clip)

    return (
        loss.detach(),
        float(terminal_utilities.mean().item()),
        float(terminal_utilities.std(unbiased=False).item()),
        float(np.mean(terminal_wealths)),
    )


def estimate_gradient_replicates_pg_loo(
    policy_state_dict: Dict[str, torch.Tensor],
    returns_df: pd.DataFrame,
    cfg: vanilla.Config,
    n_repeats: int,
) -> Dict[str, object]:
    checkpoint_cfg = vanilla.Config(**asdict(cfg))
    checkpoint_cfg.plot = False
    checkpoint_cfg.save_best_model = False

    _, env, policy = vanilla.build_market_env_policy_from_returns(returns_df, checkpoint_cfg)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    grad_vectors: List[np.ndarray] = []
    losses: List[float] = []
    batch_util_means: List[float] = []
    batch_util_stds: List[float] = []
    batch_wealth_means: List[float] = []

    for _ in range(n_repeats):
        episodes = [vanilla.collect_reinforce_episode(env, policy) for _ in range(checkpoint_cfg.batch_size)]
        loss, util_mean, util_std, wealth_mean = compute_gradient_estimate_from_episodes_pg_loo(
            episodes, policy, checkpoint_cfg
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


def train_pg_loo(cfg: vanilla.Config) -> Tuple[nn.Module, vanilla.TrainingResult]:
    vanilla.set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / cfg.config_name, "w", encoding="utf-8") as f:
        json.dump(vanilla.config_to_serializable_dict(cfg), f, indent=2)

    returns_df, _, env, policy = vanilla.build_market_env_policy(cfg)
    returns_df.to_csv(out_dir / cfg.returns_used_name)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)

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
        episodes = [vanilla.collect_reinforce_episode(env, policy) for _ in range(cfg.batch_size)]
        print(f"Finished collecting episodes for iteration {iteration}. Updating policy...", flush=True)

        loss, avg_train_u, std_train_u, avg_train_w = compute_gradient_estimate_from_episodes_pg_loo(
            episodes, policy, cfg
        )
        grad_vector = vanilla.flatten_gradients(policy)
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
                f"grad norm = {grad_norm: .6f}",
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
                        extra={"best_iteration": iteration},
                    )

        if iteration in checkpoint_iterations:
            print(f"Estimating repeated gradient statistics at iteration {iteration}...", flush=True)
            rng_state = vanilla.save_rng_state()
            checkpoint_data = estimate_gradient_replicates_pg_loo(
                policy_state_dict={k: v.detach().clone() for k, v in policy.state_dict().items()},
                returns_df=returns_df,
                cfg=cfg,
                n_repeats=cfg.gradient_repeats,
            )
            vanilla.restore_rng_state(rng_state)

            grad_npz_path = out_dir / f"pg_loo_gradient_checkpoint_iter_{iteration:04d}.npz"
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
        "Policy gradient + LOO baseline (10-stock benchmark)",
        cfg.training_plot_name,
        cfg,
    )

    vanilla.save_checkpoint(
        out_dir / cfg.final_model_name,
        policy,
        cfg,
        result,
        extra={"final_iteration": cfg.n_iterations},
    )
    return policy, result


if __name__ == "__main__":
    cfg = make_pg_loo_config(vanilla.Config())
    vanilla.print_run_header("Policy gradient + LOO baseline", cfg)
    policy, result = train_pg_loo(cfg)

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
