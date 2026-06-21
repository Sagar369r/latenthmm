"""
PPO training for RLActor using historical real market context replay.

Architecture:
  - Actor: RLActor (direction logit + Beta size distribution)
  - Critic: separate MLP value estimator
  - Reward: Sortino contribution - RL_DRAWDOWN_PENALTY * dd - RL_SLIPPAGE_PENALTY * slippage

Training procedure (per episode):
  1. Sample a start index from the real context replay buffer.
  2. For steps_per_ep:
     a. Fetch the latent state from the SDE (or precomputed buffer).
     b. Actor samples action (direction, size_fraction).
     c. Compute reward from ACTUAL subsequent price returns.
     d. Accumulate trajectory.
  3. Update with PPO + GAE.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from collections import deque
import numpy as np
import os
import logging
from v7_engine.config import CHECKPOINT_RL, MODELS_DIR
from v7_engine.ebm.rl_actor import RLActor
from v7_engine.sde.sde_model import NeuralSDE
from v7_engine.config import (
    SDE_LATENT_DIM, EMBEDDING_DIM,
    RL_EPISODES, RL_GAMMA, RL_LR_ACTOR, RL_LR_CRITIC,
    RL_ENTROPY_COEF, RL_DRAWDOWN_PENALTY, RL_SLIPPAGE_PENALTY,
    SLIPPAGE_PIPS
)
from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
from v7_engine.training.train_sde import prepare_sequences_from_ticks
from v7_engine.embedding.feature_vector import TickFeatureVector

logger = logging.getLogger("train_rl")


# ── CRITIC ─────────────────────────────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, input_dim: int = SDE_LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── REWARD FUNCTION ─────────────────────────────────────────────────────────────
def compute_reward(
    actual_return: float,
    size_fraction: float,
    direction: int,
) -> float:
    # Single-step reward approximation
    adj_return = actual_return * direction * size_fraction
    
    # Simple penalisation for single step
    dd_penalty = RL_DRAWDOWN_PENALTY * max(0.0, -adj_return)
    slippage_penalty = RL_SLIPPAGE_PENALTY * (SLIPPAGE_PIPS / 10000.0) * size_fraction
    
    reward = adj_return - dd_penalty - slippage_penalty
    return float(np.clip(reward, -10.0, 10.0))


# ── GAE ────────────────────────────────────────────────────────────────────────
def compute_gae(
    rewards: list[float],
    values: list[float],
    dts: list[float],
    gamma_base: float = 0.998,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Continuous-Time GAE.
    gamma_t = gamma_base ** dt
    """
    T       = len(rewards)
    advs    = [0.0] * T
    gae     = 0.0
    next_v  = 0.0

    for t in reversed(range(T)):
        gamma_t = gamma_base ** dts[t]
        delta = rewards[t] + gamma_t * next_v - values[t]
        gae   = delta + gamma_t * lam * gae
        advs[t] = gae
        next_v  = values[t]

    advs_t    = torch.tensor(advs, dtype=torch.float32)
    returns_t = advs_t + torch.tensor(values, dtype=torch.float32)

    if advs_t.std() > 1e-8:
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    return advs_t, returns_t


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_rl(
    contexts: np.ndarray,
    actual_returns: np.ndarray,
    valid_tns: np.ndarray,
    sde: NeuralSDE,
    ebm=None,
    episodes:       int   = RL_EPISODES,
    steps_per_ep:   int   = 20,
    ppo_epochs:     int   = 4,
    clip_epsilon:   float = 0.2,
    save_every:     int   = 500,
    save_path:      str   = None,
    device_str:     str   = None,
) -> RLActor:
    
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training RL on {device}")

    actor  = RLActor().to(device)
    critic = Critic().to(device)

    opt_actor  = torch.optim.Adam(actor.parameters(),  lr=RL_LR_ACTOR, weight_decay=1e-4)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=RL_LR_CRITIC, weight_decay=1e-4)

    os.makedirs("models", exist_ok=True)
    best_ep_return = -float("inf")
    return_history = []
    
    N = len(contexts)
    if N <= steps_per_ep:
        raise ValueError("Not enough data for episode length.")

    for ep in range(episodes):
        actor.train()
        critic.train()

        start_idx = np.random.randint(0, N - steps_per_ep)
        
        # Batch extract contexts for the episode
        idx_end = start_idx + steps_per_ep
        ctx_batch = torch.tensor(contexts[start_idx:idx_end], dtype=torch.float32, device=device)
        
        # SDE forward on the entire batch at once
        with torch.no_grad():
            _, final_states, _, _ = sde(ctx_batch)
            if ebm is not None:
                energy = ebm(final_states)
            else:
                energy = torch.zeros(len(final_states), device=device)
            
        # Actor and Critic forward on the entire batch
        dir_logits, alphas, beta_ps = actor(final_states)
        dir_dists  = torch.distributions.Bernoulli(logits=dir_logits)
        size_dists = torch.distributions.Beta(alphas, beta_ps)
        
        dir_samples  = dir_dists.sample()
        size_samples = size_dists.sample()
        
        log_p_dirs  = dir_dists.log_prob(dir_samples)
        log_p_sizes = size_dists.log_prob(size_samples.clamp(1e-4, 1 - 1e-4))
        
        values = critic(final_states).squeeze(-1)
        
        # Calculate exact immediate simple return (no look-ahead summing, GAE handles future discounting)
        # Using continuous-time logic with stride aggregation.
        # actual_returns are stride-spaced returns, we just take them directly
        # and compute exact dts between contexts using valid_tns
        ret_batch = actual_returns[start_idx : idx_end]
        if len(ret_batch) < steps_per_ep:
            ret_batch = np.pad(ret_batch, (0, steps_per_ep - len(ret_batch)))
            
        # Convert log returns to simple fractional returns
        ret_simple = np.exp(ret_batch) - 1.0
        
        # Vectorized rewards
        directions = (dir_samples.cpu().numpy() * 2) - 1
        sizes      = size_samples.cpu().numpy()
        
        adj_returns = directions * ret_simple * sizes
        
        # ── Dopamine & Prop Firm Constraints ──────────────────────────────────
        from v7_engine.config import EBM_ENERGY_THRESHOLD
        
        toxicity = (energy / EBM_ENERGY_THRESHOLD).clamp(0.0, 5.0).cpu().numpy()
        
        wins = np.maximum(0.0, adj_returns)
        losses = np.minimum(0.0, adj_returns)
        
        # 1. Alignment Dopamine: Boost reward for clean, profitable trades
        clean_multiplier = 1.0 + np.clip(1.0 - toxicity, 0.0, 1.0)
        
        # 2. Toxicity Penalty (Electric Shock): Massive penalty for trading in toxic flow
        toxic_violation = (toxicity > 1.0).astype(float) * sizes
        toxicity_penalties = toxic_violation * 0.5  # Heavy fixed penalty
        
        # 3. Prop Firm Drawdown Penalty (Calmar Focus): Exponential penalty on losses
        dd_penalties = RL_DRAWDOWN_PENALTY * (np.abs(losses) ** 1.5)
        
        slippage_penalties = RL_SLIPPAGE_PENALTY * (SLIPPAGE_PIPS / 10000.0) * sizes
        
        rewards = (wins * clean_multiplier) + losses - dd_penalties - toxicity_penalties - slippage_penalties
        rewards = np.clip(rewards, -1.0, 1.0)
        
        tns_slice = valid_tns[start_idx : idx_end + 1]
        if len(tns_slice) < steps_per_ep + 1:
            tns_slice = np.pad(tns_slice, (0, steps_per_ep + 1 - len(tns_slice)), mode='edge')
            
        # dts in seconds
        dts_batch = (tns_slice[1:] - tns_slice[:-1]) / 1e9
        dts_batch = np.clip(dts_batch, 0.0, 60.0).tolist()
        
        # Convert back to lists for GAE processing
        states_list   = list(final_states.detach())
        actions_dir   = list(dir_samples)
        actions_sz    = list(size_samples)
        log_probs_dir = list(log_p_dirs)
        log_probs_sz  = list(log_p_sizes)
        
        # Detach values for GAE since advantages/returns should be constant targets
        val_list      = values.detach().cpu().numpy().tolist()
        rewards_list  = list(rewards)
        
        # Calculate full episode return
        ep_return = float(np.sum(rewards))

        # ── 4. GAE ────────────────────────────────────────────────────────────
        advantages, returns = compute_gae(rewards_list, val_list, dts_batch)
        advantages = advantages.to(device)
        returns    = returns.to(device)

        old_log_p_dir  = torch.stack(log_probs_dir).detach().to(device)
        old_log_p_size = torch.stack(log_probs_sz).detach().to(device)
        old_log_p      = old_log_p_dir + old_log_p_size

        states_tensor  = torch.stack(states_list, dim=0).to(device)
        actions_dir_t  = torch.stack(actions_dir).detach().to(device)
        actions_sz_t   = torch.stack(actions_sz).detach().to(device)

        for _ in range(ppo_epochs):
            dir_logit, alpha, beta_p = actor(states_tensor)
            dir_dist  = torch.distributions.Bernoulli(logits=dir_logit)
            size_dist = torch.distributions.Beta(alpha, beta_p)

            log_p_dir_new  = dir_dist.log_prob(actions_dir_t)
            log_p_size_new = size_dist.log_prob(actions_sz_t.clamp(1e-4, 1 - 1e-4))
            log_p_new      = log_p_dir_new + log_p_size_new

            ratio = torch.exp(log_p_new - old_log_p)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy = dir_dist.entropy().mean() + size_dist.entropy().mean()
            actor_loss -= RL_ENTROPY_COEF * entropy

            values_pred = critic(states_tensor)
            critic_loss = nn.functional.mse_loss(values_pred, returns)

            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()

        if ep % 100 == 0:
            logger.info(f"Episode {ep:5d} | Return: {ep_return:7.3f} | Actor Loss: {actor_loss.item():.4f}")
            
        return_history.append({"episode": ep, "return": ep_return, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()})

        if ep_return > best_ep_return:
            best_ep_return = ep_return
            if save_path:
                torch.save(actor.state_dict(), save_path)

        if save_every and ep % save_every == 0 and ep > 0:
            save_dir = os.path.dirname(save_path) if save_path else "models"
            torch.save(actor.state_dict(), os.path.join(save_dir, f"rl_ep{ep}.pth"))

    import json
    with open("rl_return_history.json", "w") as f:
        json.dump(return_history, f, indent=4)
        
    logger.info(f"RL training complete. Best return: {best_ep_return:.3f}. Saved to {save_path or 'memory'}")
    return actor


def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    args = parser.parse_args()
    
    from v7_engine.config import CHECKPOINT_SDE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── 1. Load Pre-trained SDE ───────────────────────────────────────────────
    sde = NeuralSDE().to(device)
    if os.path.exists(CHECKPOINT_SDE):
        sde.load_state_dict(torch.load(CHECKPOINT_SDE, map_location=device, weights_only=True))
        logger.info(f"Loaded frozen SDE from {CHECKPOINT_SDE}")
    else:
        logger.warning(f"SDE checkpoint not found at {CHECKPOINT_SDE}, using untrained SDE!")
    sde.eval()

    logger.info(f"Loading Dukascopy ticks for {args.symbol} from {args.start} to {args.end}...")
    loader = DukascopyLoader()
    try:
        ticks = loader.load(args.symbol, args.start, args.end)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    from v7_engine.config import CHECKPOINT_WELFORD, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD, EMBEDDING_DIM
    from v7_engine.ingestion.welford import WelfordNormaliser
    if os.path.exists(CHECKPOINT_WELFORD):
        wdata = np.load(CHECKPOINT_WELFORD)
        welford = WelfordNormaliser.from_state(
            {"mean": wdata["mean"], "m2": wdata["m2"], "n": int(wdata["n"])},
            dim=EMBEDDING_DIM,
            clip_sigma=WELFORD_CLIP_SIGMA,
            min_std=WELFORD_MIN_STD,
        )
        welford.is_warm = True
    else:
        raise FileNotFoundError(f"Missing {CHECKPOINT_WELFORD}. Must train SDE first.")

    logger.info("Extracting feature sequences...")
    seqs, _, rets, _, welford, tns = prepare_sequences_from_ticks(
        ticks, TickFeatureVector, seq_len=64, stride=10, normaliser=welford
    )

    if len(seqs) == 0:
        logger.error("Not enough data to train RL.")
        return

    train_rl(contexts=seqs, actual_returns=rets, valid_tns=tns, sde=sde, episodes=500, device_str=device.type)


if __name__ == "__main__":
    main()
