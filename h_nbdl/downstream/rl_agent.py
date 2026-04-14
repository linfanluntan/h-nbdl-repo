"""
RL Agent with Dictionary-Based State Representations.

Implements a Soft Actor-Critic (SAC) agent whose state space is defined
by the H-NBDL posterior representation r_{ij} = (E[z⊙s], Var[z⊙s]).
Supports Thompson sampling for exploration via posterior uncertainty.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class RLConfig:
    """Configuration for the dictionary-based RL agent."""
    state_dim: int = 50          # 2 * K_eff (mean + var)
    action_dim: int = 5          # number of discrete/continuous actions
    hidden_dim: int = 256
    gamma: float = 0.99          # discount factor
    tau_target: float = 0.005    # soft target update
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    alpha_entropy: float = 0.2   # SAC entropy coefficient
    buffer_size: int = 100_000
    batch_size: int = 256
    thompson_sampling: bool = True


class DictionaryRLAgent:
    """RL agent that operates on H-NBDL dictionary representations.

    The state at each step is the uncertainty-aware representation:
        s_t = (E_q[z_t ⊙ s_t], Var_q[z_t ⊙ s_t])

    If Thompson sampling is enabled, the agent samples from the
    posterior over representations to drive exploration.

    Parameters
    ----------
    config : RLConfig
        Agent configuration.
    n_sites : int
        Number of sites (for transfer learning).
    """

    def __init__(self, config: RLConfig, n_sites: int = 1):
        self.config = config
        self.n_sites = n_sites
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if stable-baselines3 is available."""
        self._has_sb3 = False
        try:
            import stable_baselines3
            self._has_sb3 = True
        except ImportError:
            pass

    def build_state(
        self,
        r_mean: np.ndarray,
        r_var: np.ndarray,
        thompson: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Construct the RL state from H-NBDL representations.

        Parameters
        ----------
        r_mean : np.ndarray, shape (K,)
            Posterior mean of z ⊙ s.
        r_var : np.ndarray, shape (K,)
            Posterior variance of z ⊙ s.
        thompson : bool
            If True, sample from posterior for exploration.
        rng : Generator

        Returns
        -------
        state : np.ndarray, shape (2K,)
        """
        if thompson and rng is not None:
            # Thompson sampling: perturb mean by posterior uncertainty
            r_sample = rng.normal(r_mean, np.sqrt(np.maximum(r_var, 1e-8)))
            return np.concatenate([r_sample, r_var])
        return np.concatenate([r_mean, r_var])

    def train(
        self,
        env,
        nbdl_model,
        total_timesteps: int = 500_000,
        seed: int = 42,
    ) -> Dict:
        """Train the SAC agent in the given environment.

        Parameters
        ----------
        env : gymnasium.Env
            Environment with observation_space matching raw data dim.
        nbdl_model : HierarchicalNBDL
            Trained H-NBDL model for encoding observations.
        total_timesteps : int
        seed : int

        Returns
        -------
        results : dict
            Training metrics.
        """
        if not self._has_sb3:
            raise ImportError(
                "stable-baselines3 required for RL training. "
                "Install with: pip install stable-baselines3"
            )

        from stable_baselines3 import SAC
        import gymnasium as gym

        # Wrap environment to use dictionary representations
        wrapped_env = DictionaryObsWrapper(env, nbdl_model, self.config)

        model = SAC(
            "MlpPolicy",
            wrapped_env,
            learning_rate=self.config.lr_actor,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            tau=self.config.tau_target,
            ent_coef=self.config.alpha_entropy,
            verbose=1,
            seed=seed,
        )

        model.learn(total_timesteps=total_timesteps)

        return {
            "model": model,
            "total_timesteps": total_timesteps,
        }


class DictionaryObsWrapper:
    """Gymnasium wrapper that replaces raw observations with H-NBDL representations.

    This wrapper intercepts the environment's observations, passes them
    through the trained H-NBDL encoder, and returns the uncertainty-aware
    representation as the RL agent's state.
    """

    def __init__(self, env, nbdl_model, config: RLConfig):
        self.env = env
        self.nbdl_model = nbdl_model
        self.config = config
        self.rng = np.random.default_rng(42)

        # Update observation space
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config.state_dim,), dtype=np.float32
        )
        self.action_space = env.action_space

    def _encode(self, obs: np.ndarray, site_id: int = 0) -> np.ndarray:
        """Encode a raw observation into dictionary representation."""
        import torch
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        sid = torch.tensor([site_id], dtype=torch.long)
        r_mean, r_var = self.nbdl_model.encode(x, sid)
        r_mean = r_mean.squeeze(0).numpy()
        r_var = r_var.squeeze(0).numpy()

        agent = DictionaryRLAgent(self.config)
        return agent.build_state(
            r_mean, r_var,
            thompson=self.config.thompson_sampling,
            rng=self.rng
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._encode(obs), reward, terminated, truncated, info
