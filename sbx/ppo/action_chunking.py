import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from typing import Optional, TypeVar, Union
from functools import partial
from gymnasium import spaces

from stable_baselines3.common.utils import explained_variance

from sbx.ppo.ppo import PPO

ACPPO_Self = TypeVar("ACPPO_Self", bound="ACPPO")


class ACPPO(PPO):
    """PPO variant using action chunking to reduce variance.

    This implementation groups actions into fixed-size chunks and
    optimizes a surrogate objective using a chunked advantage as
    described in the user documentation.
    """

    def __init__(
        self,
        policy: Union[str, type["PPOPolicy"]],
        env: Union["GymEnv", str],
        *args,
        chunk_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(policy, env, *args, **kwargs)
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = chunk_size

    @staticmethod
    @partial(jax.jit, static_argnames=["normalize_advantage"])
    def _one_update_chunk(
        actor_state: TrainState,
        vf_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_prob: np.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
    ):
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size, chunk_size = observations.shape[0], observations.shape[1]
        flat_obs = observations.reshape((batch_size * chunk_size, *observations.shape[2:]))
        flat_actions = actions.reshape((batch_size * chunk_size, *actions.shape[2:]))

        def actor_loss(params):
            dist = actor_state.apply_fn(params, flat_obs)
            log_prob = dist.log_prob(flat_actions)
            entropy = dist.entropy()
            log_prob = jnp.reshape(log_prob, (batch_size, chunk_size))
            entropy = jnp.reshape(entropy, (batch_size, chunk_size))
            log_prob_sum = jnp.sum(log_prob, axis=1)
            entropy_sum = jnp.sum(entropy, axis=1)

            ratio = jnp.exp(log_prob_sum - old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.mean(jnp.minimum(policy_loss_1, policy_loss_2))
            entropy_loss = -jnp.mean(entropy_sum)
            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss, (ratio, policy_loss, entropy_loss)

        (pg_loss_value, (ratio, policy_loss, entropy_loss)), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params
        )
        actor_state = actor_state.apply_gradients(grads=grads)

        def critic_loss(params):
            vf_values = vf_state.apply_fn(params, observations[:, 0]).flatten()
            return vf_coef * ((returns - vf_values) ** 2).mean()

        vf_loss_value, grads = jax.value_and_grad(critic_loss)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)
        return (actor_state, vf_state), (pg_loss_value, policy_loss, entropy_loss, vf_loss_value, ratio)

    def train(self) -> None:  # type: ignore[override]
        if self.target_kl is None:
            self._update_learning_rate(
                [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                learning_rate=self.lr_schedule(self._current_progress_remaining),
            )
        clip_range = self.clip_range_schedule(self._current_progress_remaining)
        n_updates = 0
        mean_clip_fraction = 0.0
        mean_kl_div = 0.0

        obs_buf = self.rollout_buffer.observations
        act_buf = self.rollout_buffer.actions
        rew_buf = self.rollout_buffer.rewards
        val_buf = self.rollout_buffer.values
        logp_buf = self.rollout_buffer.log_probs
        ret_buf = self.rollout_buffer.returns

        chunks_obs = []
        chunks_actions = []
        chunks_adv = []
        chunks_ret = []
        chunks_logp = []

        ep_starts = self.rollout_buffer.episode_starts  # shape: [n_steps, n_envs]
        h = self.chunk_size

        for env_idx in range(self.n_envs):
            starts = np.asarray(ep_starts[:, env_idx], dtype=bool)
            for start in range(0, self.n_steps - h + 1, h):
                end = start + h
                # skip if a new episode starts inside (start+1 ... end-1)
                if starts[start+1:end].any():
                    continue

                obs_seq = obs_buf[start:end, env_idx]
                act_seq = act_buf[start:end, env_idx]
                logp_seq = logp_buf[start:end, env_idx]
                rewards = rew_buf[start:end, env_idx]
                value_start = val_buf[start, env_idx]

                # bootstrap only if we did not hit a terminal at 'end-1'
                if end < self.n_steps and not starts[end]:
                    value_end = val_buf[end, env_idx]
                else:
                    value_end = 0.0  # true terminal or rollout boundary -> no bootstrap

                # h-step advantage
                discount = 1.0
                chunk_adv = 0.0
                for r in rewards:
                    chunk_adv += discount * r
                    discount *= self.gamma
                chunk_adv += (self.gamma ** h) * value_end - value_start

                chunks_obs.append(obs_seq)
                chunks_actions.append(act_seq)
                chunks_adv.append(chunk_adv)
                chunks_ret.append(ret_buf[start, env_idx])
                chunks_logp.append(logp_seq.sum())

        chunks_obs = np.array(chunks_obs)
        chunks_actions = np.array(chunks_actions)
        chunks_adv = np.array(chunks_adv)
        chunks_ret = np.array(chunks_ret)
        chunks_logp = np.array(chunks_logp)

        if isinstance(self.action_space, spaces.Discrete):
            chunks_actions = chunks_actions.squeeze(-1).astype(np.int32)
        elif isinstance(self.action_space, (spaces.MultiDiscrete, spaces.MultiBinary)):
            chunks_actions = chunks_actions.astype(np.int32)
        

        data_size = len(chunks_adv)
        for _ in range(self.n_epochs):
            indices = np.random.permutation(data_size)
            for start in range(0, data_size, self.batch_size):
                mb_inds = indices[start : start + self.batch_size]
                (self.policy.actor_state, self.policy.vf_state), (
                    pg_loss,
                    policy_loss,
                    entropy_loss,
                    value_loss,
                    ratio,
                ) = self._one_update_chunk(
                    actor_state=self.policy.actor_state,
                    vf_state=self.policy.vf_state,
                    observations=chunks_obs[mb_inds],
                    actions=chunks_actions[mb_inds],
                    advantages=chunks_adv[mb_inds],
                    returns=chunks_ret[mb_inds],
                    old_log_prob=chunks_logp[mb_inds],
                    clip_range=clip_range,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    normalize_advantage=self.normalize_advantage,
                )
                n_updates += 1
                eps = 1e-7
                approx_kl_div = jnp.mean((ratio - 1.0 + eps) - jnp.log(ratio + eps)).item()
                clip_fraction = jnp.mean(jnp.abs(ratio - 1) > clip_range).item()
                mean_clip_fraction += (clip_fraction - mean_clip_fraction) / n_updates
                mean_kl_div += (approx_kl_div - mean_kl_div) / n_updates
                if self.target_kl is not None:
                    self.adaptive_lr.update(approx_kl_div)
                    self._update_learning_rate(
                        [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                        learning_rate=self.adaptive_lr.current_adaptive_lr,
                    )

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),  # type: ignore[attr-defined]
            self.rollout_buffer.returns.flatten(),  # type: ignore[attr-defined]
        )
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/approx_kl", mean_kl_div)
        self.logger.record("train/clip_fraction", mean_clip_fraction)
        self.logger.record("train/pg_loss", pg_loss.item())
        self.logger.record("train/explained_variance", explained_var)

        try:
            log_std = self.policy.actor_state.params["params"]["log_std"]
            self.logger.record("train/std", np.exp(log_std).mean().item())
        except KeyError:
            pass
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def learn(
        self: ACPPO_Self,
        total_timesteps: int,
        callback: Optional["MaybeCallback"] = None,
        log_interval: int = 1,
        tb_log_name: str = "ACPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> ACPPO_Self:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )