# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import Normal
import torch.nn as nn
from tqdm import trange
import wandb

def asdict(config):
    dic = {}
    config_dict = config.__dict__
    for key, value in config_dict.items():
        if not key.startswith('__'):
            dic[key] = value
    return dic


@dataclass
class TrainConfig:
    # wandb params
    project: str = "Corrupt_rate_ablation"
    group: str = "EDAC"
    name: str = "EDAC"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = "halfcheetah-medium-v2"
    # env_name: str = 'walker2d-medium-replay-v2' 
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 50
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 4
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cuda"

    ###### corruption
    corruption_reward: bool = False
    corruption_dynamics: bool = False
    corruption_obs: bool = False
    corruption_acts: bool = False
    random_corruption: bool = False
    corruption_range: float = 0.5
    corruption_rate: float = 0.1  



# general utils
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
    )
    # wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(deterministic_torch)
    except:
        torch.set_deterministic(deterministic_torch)



def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        raise NotImplementedError


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class EDAC:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        eta: float = 1.0,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",  # noqa
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma
        self.eta = eta

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        assert action_log_prob.shape == q_value_min.shape
        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss, batch_entropy, q_value_std

    def _critic_diversity_loss(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        num_critics = self.critic.num_critics
        # almost exact copy from the original implementation, only style changes:
        # https://github.com/snu-mllab/EDAC/blob/198d5708701b531fd97a918a33152e1914ea14d7/lifelong_rl/trainers/q_learning/sac.py#L192

        # [num_critics, batch_size, *_dim]
        state = state.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        action = (
            action.unsqueeze(0)
            .repeat_interleave(num_critics, dim=0)
            .requires_grad_(True)
        )
        # [num_critics, batch_size]
        q_ensemble = self.critic(state, action)

        q_action_grad = torch.autograd.grad(
            q_ensemble.sum(), action, retain_graph=True, create_graph=True
        )[0]
        q_action_grad = q_action_grad / (
            torch.norm(q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10
        )
        # [batch_size, num_critics, action_dim]
        q_action_grad = q_action_grad.transpose(0, 1)

        masks = (
            torch.eye(num_critics, device=self.device)
            .unsqueeze(0)
            .repeat(q_action_grad.shape[0], 1, 1)
        )
        # removed einsum as it is usually slower than just torch.bmm
        # [batch_size, num_critics, num_critics]
        q_action_grad = q_action_grad @ q_action_grad.permute(0, 2, 1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1, 2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        critic_loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        diversity_loss = self._critic_diversity_loss(state, action)

        loss = critic_loss + self.eta * diversity_loss

        return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.array(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


# @pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    eval_env = wrap_env(gym.make(config.env_name))
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(eval_env)

    ##### corrupt
    if (config.corruption_reward or config.corruption_dynamics or config.corruption_obs or config.corruption_acts):
        if config.random_corruption:
            print('random corruption')
            random_num = np.random.random(dataset["rewards"].shape)
            indexs = np.where(random_num < config.corruption_rate)

            ##### attack indexes
            # attack_indexes[indexs] = 1.0

            if config.corruption_dynamics: # corrupt dynamics
                print('attack dynamics')
                std = dataset["next_observations"].std(axis=0).reshape(1,state_dim)
                dataset["next_observations"][indexs] += \
                            np.random.uniform(-config.corruption_range, config.corruption_range, size=(indexs[0].shape[0], state_dim)) * std
            
            if config.corruption_reward: # corrupt rewards
                print('attack reward')
                dataset["rewards"][indexs] = \
                            np.random.uniform(-config.corruption_range, config.corruption_range, size=indexs[0].shape[0])
            
            if config.corruption_obs:
                print('attack observation')
                std = dataset["observations"].std(axis=0).reshape(1,state_dim)
                dataset["observations"][indexs] += np.random.uniform(-config.corruption_range, config.corruption_range, size=(indexs[0].shape[0], state_dim)) * std
            
            if config.corruption_acts:
                print('attack actions')
                std = dataset['actions'].std(axis=0).reshape(1, action_dim)
                dataset["actions"][indexs] += np.random.uniform(-config.corruption_range, config.corruption_range, size=(indexs[0].shape[0], action_dim)) * std

        else:
            print('adversarial corruption')
            import pdb;pdb.set_trace()
            # if config.corruption_reward:
            #     print('attack reward')
            #     random_num = np.random.random(dataset["rewards"].shape)
            #     indexs = np.where(random_num < config.corruption_rate)
            #     # corrupt rewards
            #     dataset["rewards"][indexs] *= - config.corruption_range
            
            # if config.corruption_dynamics:
            #     print('attack dynamics')
            #     env_dir = 'walker2d_new_diverse' if config.env_name.startswith('walker2d') else 'halfcheetah'
            #     print('loading path {}'.format(os.path.join('./log_attack_data/{}/'.format(env_dir), "attack_data_corrupt{}_rate{}.pt".format(config.corruption_range, config.corruption_rate))))
            #     data_dict = torch.load(os.path.join('./log_attack_data/{}/'.format(env_dir), "attack_data_corrupt{}_rate{}.pt".format(config.corruption_range, config.corruption_rate)))
            #     attack_indexs, next_observations  = data_dict['index'], data_dict['next_observations']
            #     dataset["next_observations"][attack_indexs] = next_observations

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(dataset)

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    trainer = EDAC(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        eta=config.eta,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)

        # if (epoch+1) % 1000 == 0 and config.checkpoints_path is not None:
        #     torch.save(
        #         trainer.state_dict(),
        #         os.path.join(config.checkpoints_path, f"{epoch}.pt"),
        #     )

    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--corruption_reward', action='store_true', default=False)
    parser.add_argument('--corruption_dynamics', action='store_true', default=False)
    parser.add_argument('--corruption_acts', action='store_true', default=False)
    parser.add_argument('--corruption_obs', action='store_true', default=False)
    parser.add_argument('--random_corruption', action='store_true', default=False)
    parser.add_argument('--corruption_range', type=float, default=0.3)
    parser.add_argument('--corruption_rate', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    ### modify config
    TrainConfig.env_name = args.env_name
    TrainConfig.train_seed = args.seed
    TrainConfig.corruption_reward = args.corruption_reward
    TrainConfig.corruption_dynamics = args.corruption_dynamics
    TrainConfig.corruption_acts = args.corruption_acts
    TrainConfig.corruption_obs = args.corruption_obs
    TrainConfig.random_corruption = args.random_corruption
    TrainConfig.corruption_range = args.corruption_range
    TrainConfig.corruption_rate = args.corruption_rate

    if TrainConfig.corruption_reward:
        group_name_center = 'reward' 
    elif TrainConfig.corruption_dynamics:
        group_name_center = 'dynamics'
    elif TrainConfig.corruption_acts:
        group_name_center = 'actions'
    else:
        group_name_center = 'observations'

    ## modify config
    group_name_center = 'random_' + group_name_center if TrainConfig.random_corruption else 'adversarial_' + group_name_center
    TrainConfig.group = TrainConfig.group + '-{}'.format(group_name_center) 
    TrainConfig.group = TrainConfig.group + '_{}'.format(TrainConfig.env_name.split('-')[0])
    TrainConfig.name = "EDAC_baseline_corrupt{}_{}_seed{}".format(TrainConfig.corruption_range, TrainConfig.corruption_rate, TrainConfig.train_seed)
    TrainConfig.name = f"{TrainConfig.name}-{TrainConfig.env_name}-{str(uuid.uuid4())[:8]}"
    if TrainConfig.checkpoints_path is not None:
        TrainConfig.checkpoints_path = os.path.join(TrainConfig.checkpoints_path, TrainConfig.name)
    
    train(TrainConfig)
