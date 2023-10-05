from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import random
import uuid
import math

import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import trange
from attack import attack_dataset
from RIQL_config import get_config

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def asdict(config):
    dic = {}
    config_dict = config.__dict__
    for key, value in config_dict.items():
        if not key.startswith('__'):
            dic[key] = value
    return dic


@dataclass
class TrainConfig:
    # Experiment
    eval_episodes: int = 10 
    eval_every: int = 10  
    log_every: int = 100
    n_episodes: int = 10 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 3000     
    num_updates_on_epoch: int = 1000
    eval_freq: int = int(1e4)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(3e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    soft_target_update_rate: float = 0.005  # Target network update rate
    learning_rate: float = 3e-4  # Learning rate for all networks
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize observation
    # Wandb logging
    project: str = "Robust_Offline_RL"
    group: str = "RIQL"
    name: str = "RIQL"
    env_name: str = 'walker2d-medium-replay-v2' 
    train_seed: int = 0  
    eval_seed: int = 42
    flag: str = 'test'
    sigma: float = 1.0
    num_actors: int = 1
    num_critics: int = 5
    quantile: float = 0.25

    ###### corruption
    corrupt_reward: bool = False
    corrupt_dynamics: bool = False
    corrupt_obs: bool = False
    corrupt_acts: bool = False
    corruption_mode: str = 'random'
    corruption_range: float = 1.0
    corruption_rate: float = 0.3  


def soft_update(target: nn.Module, source: nn.Module, soft_target_update_rate: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - soft_target_update_rate) * target_param.data + soft_target_update_rate * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
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
        attack_indexes: torch.Tensor,
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
        self._attack_indexes = torch.tensor(attack_indexes)

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

    def sample_index(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        attack_indexes = self._attack_indexes[indices]
        return [states, actions, rewards, next_states, dones, attack_indexes]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


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


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
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
    return np.asarray(episode_rewards)


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
    lengths.append(ep_len) 
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def huber_loss(diff, sigma=1):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> MultivariateNormal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()



class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(
                self(state) * self.max_action, -self.max_action, self.max_action
            )
            .cpu()
            .data.numpy()
            .flatten()
        )

class VectorizedQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        num_critics: int = 5,
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(VectorizedLinear(dims[i], dims[i + 1], num_critics))
            model.append(nn.ReLU())
        model.append(VectorizedLinear(dims[-1], 1, num_critics))
        self.critic = nn.Sequential(*model)

        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values




class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)



class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        soft_target_update_rate: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.soft_target_update_rate = soft_target_update_rate

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict, attack_indexes) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            ### calculate target Q with quantile estimator
            target_q_all = self.q_target(observations, actions)
            target_q = torch.quantile(target_q_all.detach(), TrainConfig.quantile, dim=0)

            target_q_std = target_q_all.detach().std(dim=0)
            target_diff = target_q_all.detach().mean(dim=0) - target_q
        
        log_dict['attack_target_Q_std'] = torch.mean(target_q_std[torch.where(attack_indexes == 1)]).item()
        log_dict['clean_target_Q_std'] = torch.mean(target_q_std[torch.where(attack_indexes == 0)]).item()
        log_dict['attack_target_Q_diff'] = torch.mean(target_diff[torch.where(attack_indexes == 1)]).item()
        log_dict['clean_target_Q_diff'] = torch.mean(target_diff[torch.where(attack_indexes == 0)]).item()

        v = self.vf(observations)
        adv = target_q.detach() - v
        ######### average V
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)

        log_dict["value_loss"] = v_loss.item()
        log_dict["v_mean"] = torch.mean(v).item()
        log_dict["advantage_mean"] = torch.mean(adv).item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v,
        observations,
        actions,
        rewards,
        terminals,
        log_dict,
        attack_indexes,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf(observations, actions)
        #### Huber loss for Q functions
        # target clipping
        targets = torch.clamp(targets, -100, 1000).view(1, targets.shape[0])
        q_loss = huber_loss(targets.detach() - qs, sigma=TrainConfig.sigma).mean()

        log_dict["q_loss"] = q_loss.item()
        log_dict['q_mean'] = torch.mean(qs).item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        # Update target Q network
        soft_update(self.q_target, self.qf, self.soft_target_update_rate)

    def _update_policy(self, adv, observations, actions, log_dict, attack_indexes):
        batch_size, obs_dim = observations.shape[0], observations.shape[-1]
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=-1)
        else:
            raise NotImplementedError

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_loss = torch.mean(exp_adv * bc_losses)  

        log_dict["actor_loss"] = policy_loss.item()
        log_dict['bc_loss'] = bc_losses.mean().item()
        log_dict['exp_weights'] = exp_adv.mean().item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            attack_indexes,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict, attack_indexes)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict, attack_indexes)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict, attack_indexes)
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


# @pyrallis.wrap()
def train(config: TrainConfig):
    # Set seeds
    env = gym.make(config.env_name)
    seed = config.train_seed
    set_seed(seed, env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)

    ##### corrupt offline dataset
    attack_indexes = np.zeros(dataset["rewards"].shape)
    if (config.corrupt_reward or config.corrupt_dynamics or config.corrupt_obs or config.corrupt_acts):
        dataset, indexes = attack_dataset(config, dataset)
        attack_indexes[indexes] = 1.0


    if config.normalize:
        state_mean, state_std = compute_mean_std(np.concatenate([dataset["observations"], dataset['next_observations']], axis=0), eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    print('state mean: ', state_mean)
    print('state std: ', state_std)

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        attack_indexes,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)
    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)


    print('num_critics: {}'.format(config.num_critics))
    q_network = VectorizedQ(state_dim, action_dim, num_critics=config.num_critics).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, action_dim, max_action)
    ).to(config.device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.learning_rate)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training RIQL, Env: {config.env_name}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = replay_buffer.sample_index(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)
            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **log_dict})
            total_updates += 1

        # Evaluate episode
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
            )
            eval_returns = eval_scores
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            normalized_score = env.get_normalized_score(eval_scores) * 100.0
            eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
            eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)
        
        if (epoch+1) % 1000 == 0 and config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--corrupt_reward', action='store_true', default=False)
    parser.add_argument('--corrupt_dynamics', action='store_true', default=False)
    parser.add_argument('--corrupt_acts', action='store_true', default=False)
    parser.add_argument('--corrupt_obs', action='store_true', default=False)
    parser.add_argument('--corruption_mode', type=str, default='random')
    parser.add_argument('--corruption_range', type=float, default=0.3)
    parser.add_argument('--corruption_rate', type=float, default=0.1)
    parser.add_argument('--checkpoints_path', type=str, default=None)
    args = parser.parse_args()
    print(args)

    ### modify config
    TrainConfig.env_name = args.env_name
    TrainConfig.train_seed = args.seed
    TrainConfig.corrupt_reward = args.corrupt_reward
    TrainConfig.corrupt_dynamics = args.corrupt_dynamics
    TrainConfig.corrupt_acts = args.corrupt_acts
    TrainConfig.corrupt_obs = args.corrupt_obs
    TrainConfig.corruption_mode = args.corruption_mode
    TrainConfig.corruption_range = args.corruption_range
    TrainConfig.corruption_rate = args.corruption_rate
    TrainConfig.checkpoints_path = args.checkpoints_path

    ## modify config
    if TrainConfig.corrupt_reward:
        group_name_center = 'reward' 
    elif TrainConfig.corrupt_dynamics:
        group_name_center = 'dynamics'
    elif TrainConfig.corrupt_acts:
        group_name_center = 'actions'
    elif TrainConfig.corrupt_obs:
        group_name_center = 'observations'
    else:
        group_name_center = 'no_attack'

    get_config(TrainConfig)
    TrainConfig.flag = 'RIQL_ensembleQ{}_huber{}_normobs{}_quantile{}'.format(
                        TrainConfig.num_critics, TrainConfig.sigma, TrainConfig.normalize, TrainConfig.quantile)
    print('flag: {}'.format(TrainConfig.flag))

    group_name_center = '{}_'.format(TrainConfig.corruption_mode) + group_name_center 
    TrainConfig.group = TrainConfig.group + '-{}'.format(group_name_center) + '_{}'.format(TrainConfig.env_name.split('-')[0])
    TrainConfig.name = "{}_corrupt{}_{}_seed{}".format(TrainConfig.flag, TrainConfig.corruption_range, TrainConfig.corruption_rate, TrainConfig.train_seed)

    TrainConfig.name = f"{TrainConfig.name}-{TrainConfig.env_name}-{str(uuid.uuid4())[:8]}"
    if TrainConfig.checkpoints_path is not None:
        TrainConfig.checkpoints_path = os.path.join(TrainConfig.checkpoints_path, TrainConfig.name)
    
    train(TrainConfig)






