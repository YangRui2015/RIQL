# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import random
import uuid

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
    device: str = "cuda"
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
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = True  # Use deterministic actor
    normalize: bool = False # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "Corrupt_RIQL_SCALE2"
    # project: str = "Model_Prepare"
    group: str = "IQL"
    name: str = "IQL"
    # env: str = "halfcheetah-medium-v2"  
    env_name: str = 'walker2d-medium-replay-v2' 
    train_seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 42
    flag: str = 'IQL'

    ###### corruption
    corruption_reward: bool = False
    corruption_dynamics: bool = False
    corruption_obs: bool = False
    corruption_acts: bool = False
    random_corruption: bool = False
    corruption_range: float = 0.5
    corruption_rate: float = 0.1  


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


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
    # wandb.run.save()


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



def asymmetric_l2_loss(u: torch.Tensor, tau: float, keep_dim=False) -> torch.Tensor:
    if keep_dim:
        return torch.abs(tau - (u < 0).float()) * u**2
    else:
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
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class QFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim+action_dim, *([hidden_dim] * n_hidden), 1]
        self.q = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.q(sa)


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
        tau: float = 0.005,
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
        self.tau = tau

        self.total_it = 0
        self.device = device
        self.clip_max = 50
    
    # def _get_kai(self, x, N):
    #     mean_x = np.mean(x)
    #     residual = x - mean_x
    #     kai = N * np.sum(residual ** 4) / (np.sum(residual ** 2) ** 2)
    #     return kai
    
    # def _save_values(self, x, N, name):
    #     import pickle
    #     hist, bins = np.histogram(x, bins=200)
    #     kai = self._get_kai(x, N)
    #     save_dict = {"hist": hist, "bins": bins, "kai": kai}
    #     with open(os.path.join(TrainConfig.checkpoints_path, '{}.pkl'.format(name)), 'wb') as f:
    #         pickle.dump(save_dict, f)


    # def check_values(self, batch, epoch):
    #     (
    #         observations,
    #         actions,
    #         rewards,
    #         next_observations,
    #         dones,
    #     ) = batch
    #     with torch.no_grad():
    #         target_q = self.q_target(observations, actions)
    #         next_v = self.vf(next_observations)
    #         v = self.vf(observations)
    #         adv = target_q.detach() - v
    #         v_loss = asymmetric_l2_loss(adv, self.iql_tau, keep_dim=True)

    #     N = len(observations)
    #     self._save_values(target_q.detach().cpu().numpy(), N, 'vtarget_epoch{}'.format(str(epoch)))
    #     self._save_values(v.detach().cpu().numpy(), N, 'v_epoch{}'.format(str(epoch)))
    #     self._save_values(v_loss.detach().cpu().numpy(), N, 'vloss_epoch{}'.format(str(epoch)))
    #     self._save_values(adv.detach().cpu().numpy(), N, 'adv_epoch{}'.format(str(epoch)))

        
    #     rewards = rewards.squeeze(dim=-1)
    #     dones = dones.squeeze(dim=-1)

    #     with torch.no_grad():
    #         targets = rewards + (1.0 - dones.float()) * self.discount * next_v.detach()
    #         qs = self.qf.both(observations, actions)
    #         q_mean = (qs[0] + qs[1]) / 2
    #         q_loss = ((qs[0] - targets) ** 2 + (qs[1] - targets) ** 2) / len(qs)

    #     self._save_values(targets.detach().cpu().numpy(), N, 'qtarget_epoch{}'.format(str(epoch)))
    #     self._save_values(q_mean.detach().cpu().numpy(), N, 'q_epoch{}'.format(str(epoch)))
    #     self._save_values(q_loss.detach().cpu().numpy(), N, 'qloss_epoch{}'.format(str(epoch)))


    #     with torch.no_grad():
    #         exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
    #         policy_out = self.actor(observations)
    #         bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
    #         policy_loss = exp_adv * bc_losses

    #     self._save_values(exp_adv.detach().cpu().numpy(), N, 'expadv_epoch{}'.format(str(epoch)))
    #     self._save_values(bc_losses.detach().cpu().numpy(), N, 'bcloss_epoch{}'.format(str(epoch)))
    #     self._save_values(policy_loss.detach().cpu().numpy(), N, 'policyloss_epoch{}'.format(str(epoch)))
    

        

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q.detach() - v
        ####################################### add clamp
        # adv = torch.clamp(adv, -self.clip_max, self.clip_max)
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)

        log_dict["value_loss"] = v_loss.item()
        log_dict["v_mean"] = torch.mean(v).item()
        log_dict["advantage_mean"] = torch.mean(adv).item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        ############################ add clip norm
        # torch.nn.utils.clip_grad_norm_(self.vf.parameters(), 1)
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
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        # q = self.qf(observations, actions)
        # q_loss = F.mse_loss(q, targets) 

        log_dict["q_loss"] = q_loss.item()
        log_dict['q_mean'] = (torch.mean(qs[0]).item() + torch.mean(qs[1]).item()) / 2
        # log_dict['q_mean'] = torch.mean(q).item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        ############################ add clip norm
        # torch.nn.utils.clip_grad_norm_(self.qf.parameters(), 1)
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)

        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        ############################ add clip norm
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
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
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

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
    wandb_init(asdict(config))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    ##### corrupt
    if (config.corruption_reward or config.corruption_dynamics or config.corruption_obs or config.corruption_acts):
        if config.random_corruption:
            print('random corruption')
            random_num = np.random.random(dataset["rewards"].shape)
            indexs = np.where(random_num < config.corruption_rate)
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
            if config.corruption_reward:
                print('attack reward')
                random_num = np.random.random(dataset["rewards"].shape)
                indexs = np.where(random_num < config.corruption_rate)
                # corrupt rewards
                dataset["rewards"][indexs] *= - config.corruption_range
            
            if config.corruption_dynamics:
                print('attack dynamics')
                env_dir = 'walker2d_new_diverse' if config.env_name.startswith('walker2d') else 'halfcheetah'
                print('loading path {}'.format(os.path.join('./log_attack_data/{}/'.format(env_dir), "attack_data_corrupt{}_rate{}.pt".format(config.corruption_range, config.corruption_rate))))
                data_dict = torch.load(os.path.join('./log_attack_data/{}/'.format(env_dir), "attack_data_corrupt{}_rate{}.pt".format(config.corruption_range, config.corruption_rate)))
                attack_indexs, next_observations  = data_dict['index'], data_dict['next_observations']
                dataset["next_observations"][attack_indexs] = next_observations

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

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
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)


    q_network = TwinQ(state_dim, action_dim).to(config.device)
    # q_network = QFunction(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, action_dim, max_action)
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env_name}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor


    # evaluations = []
    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = replay_buffer.sample(config.batch_size)
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
    # parser.add_argument('--iql_deterministic', action='store_true', default=False)
    parser.add_argument('--corruption_reward', action='store_true', default=False)
    parser.add_argument('--corruption_dynamics', action='store_true', default=False)
    parser.add_argument('--corruption_acts', action='store_true', default=False)
    parser.add_argument('--corruption_obs', action='store_true', default=False)
    parser.add_argument('--random_corruption', action='store_true', default=False)
    parser.add_argument('--corruption_range', type=float, default=0.0)
    parser.add_argument('--corruption_rate', type=float, default=0.0)
    parser.add_argument('--checkpoints_path', type=str, default=None)
    args = parser.parse_args()
    print(args)

    ### modify config
    TrainConfig.env_name = args.env_name
    TrainConfig.train_seed = args.seed
    # TrainConfig.iql_deterministic = args.iql_deterministic
    TrainConfig.corruption_reward = args.corruption_reward
    TrainConfig.corruption_dynamics = args.corruption_dynamics
    TrainConfig.corruption_acts = args.corruption_acts
    TrainConfig.corruption_obs = args.corruption_obs
    TrainConfig.random_corruption = args.random_corruption
    TrainConfig.corruption_range = args.corruption_range
    TrainConfig.corruption_rate = args.corruption_rate
    TrainConfig.checkpoints_path = args.checkpoints_path

    ## modify config
    if TrainConfig.corruption_reward:
        group_name_center = 'reward' 
    elif TrainConfig.corruption_dynamics:
        group_name_center = 'dynamics'
    elif TrainConfig.corruption_acts:
        group_name_center = 'actions'
    elif TrainConfig.corruption_obs:
        group_name_center = 'observations'
    else:
        group_name_center = 'no_attack'

    group_name_center = 'random_' + group_name_center if TrainConfig.random_corruption else 'adversarial_' + group_name_center
    TrainConfig.group = TrainConfig.group + '-{}'.format(group_name_center) 
    TrainConfig.group = TrainConfig.group + '_{}'.format(TrainConfig.env_name.split('-')[0])
    TrainConfig.name = "IQL_corrupt{}_{}_single_baseline_seed{}".format(TrainConfig.corruption_range, TrainConfig.corruption_rate, TrainConfig.train_seed)
    # TrainConfig.name = "IQL_corrupt{}_{}_baseline_norm2_seed{}".format(TrainConfig.corruption_range, TrainConfig.corruption_rate, TrainConfig.train_seed)
    TrainConfig.name = f"{TrainConfig.name}-{TrainConfig.env_name}-{str(uuid.uuid4())[:8]}"
    if TrainConfig.checkpoints_path is not None:
        TrainConfig.checkpoints_path = os.path.join(TrainConfig.checkpoints_path, TrainConfig.group, TrainConfig.name)
    
    train(TrainConfig)