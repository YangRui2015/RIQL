from typing import Dict

import os
import gym
import d4rl
import torch
import torch.nn as nn
import numpy as np


NAME_DICT = {
    "obs": "observations",
    "act": "actions",
    "rew": "rewards",
    "next_obs": "next_observations",
}

MODEL_PATH = {
    "EDAC": "model_path_for_attack",
}

dataset_path = "./adversarial_data/"



class Attack:
    def __init__(
        self,
        env_name: str,
        agent_name: str,
        dataset: Dict[str, np.ndarray],
        model_path: str,
        dataset_path: str,
        update_times: int = 100,
        step_size: float = 0.01,
        force_attack: bool = False,
        resample_indexs: bool = False,
        seed: int = 2023,
        device: str = "cpu",
    ):
        self.env_name = env_name
        self.agent_name = agent_name
        self.dataset = dataset
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.update_times = update_times
        self.step_size = step_size
        self.force_attack = force_attack
        self.device = device
        self.resample_indexs = resample_indexs

        self._np_rng = np.random.RandomState(seed)
        self._th_rng = torch.Generator()
        self._th_rng.manual_seed(seed)

        self.attack_indexs = None
        self.original_indexs = None

        env = gym.make(env_name)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        env.close()

    def set_attack_config(
        self,
        corruption_name,
        corruption_tag,
        corruption_rate,
        corruption_range,
        corruption_random,
    ):
        self.corruption_tag = NAME_DICT[corruption_tag]
        self.corruption_rate = corruption_rate
        self.corruption_range = corruption_range
        self.corruption_random = corruption_random
        self.new_dataset_path = os.path.expanduser(
            os.path.join(self.dataset_path, self.env_name)
        )
        attack_mode = "random" if self.corruption_random else "adversarial"
        self.new_dataset_file = f"{self.agent_name}_{attack_mode}_{corruption_name}range{corruption_range}_rate{corruption_rate}.pth"

        self.corrupt_func = getattr(self, f"corrupt_{corruption_tag}")
        self.loss_Q = getattr(self, f"loss_Q_for_{corruption_tag}")
        if self.attack_indexs is None or self.resample_indexs:
            self.attack_indexs, self.original_indexs = self.sample_indexs()

    def load_model(self):
        model_path = self.model_path
        state_dict = torch.load(model_path, map_location=self.device)
        if self.agent_name == "IQL":
            from IQL_adversarial import GaussianPolicy, DeterministicPolicy, TwinQ

            self.actor = (
                DeterministicPolicy(
                    self.state_dim, self.action_dim, self.max_action, n_hidden=2
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                TwinQ(self.state_dim, self.action_dim, n_hidden=2)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["qf"])
        elif self.agent_name == "CQL":
            from CQL import TanhGaussianPolicy, CriticFunctions

            self.actor = (
                TanhGaussianPolicy(
                    self.state_dim,
                    self.action_dim,
                    max_action=self.max_action,
                    orthogonal_init=True,
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                CriticFunctions(self.state_dim, self.action_dim, orthogonal_init=True)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.critic_1.load_state_dict(state_dict["critic1"])
            self.critic.critic_2.load_state_dict(state_dict["critic2"])
        elif self.agent_name == "EDAC":
            from EDAC import Actor, VectorizedCritic

            self.actor = (
                Actor(
                    self.state_dim,
                    self.action_dim,
                    hidden_dim=256,
                    n_hidden=2,
                    max_action=self.max_action,
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                VectorizedCritic(self.state_dim, self.action_dim, hidden_dim=256)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["critic"])
        elif self.agent_name == "MSG":
            from MSG import Actor, VectorizedCritic

            self.actor = (
                Actor(
                    self.state_dim,
                    self.action_dim,
                    hidden_dim=256,
                    n_hidden=2,
                    max_action=self.max_action,
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                VectorizedCritic(self.state_dim, self.action_dim, hidden_dim=256)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["critic"])
        else:
            raise NotImplementedError
        print(f"Load model from {model_path}")

    
    def sample_indexs(self):
        # if indexs is None:
        indexs = np.arange(len(self.dataset["rewards"]))
        random_num = self._np_rng.random(len(indexs))
        attacked = np.where(random_num < self.corruption_rate)[0]
        original = np.where(random_num >= self.corruption_rate)[0]
        return indexs[attacked], indexs[original]

    def sample_para(self, shape, std):
        return (
            2
            * self.corruption_range
            * std
            * (torch.rand(shape, generator=self._th_rng).to(self.device) - 0.5)
        )

    def sample_data(self, shape):
        return self._np_rng.uniform(-self.corruption_range, self.corruption_range, size=shape)


    def optimize_para(self, para, std, obs, act=None):
        for _ in range(self.update_times):
            para = torch.nn.Parameter(para.clone(), requires_grad=True)
            optimizer = torch.optim.Adam([para], lr=self.step_size * self.corruption_range)
            loss = self.loss_Q(para, obs, act, std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            para = torch.clamp(para, -self.corruption_range, self.corruption_range).detach()
        return para * std

    def loss_Q_for_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        qvalue = self.critic(noised_obs, action)
        return qvalue.mean()

    def loss_Q_for_act(self, para, observation, action, std):
        noised_act = action + para * std
        qvalue = self.critic(observation, noised_act)
        return qvalue.mean()

    def loss_Q_for_next_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        action = self.actor(noised_obs).detach()
        qvalue = self.critic(observation, action)
        return qvalue.mean()

    def loss_Q_for_rew(self):
        # Just Placeholder
        raise NotImplementedError

    def split_gradient_attack(self, original_obs_torch, original_act_torch, std_torch):
        if self.corruption_tag == 'observations' or self.corruption_tag == 'next_observations':
            attack_data = np.zeros(original_obs_torch.shape)
        elif self.corruption_tag == 'actions':
            attack_data = np.zeros(original_act_torch.shape)
        else:
            raise NotImplementedError

        split = 10
        pointer = 0
        M = original_obs_torch.shape[0]
        for i in range(split):
            number = M // split if i < split - 1 else M - pointer
            temp_act = original_act_torch[pointer : pointer + number]
            temp_obs = original_obs_torch[pointer : pointer + number]

            if self.corruption_tag == 'observations' or self.corruption_tag == 'next_observations':
                para = self.sample_para(temp_obs.shape, std_torch)
            elif self.corruption_tag == 'actions':
                para = self.sample_para(temp_act.shape, std_torch)
            else:
                raise NotImplementedError

            para = self.optimize_para(para, std_torch, temp_obs, temp_act)
            noise = para.cpu().numpy()
            if self.corruption_tag == 'observations' or self.corruption_tag == 'next_observations':
                attack_data[pointer : pointer + number] = noise + temp_obs.cpu().numpy()
            elif self.corruption_tag == 'actions':
                attack_data[pointer : pointer + number] = noise + temp_act.cpu().numpy()
            else:
                raise NotImplementedError

            pointer += number

        return attack_data

    
    def corrupt_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_obs = original_obs + self.sample_data(original_obs.shape) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()
            original_act = self.dataset["actions"][self.attack_indexs].copy()
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            std_torch = torch.from_numpy(self.dataset[self.corruption_tag].std(axis=0)).view(1, -1).to(self.device)

            # adversarial attack obs
            attack_obs = self.split_gradient_attack(original_obs_torch, original_act_torch, std_torch)

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")
            self.save_dataset(attack_obs)

        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset

    def corrupt_act(self, dataset):
        # load original act
        original_act = self.dataset[self.corruption_tag][self.attack_indexs].copy()

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_act = original_act + self.sample_data(original_act.shape) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()
        
            original_obs = self.dataset["observations"][self.attack_indexs].copy()
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            std_torch = torch.from_numpy(self.dataset[self.corruption_tag].std(axis=0)).view(1, -1).to(self.device)

            # adversarial attack act
            attack_act = self.split_gradient_attack(original_obs_torch, original_act_torch, std_torch)

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")
            self.save_dataset(attack_act)

        dataset[self.corruption_tag][self.attack_indexs] = attack_act
        return dataset

    def corrupt_rew(self, dataset):
        # load original rew
        original_rew = self.dataset[self.corruption_tag][self.attack_indexs].copy()

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_rew = self.sample_data(original_rew.shape) * 30
            print(f"Random attack {self.corruption_tag}")
        else:
            attack_rew = original_rew.copy() * -self.corruption_range
            print(f"Adversarial attack {self.corruption_tag}")

            self.save_dataset(attack_rew)
        dataset[self.corruption_tag][self.attack_indexs] = attack_rew
        return dataset

    def corrupt_next_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_obs = original_obs + self.sample_data(original_obs.shape) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()
            original_act = self.dataset["actions"][self.attack_indexs].copy()
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            std_torch = torch.from_numpy(self.dataset[self.corruption_tag].std(axis=0)).view(1, -1).to(self.device)

            # adversarial attack obs
            attack_obs = self.split_gradient_attack(original_obs_torch, original_act_torch, std_torch)

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")

            self.save_dataset(attack_obs)

        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset

    def clear_gpu_cache(self):
        self.actor.to("cpu")
        self.critic.to("cpu")
        torch.cuda.empty_cache()

    def save_dataset(self, attack_datas):
        ### save data
        save_dict = {}
        save_dict["attack_indexs"] = self.attack_indexs
        save_dict["original_indexs"] = self.original_indexs
        save_dict[self.corruption_tag] = attack_datas
        if not os.path.exists(self.new_dataset_path):
            os.makedirs(self.new_dataset_path)
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        torch.save(save_dict, dataset_path)
        print(f"Save attack dataset in {dataset_path}")

    def get_original_data(self, indexs):
        dataset = {}
        dataset["observations"] = self.dataset["observations"][indexs]
        dataset["actions"] = self.dataset["actions"][indexs]
        dataset["rewards"] = self.dataset["rewards"][indexs]
        dataset["next_observations"] = self.dataset["next_observations"][indexs]
        dataset["terminals"] = self.dataset["terminals"][indexs]
        return dataset

    def attack(self, dataset):
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        if os.path.exists(dataset_path) and not self.force_attack:
            new_dataset = torch.load(dataset_path)
            print(f"Load new dataset from {dataset_path}")
            original_indexs, attack_indexs, attack_datas = (
                new_dataset["original_indexs"],
                new_dataset["attack_indexs"],
                new_dataset[self.corruption_tag],
            )
            ori_dataset = self.get_original_data(original_indexs)
            dataset[self.corruption_tag][attack_indexs] = attack_datas
            self.attack_indexs = attack_indexs
            return ori_dataset, dataset
        else:
            ori_dataset = self.get_original_data(self.original_indexs)
            att_dataset = self.corrupt_func(dataset)
            return ori_dataset, att_dataset


def attack_dataset(config, dataset, use_original=False): 
    corruption_agent = 'EDAC'
    attack_agent = Attack(
        env_name=config.env_name,
        agent_name=corruption_agent,
        dataset=dataset,
        model_path=MODEL_PATH[corruption_agent],
        dataset_path=dataset_path, 
        resample_indexs=True,
        force_attack=False , 
        device=config.device,
        seed=config.train_seed,
    )
    corruption_random = config.corruption_mode == "random"
    attack_params = {
        "corruption_rate": config.corruption_rate,
        "corruption_range": config.corruption_range,
        "corruption_random": corruption_random,
    }
    name = ""
    #### the ori_dataset refers to the part of unattacked data
    ### the att_dataset refers to attacked data + unattacked data
    if config.corruption_obs:
        name += "obs_"
        attack_agent.set_attack_config(name, "obs", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset

    if config.corruption_acts:
        name += "act_"
        attack_agent.set_attack_config(name, "act", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset

    if config.corruption_reward:
        name += "rew_"
        attack_agent.set_attack_config(name, "rew", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset

    if config.corruption_dynamics:
        name += "next_obs_"
        attack_agent.set_attack_config(name, "next_obs", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if use_original else att_dataset


    return dataset, attack_agent.attack_indexs