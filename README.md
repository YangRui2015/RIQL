# Towards Robust Offline RL under Diverse Data Corruption

This repo contains the official implemented Robust IQL (RIQL) algorithm for the **ICLR 2024 spotlight paper** (⭐ top 5%), <ins>"Towards Robust Offline Reinforcement Learning under Diverse Data Corruption"</ins>. This code is implemented based on the open-sourced [CORL library](https://github.com/tinkoff-ai/CORL).



## Getting started
Install torch>=1.7.1, gym, mujoco_py, d4rl, pyrallis, wandb, tqdm.


### Under Random Data Corruption
Run RIQL with random observation corruption:
```bash
CUDA_VISIBLE_DEVICES=${gpu} python RIQL.py --corruption_mode random  --corrupt_obs --corruption_range ${corruption_range} --corruption_rate ${corruption_rate}  --env_name ${env_name} --seed ${seed} 
```
'env_name' can be 'halfcheetah-medium-replay-v2',  'walker2d-medium-replay-v2', 'hopper-medium-replay-v2', .... 

'corruption\_range' and 'corruption\_rate' are set to 1.0 and 0.3 by default. 

Replace '--corrupt_obs' with '--corrupt_reward', '--corrupt_acts', and '--corrupt_dynamics' to enforce corruption on rewards, actions, and dynamics.

### Under Adversarial Data Corruption

Run RIQL with adversarial observation corruption:
```bash
CUDA_VISIBLE_DEVICES=${gpu} python RIQL.py --corruption_mode adversarial --corruption_obs --corruption_range ${corruption_range} --corruption_rate ${corruption_rate}  --env_name ${env_name} --seed ${seed} 
```

The adversarial attacks on obs, actions, and next-obs require performing gradient-based attack and will save the corrupted data. After saving the corrupted data, we will load these data for later training.


### Clean Data
To run the algorithm with a clean dataset, you can run the following command without specifying the corruption-related parameters
```bash
CUDA_VISIBLE_DEVICES=${gpu} python RIQL.py  --env_name ${env_name} --seed ${seed} 
```


### Baselines
You can replace the RIQL.py with other baselines, such as IQL.py, CQL.py, EDAC.py, and MSG.py, to run IQL, CQL, EDAC, and MSG. 



## Citation
If you find our work helpful for your research, please cite:
```
@inproceedings{yang2023towards,
  title={Towards Robust Offline Reinforcement Learning under Diverse Data Corruption},
  author={Yang, Rui and Zhong, Han and Xu, Jiawei and Zhang, Amy and Zhang, Chongjie and Han, Lei and Zhang, Tong},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=5hAMmCU0bK}
}
```





