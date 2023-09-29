# Towards Robust Offline RL under Diverse Data Corruption

This repo contains the official implemented Robust IQL (RIQL) algorithm for the paper "Towards Robust Offline RL under Diverse Data Corruption". This code is implemented based on CORL (Research-oriented Deep Offline Reinforcement Learning Library).



## Getting started
Install torch>=1.13, gym, mujoco_py, d4rl, pyrallis, wandb, tqdm


### Under Random Attack
Run RIQL with random reward corruption:
```bash
CUDA_VISIBLE_DEVICES=${gpu} python RIQL.py --random_corruption  --corruption_reward --corruption_range ${corruption_range} --corruption_rate ${corruption_rate}  --env_name ${env_name} --seed ${seed} --use_UW 
```
\${env_name}\$ can be 'halfcheetah-medium-v2' and 'walker2d-medium-replay-v2'.\${corruption_range}\$ and \${corruption_rate}\$ are hyperparameters listed in our appendix. 

Run RIQL with random dynamics corruption:
```bash
CUDA_VISIBLE_DEVICES=${gpu} python RIQL.py --random_corruption  --corruption_dynamics --corruption_range ${corruption_range} --corruption_rate ${corruption_rate}  --env_name ${env_name} --seed ${seed} --use_UW 
```

### Under Adversarial Attack

Run RIQL with adversarial reward corruption:
```bash
CUDA_VISIBLE_DEVICES=${gpu} python RIQL.py --corruption_reward --corruption_range ${corruption_range} --corruption_rate ${corruption_rate}  --env_name ${env_name} --seed ${seed} --use_UW 
```

Run RIQL with adversarial dynamics corruption:
```bash
CUDA_VISIBLE_DEVICES=${gpu} python RIQL.py  --corruption_dynamics --corruption_range ${corruption_range} --corruption_rate ${corruption_rate}  --env_name ${env_name} --seed ${seed} --use_UW 
```
Note the adversarial dynamics attack needs to load an offline dataset in the 'load_attack_data' directory with corresponding attack ratio and attack scale. 

## Baselines
You can replace the RIQL.py with other baselines, such as IQL.py, EDAC.py, and MSG.py, to run IQL, EDAC, and MSG. 



