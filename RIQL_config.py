
def get_config(TrainConfig):
    TrainConfig.num_critics = 5
    key = TrainConfig.env_name.split('-')[0]
    
    if TrainConfig.corruption_mode == 'random':
        if TrainConfig.corrupt_obs:
            TrainConfig.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.25, 
            }[key]
            if key == 'hopper':
                TrainConfig.num_critics = 3

        elif TrainConfig.corrupt_acts:
            TrainConfig.sigma = {
                'halfcheetah': 0.5,
                'walker2d': 0.5,
                'hopper': 0.1,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
            if key == 'halfcheetah':
                TrainConfig.num_critics = 3

        elif TrainConfig.corrupt_reward:
            TrainConfig.sigma = {
                'halfcheetah': 3.0,
                'walker2d': 3.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
            if key == 'hopper':
                TrainConfig.num_critics = 3

        elif TrainConfig.corrupt_dynamics:
            TrainConfig.sigma = {
                'halfcheetah': 3.0,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.25,
                'hopper': 0.5,
            }[key]
            if key == 'walker2d':
                TrainConfig.num_critics = 3
    
    elif TrainConfig.corruption_mode == 'adversarial':
        if TrainConfig.corrupt_obs:
            TrainConfig.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.25, 
            }[key]
        elif TrainConfig.corrupt_acts:
            TrainConfig.sigma = {
                'halfcheetah': 1.0,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
        elif TrainConfig.corrupt_reward:
            TrainConfig.sigma = {
                'halfcheetah': 1.0,
                'walker2d': 3.0,
                'hopper': 0.1,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
        elif TrainConfig.corrupt_dynamics:
            TrainConfig.sigma = {
                'halfcheetah': 1.0,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.5,
            }[key]
        