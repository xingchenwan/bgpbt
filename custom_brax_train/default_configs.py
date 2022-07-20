# Default configs provided by brax using PPO
PPO_DEFAULT_CONFIGS = {
    'ant': {
        'num_timesteps': 32_000_000,
        'log_frequency': 20,
        'reward_scaling': 10,
        'episode_length': 1000,
        'normalize_observations': True,
        "action_repeat": 1, "unroll_length": 5, "num_minibatches": 32,
        "num_update_epochs": 4, "discounting": 0.97, "learning_rate": 3e-4,
        "entropy_cost": 1e-2, "num_envs": 2048, "batch_size": 1024
    },
    'humanoid': {
        'num_timesteps': 52_000_000,
        'log_frequency': 20,
        'reward_scaling': 0.1,
        'episode_length': 1000,
        'normalize_observations': True,
        "action_repeat": 1, "unroll_length": 10, "num_minibatches": 32,
        "num_update_epochs": 8, "discounting": 0.97, "learning_rate": 3e-4,
        "entropy_cost": 1e-3, "num_envs": 2048, "batch_size": 1024
    },
    'halfcheetah': {
        'num_timesteps': 100_000_000,
        'log_frequency': 20,
        'reward_scaling': 1,
        'episode_length': 1000,
        'normalize_observations': True,
        "action_repeat": 1, "unroll_length": 20, "num_minibatches": 32,
        "num_update_epochs": 8, "discounting": 0.95, "learning_rate": 3e-4,
        "entropy_cost": 1e-3, "num_envs": 2048, "batch_size": 512
    },
    'fetch': {
        'num_timesteps': 100_000_000,
        'log_frequency': 20,
        'reward_scaling': 5,
        'episode_length': 1000,
        'normalize_observations': True,
        "action_repeat": 1, "unroll_length": 20, "num_minibatches": 32,
        "num_update_epochs": 4, "discounting": 0.997, "learning_rate": 3e-4,
        "entropy_cost": 1e-3, "num_envs": 2048, "batch_size": 256
    },
    'grasp': {
        'num_timesteps': 600_000_000,
        'log_frequency': 20,
        'reward_scaling': 10,
        'episode_length': 1000,
        'normalize_observations': True,
        "action_repeat": 1, "unroll_length": 20, "num_minibatches": 32,
        "num_update_epochs": 2, "discounting": 0.99, "learning_rate": 3e-4,
        "entropy_cost": 1e-3, "num_envs": 2048, "batch_size": 256
    },
    'ur5e': {
        'num_timesteps': 20_000_000,
        'log_frequency': 20,
        'reward_scaling': 10,
        'episode_length': 1000,
        'normalize_observations': True,
        "action_repeat": 1, "unroll_length": 5, "num_minibatches": 32,
        "num_update_epochs": 4, "discounting": 0.95, "learning_rate": 2e-4,
        "entropy_cost": 1e-2, "num_envs": 2048, "batch_size": 1024,
        "max_devices_per_host": 8,
    },
    'reacher': {
        'num_timesteps': 100_000_000,
        'log_frequency': 20,
        'reward_scaling': 5,
        'episode_length': 1000,
        'normalize_observations': True,
        "action_repeat": 4, "unroll_length": 50, "num_minibatches": 32,
        "num_update_epochs": 8, "discounting": 0.95, "learning_rate": 3e-4,
        "entropy_cost": 1e-3, "num_envs": 2048, "batch_size": 256,
        "max_devices_per_host": 8,
    },
}
