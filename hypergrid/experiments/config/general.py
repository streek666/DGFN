from ml_collections.config_dict import ConfigDict


def get_config(seed : str):
    config = ConfigDict(
        {
            'seed': int(seed),
            'device': 'cuda',
            'validation_interval': 100,
            'validation_samples': 80000,
            'wandb_project': 'hypergrid-hard',    # if empty, do not use wandb
            'n_envs': 16,
            'n_trajectories': 800000
        }
    )

    return config
