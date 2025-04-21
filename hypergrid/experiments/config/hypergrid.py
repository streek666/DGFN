from ml_collections.config_dict import ConfigDict


def get_config(env_type):

    env_config = {
        "standard" : ConfigDict({
            'env_name': 'Hypergrid',
            'reward_type': 'standard',
            'ndim': 4,
            'height': 20,
            'R0': 0.001,
            'R1': 0.5,
            'R2': 2.0
        }),
        "hard" : ConfigDict({
            'env_name': 'Hypergrid',
            'reward_type': 'hard',
            'ndim': 4,
            'height': 20,
            'R0': 0.0001,
            'R1': 1.0,
            'R2': 3.0
        }),
        "hard2" : ConfigDict({
            'env_name': 'Hypergrid',
            'reward_type': 'hard',
            'ndim': 4,
            'height': 20,
            'R0': 0.0001,
            'R1': 1.0,
            'R2': 1.5
        }),
        "hard3" : ConfigDict({
            'env_name': 'Hypergrid',
            'reward_type': 'hard',
            'ndim': 4,
            'height': 20,
            'R0': 0.0001,
            'R1': 1.0,
            'R2': 1.2
        }),
        "hard4" : ConfigDict({
            'env_name': 'Hypergrid',
            'reward_type': 'hard',
            'ndim': 4,
            'height': 20,
            'R0': 0.0001,
            'R1': 1.0,
            'R2': 0.01
        })
        ,
        "hard5" : ConfigDict({
            'env_name': 'Hypergrid',
            'reward_type': 'hard',
            'ndim': 5,
            'height': 20,
            'R0': 0.0001,
            'R1': 0.01,
            'R2': 1.00
        })
        ,
        "hard6" : ConfigDict({
            'env_name': 'Hypergrid',
            'reward_type': 'hard6',
            'ndim': 5,
            'height': 20,
            'R0': 0.0001,
            'R1': -0.000099,
            'R2': 0.999999
        })
    }

    return env_config[env_type]
