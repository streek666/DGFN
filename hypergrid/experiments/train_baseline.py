r"""
Adapted from torchgfn library
https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid.py
"""

import torch
import wandb
from tqdm import tqdm, trange
import numpy as np

from gfn.containers import ReplayBuffer
from gfn.gflownet import (
    DBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
    FMGFlowNet,
)
from gfn.env import Env
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from experiments.utils import validate
from gfn.utils.modules import DiscreteUniform, NeuralNet
from ml_collections.config_dict import ConfigDict

from gfn.loss import F_alpha, G_alpha, H_alpha, V_alpha, G_Loss, F_Loss

def train_baseline(
    env: Env,
    experiment_name: str,
    general_config: ConfigDict,
    algo_config: ConfigDict
):
    use_wandb = len(general_config.wandb_project) > 0

    if algo_config.uniform_pb:
        experiment_name += '_uniform-pb'
    else:
        experiment_name += '_learnt-pb'

    # Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet = None
    pb_module = None
    # We need a DiscretePFEstimator and a DiscretePBEstimator

    pf_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )
    if not algo_config.uniform_pb:
        pb_module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=algo_config.net.hidden_dim,
            n_hidden_layers=algo_config.net.n_hidden,
            torso=pf_module.torso if algo_config.tied else None,
        )
    if algo_config.uniform_pb:
        pb_module = DiscreteUniform(env.n_actions - 1)

    assert (
        pf_module is not None
    ), f"pf_module is None. Command-line arguments: {algo_config}"
    assert (
        pb_module is not None
    ), f"pb_module is None. Command-line arguments: {algo_config}"
    
    pf_estimator = DiscretePolicyEstimator(env=env, module=pf_module, forward=True)
    pb_estimator = DiscretePolicyEstimator(env=env, module=pb_module, forward=False)
    
    if algo_config.name in ("DetailedBalance", "SubTrajectoryBalance", "TrajectoryBalance", "FlowMatching"):
        if algo_config.loss_type == 'G':
            loss_fn = G_Loss(G_alpha(algo_config.alpha), clip=algo_config.logr_clip)
        elif algo_config.loss_type == 'H':
            loss_fn = G_Loss(H_alpha(algo_config.alpha), clip=algo_config.logr_clip)
        elif algo_config.loss_type == 'V':
            loss_fn = G_Loss(V_alpha(algo_config.alpha), clip=algo_config.logr_clip)
        else:
            loss_fn = F_Loss(F_alpha(algo_config.alpha), clip=algo_config.logr_clip)
    
    if algo_config.name in ("DetailedBalance", "SubTrajectoryBalance"):
        # We need a LogStateFlowEstimator
        assert (
            pf_estimator is not None
        ), f"pf_estimator is None. Arguments: {algo_config}"
        assert (
            pb_estimator is not None
        ), f"pb_estimator is None. Arguments: {algo_config}"

        module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=1,
            hidden_dim=algo_config.net.hidden_dim,
            n_hidden_layers=algo_config.net.n_hidden,
            torso=pf_module.torso if algo_config.tied else None,
        )

        logF_estimator = ScalarEstimator(env=env, module=module)
        if algo_config.name == "DetailedBalance":
            gflownet = DBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
                on_policy=True if algo_config.replay_buffer_size == 0 else False,
                loss_fn=loss_fn,
            )
        else:
            gflownet = SubTBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                logF=logF_estimator,
                on_policy=True if algo_config.replay_buffer_size == 0 else False,
                weighting=algo_config.subTB_weighting,
                lamda=algo_config.subTB_lambda,
                loss_fn=loss_fn,
            )
    elif algo_config.name == "TrajectoryBalance":
        gflownet = TBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            on_policy=True if algo_config.replay_buffer_size == 0 else False,
            loss_fn=loss_fn,
        )
    elif algo_config.name == "FlowMatching":
        print(loss_fn)
        gflownet = FMGFlowNet(
            logF=pf_estimator,
            loss_fn=loss_fn,
        )

    # Initialize the replay buffer ?

    replay_buffer = None
    if algo_config.replay_buffer_size > 0:
        if algo_config.name in ("TrajectoryBalance", "SubTrajectoryBalance"):
            objects_type = "trajectories"
        elif algo_config.name in ("DetailedBalance"):
            objects_type = "transitions"
        elif algo_config.name in ("FlowMatching"):
            objects_type = "states"
        else:
            raise NotImplementedError(f"Unknown GFN: {algo_config.name}")
        replay_buffer = ReplayBuffer(
            env, objects_type=objects_type, capacity=algo_config.replay_buffer_size
        )

    # 3. Create the optimizer

    # Policy parameters have their own LR.
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": algo_config.learning_rate,
        }
    ]
    param_list = params[0]["params"]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": algo_config.learning_rate_Z,
            }
        )
        param_list = param_list + params[-1]["params"]

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    kl_history, l1_history, nstates_history = [], [], []

    # Train loop
    n_iterations = general_config.n_trajectories // general_config.n_envs
    with trange(n_iterations) as pbar:
        for iteration in pbar:
            trajectories = gflownet.sample_trajectories(n_samples=general_config.n_envs)
            training_samples = gflownet.to_training_samples(trajectories)
            if replay_buffer is not None:
                with torch.no_grad():
                    replay_buffer.add(training_samples)
                    training_objects = replay_buffer.sample(
                        n_trajectories=general_config.n_envs
                    )
            else:
                training_objects = training_samples

            optimizer.zero_grad()
            loss = gflownet.loss(training_objects)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(param_list, algo_config.grad_clip)
            optimizer.step()

            visited_terminating_states.extend(trajectories.last_states)

            states_visited += len(trajectories)

            to_log = {"loss": loss.item(), "states_visited": states_visited, 'grad_norm':grad_norm.item()}
            pbar.set_description(f"loss: {loss.item()} grad: {grad_norm.item()}")
            if use_wandb:
                wandb.log(to_log, step=iteration)
            if (iteration + 1) % general_config.validation_interval == 0:
                validation_info = validate(
                    env,
                    gflownet,
                    general_config.validation_samples,
                    visited_terminating_states,
                )
                if use_wandb:
                    wandb.log(validation_info, step=iteration)
                to_log.update(validation_info)
                tqdm.write(f"{iteration}: {to_log}")

                kl_history.append(to_log["kl_dist"])
                l1_history.append(to_log["l1_dist"])
                nstates_history.append(to_log["states_visited"])

            if (iteration + 1) % 1000 == 0:
                np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
                np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
                np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))

    np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
    np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
    np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))
