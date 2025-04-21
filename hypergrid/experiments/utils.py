import torch
import numpy as np
from gfn.gflownet import TBGFlowNet

from torchtyping import TensorType as TT

from collections import Counter
from typing import Dict, Optional

from gfn.env import Env
from gfn.gflownet import GFlowNet
from gfn.states import States


def get_terminating_state_dist_pmf(env: Env, states: States) -> TT["n_states", float]:
    states_indices = env.get_terminating_states_indices(states).cpu().numpy().tolist()
    counter = Counter(states_indices)
    counter_list = [
        counter[state_idx] if state_idx in counter else 0
        for state_idx in range(env.n_terminating_states)
    ]

    return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)


def validate(
    env: Env,
    gflownet: GFlowNet,
    n_validation_samples: int = 20000,
    visited_terminating_states: Optional[States] = None,
) -> Dict[str, float]:
    """Evaluates the current gflownet on the given environment.

    This is for environments with known target reward. The validation is done by
    computing the l1 distance between the learned empirical and the target
    distributions.

    Args:
        env: The environment to evaluate the gflownet on.
        gflownet: The gflownet to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training. If given, the pmf is obtained from
            these last n_validation_samples states. Otherwise, n_validation_samples are resampled for evaluation.

    Returns: A dictionary containing the l1 validation metric. If the gflownet
        is a TBGFlowNet, i.e. contains LogZ, then the (absolute) difference
        between the learned and the target LogZ is also returned in the dictionary.
    """

    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
    if isinstance(true_dist_pmf, torch.Tensor):
        true_dist_pmf = true_dist_pmf.cpu()
    else:
        # The environment does not implement a true_dist_pmf property, nor a log_partition property
        # We cannot validate the gflownet
        return {}

    logZ = None
    if isinstance(gflownet, TBGFlowNet):
        logZ = gflownet.logZ.item()
    if visited_terminating_states is None:
        terminating_states = gflownet.sample_terminating_states(n_validation_samples)
    else:
        terminating_states = visited_terminating_states[-n_validation_samples:]

    final_states_dist_pmf = get_terminating_state_dist_pmf(env, terminating_states)
    
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    kl_dist = (true_dist_pmf * torch.log(true_dist_pmf / (final_states_dist_pmf + 1e-9))).sum().item()
    num_modes = get_terminating_state_mode_counts(env, terminating_states)
    validation_info = {"l1_dist": l1_dist, "kl_dist": kl_dist, "num_modes": num_modes}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)
    return validation_info

def get_terminating_state_mode_counts(env, states):
    visited = [0] * (2**env.ndim)
    pow2 = torch.pow(torch.tensor([2]*env.ndim), torch.arange(env.ndim))
    for state in states:
        state_raw = state.tensor
        ax = abs(state_raw / (env.height - 1) - 0.5).cpu()
        bx = (state_raw > 0.5*(env.height - 1)).long().cpu()
        if ((0.3 < ax) * (ax < 0.4)).prod(-1) == 1:
            visited[torch.dot(bx, pow2).item()]=1
    return np.sum(visited)
        
