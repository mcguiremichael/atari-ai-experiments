
from mcts import run_mcts

import numpy as np
import torch

def simple_repr(s, a):

    return torch.ones((1, 10))

def simple_dynamics(s, a):

    if a == 3:
        reward = 0
    else:
        reward = 0

    out = s.clone()
    out[0,a] += 1

    return out, torch.Tensor([[reward]])

def simple_policy(obs):

    policy = torch.zeros((1,6))
    policy[0,3] = 1
    policy[0,[0,1,2,4,5]] = 0.1
    policy = policy / policy.sum()
    value = 100 * np.random.random() - 50

    #return torch.ones((1,6)) / 6, torch.Tensor([[value]])
    print(policy)
    return policy, torch.Tensor([[value]])


def trivial_experiment():

    output_policy, _, _ = run_mcts(
        simple_repr,
        simple_policy,
        simple_dynamics,
        torch.ones((1,10)),
        torch.zeros((1, 5)),
        50,
        1,
        5,
        6
    )

    print(output_policy)

if __name__ == "__main__":
    trivial_experiment()
