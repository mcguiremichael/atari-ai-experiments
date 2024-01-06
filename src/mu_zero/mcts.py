
import torch
import torch.nn as nn
import numpy as np

import time

from typing import Tuple

def run_mcts(repr_net : nn.Module,
             policy_net : nn.Module,
             dynamics_net : nn.Module,
             state : torch.Tensor,
             action_history : torch.Tensor,
             max_search_iter : int,
             max_search_t : float,
             search_depth : int,
             action_space : int) -> Tuple[
                 torch.Tensor,
                 torch.Tensor,
                 torch.Tensor
             ]:
    """_summary_

    Args:
        repr_net (nn.Module): Network for generating state embedding
        policy_net (nn.Module): Network for producing policy, value estimate, reward estimate
        dynamics_net (nn.Module): Network for predictiong state change from state + action
        state (torch.Tensor): State input
        max_search_iter (int): Maximum number of searches to run
        max_search_t (float): Maximum amount of time to run searching for
        search_depth (int): Maximum depth in search tree to traverse
        action_space (int): Number of actions
        
    Returns:
        Tuple[ torch.Tensor, torch.Tensor ]: Raw network policy, derived policy after search
    """

    N_sa = [{} for i in range(len(state))]
    Q_sa = [{} for i in range(len(state))]
    R_sa = [{} for i in range(len(state))]
    O_sa = [{} for i in range(len(state))]

    num_search_iters = 0
    start_search_t = time.time()

    o_t = repr_net(state, action_history)
    p_t, v_t = policy_net(o_t)

    while (
        (num_search_iters < max_search_iter) and \
        (time.time() - start_search_t) < max_search_t
    ):
        
        unroll_mcts(policy_net,
                    dynamics_net,
                    search_depth,
                    action_space,
                    N_sa,
                    Q_sa,
                    R_sa,
                    O_sa,
                    o_t,
                    p_t=p_t,
                    v_t=v_t)
        
    output_policy = np.zeros_like(p_t)

    N = len(o_t)
    for i in range(N):
        curr_state = o_t[i]
        curr_state_hashed = hash_state(curr_state)
        available_actions = list(range(action_space))

        counts = np.array([N_sa[i][(curr_state_hashed, a)] for a in available_actions])
        policy = counts / np.sum(counts)

        output_policy[i] = policy

    return output_policy, v_t, o_t
        
def unroll_mcts(policy_net,
                dynamics_net,
                search_depth,
                action_space,
                N_sa,
                Q_sa,
                R_sa,
                O_sa,
                o_t,
                p_t,
                v_t,
                c1=1.25,
                c2=19652):
    
    states, \
    next_states, \
    policies, \
    actions, \
    rewards, \
    values = run_selection_and_expansion(policy_net,
                                         dynamics_net,
                                         search_depth,
                                         action_space,
                                         N_sa,
                                         Q_sa,
                                         R_sa,
                                         O_sa,
                                         o_t,
                                         p_t,
                                         v_t,
                                         c1=c1,
                                         c2=c2)

    final_policy = run_backup(states,
                              next_states,
                              policies,
                              actions,
                              rewards,
                              values,
                              action_space,
                              N_sa,
                              Q_sa,
                              R_sa,
                              O_sa)
    
    return final_policy
    
def run_selection_and_expansion(policy_net,
                                dynamics_net,
                                search_depth,
                                action_space,
                                N_sa,
                                Q_sa,
                                R_sa,
                                O_sa,
                                o_t,
                                p_t,
                                v_t,
                                c1=1.25,
                                c2=19652):
    
    states = []
    next_states = []
    policies = []
    actions = []
    rewards = []
    values = []

    for i in range(search_depth):

        states.append(o_t.cpu().numpy())

        actions_per_env = [
            select_action_UCT(
                env_o_t,
                p_t[j],
                N_sa[j],
                Q_sa[j],
                action_space,
                c1=c1,
                c2=c2
            ) for j, env_o_t in enumerate(o_t)
        ]

        actions_per_env = torch.from_numpy(
            np.array(actions_per_env).astype(np.int32).reshape((-1, 1))
        ).to(o_t.device)

        o_t, r_t = dynamics_net(o_t, actions_per_env)

        p_t, v_t = policy_net(o_t)

        policies.append(p_t.cpu().numpy())
        next_states.append(o_t.cpu().numpy())
        actions.append(actions_per_env.cpu().numpy().flatten())
        rewards.append(r_t.cpu().numpy().flatten())
        values.append(v_t.cpu().numpy().flatten())

    states = np.array(states).swapaxes(0,1)
    next_states = np.array(next_states).swapaxes(0,1)
    policies = np.array(policies).swapaxes(0,1)
    actions = np.array(actions).T
    rewards = np.array(rewards).T
    values = np.array(values).T

    return states, next_states, policies, actions, rewards, values

def run_backup(states,
               next_states,
               policies,
               actions,
               rewards,
               values,
               action_space,
               N_sa,
               Q_sa,
               R_sa,
               O_sa,
               gamma=0.99):
    
    N, T = rewards.shape

    Gk = np.zeros((N, T))
    for i in range(T):
        adjusted_i = T - i - 1

        gammas = np.array([gamma ** j for j in range(i + 1)]).reshape((1,-1))
        v_gamma = gamma ** (i + 1)

        Gk[:,adjusted_i] = np.sum(rewards[:,adjusted_i:] * gammas, axis=1) + v_gamma * values[:,adjusted_i]

    for i in range(N):
        for j in range(T):
            
            curr_state = states[i,j]
            curr_action = actions[i,j]
            hashed_state = hash_state(curr_state)

            sa = (hashed_state, curr_action)

            Q_sa[i][sa] = (N_sa[i][sa] * Q_sa[i][sa] + Gk[i,j]) / (N_sa[i][sa] + 1)
            N_sa[i][sa] += 1

def select_action_UCT(o_t,
                      p_t,
                      N_sa,
                      Q_sa,
                      action_space,
                      c1=1.25,
                      c2=19652):
    
    computed_q_values = list(Q_sa.values())
    min_q, max_q = 0.0, 1.0
    if len(computed_q_values) > 0:
        min_q = min(computed_q_values)
        max_q = max(computed_q_values)
    
    hashed_o_t = hash_state(o_t)
    actions = list(range(action_space))
    p_t = p_t.cpu().numpy().flatten()
    q_t = np.zeros(len(p_t))
    n_t = np.zeros(len(p_t)).astype(np.int32)

    for a in actions:
        if (hashed_o_t, a) not in N_sa:
            N_sa[(hashed_o_t, a)] = 0
        if (hashed_o_t, a) not in Q_sa:
            Q_sa[(hashed_o_t, a)] = 0

        q_t[a] = Q_sa[(hashed_o_t, a)]
        n_t[a] = N_sa[(hashed_o_t, a)]

    adjusted_q_t = (q_t - min_q) / (max_q - min_q)

    policy_weight = ((sum(n_t) ** 0.5) / (1 + n_t)) * (c1 + np.log((sum(n_t) + c2 + 1) / c2))

    UCT_vec = adjusted_q_t + (p_t * policy_weight)

    action = np.argmax(UCT_vec)

    return action

def hash_state(s):

    if type(s) == torch.Tensor:
        s = s.cpu().numpy()

    return s.tobytes()
