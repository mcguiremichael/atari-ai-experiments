
from mu_zero_memory import MultiEnvReplayMemory
from mu_zero_models import AtariRepresentationNetwork, AtariDynamicsNet, AtariPolicyNet
from mcts import run_mcts

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import random
from typing import Tuple
import os

class MuZeroAgent:

    def __init__(self,
                 model_config,
                 frame_size : Tuple[int, int, int],
                 history_size : int,
                 action_size : int,
                 batch_size : int,
                 max_search_iter : int,
                 max_search_t : float,
                 search_depth : int,
                 num_envs : int,
                 per_env_memory_length : int,
                 learning_rate : float = 1e-3,
                 c1 : float = 1.25,
                 c2 : float = 19652,
                 gamma : float = 0.99,
                 wp : float = 1.0,
                 wv : float = 1.0,
                 wr : float = 1.0):

        self.model_config = model_config
        self.frame_size = frame_size
        self.history_size = history_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.max_search_iter = max_search_iter
        self.max_search_t = max_search_t
        self.search_depth = search_depth
        self.num_envs = num_envs
        self.per_env_memory_length = per_env_memory_length
        self.learning_rate = learning_rate
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma
        self.wp = wp
        self.wv = wv
        self.wr = wr

        self.build_networks()

        self.policy_loss_fn = nn.MSELoss()
        self.value_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.MSELoss()

        self.memory = MultiEnvReplayMemory(self.num_envs,
                                           self.per_env_memory_length,
                                           self.history_size,
                                           self.search_depth)

        self.checkpoint_counter = 0

        self.optimizer = optim.Adam(params=list(self.representation_net.parameters()) + \
                                            list(self.dynamics_net.parameters()) + \
                                            list(self.policy_net.parameters()),
                                    lr=learning_rate,
                                    eps=1e-8)

    def build_networks(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.representation_net = AtariRepresentationNetwork(self.frame_size,
                                                             self.history_size,
                                                             self.action_size).to(self.device)

        self.dynamics_net = AtariDynamicsNet(self.frame_size,
                                             self.model_config["dynamics_num_resnet_blocks"],
                                             self.action_size).to(self.device)

        self.policy_net = AtariPolicyNet(self.action_size).to(self.device)

    def get_action(self,
                   state,
                   action_history,
                   action_availabilities=None,
                   hidden_state=None,
                   use_search : bool = True):

        self.eval()

        if (len(state.shape)) == 3:
            state = torch.from_numpy(state).to(self.device).unsqueeze(0)
        else:
            state = torch.from_numpy(state).to(self.device)

        if len(action_history.shape) == 1:
            action_history = torch.from_numpy(action_history).to(self.device).unsqueeze(0)
        else:
            action_history = torch.from_numpy(action_history).to(self.device)

        with torch.no_grad():

            if use_search:
                probs, val = self.generate_policy_with_search(state, action_history)
            else:
                repr = self.representation_net(state, action_history)
                probs, val = self.policy_net(repr)
                probs = probs.detach().cpu().numpy()
                val = val.detach().cpu().numpy()

        probs = probs
        val = val.flatten()
        action = self.select_action(probs, action_availabilities)

        return action

    def select_action(self,
                      probs,
                      action_availabilities=None):

        probs = probs.copy()
        print(probs)
        if action_availabilities is not None:
            for i, avail in enumerate(action_availabilities):
                probs[i] = probs[i] * avail
                probs[i] = probs[i] / np.sum(probs[i])

        outs = []
        for j in range(len(probs)):
            candidate = random.random()
            total = probs[j,0]
            i = 0
            while (total < candidate and total < 1.0 and i < len(probs[j])-1):
                i += 1
                total += probs[j,i]
            outs.append(i)

            if action_availabilities is not None:
                assert(action_availabilities[j][i])
        return outs

    def generate_policy_with_search(self,
                                    state : torch.Tensor,
                                    action_history : torch.Tensor):

        policy, values, _ = run_mcts(self.representation_net,
                                     self.policy_net,
                                     self.dynamics_net,
                                     state,
                                     action_history,
                                     self.max_search_iter,
                                     self.max_search_t,
                                     self.search_depth,
                                     self.action_size)

        return policy, values

    def run_training_step(self, batches_per_train_step : int):

        self.train()

        for batch_idx in range(batches_per_train_step):

            mini_batch = self.memory.sample_mini_batch(self.batch_size)

            # (N, history_size + search_depth - 1, H, W)
            # (N, history_size + search_depth), note: actions are shifted by one in reverse relative to states, i.e. action i+1 was taken at state i
            # (N, search_depth - 1)
            # (N, search_depth - 1)
            history, \
            actions, \
            rewards, \
            dones = mini_batch

            history = torch.from_numpy(history).to(self.device).float()
            actions = torch.from_numpy(actions).to(self.device).long()
            rewards = torch.from_numpy(rewards).to(self.device).float()
            dones = torch.from_numpy(dones).to(self.device).bool()

            batch_size = len(history)

            # Output shape is (N * search_depth, history_size, ...)
            stacked_states = stack_subsequences(history,
                                                self.history_size)

            # Reshape to allow search depth sampling
            states = stacked_states.reshape((batch_size,
                                             self.search_depth,
                                             *stacked_states.shape[1:]))

            stacked_action_histories = stack_subsequences(actions[:,:-1],
                                                          self.history_size)

            action_histories = stacked_action_histories.reshape((batch_size,
                                                                 self.search_depth,
                                                                 *stacked_action_histories.shape[1:]))

            actions = actions[:,self.history_size:]

            #encoded_states = self.representation_net(stacked_states,
            #                                         stacked_actions)
            #predicted_policies, predicted_values = self.policy_net(encoded_states)

            # Inputs and outputs are stacked, meaning batch size and search_depth dimensions are flattened into one

            # Output shapes are
            # - (N * search_depth, ...)
            # - (N * search_depth, 1)
            # - (N * search_depth, ...)
            stacked_search_policy, stacked_predicted_values, stacked_encoded_states = run_mcts(
                self.representation_net,
                self.policy_net,
                self.dynamics_net,
                stacked_states,
                stacked_action_histories,
                self.max_search_iter,
                self.max_search_t,
                self.search_depth,
                self.action_size
            )

            sequential_search_policy = stacked_search_policy.reshape((batch_size,
                                                                      self.search_depth,
                                                                      *stacked_search_policy.shape[1:]))

            sequential_search_policy = torch.from_numpy(sequential_search_policy).to(self.device).float()

            sequential_predicted_rewards = []
            sequential_predicted_policies = []
            sequential_predicted_values = []

            # @TODO: Make this right
            # Inputs have shape (N, history_size * ...) and (N, history_size)
            encoded_states = self.representation_net(states[:,0],
                                                     action_histories[:,0])

            curr_predicted_policy, curr_predicted_values = self.policy_net(encoded_states)
            sequential_predicted_policies.append(curr_predicted_policy)
            sequential_predicted_values.append(curr_predicted_values)

            for i in range(self.search_depth):
                encoded_states, curr_predicted_rewards = self.dynamics_net(encoded_states,
                                                                           actions[:,i])

                sequential_predicted_rewards.append(curr_predicted_rewards.flatten())

                # Very important this value function estimation occurs AFTER dynamics_net applied...
                # This generates the final V term in the value target generation equation

                curr_predicted_policy, curr_predicted_values = self.policy_net(encoded_states)
                sequential_predicted_values.append(curr_predicted_values)

                # Don't run on last iteration
                if i < (self.search_depth - 1):
                    sequential_predicted_policies.append(curr_predicted_policy)

            sequential_predicted_rewards = torch.stack(sequential_predicted_rewards, dim=1)
            sequential_predicted_policies = torch.stack(sequential_predicted_policies, dim=1)
            sequential_predicted_values = torch.cat(sequential_predicted_values, dim=1)

            # vtarg value bootstraps are from next states
            vtargs = self.compute_vtargs(rewards,
                                         sequential_predicted_values[:,-1].detach())

            # predicted_values we are trying to optimize from from curr_states, not next_states, thus we drop last ones
            predicted_values = sequential_predicted_values[:,:-1].reshape((batch_size, self.search_depth))

            loss_est, policy_loss, value_loss, reward_loss = self.loss(sequential_predicted_policies,
                                                                       sequential_predicted_rewards,
                                                                       predicted_values,
                                                                       sequential_search_policy,
                                                                       rewards,
                                                                       vtargs)

            self.optimizer.zero_grad()
            loss_est.backward()
            self.optimizer.step()

    def compute_vtargs(self,
                       rewards : torch.Tensor,
                       last_predicted_values : torch.Tensor):

        vtargs = torch.zeros_like(rewards).float()

        for i in range(self.search_depth):
            adjusted_i = (self.search_depth - i) - 1

            gammas = torch.Tensor([self.gamma ** j for j in range(i+1)]).to(rewards.device)
            v_gamma = self.gamma ** (i + 1)
            vtargs[:, adjusted_i] = torch.sum(rewards[:, adjusted_i:] * gammas, dim=1) + v_gamma * last_predicted_values

        return vtargs

    def save_checkpoint(self, output_folder : str):

        os.makedirs(output_folder, exist_ok=True)

        torch.save(
            self.policy_net.state_dict(),
            os.path.join(output_folder, f"policy_{self.checkpoint_counter:04d}.pth")
        )

        torch.save(
            self.representation_net.state_dict(),
            os.path.join(output_folder, f"representation_{self.checkpoint_counter:04d}.pth")
        )

        torch.save(
            self.dynamics_net.state_dict(),
            os.path.join(output_folder, f"dynamics_{self.checkpoint_counter:04d}.pth")
        )

        self.checkpoint_counter += 1

    def loss(self,
             predicted_policy,
             predicted_rewards,
             predicted_values,
             search_policy,
             observed_rewards,
             observed_values):

        policy_loss = self.policy_loss_fn(predicted_policy,
                                          search_policy)

        value_loss = self.value_loss_fn(predicted_values,
                                        observed_values)

        reward_loss = self.reward_loss_fn(predicted_rewards,
                                          observed_rewards)

        loss = self.wp * policy_loss + self.wv * value_loss + self.wr * reward_loss

        return loss, policy_loss.cpu().item(), value_loss.cpu().item(), reward_loss.cpu().item()

    def train(self):
        self.policy_net.train()
        self.dynamics_net.train()
        self.representation_net.train()

    def eval(self):
        self.policy_net.eval()
        self.dynamics_net.eval()
        self.representation_net.eval()

def stack_subsequences(x : torch.Tensor,
                       subsequence_length):

    N = x.shape[0]
    DK = x.shape[1]
    K = DK - subsequence_length + 1

    subsequences = [x[:,i:i+subsequence_length] for i in range(K)]

    output_tensor = torch.stack(subsequences, dim=1).flatten(start_dim=0, end_dim=1)

    return output_tensor
