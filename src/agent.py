import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmarks = True
from memory import ReplayMemory
from model import *
from utils import *
from config import *
import time
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self,
                 action_size,
                 mode='PPO',
                 c1=1.0,
                 c2=0.01):
        print(f"Using device {device}")
        self.load_model = False
        self.mode = mode

        self.action_size = action_size
        self.loss = nn.MSELoss()

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.lam = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.eps_denom = 1.0e-8
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000
        self.c1 = c1      # Weight for value loss
        self.c2 = c2      # Weight for entropy loss
        self.num_epochs = 3
        self.num_epochs_trained = 0
        self.num_frames_trained = 0

        # Generate the memory
        self.memory = ReplayMemory(mode=mode)

        # Create the policy net and the target net
        if (mode == 'PPO'):
            self.policy_net = PPO(action_size, HISTORY_SIZE)
            self.target_net = PPO(action_size, HISTORY_SIZE)
        elif (mode == 'PPO_MHDPA'):
            self.policy_net = PPO_MHDPA_160(action_size, HISTORY_SIZE)
            self.target_net = PPO_MHDPA_160(action_size, HISTORY_SIZE)
        elif (mode == "PPO_MHDPA_2D"):
            self.policy_net = PPO_MHDPA_160_ACTION_2D((3, 9, 9), HISTORY_SIZE)
            self.target_net = PPO_MHDPA_160_ACTION_2D((3, 9, 9), HISTORY_SIZE)
        elif (mode == 'PPO_LSTM'):
            self.policy_net = PPO_LSTM(action_size, 64, 1)
            self.target_net = PPO_LSTM(action_size, 64, 1)

        self.policy_net.to(device)
        self.target_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate, eps=1e-8)

        # Added for learning rate decay
        self.lr_min = learning_rate / 10
        self.clip_min = clip_param / 10
        self.clip_param = clip_param
        self.decay_rate = 10000000



        # initialize target net
        self.update_target_net()

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using action probabilities"""
    def get_action(self,
                   state,
                   action_availabilities=None,
                   hidden_state=None):
        if (len(state.shape)) == 3:
            state = torch.from_numpy(state).to(device).unsqueeze(0)
        else:
            state = torch.from_numpy(state).to(device)
        if self.mode == "PPO_LSTM":
            probs, val, hidden_state = self.policy_net(state, hidden_state)
        else:
            probs, val = self.policy_net(state)
        probs = probs.detach().cpu().numpy()
        val = val.detach().cpu().numpy().flatten()
        action = self.select_action(probs, action_availabilities)
        return action, val, hidden_state

    def select_action(self,
                      probs,
                      action_availabilities=None):

        probs = probs.copy()
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

    def get_attention_maps(self, state):
        if (len(state.shape)) == 3:
            state = torch.from_numpy(state).to(device).unsqueeze(0)
        else:
            state = torch.from_numpy(state).to(device)

        assert(self.mode == "PPO_MHDPA")
        return self.policy_net.forward_attention(state)


    def entropy(self, probs):
        return - (torch.clamp(probs, self.eps_denom) * torch.log(torch.clamp(probs, self.eps_denom))).mean()
        #return - (probs * torch.log(probs)).mean()

    def train_policy_net(self, frame, frame_next_vals):

        self.policy_net.train()

        for param_group in self.optimizer.param_groups:
            curr_lr = param_group['lr']
        print("Training network. lr: %f. clip: %f" % (curr_lr, self.clip_param))

        #with detect_anomaly():
        #with True:

        # Memory computes targets for value network, and advantag es for policy iteration
        self.memory.compute_vtargets_adv(self.discount_factor, self.lam, frame_next_vals)



        # Should be integer. len(self.memory) should be a multiple of batch_size.
        num_iters = int(len(self.memory.indices) / batch_size)

        """
        lambda1 = lambda epoch: self.lr_min + (learning_rate - self.lr_min) * ((self.decay_rate - self.num_epochs_trained) / self.decay_rate)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])
        """
        for i in range(self.num_epochs):

            pol_loss = 0.0
            vf_loss = 0.0
            ent_total = 0.0


            """
            Added for slowdown debugging

            """
            loading_time = 0.0
            forward_time = 0.0
            loss_time = 0.0
            clipping_time = 0.0
            backward_time = 0.0
            step_time = 0.0

            self.num_epochs_trained += 1


            if (self.num_frames_trained < self.decay_rate and self.num_epochs_trained % 50 == 0):
                new_lr = self.lr_min + (learning_rate - self.lr_min) * ((self.decay_rate - self.num_frames_trained) / self.decay_rate)
                del self.optimizer
                self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=new_lr)
                self.clip_param = self.clip_min + (clip_param - self.clip_min) * ((self.decay_rate - self.num_frames_trained) / self.decay_rate)



            for i in range(num_iters):




                # Loading time begin
                t1 = time.time()


                mini_batch = self.memory.sample_mini_batch(frame)
                #mini_batch = np.array(mini_batch).transpose()
                history, \
                actions, \
                rewards, \
                dones, \
                vtargs, \
                rets, \
                advs, \
                hiddens = zip(*mini_batch)

                history = np.stack(history, axis=0)
                states = np.float32(history[:, :HISTORY_SIZE, :, :]) / 255.
                actions = np.array(list(actions))
                rewards = np.array(list(rewards))
                next_states = np.float32(history[:, 1:, :, :]) / 255.
                dones = np.array(dones)
                v_returns = np.array(rets)
                advantages = np.array(advs)

                if (self.mode == 'PPO_LSTM'):
                    hiddens = torch.from_numpy(np.stack(hiddens, axis=0)).to(device)
                    hiddens.requires_grad = True




                ### Converts everything to tensor
                states = torch.from_numpy(states).to(device)
                actions = torch.from_numpy(actions).to(device)
                rewards = torch.from_numpy(rewards).to(device).float()
                next_states = torch.from_numpy(next_states).to(device)
                dones = torch.from_numpy(np.uint8(dones)).to(device)
                v_returns = torch.from_numpy(v_returns).to(device).float()
                advantages = torch.from_numpy(advantages).to(device).float()





                # Normalize advantages
                # advantages = (advantages - advantages.mean()) / (torch.clamp(advantages.std(), self.eps_denom))


                # Loading time end
                loading_time += (time.time() - t1)

                # Forward time begin
                t1 = time.time()

                ### Get probs, val from current state for curr, old policies
                n = states.shape[0]
                actions = actions.reshape((n, 1))

                if (self.mode == 'PPO_LSTM'):
                    curr_probs, curr_vals, _ = self.policy_net.forward_unroll(states, hiddens, unroll_steps=HISTORY_SIZE)
                    with torch.no_grad():
                        old_probs, old_vals, _ = self.target_net.forward_unroll(states, hiddens, unroll_steps=HISTORY_SIZE)
                else:
                    curr_probs, curr_vals = self.policy_net(states)
                    with torch.no_grad():
                        old_probs, old_vals = self.target_net(states)

                curr_vals = curr_vals.flatten()
                curr_prob_select = curr_probs.gather(1, actions).reshape((n,))
                old_prob_select = old_probs.gather(1, actions).reshape((n,))

                # Forward time end
                forward_time += (time.time() - t1)


                # Loss time begin
                t1 = time.time()

                ### Compute ratios
                ratio = torch.exp(torch.log(torch.clamp(curr_prob_select, self.eps_denom)) - torch.log(torch.clamp(old_prob_select.detach(), self.eps_denom)))
                ratio_adv = ratio * advantages.detach()
                bounded_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages.detach()



                pol_avg = - ((torch.min(ratio_adv, bounded_adv)).mean())

                """
                if (i == 0):
                    print(ratio, ratio_adv, bounded_adv, pol_avg)
                """
                ### Compute value and loss
                value_loss = self.loss(curr_vals, v_returns.detach())

                ent = self.entropy(curr_probs)

                total_loss = pol_avg + self.c1 * value_loss - self.c2 * ent

                # Loss time end
                loss_time += (time.time() - t1)

                # backward time begin
                t1 = time.time()

                self.optimizer.zero_grad()
                total_loss.backward()

                max_grad = torch.Tensor([0])
                min_grad = torch.Tensor([0])
                for name, param in self.policy_net.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        max_grad = torch.max(max_grad, torch.max(param.grad).cpu())
                        min_grad = torch.min(min_grad, torch.min(param.grad).cpu())
                        if torch.isnan(param.grad).any():
                            print(name, torch.min(param.grad), torch.max(param.grad), torch.isnan(param.grad).any())
                            raise NotImplementedError

                print(max_grad.cpu().item(),
                      min_grad.cpu().item())

                #backward time end
                backward_time += (time.time() - t1)

                # Clipping time begin
                t1 = time.time()

                clip_grad_norm_(self.policy_net.parameters(), 50.0)

                # Clipping time end
                clipping_time += (time.time() - t1)


                # step time begin
                t1 = time.time()

                self.optimizer.step()

                # step time end
                step_time += (time.time() - t1)


                pol_loss += pol_avg.detach().cpu().item()
                vf_loss += value_loss.detach().cpu().item()
                ent_total += ent.detach().cpu().item()

            pol_loss /= num_iters
            vf_loss /= num_iters
            ent_total /= num_iters
            print("Iteration %d: Policy loss: %f. Value loss: %f. Entropy: %f." % (self.num_epochs_trained, pol_loss, vf_loss, ent_total))


            """
            total_time = loading_time + forward_time + loss_time + clipping_time + backward_time + step_time

            print("load: %f\nforward: %f\nloss: %f\nclip: %f\nbackward: %f\nstep: %f\ntotal: %f\n" % (loading_time, forward_time, loss_time, clipping_time, backward_time, step_time, total_time))
            """

        self.num_frames_trained += train_frame

        self.policy_net.eval()

    def init_hidden(self):
        return self.policy_net.init_hidden()

    def normalize(self, x):
        return (x - torch.mean(x)) / (torch.std(x) + self.eps_denom)

    def clip_gradients(self, clip):

        ### Clip the gradients of self.policy_net
        for param in self.policy_net.parameters():
            if param.grad is None:
                continue
            #print(torch.max(param.grad.data), torch.min(param.grad.data))
            param.grad.data = param.grad.data.clamp(-clip, clip)

    def displayStack(self, state):
        #state = state.reshape(INPUT_SHAPE)
        self.displayImage(state[0,:,:])
        self.displayImage(state[1,:,:])
        self.displayImage(state[2,:,:])
        self.displayImage(state[3,:,:])


    def displayImage(self, image):
        plt.imshow(image)
        plt.show()
