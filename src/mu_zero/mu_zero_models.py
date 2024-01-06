
from model import ResnetBlock

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from typing import Tuple

class AtariRepresentationNetwork(nn.Module):
    
    def __init__(self,
                 frame_shape : Tuple[int, int, int],
                 history_size : int,
                 max_action : int):


        super(AtariRepresentationNetwork, self).__init__()

        self.frame_shape = frame_shape
        self.history_size = history_size
        self.max_action = max_action
        h, w = self.frame_shape
        channels_per_frame = (1 + 1) # Add 1 for actions
        self.input_channels = channels_per_frame * history_size

        self.layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, stride=2, kernel_size=3, padding=1),
            ResnetBlock(128, 2),
            ResnetBlock(128, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ResnetBlock(256, 2),
            ResnetBlock(256, 2),
            ResnetBlock(256, 2),
            nn.AvgPool2d(2),
            ResnetBlock(256, 2),
            ResnetBlock(256, 2),
            ResnetBlock(256, 2),
            nn.AvgPool2d(2)
        )

    def forward(self,
                states : torch.Tensor,
                actions : torch.Tensor):
        
        actions = actions / self.max_action

        encoded_actions = actions.reshape((*actions.shape, 1, 1)) * torch.ones((1, 1, self.frame_shape[0], self.frame_shape[1])).to(actions.device)
        concattenated_input = torch.cat([states, encoded_actions], dim=1)

        output = self.layers(concattenated_input)

        return output
    
class AtariDynamicsNet(nn.Module):

    def __init__(self,
                 input_shape : Tuple[int, int],
                 num_resnet_blocks : int = 8,
                 action_space : int = 18):

        super(AtariDynamicsNet, self).__init__()

        self.input_shape = input_shape
        self.num_resnet_blocks = num_resnet_blocks
        self.action_space = action_space

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(256 + self.action_space, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            *[
                ResnetBlock(256, 2) for i in range(num_resnet_blocks)
            ]
        )

        self.reward_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self,
                states : torch.Tensor,
                actions : torch.Tensor):
        
        action_encoding = torch.zeros((len(actions), self.action_space, *states.shape[2:])).to(actions.device)
        action_encoding[:,actions] = 1
        combined_input = torch.cat([states, action_encoding], dim=1)

        features = self.feature_extractor(combined_input)
        reward_prediction = self.reward_head(features)

        state_repr = features

        return state_repr, reward_prediction


class AtariPolicyNet(nn.Module):
    
    def __init__(self,
                 action_space : int):

        super(AtariPolicyNet, self).__init__()

        self.action_space = action_space

        self.feature_reduction_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self,
                obs):
        
        features = self.feature_reduction_conv(obs)
        policy = self.policy_head(features)
        value = self.value_head(features)

        return policy, value
