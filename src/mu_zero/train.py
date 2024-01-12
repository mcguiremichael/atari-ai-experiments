
from env import GameEnv
from mu_zero_agent import MuZeroAgent
from generate_replay_memory import generate_gameplay_data

import argparse

import yaml
import importlib
from typing import Tuple, Dict

def train_agent(env_name : str,
                gym_factory,
                num_envs : int,
                model_config : Dict,
                output_folder : str,
                resize_shape : Tuple[int, int],
                evaluation_reward_length : int,
                training_frame_count : int,
                frame_start_training,
                train_step_interval : int,
                batches_per_train_step : int,
                checkpoint_interval : int,
                history_size : int,
                max_search_iter : int,
                max_search_t : float,
                search_depth : int,
                batch_size : int,
                replay_memory_length : int,
                wp=1.0,
                wv=0.1,
                wr=0.1,
                c1=1.25,
                c2=19652):

    envs = init_envs(env_name,
                     gym_factory,
                     num_envs,
                     history_size,
                     resize_shape)

    action_size = envs[0].action_space.n

    frame_size = resize_shape

    per_env_memory_length = replay_memory_length // num_envs

    agent = MuZeroAgent(model_config,
                        frame_size,
                        history_size,
                        action_size,
                        batch_size,
                        max_search_iter,
                        max_search_t,
                        search_depth,
                        num_envs,
                        per_env_memory_length,
                        c1=c1,
                        c2=c2,
                        wp=wp,
                        wv=wv,
                        wr=wr)

    run_training(agent,
                 envs,
                 output_folder,
                 training_frame_count,
                 frame_start_training,
                 train_step_interval,
                 batches_per_train_step,
                 checkpoint_interval,
                 history_size,
                 c1=c1,
                 c2=c2)

def run_training(agent,
                 envs,
                 output_folder : str,
                 training_frame_count : int,
                 frame_start_training : int,
                 train_step_interval : int,
                 batches_per_train_step : int,
                 checkpoint_interval : int,
                 history_size : int,
                 c1=1.25,
                 c2=19652):
    """_summary_

    Args:
        agent (_type_): _description_
        envs (_type_): _description_
        output_folder (str): _description_
        training_frame_count (int): _description_
        train_step_interval (int): Number of data generation steps between running training epochs
        history_size (int): _description_
        c1 (float, optional): _description_. Defaults to 1.25.
        c2 (int, optional): _description_. Defaults to 19652.
    """

    num_envs = len(envs)
    assert(train_step_interval % num_envs == 0)

    gameplay_generator = generate_gameplay_data(agent,
                                                envs,
                                                history_size,
                                                0)

    frame = 0
    while frame < training_frame_count:

        gameplay_snapshot = next(gameplay_generator)
        frame += len(gameplay_snapshot)

        for env_snapshot in gameplay_snapshot:
            agent.memory.push(*env_snapshot)

        if (frame > frame_start_training and frame % train_step_interval == 0):
            agent.run_training_step(batches_per_train_step)

        if (frame > 0 and frame % checkpoint_interval == 0):
            agent.save_checkpoint(output_folder)

def init_envs(env_name,
              gym_factory,
              num_envs,
              history_size : int,
              resize_shape : Tuple[int, int]):


    vis_env_idx = 0
    envs = []
    for i in range(num_envs):
        if i == vis_env_idx:
            render_mode="human"
        else:
            render_mode="rgb_array"
        envs.append(GameEnv(env_name,
                            render_mode=render_mode,
                            gym_factory=gym_factory,
                            history_size=history_size,
                            resize_shape=resize_shape))

    return envs

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_config")

    return parser.parse_args()

def main():

    args = parse_args()

    train_config = args.train_config
    with open(train_config, "r") as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    env_name = train_config["env_name"]
    gym_factory = train_config["gym_factory"]
    num_envs = train_config["num_envs"]
    model_config = train_config["model_config"]
    resize_shape = train_config["resize_shape"]
    output_folder = train_config["output_folder"]
    evaluation_reward_length = train_config["evaluation_reward_length"]
    training_frame_count = train_config["training_frame_count"]
    frame_start_training = train_config["frame_start_training"]
    history_size = train_config["history_size"]
    max_search_iter = train_config["max_search_iter"]
    max_search_t = train_config["max_search_t"]
    train_step_interval = train_config["train_step_interval"]
    batches_per_train_step = train_config["batches_per_train_step"]
    checkpoint_interval = train_config["checkpoint_interval"]
    search_depth = train_config["search_depth"]
    batch_size = train_config["batch_size"]
    replay_memory_length = train_config["replay_memory_length"]
    wp = train_config.get("policy_loss_weight", 1.0)
    wv = train_config.get("value_loss_weight", 0.1)
    wr = train_config.get("reward_loss_weight", 0.1)
    c1 = train_config.get("c1", 1.25)
    c2 = train_config.get("c2", 19652)

    gym_factory = importlib.import_module(gym_factory)

    train_agent(
        env_name,
        gym_factory,
        num_envs,
        model_config,
        output_folder,
        resize_shape,
        evaluation_reward_length,
        training_frame_count,
        frame_start_training,
        train_step_interval,
        batches_per_train_step,
        checkpoint_interval,
        history_size,
        max_search_iter,
        max_search_t,
        search_depth,
        batch_size,
        replay_memory_length,
        wp=wp,
        wv=wv,
        wr=wr,
        c1=c1,
        c2=c2
    )

if __name__ == "__main__":
    main()
