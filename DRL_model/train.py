import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from Env import environment
from Model import TD3
from itertools import count
import torch

def train_model(epochs, start_train, frame_size, env, agent, directory, batch_size):
    reward_list = []
    patience = 30
    es_count = 0

    for epoch in range(epochs):
        total_reward = 0
        env.reset()

        # Stack the frames.
        state = []
        for i in range(4):
            state.append(env.to_frame(frame_size, frame_size).squeeze().copy() / 255)
        state = np.array(state)

        for t in count():
            reward = 0

            if epoch < start_train:
                action_r = (np.random.normal(0, 0.2, size=1)).clip(0, 1)
                action_theta = np.random.normal(0, 0.4, size=1).clip(-1, 1)
            else:
                action_r, action_theta = agent.select_action(state, noise=0.1)
                action_r = np.array([action_r])
                action_theta = np.array([action_theta])

            next_state = []
            for _ in range(4):
                next_tmp, reward_tmp, done = env.step(action_r, action_theta)
                next_tmp = env.to_frame(frame_size, frame_size).squeeze().copy() / 255
                next_state.append(next_tmp)
                reward += reward_tmp
            next_state = np.array(next_state)

            action = np.array([action_r.item(), action_theta.item()], dtype=float)
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))
            state = next_state.copy()

            total_reward += reward

            if done:
                break
        
        print('\rEpisode : {}, Total Step : {}, Total Reward : {:.2f}'.format(epoch, env.count, total_reward), end='')

        if epoch == start_train:
            print('')
            print('=' * 50)
            print('***** Now Train begins.. *****')
            print('=' * 50)

        reward_list.append(total_reward)

        if epoch > start_train:
            agent.update(batch_size, epoch-start_train)

        if epoch > start_train and total_reward == max(reward_list):
            agent.save(directory=str(directory), epoch=epoch)
            agent.save(directory=str(directory), epoch='Best')

        # early stopping
        if len(reward_list) > 2 and (reward_list[-1] < reward_list[-2]):
            es_count += 1
        else:
            es_count = 0

        if es_count > patience:
            np.save(Path(directory, f'Learning_Curve.npy'), reward_list)
            break


def test_model(frame_size, env, agent, directory, device):
    agent.load(str(directory), 'Best', device=device)

    test_reward = 0
    RL_action = []
    RL_pos = []
    for episode in range(1):
        env.reset()

        state = []
        for i in range(4):
            state.append(env.to_frame(frame_size, frame_size).squeeze().copy() / 255)
        state = np.array(state)

        for step_ in count():
            reward = 0

            action_r, action_theta = agent.select_action(state, noise=0)
            action_r = np.array([action_r])
            action_theta = np.array([action_theta])

            next_state = []
            for _ in range(4):
                RL_pos.append(env.state)
                RL_action.append([action_r, action_theta])
                next_tmp, reward_tmp, done, _ = env.step(action_r, action_theta)
                next_tmp = env.to_frame(frame_size, frame_size).squeeze().copy() / 255
                next_state.append(next_tmp)
                reward += reward_tmp
            next_state = np.array(next_state)

            test_reward += reward

            if done or step_ > env.parameter.duration:
                print('Episode : {}, Reward : {:.2f}, Step : {}'.format(int(episode), test_reward, int(env.count)))
                test_reward = 0
                break

            state = next_state.copy()
    RL_action = np.array(RL_action).squeeze()
    np.save(Path(directory, 'DRL_action.npy'), np.array(RL_action))
    np.save(Path(directory, 'DRL_position.npy'), np.array(RL_pos))


if __name__ == '__main__':

    env = environment.Load()
    agent = TD3.make_model()
    root_dir = '/Users/imlim/Documents/Project/BRAIN_RL'

    model_name = input("Please Enter the model's name : ")

    directory = Path(root_dir, 'Result', 'DRL_model', model_name)
    directory.mkdir(exist_ok=True, parents=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_episode = 10000
    start_train = 0
    batch_size = 128
    tau = 0.01
    frame_size = 84

    train_model(max_episode, start_train, frame_size, env, agent, directory, batch_size)
    test_model(frame_size, env, agent, directory, device)

