import gym
import gym_matching
import numpy as np
import argparse
import time
from baselines.common.misc_util import boolean_flag
from collections import deque


def run(env_id, seed, evaluation, nb_epochs, nb_rollout_steps):
    assert env_id in ['Matching-v2', 'Matching-v3']  # only works for shadow prices env.
    env = gym.make(env_id)
    start_time = time.time()
    obs = env.reset()
    action_shape = env.action_space.shape
    coefficients = np.random.rand(32)
    discount = 0.9
    lr = 0.001
    epoch_episodes = 0
    episode_reward = 0.
    episode_step = 0
    episodes = 0
    t = 0
    epoch_episode_rewards = []
    episode_rewards_history = deque(maxlen=5)
    epoch_episode_steps = []

    for epoch in range(nb_epochs):
        for t_rollout in range(nb_rollout_steps):
            # action = 0 * np.ones(action_shape)
            action = coefficients
            new_obs, r, done, info = env.step(action)

            error = np.dot(coefficients, obs) - r - discount * np.dot(coefficients, new_obs)
            coefficients -= error * obs * lr
            t += 1
            episode_reward += r
            episode_step += 1

            obs = new_obs

            if done:
                # Episode done.
                epoch_episode_rewards.append(episode_reward)
                episode_rewards_history.append(episode_reward)
                epoch_episode_steps.append(episode_step)
                episode_reward = 0.
                episode_step = 0
                epoch_episodes += 1
                episodes += 1
                obs = env.reset()
                break
        if epoch * 20 % nb_epochs == 0:
            print("Mean reward: {}".format(np.mean(episode_rewards_history)))
            print(coefficients)
    print("Mean reward: {}".format(np.mean(epoch_episode_rewards)))

    print(time.time() - start_time)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='Matching-v3')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=50)
    boolean_flag(parser, 'evaluation', default=True)
    parser.add_argument('--nb-rollout-steps', type=int, default=1000)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)
