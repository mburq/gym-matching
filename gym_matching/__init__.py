from gym.envs.registration import register

register(
    id='matching-v0',
    entry_point='gym_matching.envs:MatchingEnv',
    )
