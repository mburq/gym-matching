from gym.envs.registration import register

register(
    id='Matching-v0',
    entry_point='gym_matching.envs:MatchingEnv',
    timestep_limit=200,
    reward_threshold=10.,
    nondeterministic=True,
    )
