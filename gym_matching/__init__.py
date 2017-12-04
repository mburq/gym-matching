from gym.envs.registration import register

register(
    id='Matching-v0',
    entry_point='gym_matching.envs:TypeMatchingEnv',
    timestep_limit=100,
    reward_threshold=1.0,
    nondeterministic=True,
    )
