from gym.envs.registration import register

register(
    id='Matching-v0',
    entry_point='gym_matching.envs:TypeMatchingEnvDiscrete',
    timestep_limit=100,
    reward_threshold=1.0,
    nondeterministic=True,
    )

register(
    id='Matching-v1',
    entry_point='gym_matching.envs:TypeMatchingEnvContinuous',
    timestep_limit=100,
    reward_threshold=1.0,
    nondeterministic=True,
    )

register(
    id='Matching-v2',
    entry_point='gym_matching.envs:TypeMatchingEnvShadow',
    timestep_limit=100,
    reward_threshold=1.0,
    nondeterministic=True,
    )
