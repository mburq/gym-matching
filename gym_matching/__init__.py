from gym.envs.registration import register

register(
    id='Matching-v0',
    entry_point='gym_matching.envs:MatchingEnv',
    timestep_limit=200,
    reward_threshold=10.,
    nondeterministic=True,
    )

register(
    id='Matching-v1',
    entry_point='gym_matching.envs:KidneyMatchingEnv',
    timestep_limit=200,
    reward_threshold=1.0,
    nondeterministic=True,
    )
