from gym.envs.registration import register
register(
    id='NavEnv-v0',
    entry_point='gym_nav.envs:NavEnvFlat',
)
register(
    id='NavEnv-v1',
    entry_point='gym_nav.envs:NavEnv',
)
register(
    id='MorrisEnv-v0',
    entry_point='gym_nav.envs:MorrisNav'
)
register(
    id='Gridworld-v0',
    entry_point='gym_nav.envs:GridworldNav'
)