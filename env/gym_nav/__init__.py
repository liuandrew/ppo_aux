from gym.envs.registration import register
register(
    id='NavEnv-v0',
    entry_point='gym_nav.envs:NavEnvFlat',
)
register(
    id='ShortcutNav-v0',
    entry_point='gym_nav.envs:ShortcutNavEnv',
)
register(
    id='ExploreNav-v0',
    entry_point='gym_nav.envs:ExploreMWM',
)
register(
    id='PlumNav-v0',
    entry_point='gym_nav.envs:PlumNavEnv',
)