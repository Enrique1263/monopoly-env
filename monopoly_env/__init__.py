from gym.envs.registration import register

register(
    id='MonopolyEnv-v0',
    entry_point='monopoly_env.envs.game:MonopolyEnv',
)
