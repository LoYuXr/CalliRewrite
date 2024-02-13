from gym.envs.registration import register

register(
		id='CalliEnv-v0',
		entry_point='Callienv.envs:CalliEnv',
)