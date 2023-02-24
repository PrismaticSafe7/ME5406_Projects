import numpy as np

class GenericTrainer():
	'''
	Generic Template with variables and functions for all other 
	reinforcement training algo
	'''

	def __init__(self, env):
		# Initialize environment (Frozen Lake env will be used)
		self.env = env
		self.num_action = env.num_action
		self.num_state = env.num_state

		self.num_episodes = 10000	# Total number of episodes
		self.max_steps = 50000		# Max number of steps in 1 episode

		# Tuning Parameters for learner
		self.lr = 0.1		# Learning Rate
		self.gamma = 0.9 	# Discount factor
		self.epsilon = 0.1 	# Exploration rate

		# Initialization of tables - Policy and Q table
			# Policy table - Provides policy that will be cho
			# Q Table - table that stores Q value (expected reward for each action
			# 			in a particular state of environment)
		self.Q_table = np.zeros((self.num_state, self.num_action)) # Initializing of Q_table
		self.P_table = {
			s: [1/self.num_action] * self.num_action for s in range(self.num_state)
		} # Initializing of Policy table

		self.episodes_reward = [] # Reward received in each episode