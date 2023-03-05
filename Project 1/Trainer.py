import time
import numpy as np
import random as rd
import pandas as pd

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

		self.num_episodes = 3000	# Total number of episodes
		self.max_steps = 50000		# Max number of steps in 1 episode

		# Tuning Parameters for learner
		self.alpha = 0.1	# Learning Rate
		self.gamma = 0.9 	# Discount factor
		self.epsilon = 0.1 	# Exploration rate
		self.tolerance = 0.005 # Difference in Qreward must exceed tolerance to be updated

		# Initialization of tables - Policy and Q table
			# Q Table - table that stores Q value (expected reward for each action
			# 			in a particular state of environment)
		self.Q_table = np.zeros((self.num_state, self.num_action)) # Initializing of Q_table
		self.policy = np.zeros(self.num_state, dtype=int) 

		self.episodes_reward = [] # Reward received in each episode

	def getNextAction(self, state):
		'''
		Chooses between either:
			1) Exploration  (chance <= epsilon)
			2) Exploitation (chance > epsilon)

		returns:
			- int: Chosen action value 
		'''

		# Produce a random number from [0,1)
		chance = np.random.random()

		# if chance is higher than epsilon, (epsilon,1], choose the best action
		# else, randomly choose an action
		if self.epsilon < chance:
			action = np.argmax(self.Q_table[state, :])
		else:
			action = rd.randint(0,3)
		
		return action

	def test(self):
		for episode in range(3):
			state = self.env.reset()
			terminate = False
			noActionTaken = 0
			time.sleep(1)

			while not terminate:
				action = np.argmax(self.Q_table[state, :])
				new_state, reward, terminate, info = self.env.step(action)
				state = new_state

				if terminate:
					self.env.render()
					if reward == 1:
						print("Goal Accomplished")
					else:
						print("You Fell in the hole, Game Over")
				else:
					noActionTaken += 1
					if noActionTaken == self.max_steps:
						self.env.render()
						terminate = True
						print("Failed to find goal")
	
	def data_conversion(self, data):
		keys = []
		for key in data:
			keys.append(key)
		df = pd.DataFrame(data,columns = keys)
		return df


class FVMonteCarlo(GenericTrainer):
	# Does not terminate until it reaches terminal state
	def __init__(self, env):
		super().__init__(env)
		self.G_table = {(s,a): [] for s in range(self.num_state) for a in range(self.num_action)}

	def train(self, ep_no = 1):
		'''
		Trains the model and output the final Q_table

		Outputs:
			- Training statistics (dictionary)
		'''		
		
		reward_var= "FVMC_Rewards" + str(ep_no)  # Reward received in each episode
		step_var = "FVMC_noSteps" + str(ep_no)  # No. of steps taken in each episode
		acc_var ="FVMC_Accuracy" + str(ep_no) # Accuracy of data in each episode
		data = {reward_var:[], step_var:[], acc_var:[]}

		for i in range(self.num_episodes):
			# Infrom us the number of episodes we are going throug
			if (i % 100) == 0:
				print(i)

			# Reset environment
			state = self.env.reset()

			# Initialize data 
			G_episode = []		# List to store tuple of (state, action, reward)
			eps_reward = 0		# Total reward earned during episode
			terminate = False	# Bool to terminate episode
			noActionTaken = 0	# total no. of action taken during the episode
			G = 0

			while not terminate:
				action = self.getNextAction(state)
				new_state, reward, terminate, info = self.env.step(action)
				
				eps_reward += reward
				G_episode.append((state, action, reward))

				state = new_state

				if terminate == False:
					noActionTaken += 1					

				elif terminate == True:
					data[step_var].append(noActionTaken)
					if reward != 1:
						data[acc_var].append(0)
					elif reward == 1:
						data[acc_var].append(1)

			G_data = {}
			for (state, action, reward) in reversed(G_episode):
				G = self.gamma * G + reward
				G_data[(state,action)] = G 

			for key in G_data:
				state, action = key
				self.G_table[key].append(G_data[key])
				self.Q_table[state][action] = np.mean(self.G_table[key])
			
			self.episodes_reward.append(eps_reward)
			data[reward_var].append(eps_reward)

		
			for s in range(self.num_state):
				curr_best = np.argmax(self.Q_table[s])
				self.policy[s] = curr_best
		
		data_df = self.data_conversion(data)

		return data_df, data

class SARSA(GenericTrainer):
	def __init__(self, env):
		super().__init__(env)
	
	def train(self, ep_no=1):
		reward_var= "SARSA_Rewards" + str(ep_no)  # Reward received in each episode
		step_var = "SARSA_noSteps" + str(ep_no)  # No. of steps taken in each episode
		acc_var ="SARSA_Accuracy" + str(ep_no) # Accuracy of data in each episode
		data = {reward_var:[], step_var:[], acc_var:[]}

		for i in range(self.num_episodes):
			# Infrom us the number of episodes we are going throug
			if (i % 100) == 0:
				print(i)

			# Reset environment
			state = self.env.reset()

			# Initialize data 
			eps_reward = 0		# Total reward earned during episode
			terminate = False	# Bool to terminate episode
			noActionTaken = 0	# total no. of action taken during the episode
			action = self.getNextAction(state)
			
			while not terminate:
				new_state, reward, terminate, info = self.env.step(action)
				new_action = self.getNextAction(new_state)

				self.Q_table[state][action] += self.alpha * (reward + self.gamma * self.Q_table[new_state][new_action] - self.Q_table[state][action])

				state = new_state
				action = new_action
				eps_reward += reward

				if terminate == False:
					noActionTaken += 1					
					if noActionTaken == (self.max_steps):
						# Initiate failure to find goal/hole found
						terminate = True
						data[step_var].append(self.max_steps)
						data[acc_var].append(0)

				elif terminate == True:
					data[step_var].append(noActionTaken)
					if reward != 1:
						data[acc_var].append(0)
					elif reward == 1:
						data[acc_var].append(1)
			
			self.episodes_reward.append(eps_reward)
			data[reward_var].append(eps_reward)
		
		self.policy = np.argmax(self.Q_table, axis=1)
		data_df = self.data_conversion(data)

		return data_df, data


class QLearning(GenericTrainer):
	def __init__(self, env):
		super().__init__(env)
	
	def train(self, ep_no=1):
		reward_var= "Q_Rewards" + str(ep_no)  # Reward received in each episode
		step_var = "Q_noSteps" + str(ep_no)  # No. of steps taken in each episode
		acc_var ="Q_Accuracy" + str(ep_no) # Accuracy of data in each episode
		data = {reward_var:[], step_var:[], acc_var:[]}
		
		for i in range(self.num_episodes):
			# Infrom us the number of episodes we are going throug
			if (i % 100) == 0:
				print(i)

			# Reset environment
			state = self.env.reset()

			# Initialize data 
			eps_reward = 0		# Total reward earned during episode
			terminate = False	# Bool to terminate episode
			noActionTaken = 0	# total no. of action taken during the episode

			while not terminate:
				action = self.getNextAction(state)
				new_state, reward, terminate, info = self.env.step(action)
				
				self.Q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q_table[new_state])) - self.Q_table[state][action]

				state = new_state
				eps_reward += reward

				if terminate == False:
					noActionTaken += 1					
					if noActionTaken == (self.max_steps):
						# Initiate failure to find goal/hole found
						terminate = True
						data[step_var].append(self.max_steps)
						data[acc_var].append(0)

				elif terminate == True:
					data[step_var].append(noActionTaken)
					if reward != 1:
						data[acc_var].append(0)
					elif reward == 1:
						data[acc_var].append(1)
			
			self.episodes_reward.append(eps_reward)
			data[reward_var].append(eps_reward)
		
		self.policy = np.argmax(self.Q_table, axis=1)
		data_df = self.data_conversion(data)

		return data_df, data