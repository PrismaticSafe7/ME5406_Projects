import numpy as np
import random

import sys
from io import StringIO

# Possible action of agent
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Providing valid maps for testings
DEFAULT_MAPS = {}
DEFAULT_MAPS["4x4"] = ["SFFF", "FHFH", "FFFH", "HFFG"]
DEFAULT_MAPS["8x8"] = ["SFFFFFFF",
				        "FFFFFFFF",
				        "FFFHFFFF",
				        "FFFFFHFF",
				        "FFFHFFFF",
				        "FHHFFFHF",
				        "FHFFHFHF",
				        "FFFHFFFG"]
DEFAULT_MAPS["10x10"] = ["SFFFHFFFHH", 
				        "FFFFHHFFFF", 
				        "FFFFFFFFFF", 
				        "HHFFFFHFFH", 
				        "FFHFFFFFFF", 
				        "FHFFFHFFFF", 
				        "HFFFFHFFFF", 
				        "HFFHFFHFFF", 
				        "FHHFFHFFFH", 
				        "HHHFHFFFFG"
				        ]

# Checking newly generated map for valid path using Breadth-First-Search
def mapValidity(map, size):
	frontier = []
	explored = set()
	frontier.append((0,0))
	# [Right, Down, Left, Up]
	exploration_paths = [(1,0), (0,1), (-1,0), (0,-1)]

	while frontier:
		loc = frontier.pop()
		row, col = loc
		if loc not in explored:
			# add explored location
			explored.add(loc)

			for x,y in exploration_paths:
				X_loc = row + x
				Y_loc = col + y
				if 0 <= X_loc < size:
					if 0 <= Y_loc < size:
						if map[X_loc][Y_loc] == "G":
							return True
						elif map[X_loc][Y_loc] != "H":
							frontier.append((X_loc, Y_loc))

	return False


def randomMapGenerator(size = 4, p=0.25):
	validMap = False
	numHole = 0

	while validMap == False:
		generatedMap = np.random.choice(["F","H"], (size,size), p=[1-p, p])
		generatedMap[0][0] = "S"
		generatedMap[-1][-1] = "G"
		validMap = mapValidity(generatedMap, size)

		# find number of holes in map
		numHole = np.count_nonzero(generatedMap == "H")
		print(numHole)

		if numHole != (size**2 * p):
			numHole = 0
			validMap = False

	print(generatedMap)

	return ["".join(x) for x in generatedMap]


class FrozenLakeEnv:
	'''
    A Frozen Lake environment where you and your friends were playing frisbee nearby.

    You accidentally threw the frisbee (which is also limited edition that cannot
    be found anywhere else in the world) towards the lake while trying to impress 
    your friends and so, you ended up in this undesirable situation where you have
    to risk falling into the lake to retrieve the frisbee.

    The ice could be slippery as well, depending on the weather a few days ago.

    Default surface is described using a grid as shown:

        S F F F
        F H F H
        F F F H
        H F F G

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reached the goal or fell into a hole.
    You will receive +1 reward for reaching the frisbee, -1 reward for falling into
    a hole, and 0 otherwise.
    '''

	def __init__(self, mapData="Default", mapSize=4, isSlippery=False, slipChance=0.0):
    	#if no map data provided, use default maps
		if mapData == "Default":
			if mapSize == 4:
				mapData = DEFAULT_MAPS["4x4"]
			elif mapSize == 8:
				mapData = DEFAULT_MAPS["8x8"]
			elif mapSize == 10:
				mapData = DEFAULT_MAPS["10x10"]
    	
		elif mapData == "newMap":
			mapData = randomMapGenerator(mapSize)

		else:
			mapData = DEFAULT_MAPS["4x4"]

    	#checks for whether floor is slippery, updates slipchance if it is:
		if isSlippery:
			if slipChance != 0:
				self.slipChance = slipChance
			else:
				self.slipChance = 0.2

		self.mapData = np.asarray(mapData, dtype="c")

    	# Initialize data:
		self.lastAction = None
		self.num_row = mapSize 
		self.num_col = mapSize
		self.num_state = self.num_row * self.num_col
		self.action_space = [LEFT, DOWN, RIGHT, UP]
		self.num_action = len(self.action_space)

    	# Store location of all starting points "S" in the map
		self.initialStates = [(x,y) for x in range(self.num_row) for y in range(self.num_col) 
    							if mapData[x][y]=="S"]

    	# When environment initialized, current state == None. Agent needs to perform reset()
    	# to update and pick the initial start state
		self.currState = None

		'''
    	Create dictionary of lists for the probability matrix of every state-action pair
    	where P[state][action] == [(probability, nextState, reward, terminate), ...]
    	
		Probability of action leading to nextexpected state is 1 when not slippery
		Else, there is a ?? chance of moving to an unintended state 
    	'''
		self.probabilityMatrix = {}
		for row in range(self.num_row):
			for col in range(self.num_col):
				s = (row * self.num_col) + col 
				self.probabilityMatrix[s] = {}
				for a in range(self.num_action):
					self.probabilityMatrix[s][a] = []

		def update_prob_matrix(row, col, action):
			newRow, newCol = row, col
			reward = 0
			terminate = False
    		
    		#Update Col/Row based on action chosen
			if action == LEFT:
				newCol = max(col-1, 0)
			if action == RIGHT:
				newCol = min(col+1, mapSize)
			if action == UP:
				newRow = max(0, row+1)
			if action == DOWN:
				newRow = min(row+1, mapSize)
    		
			# Update state and cell type
			newState = newRow * self.num_col + newCol
			newCellType = self.mapData[newRow][newCol]

    		# Check if cell is a hole or goal, else pass
			if newCellType == "G":
				terminate = True
				reward = 1
			elif newCellType == "H":
				terminate = True
				reward = -1
    		
			return newState, reward, terminate

		for row in range(self.num_row):
			for col in range(self.num_col):
    			# State in 1D
				s = row * self.num_col + col

				for a in range(self.num_action):
					P = self.probabilityMatrix[s][a]
					spot = mapData[row][col]

    				# Check if spot is goal or hole
    				# Actions taken on those spots will not change agent's status and terminate episode
					if spot == "G" or spot == "H":
						P.append((1.0, s, 0, True))
					else:
						if isSlippery:
    						# Random actions 
							for randomAction in [(a-1) % 4, a, (a+1) % 4]:
								newState, reward, terminate = update_prob_matrix(row,col,randomAction)
								if randomAction == a:
									P.append((1.0 - self.slipChance, newState, reward, terminate))
								else:
									P.append((self.slipChance, newState, reward, terminate))
						else:
							newState, reward, terminate = update_prob_matrix(row,col,a)
							P.append((1.0, newState, reward, terminate))


	def reset(self):
		'''
    	Reset state to the randomly selected intial state
    	Other parameters of environment remain same

    	return: index of initial state
    	'''

		self.lastAction = None
		self.currState = random.choice(self.initialStates)

		return self.currState
	
	def randomAction(self):
		'''
		Choose random action
		
		return: a (random action)
		'''
		a = self.action_space[np.random.randint(self.num_action)]
		return a

	def step(self, action):
		'''
    	Update state given an action and current state
    	
		Parameters:
			Action a: taken (LEFT,RIGHT,UP,DOWN)

    	Return: 
    	Tuple: (next valid state, reward, termination)
		
    	'''

    	# Store all possible transitions for state-action pair
		transitions = self.probabilityMatrix[self.currState][action]
    	# Store all probability values for each possible transition
		prob_transitions = [t[0] for t in transitions]

		self.lastAction = action

		result = 0
		result_sum = prob_transitions[0]
		chance = np.random.random()

		'''
		Comments to be added here to explain code
		'''
		while result_sum < chance:
			result += 1
			result_sum += prob_transitions[result]

		probability, newState, reward, terminate = transitions[result]
		self.currState = newState
		return (newState, reward, terminate, 0)
	
	def render(self, mode="human"):
		'''
		Rendering method from openAI environment
		'''

		def colorize(string, color, bold=False, highlight=False):
			"""Return string surrounded by appropriate terminal color codes to
            print colorized text.  Valid colors: gray, red, green, yellow,
            blue, magenta, cyan, white, crimson
            """
			color2num = dict(
                gray=30,
                red=31,
                green=32,
                yellow=33,
                blue=34,
                magenta=35,
                cyan=36,
                white=37,
                crimson=38,
            )

			attr = []
			num = color2num[color]
			if highlight:
				num += 10
			attr.append(str(num))
			if bold:
				attr.append("1")
			attrs = ";".join(attr)
			return "\x1b[%sm%s\x1b[0m" % (attrs, string)

		outfile = StringIO() if mode == "ansi" else sys.stdout

		row, col = self.currState // self.num_col, self.currState % self.num_col
		desc = self.mapData.tolist()
		desc = [[c.decode("utf-8") for c in line] for line in desc]
		desc[row][col] = colorize(desc[row][col], "red", highlight=True)
		if self.lastAction is not None:
			outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastAction])
            )
		else:
			outfile.write("\n")
		outfile.write("\n".join("".join(line) for line in desc) + "\n")