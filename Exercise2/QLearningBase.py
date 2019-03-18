#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.gamma = discountFactor
		self.setEpsilon(epsilon)
		self.setLearningRate(learningRate)
		self.Q_table = {}
		self.alpha = learningRate
		self.initVals = initVals
		self.Q_target = 0


		

	def learn(self):
		if self.status != 0:
			self.Q_target = self.reward
		else:
			self.Q_target = self.reward + self.gamma * self.Q_table[self.nextState].max()  # next state is not terminal
		difference = self.Q_target - self.Q_table[self.state][self.possibleActions.index(self.action)]
		self.Q_table[self.state][self.possibleActions.index(self.action)] += self.alpha * difference
		return self.alpha * difference


	def act(self):
		if np.random.rand() < self.epsilon:
			action = np.random.choice(self.possibleActions)
			return action
		else:
			actionlist = self.Q_table[self.state]
			action = self.possibleActions[np.random.choice(np.where(actionlist==np.max(actionlist))[0])]
			return action



	def toStateRepresentation(self, state):
		return tuple(state)

	def setState(self, state):
		self.state = state
		if state not in self.Q_table.keys():
			self.Q_table[state] = np.ones(5)*self.initVals		

	def setExperience(self, state, action, reward, status, nextState):
		if nextState not in self.Q_table.keys():
			self.Q_table[nextState] = np.ones(5)*self.initVals
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState
		self.state = state


	def setLearningRate(self, learningRate):
		self.alpha = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def reset(self):
		self.action = []
		self.reward = 0
		self.status = ()
		self.Q_target = 0
	
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.episode_num = episodeNumber
		if numTakenActions < 10:
			epsilon = 1
		else:
			epsilon = 1.0/(numTakenActions//10)
		# if numTakenActions < 30 :
		# 	epsilon = min(1,1-(episodeNumber-100)/100.0)
		# else:
		# 	epsilon = min(1,1-(episodeNumber-100)/100.0)/(numTakenActions//10)
		learningRate = self.alpha

		return learningRate, epsilon			


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status == 0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	
