#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np

		
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.gamma = discountFactor
		self.setEpsilon(epsilon)
		self.setLearningRate(learningRate)
		self.Q_table = {}
		self.alpha = learningRate
		self.initVals = initVals
		self.Q_target = 0

	def setExperience(self, state, action, reward, status, nextState):
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState[0]
		self.state = state[0]
		if nextState not in self.Q_table.keys():
			self.Q_table[self.nextState] = np.ones(len(self.possibleActions))*self.initVals


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
		list_state = []
		tuple_state = []
		for i in state[0]:
			list_state.append(tuple(i))
		tuple_state.append(tuple(list_state[:]))
		for i in range(len(state)-1):
			tuple_state.append(tuple(state[i+1][0]))
		return tuple(tuple_state)


	def setState(self, state):
		self.state = state[0]
		# print(self.state)
		if state not in self.Q_table.keys():
			self.Q_table[self.state] = np.ones(len(self.possibleActions)) * self.initVals
		# print(self.Q_table.keys())

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

		
	def setLearningRate(self, learningRate):
		self.alpha = learningRate

		
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
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				# print(obsCopy)
				# print(isinstance(obsCopy,list))

				stateCopies.append(obsCopy)

				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))

				actions.append(agents[agentIdx].act())


			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation
				
