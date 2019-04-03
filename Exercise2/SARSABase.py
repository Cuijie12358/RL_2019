#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import logging
import logging.config
from hfo import *

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.alpha = learningRate
		self.gamma = discountFactor
		self.epsilon = epsilon
		self.initVals = initVals
		self.Q_table = {}
		self.Q_target = 0
		self.episode = []


	def learn(self):
		state_,action_,reward_ = self.episode[-1]
		self.state, self.action, self.reward = self.episode[-2]
		if action_ != None:
			self.Q_target = self.reward + self.gamma * self.Q_table[state_][self.possibleActions.index(action_)]  # next state is not terminal
		else:
			self.Q_target = self.reward
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

	def setState(self, state):
		self.state = state
		if state not in self.Q_table.keys():
			self.Q_table[state] = np.ones(len(self.possibleActions))*self.initVals

	def setExperience(self, state, action, reward, status, nextState):
		self.episode.append([state, action, reward])
		if nextState != None:
			if nextState not in self.Q_table.keys():
				self.Q_table[nextState] = np.ones(len(self.possibleActions))*self.initVals



	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.episode_num = episodeNumber
		if episodeNumber < 500:
			epsilon = 1
		else:
			# epsilon = 1.0/(numTakenActions//10)
			epsilon = 1. * ((1 - 1 / (1 + np.exp(-numTakenActions / 250))) * 2 + 0.1)
		learningRate = self.alpha

		return learningRate, epsilon

	def toStateRepresentation(self, state):
		return tuple(state)

	def reset(self):
		self.episode = []

	def setLearningRate(self, learningRate):
		self.alpha = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99,1.0)

	# Run training using SARSA
	numTakenActions = 0

	# Logging
	logging.config.fileConfig('logconfig.ini')
	list_status = np.zeros(6)

	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			# print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

			if episode>4500:
				list_status[status] += 1
				if episode % 50 ==0:
					logging.info("GOAL:%d, CAPTURED_BY_DEFENSE:%d, OUT_OF_BOUNDS:%d, OUT_OF_TIME:%d, SERVER_DOWN:%d, RATE = %f",
							 list_status[hfo.GOAL], list_status[hfo.CAPTURED_BY_DEFENSE], list_status[hfo.OUT_OF_BOUNDS],
							 list_status[hfo.OUT_OF_TIME], list_status[hfo.SERVER_DOWN], list_status[hfo.GOAL] / (episode-4500))

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	
