#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np

		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()	
		self.gamma = discountFactor
		self.setEpsilon(epsilon)
		self.setLearningRate(learningRate)
		self.alpha = learningRate
		self.initVals = initVals
		self.Q_target = 0
		self.numTeammates = numTeammates
		self.CountActions = {}
		self.nState = {}
		self.Q_table = {}
		self.sum_action_list={}

	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState[0]
		self.state = state[0]
		self.oppoActions = oppoActions
		self.nextState = nextState[0]
		# print("reward",self,reward)


		#
		#
		# if self.state not in self.Q_table.keys():
		# 	self.Q_table[self.state] = np.zeros(index_Q)
		if self.nextState not in self.sum_action_list.keys():
			self.sum_action_list[self.nextState] = np.zeros(len(self.possibleActions))




		# C(s,a-i)
		self.index_actions = np.array(self.possibleActions.index(action))
		for anotherAction in oppoActions:
			self.index_actions = np.append(self.index_actions, self.possibleActions.index(anotherAction))
		# print("index_actions",self.index_actions)

		try:
			# print("index",np.split(self.index_actions[1:],len(self.index_actions[1:])))

			self.CountActions[self.state][np.split(self.index_actions[1:],len(self.index_actions[1:]))] += 1

		except KeyError:
			self.CountActions[self.state] = np.zeros(self.index_C)
			self.CountActions[self.state][np.split(self.index_actions[1:],len(self.index_actions[1:]))] += 1

		# print(self.CountActions[self.state])





		
	def learn(self):

		a = (1-self.alpha) * sum(self.Q_table[self.state][self.possibleActions.index(self.action)])
		b = self.alpha*(self.reward+self.gamma*self.sum_action_list[self.nextState].max())
		self.Q_target = a + b                  # next state is not terminal
		# print("Q_target???",self.Q_target)


		difference = self.Q_target - self.Q_table[self.state][np.split(self.index_actions,len(self.index_actions))]
		# print("alpha",self.alpha)
		# print("index",[np.split(self.index_actions,len(self.index_actions))])
		# print("Q_table",self.Q_table[self.state])

		# print("difference",difference)
		return difference



	def act(self):
		if np.random.rand() < self.epsilon:
			action = np.random.choice(self.possibleActions)
			return action
		else:
			# actionlist = self.Q_table[self.state]

			self.sum_action_list[self.state] = np.zeros(len(self.possibleActions))
			for i in range(len(self.possibleActions)):
				self.sum_action_list[self.state][i]= sum(self.CountActions[self.state]/self.nState[self.state]*self.Q_table[self.state][i])
				# print("sum_action_list[i]",i,self.sum_action_list[self.state][i])

			action = self.possibleActions[np.random.choice(np.where(self.sum_action_list[self.state]==np.max(self.sum_action_list[self.state]))[0])]
			return action

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate) :
		self.alpha = learningRate

	def setState(self, state):
		self.state = state[0]
		self.index_Q = tuple(list((np.ones(self.numTeammates+1)*len(self.possibleActions)).astype(int)))
		self.index_C = tuple(list((np.ones(self.numTeammates) * len(self.possibleActions)).astype(int)))

		#n(s)
		try:
			self.nState[self.state] +=1
		except KeyError:
			self.nState[self.state] = 1

		# print(self.state)
		if state[0] not in self.Q_table.keys():
			self.Q_table[state[0]] = np.ones(self.index_Q) * self.initVals
		if state[0] not in self.CountActions.keys():
			self.CountActions[state[0]] = np.zeros(self.index_C)
		if state[0] not in self.sum_action_list.keys():
			self.sum_action_list[state[0]] = np.zeros(len(self.possibleActions))
		# print(self.state)



	def toStateRepresentation(self, rawState):
		list_state = []
		tuple_state = []
		for i in rawState[0]:
			list_state.append(tuple(i))
		tuple_state.append(tuple(list_state[:]))
		for i in range(len(rawState)-1):
			tuple_state.append(tuple(rawState[i+1][0]))
		return tuple(tuple_state)
		
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
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0

	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()

			observation = nextObservation
