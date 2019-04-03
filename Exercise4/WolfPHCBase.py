#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
from tqdm import tqdm


def describe(data):
    tmp_status = data['status'][-500:]
    tmp_n_steps_episode = data['steps_in_episode'][-500:]
    ratio_goal = np.sum(np.array(tmp_status) == "GOAL") / len(tmp_status)
    ratio_oob = np.sum(np.array(tmp_status) == "OUT_OF_BOUNDS") / len(tmp_status)
    ratio_oot = np.sum(np.array(tmp_status) == "OUT_OF_TIME") / len(tmp_status)

    avg_n_steps_episode = sum(tmp_n_steps_episode) / len(tmp_n_steps_episode)

    print('============ INFO ============')
    print('%-25s:  %d' % ("TOTAL EPISODE NUM", len(data['status'])))
    print('===== LATEST 500 EPISODE =====')
    for info in zip(['GOAL', 'OUT_OF_BOUNDS', 'OUT_OF_TIME'],
                    [ratio_goal, ratio_oob, ratio_oot]):
        print('%-25s:  %.3f' % (info[0], info[1] * 100))

    print('\n')
    print('%-25s:  %.3f' % ('AVG STEPS TO FINISH', avg_n_steps_episode))

class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.gamma = discountFactor
		self.setWinDelta(winDelta)
		self.setLearningRate(learningRate)
		self.setLoseDelta(loseDelta)
		self.Q_table = {}
		self.Pi_table = {}
		self.alpha = learningRate
		self.initVals = initVals
		self.Q_target = 0
		self.C = {}
		self.AveragePi = {}
		self.delta = 0

		
	def setExperience(self, state, action, reward, status, nextState):
		self.state = state[0]
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState[0]

		if self.nextState not in self.Q_table.keys():
			self.Q_table[self.nextState] = np.ones(len(self.possibleActions))*self.initVals
			self.Pi_table[self.nextState] = np.ones(len(self.possibleActions))/len(self.possibleActions)
			self.AveragePi[self.nextState] = np.ones(len(self.possibleActions))/len(self.possibleActions)
			self.C[self.nextState] = 0





	def learn(self):
		action_index = self.possibleActions.index(self.action)
		max_a_ = np.random.choice(np.where(self.Q_table[self.nextState]==np.max(self.Q_table[self.nextState]))[0])
		self.Q_target = self.Q_table[self.state][action_index]
		self.Q_table[self.state][action_index] = self.Q_table[self.state][action_index] + self.alpha*(self.reward+self.Q_table[self.nextState][max_a_]-self.Q_table[self.state][action_index])
		return self.Q_table[self.state][action_index]-self.Q_target

	def act(self):
		action_index = np.random.choice(len(self.possibleActions),1,p=self.Pi_table[self.state])[0]
		return self.possibleActions[action_index]



	def calculateAveragePolicyUpdate(self):
		# Update Pi(s,a)
		self.C[self.state] += 1
		action_index = self.possibleActions.index(self.action)
		self.AveragePi[self.state] = self.AveragePi[self.state] + 1./self.C[self.state] * (self.Pi_table[self.state]-self.AveragePi[self.state])

		# self.AveragePi[self.state][action_index] = self.AveragePi[self.state][action_index] + 1./self.C[self.state] * (self.Pi_table[self.state][action_index]-self.AveragePi[self.state][action_index])
		return self.AveragePi[self.state]

	def calculatePolicyUpdate(self):
		if sum(self.Pi_table[self.state]*self.Q_table[self.state]) >= sum(self.AveragePi[self.state]*self.Q_table[self.state]):
			self.delta = self.winDelta
		else:
			self.delta = self.loseDelta
		best_a = np.where(self.Q_table[self.state]==self.Q_table[self.state].max())[0]
		Pmoved = 0

		# print("original_PI",self.Pi_table[self.state])

		for a_ in range(len(self.possibleActions)):

			if a_ not in list(best_a):
				# print((len(self.possibleActions)-len(best_a)))
				Pmoved += min((self.delta/(len(self.possibleActions)-len(best_a))),(self.Pi_table[self.state][a_]))
				self.Pi_table[self.state][a_] -= min(self.delta/(len(self.possibleActions)-len(best_a)),self.Pi_table[self.state][a_])
				# print("action!!!!!!!!!!!!!!!!",a_)
				# print("Pi_now",self.Pi_table[self.state])
		for a_ in best_a:
			self.Pi_table[self.state][a_] += (Pmoved/len(best_a))


		return self.Pi_table[self.state]

	
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
		if self.state not in self.Q_table.keys():
			self.Pi_table[self.state] = np.ones(len(self.possibleActions))/len(self.possibleActions)
			self.AveragePi[self.state] = np.ones(len(self.possibleActions))/len(self.possibleActions)
			self.Q_table[self.state] = np.ones(len(self.possibleActions)) * self.initVals
			self.C[self.state] = 0



	def setLearningRate(self,lr):
		self.alpha = lr
		
	def setWinDelta(self, winDelta):
		self.winDelta = winDelta
		
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		loseDelta = 0.005
		winDelta = 0.05
		if episodeNumber < 500:
			alpha = self.alpha
		else:
			alpha = 0.5 - episodeNumber * 9 / 500000
		return loseDelta, winDelta, alpha

		# self.episode_num = episodeNumber
		# if numTakenActions < 10:
		# 	self.epsilon = 1
		# else:
		# 	self.epsilon = 1.0/(numTakenActions//10)
		# # if numTakenActions < 30 :
		# # 	epsilon = min(1,1-(episodeNumber-100)/100.0)
		# # else:
		# # 	epsilon = min(1,1-(episodeNumber-100)/100.0)/(numTakenActions//10)
		# learningRate = self.alpha

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=100000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0

	status_lst = []
	n_episode = []
	data = {'status': status_lst, 'steps_in_episode': n_episode}

	for episode in tqdm(range(numEpisodes)):
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			
			observation = nextObservation

		status_lst.append(
			status[0])
		n_episode.append(numTakenActions - args.numEpisodes)
		numTakenActionCKPT = numTakenActions

		if episode % 100 == 0:
			print('action number %d, episode numer %d' % (numTakenActions, episode))

	describe(data)
