#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
from hfo import *
import logging




class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.gamma = discountFactor
		self.G = 0
		self.Returns = {}
		self.episode = []
		self.episode_num = 0
		self.Q = {}
		# self.Policy = {}
		self.epsilon = epsilon
		self.initVals = initVals
		self.New_returns = {}



	def learn(self):

		list_Q_now=[]
		self.episode.reverse()
		episode_np = np.array(self.episode)
		pop_list = []
		for i in episode_np[:,:2]:
			pop_list.append(tuple(i))


		for index in range(episode_np.shape[0]):
			self.G = self.gamma * self.G + episode_np[index,2]
			state = episode_np[index,0]
			action_index = self.possibleActions.index(episode_np[index,1])

			if state not in self.Returns.keys():
				self.Returns[state] = [[],[],[],[],[]]
			list_SA = self.Returns[state][action_index]

			SA_ = pop_list[0]
			pop_list = pop_list[1:]
			if  SA_ not in pop_list:
				list_SA.append(self.G)
				list_Q_now.append(np.average(list_SA))
				try:
					self.Q[state][action_index] = np.average(list_SA)
				except KeyError:
					self.Q[state] = np.ones(len(self.possibleActions))*self.initVals

		list_Q_now.reverse()
		return self.Q,list_Q_now


	def toStateRepresentation(self, state):
		return tuple(state)

	def setExperience(self, state, action, reward, status, nextState):
		self.episode.append([state, action, reward])


	def setState(self, state):
		self.state_now = state
		if self.state_now not in self.Q.keys():
			self.Q[self.state_now] = np.ones(len(self.possibleActions))*self.initVals


	def reset(self):
		self.G = 0
		self.Q = {}
		self.state_now = ()
		self.episode = []

	def act(self):
		# try:
		# 	action = self.possibleActions[np.random.choice(np.arange(len(self.possibleActions)),p=self.Policy[self.state_now])]
		# except KeyError:
		# 	action = np.random.choice(self.possibleActions)
		# return action
		if np.random.rand() < self.epsilon:
			action = np.random.choice(self.possibleActions)
		else:
			# try:
				action = self.possibleActions[
					np.random.choice(np.where(self.Q[self.state_now]==np.max(self.Q[self.state_now]))[0])]
				# print("!!!")
			# except KeyError:
			# 	action = np.random.choice(self.possibleActions)
			# 	print("?")
		return action





	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.episode_num = episodeNumber
		# epsilon = 1
		if episodeNumber < 500:
			epsilon = 1
		else:
			epsilon = 10/9. - episodeNumber / 4500
			# epsilon = 0.9*(5000-episodeNumber)/5000

		return epsilon



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0

	# Configure the logging system
	logging.basicConfig(
		filename='status.log',
		level=logging.INFO
	)

	# # Create a custom logger
	# logger = logging.getLogger(__name__)
	#
	# # Create handlers
	# c_handler = logging.StreamHandler()
	# f_handler = logging.FileHandler('status.log')
	# c_handler.setLevel(logging.WARNING)
	# f_handler.setLevel(logging.DEBUG)
	#
	# # Create formatters and add it to handlers
	# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
	# f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# c_handler.setFormatter(c_format)
	# f_handler.setFormatter(f_format)
	#
	# # Add handlers to the logger
	# logger.addHandler(c_handler)
	# logger.addHandler(f_handler)

	list_status = np.zeros(6)

	# Run training Monte Carlo Method
	for episode in range(numEpisodes):
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

			list_status[status] += 1
			logging.info("GOAL:%d, CAPTURED_BY_DEFENSE:%d, OUT_OF_BOUNDS:%d, OUT_OF_TIME:%d, SERVER_DOWN:%d, RATE = %f",
						 list_status[hfo.GOAL],list_status[hfo.CAPTURED_BY_DEFENSE],list_status[hfo.OUT_OF_BOUNDS],list_status[hfo.OUT_OF_TIME],list_status[hfo.SERVER_DOWN],list_status[hfo.GOAL]/episode)

		agent.learn()
