import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import numpy as np
import torch.multiprocessing as mp


def train(idx, value_network, target_value_network, optimizer, lock, counter, epsilon, discountFactor,args):
	if args.gpu and torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	Itarget = 500
	IAsyncUpdate = 20
	# episodeNumber += 1
	t = 0
	# for idx in range(num_processes):
	port = 9012 + idx * 12
	seed = 123 + idx * 10
	epsilon = 1

	hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
	hfoEnv.connectToServer()

	while True:
		done = False
		state = torch.tensor(hfoEnv.reset()).to(device)
		while not done:
			action = select_action(state, hfoEnv, value_network, epsilon)

			action_idx =hfoEnv.possibleActions.index(action)

			# print("action_idx",action_idx)
			newObservation, reward, done, status, info = hfoEnv.step(action)
			target_output = computeTargets(reward, newObservation, discountFactor, done, target_value_network)
			predict_output = computePrediction(state, action_idx, value_network)

			criterion = nn.MSELoss()
			loss = criterion(target_output, predict_output)
			loss.backward()

			with lock:
				counter.value += 1
			t += 1
			# state = newObservation
			if counter.value % Itarget == 0:
				target_value_network.load_state_dict(value_network.state_dict())
			if t % IAsyncUpdate == 0 or done:
				optimizer.step()
				optimizer.zero_grad()
			state = newObservation

			if counter.value % 1000 == 0:
				i = counter.value//1000
				directory = "output/params_"+str(i)
				saveModelNetwork(target_value_network,directory)
		if counter.value > 10000:
			break



#         hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
#         hfoEnv.connectToServer()
#         state = hfoEnv.hfo.getState()
#         valueNetwork = ValueNetwork(68,[16,16],4) # LOW_FEATURE_LEVEL
#         # print(valueNetwork)
#         targetNetwork = valueNetwork
#         epsilon = 1
#         action = Worker.select_action(state,hfoEnv,valueNetwork,epsilon)
#         # print("action:",action)
#         newObservation, reward, done, status, info = hfoEnv.step(action)
#         valueNetwork.share_mrmoey()
#
#         if done:
#             num_episode += 1

	# if args.use_gpu and torch.cuda.is_available():
	# 	device = torch.device("cuda")
	# else:
	# 	device = torch.device("cpu")
	# port = 2018 + idx * 4
	# seed = 402 + idx * 1
	# return None

	#
	# T = 0
	# t = 0
	# d_theta = 0
	# state = state


def select_action(state,agent,valueNetwork,epsilon):
	if np.random.rand() < epsilon:
		return agent.possibleActions[np.random.randint(0, 3)]
	else:
		Q_table = valueNetwork.forward(state).detach().numpy()
		# print(Q_table)
		action = np.random.choice(np.where(Q_table == np.max(Q_table))[0])
		return agent.possibleActions[action]
	return action


def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	# print(nextObservation)
	q_target = torch.empty(1, 1, dtype=torch.float)
	value = targetNetwork.forward(nextObservation).max()
	if done:
		q_target.fill_(reward)
	else:
		q_target.fill_(reward + discountFactor * value)
	# print("q_target",q_target)

	# print(np.max(value))
	return q_target

def computePrediction(state, action, valueNetwork):
	result = valueNetwork.forward(state)
	# print("action",action)
	# print("Result:",result)
	return result[:,action]




	
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)
	




