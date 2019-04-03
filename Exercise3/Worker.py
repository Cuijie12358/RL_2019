import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def train():
	# if args.use_gpu and torch.cuda.is_available():
	# 	device = torch.device("cuda")
	# else:
	# 	device = torch.device("cpu")
	# port = 2018 + idx * 4
	# seed = 402 + idx * 1
	#
	# if



def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork)
	if done:
		q_target = reward
	else:
		q_target = reward + discountFactor * max(targetNetwork.forward(nextObservation))
	return q_target


def computePrediction(state, action, valueNetwork):




	
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)
	




