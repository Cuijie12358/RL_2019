#!/usr/bin/env python3
# encoding utf-8

import torch.multiprocessing as mp
from SharedAdam import SharedAdam
from Networks import ValueNetwork
import numpy as np
from Environment import HFOEnv
from Worker import train
import argparse
import torch.optim as optim
# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment, and initialization of
# your models, torch's multiprocessing methods, etc.



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

	# Example on how to initialize global locks for processes
	# and counters.
    num_processes = 4
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    processes = []
    value_network = ValueNetwork(68,[16, 16],4)
    target_value_network = ValueNetwork(68,[16, 16],4)
    optimizer = SharedAdam(value_network.parameters())
    epsilon = 1
    discountFactor = 0.99

    # Example code to initialize torch multiprocessing.
    for idx in range(0, num_processes):
        trainingArgs = (idx, value_network, target_value_network, optimizer, lock, counter, epsilon, discountFactor, args)
        p = mp.Process(target=train, args=trainingArgs)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # num_processes = 2
    # num_episode = 0
    # processes=[]
    # while True:
    #     counter = mp.Value('i', 0)
    #     lock = mp.Lock()
    #     for idx in range(num_processes):
    #         port = 2019 + idx * 12
    #         seed = 123 + idx * 4
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

