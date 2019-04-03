import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
from JointActionLearnerBase import JointQLearningAgent
from tqdm import tqdm
import numpy as np


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



if __name__ == '__main__':
    numOpponents = 1
    numAgents = 2
    MARLEnv = DiscreteMARLEnvironment(numOpponents=numOpponents, numAgents=numAgents)
    agents = []
    for i in range(numAgents):
        agent = JointQLearningAgent(learningRate=0.1, discountFactor=0.9, epsilon=1.0,numTeammates=1)
        agents.append(agent)

    numEpisodes = 50000
    numTakenActions = 0
    numTakenActionCKPT = 0

    status_lst = []
    n_episode = []
    data = {'status': status_lst, 'steps_in_episode': n_episode}

    for episode in tqdm(range(numEpisodes)):
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0
        timeSteps = 0

        while status[0] == "IN_GAME":
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies, nextStateCopies = [], []
            for agentIdx in range(numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())
            numTakenActions += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)

            for agentIdx in range(numAgents):
                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx],
                    status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            observation = nextObservation

        status_lst.append(
            status[0])
        n_episode.append(numTakenActions - numTakenActionCKPT)
        numTakenActionCKPT = numTakenActions

        if episode % 100 == 0:
            print('action number %d, episode numer %d' % (numTakenActions, episode))

    describe(data)