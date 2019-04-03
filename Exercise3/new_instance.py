from Environment import HFOEnv
import numpy as np



port = 6000
seed = 123
hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
hfoEnv.connectToServer()

# This runs a random agent
episodeNumber = 0
while True:
    # action = np.random.randint(0, 3)
    # act = hfoEnv.possibleActions[action]
    # newObservation, reward, done, status, info = hfoEnv.step(act)
    # print(newObservation)
    # newObservation[2]=0
    state = hfoEnv.hfo.getState()




    # print(newObservation)

    # breakpoint()


    # if done:
    #     episodeNumber += 1
    # if episodeNumber!=0:
    #     break