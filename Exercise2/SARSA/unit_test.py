import unittest

# from MonteCarloBase import MonteCarloAgent
# from QLearningBase import QLearningAgent
from SARSABase import SARSAAgent


class RLTest(unittest.TestCase):
    def test_Qlearning(self):
        agent = QLearningAgent(learningRate=0.1, discountFactor=1, epsilon=1.0)
        agent.setLearningRate(0.1)

        status = 0
        for obsCopy, action, reward, nextObservation, trueUpdate in zip(
                [((1, 1), (2, 1)), ((2, 1), (2, 1)), ((1, 1), (2, 1))], ['DRIBBLE_RIGHT', 'DRIBBLE_LEFT', 'KICK'],
                [-0.4, 0.0, 1.0], [((2, 1), (2, 1)), ((1, 1), (2, 1)), 'GOAL'], [-0.04, 0.0, 0.1]):
            agent.setState(agent.toStateRepresentation(obsCopy))

            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))
            update = agent.learn()
            self.assertAlmostEqual(update, trueUpdate, places=7)

    def test_SARSA(self):
        agent = SARSAAgent(learningRate=0.1, discountFactor=1, epsilon=1.0)
        agent.setLearningRate(0.1)

        epsStart = True
        status = 0

        for obsCopy, action, reward, nextObservation, trueUpdate in zip(
                [((1, 1), (2, 1)), ((2, 1), (2, 1)), ((1, 1), (2, 1))], ['DRIBBLE_RIGHT', 'DRIBBLE_LEFT', 'KICK'],
                [-0.4, 0.0, 1.0], [((2, 1), (2, 1)), ((1, 1), (2, 1)), 'GOAL'], [None, -0.04, 0.0]):
            agent.setState(agent.toStateRepresentation(obsCopy))

            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))
            if not epsStart:
                update = agent.learn()
                self.assertAlmostEqual(update, trueUpdate, places=7)
            else:
                epsStart = False
                self.assertAlmostEqual(None, trueUpdate, places=7)

        agent.setExperience('GOAL', None, None, None, None)
        update = agent.learn()
        self.assertAlmostEqual(update, 0.1, places=7)

    def test_MonteCarlo(self):
        agent = MonteCarloAgent(discountFactor=1, epsilon=1.0)

        status = 0
        episode = 0
        for obsCopy, action, reward, nextObservation in zip(
                [((1, 1), (2, 1)), ((2, 1), (2, 1)), ((1, 1), (2, 1)), ((0, 1), (2, 1))],
                ['DRIBBLE_RIGHT', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'DRIBBLE_RIGHT'],
                [-0.4, 0.0, 0.0, 0.0], [((2, 1), (2, 1)), ((1, 1), (2, 1)), ((0, 1), (2, 1)), 'OUT_OF_TIME']):
            agent.episode_num = episode
            agent.setState(agent.toStateRepresentation(obsCopy))
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))

        _, QValueList = agent.learn()
        self.assertEqual(QValueList, [-0.4, 0, 0])

        agent.reset()
        episode +=1
        for obsCopy, action, reward, nextObservation in zip(
                [((1, 1), (2, 1)), ((0, 1), (2, 1)), ((1, 1), (2, 1)), ((0, 1), (2, 1))],
                ['DRIBBLE_RIGHT', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'DRIBBLE_RIGHT'],
                [0.0, 0.0, 0.0, 0.0], [((0, 1), (2, 1)), ((1, 1), (2, 1)), ((0, 1), (2, 1)), 'OUT_OF_TIME']):
            agent.episode_num = episode
            agent.setState(agent.toStateRepresentation(obsCopy))
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))

        _, QValueList = agent.learn()
        self.assertEqual(QValueList, [-0.2, 0, 0])


if __name__ == '__main__':
    unittest.main()
