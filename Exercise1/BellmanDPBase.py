from MDP import MDP
import numpy as np

class BellmanDPSolver(object):
	def __init__(self,discountRate):
		self.MDP = MDP()
		self.gamma = discountRate
		self.result_Value = {}
		self.new_Value = {}
		self.Optimal_Policy = {}
		self.Policy_list = {}

	def initVs(self):
		for i in self.MDP.S:
			self.result_Value[i] = 0
			self.new_Value[i] = 0
			self.Optimal_Policy[i] = []
		for i in self.MDP.A:
			self.Policy_list[i] = 0
			

	def BellmanUpdate(self):
		nextState = {}
		for i in self.MDP.S:
			self.Optimal_Policy[i] = []	
		for eachState in self.MDP.S:
			for eachAction in self.MDP.A:
				nextState = self.MDP.probNextStates(eachState,eachAction)
				for nextPossibleState in nextState.keys():
					self.Policy_list[eachAction] += nextState[nextPossibleState]*(self.MDP.getRewards(eachState,eachAction,nextPossibleState) + self.gamma * self.result_Value[nextPossibleState])
			allValue = np.array(list(self.Policy_list.values()))
			self.new_Value[eachState] = np.max(allValue)
			allPolicy = np.where(allValue==np.max(allValue))[0]
			for eachPolicy in allPolicy:
				self.Optimal_Policy[eachState].append(self.MDP.A[eachPolicy])
			for i in self.MDP.A:
				self.Policy_list[i] = 0
		self.result_Value = self.new_Value.copy()
		return self.result_Value, self.Optimal_Policy



	
		
if __name__ == '__main__':
	solution = BellmanDPSolver(0.9)
	solution.initVs()
	for i in range(2000):
		values, policy = solution.BellmanUpdate()

	print("Values : ", values)

	print("Policy : ", policy)

