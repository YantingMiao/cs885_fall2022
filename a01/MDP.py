from re import S
import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def getMaxValue(self, value, state):
        reward, transition = self.R[:, state], self.T[:, state, :]
        max_v = np.max(reward + self.discount * np.dot(transition, value))
        return max_v

    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.copy(initialV)
        iterId = 0
        while iterId < nIterations:
            epsilon = 0
            prev_V = np.copy(V)
            for state in range(self.nStates):
                V[state] = self.getMaxValue(prev_V, state)
                epsilon = max(epsilon, np.abs(V[state] - prev_V[state]))
            iterId += 1
            if epsilon < tolerance:
                break
        return [V,iterId,epsilon]

    def getOptimalAction(self, value, state):
        reward, transition = self.R[:, state], self.T[:, state, :]
        optimal_action = np.argmax(reward + self.discount * np.dot(transition, value))
        return optimal_action

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        for state in range(self.nStates):
            policy[state] = self.getOptimalAction(V, state)
        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        r_pi, transition_pi = np.zeros((self.nStates,)), np.zeros((self.nStates, self.nStates))
        for state in range(self.nStates):
            r_pi[state] = self.R[policy[state], state]
            transition_pi[state] = self.T[policy[state], state]
        v = np.dot(np.linalg.inv(np.identity(self.nStates) - self.discount * transition_pi), r_pi)
        return v

    def policyImprove(self, value, policy):
        done = True
        for state in range(self.nStates):
            original_action = policy[state]
            opt_action = self.getOptimalAction(value, state)
            if original_action != opt_action:
                policy[state] = opt_action
                done = False
        return policy, done
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.copy(initialPolicy)
        V = np.zeros(self.nStates)
        iterId = 0
        done = False
        while iterId < nIterations and not done:
            V = self.evaluatePolicy(policy)
            policy, done = self.policyImprove(V, policy)
            iterId += 1
        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.copy(initialV)
        iterId = 0
        reward, transition = np.zeros((self.nStates,)), np.zeros((self.nStates, self.nStates))
        for state in range(self.nStates):
            reward[state] = self.R[policy[state], state]
            transition[state] = self.T[policy[state], state]

        while iterId < nIterations:
            epsilon = 0
            prev_v = np.copy(V)
            V = reward + self.discount * np.dot(transition, prev_v)
            epsilon = np.linalg.norm(V - prev_v, np.inf)
            iterId += 1
            if epsilon < tolerance:
                break
        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.copy(initialPolicy)
        V = np.copy(initialV)
        iterId = 0
        epsilon = 0
        while iterId < nIterations:
            V = self.evaluatePolicyPartially(policy, V, nIterations=nEvalIterations, tolerance=tolerance)[0]
            policy, done  = self.policyImprove(V, policy)
            iterId += 1
            # if done:
            #     break
            prev_v = np.copy(V)
            for state in range(self.nStates):
                V[state] = self.getMaxValue(V, state)
            epsilon = np.linalg.norm(V - prev_v, np.inf)
            if epsilon < tolerance:
                break
        return [policy,V,iterId,epsilon]
        