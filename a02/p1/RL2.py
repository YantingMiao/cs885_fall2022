import numpy as np
import MDP

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def epsilon_greed(self, values, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(values.shape[0])
        return np.argmax(values)

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions,)
        actions_counter, epsilon_greedy_rewards = np.zeros_like(empiricalMeans), [None] * nIterations
        state = 0
        for t in range(1, nIterations + 1):
            action = self.epsilon_greed(empiricalMeans, 1/t)
            [reward, state] = self.sampleRewardAndNextState(state, action)
            epsilon_greedy_rewards[t - 1] = reward
            actions_counter[action] += 1
            empiricalMeans[action] += (reward - empiricalMeans[action]) / actions_counter[action]
        return empiricalMeans, epsilon_greedy_rewards

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions,)
        n_acticons = empiricalMeans.shape[0]
        ts_rewards = [None] * nIterations
        state = 0
        for t in range(nIterations):
            for action in range(n_acticons):
                empiricalMeans[action] = np.mean(np.random.beta(prior[action][0], prior[action][1], size=k))
            action = np.argmax(empiricalMeans)
            [reward, state] = self.sampleRewardAndNextState(state, action)
            ts_rewards[t] = reward
            prior[action][0] += reward
            prior[action][1] += (1 - reward)
        return empiricalMeans, ts_rewards
    
    def UCBSelectAction(self, values, actions_counter,timestep, epsilon=1e-08):
        temp = 2 * np.log(timestep) * np.ones_like(values)
        upper_bound = np.sqrt(np.divide(temp, actions_counter + epsilon))
        return np.argmax(values + upper_bound)

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions,)
        actions_counter, ucb_rewards = np.zeros_like(empiricalMeans), [None] * nIterations
        state = 0
        epsilon = 1e-08
        for t in range(1, nIterations + 1):
            action = self.UCBSelectAction(empiricalMeans, actions_counter, t, epsilon)
            [reward, state] = self.sampleRewardAndNextState(state, action)
            ucb_rewards[t - 1] = reward
            actions_counter[action] += 1
            empiricalMeans[action] += (reward - empiricalMeans[action]) / actions_counter[action]
        return empiricalMeans, ucb_rewards