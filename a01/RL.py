import numpy as np
import MDP

class RL:
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
        self.reward_list = None

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

    def shuffleAction(self, value_list):
        q_action_pairs = []
        for action, q in enumerate(value_list):
            q_action_pairs.append((q, action))
        np.random.shuffle(q_action_pairs)
        q_action_pairs.sort(key=lambda x:x[0], reverse=True)
        action = q_action_pairs[0][1]
        return action

    def selectOptimalAction(self, Q, state, epsilon=0, temperature=0):
        n_actions = self.mdp.nActions
        temp = Q[:, state]
        if epsilon > 0:
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                # to select one of the actions of equal Q-value at random due to Python's sort is stable
                action = self.shuffleAction(temp)
            return action
        if epsilon == 0 and temperature > 0:
            p = np.exp(temp / temperature) / np.sum(np.exp(temp / temperature))
            action = np.random.choice(n_actions, 1, p=p)[0]
            return action
        # to select one of the actions of equal Q-value at random due to Python's sort is stable
        action = self.shuffleAction(temp)
        return action

    def culmulativeReward(self, reward, episode, discount, t):
        cul_reward = (discount ** t) * reward
        self.reward_list[episode] += cul_reward
    
    def getCulmulativeReward(self):
        return self.reward_list

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random )
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        self.reward_list = [0] * nEpisodes
        Q = np.copy(initialQ)
        policy = np.zeros(self.mdp.nStates,int)
        count = np.zeros((self.mdp.nStates, self.mdp.nActions))
        for episode in range(nEpisodes):
            state = int(np.copy(s0))
            for t in range(nSteps):
                action = self.selectOptimalAction(Q, state, epsilon, temperature)
                reward, next_state = self.sampleRewardAndNextState(state, action)
                q_next = np.max(Q[:, next_state])
                count[state, action] += 1
                alpha = 1 / count[state, action]
                Q[action][state] += alpha * (reward + self.mdp.discount * q_next - Q[action][state])
                state = next_state
                self.culmulativeReward(reward, episode, self.mdp.discount, t)
                
        # extract policy
        for state in range(self.mdp.nStates):
            policy[state] = self.selectOptimalAction(Q, state, 0, 0)
        return [Q,policy] 