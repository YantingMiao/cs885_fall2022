import numpy as np
import MDP
import RL
from matplotlib import pyplot as plt
from tqdm import tqdm

n_trails = 100
n_episodes, n_steps = 200, 100
epsilon_list = [0.1, 0.3, 0.5]

def plot_curves(data, color_list, label_list):
    for i in range(data.shape[0]):
        x = np.arange(n_episodes)
        plt.plot(x, data[i], color=color_list[i], label=r'$\epsilon={}$'.format(label_list[i]))
    plt.legend(loc='best')
    plt.xlabel("# Episodes")
    plt.ylabel("Cumulative rewards")
    plt.savefig('epsilon.png')

''' Construct simple MDP as described in Lecture 1b Slides 17-18'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning
results = np.zeros((len(epsilon_list), n_episodes))
for i, epsilon in enumerate(epsilon_list):
    res = np.zeros((n_episodes,))
    print('epsilon =', epsilon)
    for _ in tqdm(range(n_trails)):
        mdp = MDP.MDP(T,R,discount)
        rlProblem = RL.RL(mdp,np.random.normal)
        [Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=n_episodes,nSteps=n_steps,epsilon=epsilon)
        res += np.array(rlProblem.reward_list)
    results[i] = res / n_trails

color_list = ['red', 'blue', 'green']
plot_curves(results, color_list, epsilon_list)