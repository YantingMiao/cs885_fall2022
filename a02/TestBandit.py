import numpy as np
import MDP
import RL2
from matplotlib import pyplot as plt
from tqdm import tqdm


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0

def plot_curves(data, colors, labels):
    for i, y in enumerate(data):
        x = np.arange(y.shape[0])
        plt.plot(x, y, color=colors[i], label=labels[i])
    plt.legend(loc="best")
    plt.xlabel("nIterations")
    plt.ylabel("Avg Rewards")
    plt.grid()
    plt.show()

def main(n_eval=1000, nIterations=200):
    T = np.array([[[1]],[[1]],[[1]]])
    R = np.array([[0.25],[0.5],[0.75]])
    discount = 0.999
    mdp = MDP.MDP(T,R,discount)
    banditProblem = RL2.RL2(mdp,sampleBernoulli)
    epsilon_avg_rewards, ts_avg_rewards, ucb_avg_rewards = np.zeros(nIterations,), np.zeros(nIterations,), np.zeros(nIterations,)
    for _ in tqdm(range(n_eval)):
        __, rewards = banditProblem.epsilonGreedyBandit(nIterations=nIterations)
        epsilon_avg_rewards += rewards

        __, rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=nIterations)
        ts_avg_rewards += rewards

        __, rewards = banditProblem.UCBbandit(nIterations=nIterations)
        ucb_avg_rewards += rewards
    epsilon_avg_rewards /= n_eval
    ts_avg_rewards /= n_eval
    ucb_avg_rewards /= n_eval
    colors = ['gold', 'deepskyblue', 'pink']
    labels = [r'$\epsilon$-greedy', 'Thompson', 'UCB']
    plot_curves([epsilon_avg_rewards, ts_avg_rewards, ucb_avg_rewards], colors, labels)

if __name__ == "__main__":
    main(1000, 200)

"""
# Multi-arm bandit problems (3 arms with probabilities 0.25, 0.5 and 0.75)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.25],[0.5],[0.75]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy
empiricalMeans, epsilon_rewards = banditProblem.epsilonGreedyBandit(nIterations=200)
print("\nepsilonGreedyBandit results")
print(empiricalMeans)

# Test Thompson sampling strategy
empiricalMeans, ts_rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
print("\nthompsonSamplingBandit results")
print(empiricalMeans)

# Test UCB strategy
empiricalMeans, ucb_rewards = banditProblem.UCBbandit(nIterations=200)
print("\nUCBbandit results")
print(empiricalMeans)
"""