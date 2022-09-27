from email.errors import ObsoleteHeaderDefect
import gym
import argparse
import torch
import numpy as np
import random
from utils import save
from cql import CQLDQN, DeepQN
from torch.utils.data import DataLoader, TensorDataset
import wandb
from collections import deque
import pickle

def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='cqldqn', help='algo name, default:cql')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='environment name, defalult: CartPole-v0')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--cql_alpha', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1e-02)
    parser.add_argument('--cql_rescaled', type=float, default=1.0)

    args = parser.parse_args()
    return args

def create_dataloader_env(batch_size=256, seed=1):
    with open('./datasets/cartPole_pure_0.0_0.pkl', 'rb') as f:
        dataset = pickle.load(f)
    tensors = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'dones': []}
    help_list = ['observations', 'actions', 'rewards', 'next_observations', 'dones']
    for i in range(len(dataset)):
        for k, v in dataset[i].items():
            if k in help_list:
                if k != 'dones' and k != 'actions':
                    tensors[k].append(torch.from_numpy(v).float())
                else:
                    tensors[k].append(torch.from_numpy(v).long())

    for k in help_list:
        tensors[k] = torch.cat(tensors[k])

    tensor_dataset = TensorDataset(tensors['observations'],
                                    tensors['actions'],
                                    tensors['rewards'][:, None],
                                    tensors['next_observations'],
                                    tensors['dones'][:, None],)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    eval_env = gym.make('CartPole-v0')
    eval_env.seed(seed)
    return dataloader, eval_env

def evaluate(env, agent, eavl_runs=5):
    avg_rewards = []
    for i in range(eavl_runs):
        state = env.reset()
        cumulative_reward = 0
        done = False
        while not done:
            action = agent.get_action(state, 0.0)
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
        avg_rewards.append(cumulative_reward)
    return np.mean(avg_rewards)

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    import os
    path = './datasets'
    if not os.path.exists(path):
        raise Exception('Download datasets first please!')
    dataloader, env = create_dataloader_env(batch_size=config.batch_size, seed=config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_last10 = deque(maxlen=10)

    with wandb.init(project='cs885a2', name=config.algo + ', alpha=' + str(config.cql_alpha) + ', env:' + config.env + ', seed=' + str(config.seed), config=config):
        if config.algo == 'cqldqn':
            agent = CQLDQN(state_size=env.observation_space.shape[0],
                           action_size=env.action_space.n,
                           hidden_size=config.hidden_size,
                           alpha=config.cql_alpha,
                           device=device,
                           cql_rescaled=config.cql_rescaled,
                           lr=config.lr,
                           tau=config.tau,)
        
        if config.algo == 'dqn':
            agent = DeepQN(state_size=env.observation_space.shape[0],
                           action_size=env.action_space.n,
                           hidden_size=config.hidden_size,
                           lr=config.lr,
                           tau=config.tau,
                           device=device)
    
        wandb.watch(agent, log='gradients', log_freq=10)
        returns = evaluate(env, agent)
        avg_last10.append(returns)
        wandb.log({'Test Returns': returns, 'Episode': 0})
        for i in range(1, config.episodes + 1):
            for states, actions, rewards, next_states, dones in dataloader:
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)   
                dones = dones.to(device)
                batch = (states, actions, rewards, next_states, dones)
                total_loss, cql_loss, bellman_error = agent.learn(batch)
                
            if i % config.eval_every == 0:
                returns = evaluate(env, agent)
                wandb.log({'Test Returns': returns, 'Episode': i})
                avg_last10.append(returns)
                print('Episode: {}, Normalized Returns: {}'.format(i, returns))
            
            wandb.log({
                'Last 10 Avg Normalized Returns': np.mean(avg_last10),
                'Total Loss': total_loss,
                'CQL Loss': cql_loss,
                'Bellman Error': bellman_error,
                'Episode': i
            })

            if i % config.save_every == 0:
                save(agent.q_net, filename='cqldqn_cartpole')

if __name__ == '__main__':
    config = set_config()
    train(config)