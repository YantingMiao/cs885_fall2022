#!/bin/bash
#SBATCH --job-name=dqn_target_freq

#SBATCH --partition=p100,t4v1,t4v2,rtx6000

#SBATCH --gres=gpu:1

#SBATCH --qos=normal

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# put your command here
python3 DQN-soln-targetnetworkupdate.py
python3 DQN-soln-DQN-soln-minibatchsize.p
python3 DQN-soln-trainepoch.py 
