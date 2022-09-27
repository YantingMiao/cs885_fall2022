#!/bin/bash
#SBATCH --job-name=cql_expert_walker2d

#SBATCH --partition=p100,t4v1,t4v2,rtx6000

#SBATCH --gres=gpu:1

#SBATCH --qos=normal

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# put your command here
python3 run.py --algo=dqn --seed=1
python3 run.py --algo=dqn --seed=2
python3 run.py --algo=dqn --seed=3
python3 run.py --algo=dqn --seed=4
python3 run.py --algo=dqn --seed=5