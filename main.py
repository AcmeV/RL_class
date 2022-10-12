import argparse
import os
import torch

from envs import *
from models import QLearningTable, SarsaTable, SarsaLambdaTable
from models import DQN, DoubleDQN, PrioritizedReplayDQN, DuelingDQN
from train import qlearning_training, sarsa_training, dqn_training

parser = argparse.ArgumentParser()
# System settings
parser.add_argument('--model-save-dir', type=str, default='./model_files/')
parser.add_argument('--log-dir', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cpu',
                    choices=('cpu', 'gpu', 'gpus'))
parser.add_argument('--gpus', type=str, default='0,1,2,3')

# env
parser.add_argument('--env', type=str, default='Snake',
                    choices=('Maze', 'Snake'))

parser.add_argument('--is-render', type=int, default=0, choices=(0, 1))

# Hyper-parameters
parser.add_argument('--model', type=str, default='PrioritizedReplayDQN',
                    choices=('QLearning', 'Sarsa', 'SarsaLambda',
                             'DQN', 'DoubleDQN', 'PrioritizedReplayDQN', 'DuelingDQN'))
parser.add_argument('--episodes', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--bsz', type=int, default=64)
parser.add_argument('--pre-training', type=int, default=1, choices=(0, 1))
parser.add_argument('--if-save', type=int, default=1, choices=(0, 1))

# dqns config
parser.add_argument('--memory', type=int, default=200000)

args = parser.parse_args()

def init_env():
    if args.env == 'Maze':
        env = Maze()
    elif args.env == 'Snake':
        env = SnakeGame()
    else:
        env = None
        print("Environment doesn't exitst")
        exit(1)
    if args.is_render == 1:
        env.is_render = True
    return env

if __name__ == '__main__':

    print(f'\nModel: {args.model} | Env: {args.env} | LR: {args.lr} | device: {args.device}\n')

    env = init_env()

    save_path = f'{args.model_save_dir}/{args.model}-{args.env}' if args.if_save == 1 else None
    load_path = f'{args.model_save_dir}/{args.model}-{args.env}'

    log_path = f'{args.log_dir}/{args.model}-{args.env}-lr_{args.lr}-memory-{args.memory}.csv'

    device = torch.device(f'cuda:0' if args.device != 'cpu' and torch.cuda.is_available() else "cpu")

    if args.model == 'QLearning':
        q_table = QLearningTable(
            actions=list(range(env.n_actions)),
            learning_rate=args.lr)
        # load pre-training model parameters
        start_episode = 0
        if os.path.exists(load_path) and args.pre_training == 1:
            start_episode = q_table.load(load_path)

        qlearning_training(env, q_table, start_episode, args.episodes, save_path, log_path)

    elif 'Sarsa' in args.model:

        if args.model == 'Sarsa':
            sarsa_table = SarsaTable(
                actions=list(range(env.n_actions)),
                learning_rate=args.lr)
        else:
            sarsa_table = SarsaLambdaTable(
                actions=list(range(env.n_actions)),
                learning_rate=args.lr)

        # load pre-training model parameters
        start_episode = 0
        if os.path.exists(load_path) and args.pre_training == 1:
            start_episode = sarsa_table.load(load_path)

        sarsa_training(env, sarsa_table, start_episode, args.episodes, save_path, log_path)

    elif 'DQN' in args.model:

        if args.model == 'DQN':
            model = DQN(env.n_actions, env.n_features, batch_size=args.bsz,
                        learning_rate=args.lr, reward_decay=0.9,
                        e_greedy=0.9, e_greedy_increment=0.00005,
                        replace_target_iter=200, memory_size=args.memory, device=device)
        elif args.model == 'DoubleDQN':

            model = DoubleDQN(env.n_actions, env.n_features, batch_size=args.bsz,
                              learning_rate=args.lr, reward_decay=0.9,
                              e_greedy=0.9, e_greedy_increment=0.00005,
                              replace_target_iter=200, memory_size=args.memory, device=device)
        elif args.model == 'DuelingDQN':

            model = DuelingDQN(env.n_actions, env.n_features, batch_size=args.bsz,
                               learning_rate=args.lr, reward_decay=0.9,
                               e_greedy=0.9, e_greedy_increment=0.00005,
                               replace_target_iter=200, memory_size=args.memory, device=device)
        else:
            model = PrioritizedReplayDQN(env.n_actions, env.n_features, batch_size=args.bsz,
                                         learning_rate=args.lr, reward_decay=0.9,
                                         e_greedy=0.9, e_greedy_increment=0.00005,
                                         replace_target_iter=200, memory_size=args.memory, device=device)
        # load pre-training model parameters
        start_episode = 0
        if os.path.exists(load_path) and args.pre_training == 1:
            start_episode = model.load(load_path)
        dqn_training(env, model, start_episode, args.episodes, save_path, log_path)

    print('training end')