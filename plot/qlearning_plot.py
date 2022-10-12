import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


font = {
    'size': 13
}

def load_log(path):
    log = pd.read_csv(path)
    episodes, steps, losses, rewards = [], [], [], []

    # for i in range(len(log['Epoch'])):
    for i in range(len(log['Episode'])):
    # for i in range(300):
        episodes.append(log['Episode'][i])
        steps.append(float(log['Step'][i]))
        losses.append(float(log['Loss'][i]))
        rewards.append(float(log['Reward'][i]))

    return episodes, rewards, steps, losses

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_ql_sarsa():
    step = 10
    paths = [
        './ql_sarsa/QLearning-Maze-lr_0.1.csv',
        './ql_sarsa/Sarsa-Maze-lr_0.1.csv',
        './ql_sarsa/SarsaLambda-Maze-lr_0.1.csv',
    ]
    colors = [
        '#96C37D', # 绿色
        '#2F7FC1', # 蓝色
        '#D8383A', # 红色
    ]

    labels = [
        'QLearning',
        'Sarsa',
        'SarsaLambda'
    ]

    plt.xlabel('Episode', font)
    plt.ylabel('Reward', font)

    for path, label, color in zip(paths, labels, colors):
        episode, reward, _, _ = load_log(path)
        e_reward = reward[:step - 1]
        e_reward.extend(moving_average(reward, step))
        plt.plot(episode, e_reward, color=color, label=label)
        plt.fill_between(x=episode, y1=reward, y2=e_reward, facecolor=color,
                         alpha=0.7)
    plt.legend(loc='lower right', fontsize=12, ncol=2)
    plt.show()

def plot_gamma():
    step = 10
    paths = [
        './gamma/0.csv',
        './gamma/0.5.csv',
        './gamma/0.7.csv',
        './gamma/0.9.csv',
    ]
    colors = [
        '#96C37D', # 绿色
        'orange',
        '#2F7FC1', # 蓝色
        '#D8383A', # 红色
    ]

    labels = [
        'γ = 0',
        'γ = 0.5',
        'γ = 0.7',
        'γ = 0.9',
    ]

    plt.xlabel('Episode', font)
    plt.ylabel('Reward', font)

    for path, label, color in zip(paths, labels, colors):
        episode, reward, _, _ = load_log(path)
        e_reward = reward[:step - 1]
        e_reward.extend(moving_average(reward, step))
        plt.plot(episode, e_reward, color=color, label=label)
        plt.fill_between(x=episode, y1=reward, y2=e_reward, facecolor=color,
                         alpha=0.7)
    plt.legend(loc='lower right', fontsize=12, ncol=2)
    plt.show()




def plot_diff(paths, labels, colors, step=100):
    plt.figure(1, figsize=(6, 5))
    plt.xlabel('Episode', font)
    plt.ylabel('Reward', font)

    for path, label, color in zip(paths, labels, colors):
        episode, reward, _, _ = load_log(path)
        e_reward = reward[:step - 1]
        e_reward.extend(moving_average(reward, step))
        plt.plot(episode, e_reward, color=color, label=label)
        plt.fill_between(x=episode, y1=reward, y2=e_reward, facecolor=color,
                         alpha=0.7)
    plt.legend(loc='upper right', fontsize=12, ncol=2)
    plt.show()

def plot_dqns():
    paths = [
        './dqns/QLearning.csv',
        './dqns/DQN.csv',
        './dqns/DoubleDQN.csv',
        './dqns/PrioritizedReplayDQN.csv',
        './dqns/DuelingDQN.csv',
    ]
    colors = [
        '#96C37D', # 绿色
        'orange',
        '#2F7FC1', # 蓝色
        '#8983bf', # 紫色
        '#D8383A', # 红色
    ]

    labels = [
        'QLearning',
        'DQN',
        'DoubleDQN',
        'PrioritizedReplayDQN',
        'DuelingDQN'
    ]

    plot_diff(paths, labels, colors)

def plot_lrs():
    step = 100
    paths = [
        './lrs/0.01.csv',
        './lrs/0.001.csv',
        './lrs/0.0001.csv',
    ]
    colors = [
        '#96C37D', # 绿色
        '#2F7FC1', # 蓝色
        '#D8383A', # 红色
    ]

    labels = [
        'LR = 10^-2',
        'LR = 10^-3',
        'LR = 10^-4',
    ]

    plot_diff(paths, labels, colors)

def plot_memorys(model='DQN'):
    paths = [
        f'./memorys/{model}-2000.csv',
        f'./memorys/{model}-20000.csv',
        f'./memorys/{model}-200000.csv',
    ]

    colors = [
        '#96C37D', # 绿色
        '#2F7FC1', # 蓝色
        '#D8383A', # 红色
    ]

    labels = [
        'Memory = 2*10^3',
        'Memory = 2*10^4',
        'Memory = 2*10^5'
    ]

    plot_diff(paths, labels, colors)

if __name__ == '__main__':
    # plot_ql_sarsa()
    # plot_gamma()
    #
    # plot_dqns()
    # plot_lrs()

    # plot_memorys('PrioritizedDQN')
    plot_memorys()
