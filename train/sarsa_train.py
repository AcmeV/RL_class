import csv
import numpy as np

def sarsa_training(env, sarsa_table, start_episode, episodes, save_path, log_path):
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Reward', 'Step'])

    observation = env.reset()
    total_steps = 1

    for episode in range(start_episode, start_episode + episodes):

        if env.has_terminal_tag:
            observation = env.reset()

        terminate = False

        step_counter = 0
        episode_losses = []
        episode_rewards = []

        action = sarsa_table.choose_action(observation)

        while not terminate:

            env.render()

            observation_, reward, terminate, info = env.step(action)

            action_ = sarsa_table.choose_action(observation_)

            episode_losses.append(sarsa_table.learn(observation, action, reward, observation_, action_))

            episode_rewards.append(reward)

            step_counter += 1

            observation = observation_
            step_counter += 1
            total_steps += 1
            action = action_

            if not env.has_terminal_tag and total_steps % 2000 == 0:
                terminate = True

        episode_loss = np.mean(episode_losses)
        episode_reward = np.mean(episode_rewards)

        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        file.flush()
        print(f'Episode: {episode} | Loss: {episode_loss: .4f} | Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        sarsa_table.save(start_episode + episodes, save_path)

    file.close()
    env.close()