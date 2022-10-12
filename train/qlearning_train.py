import csv
import numpy as np

def qlearning_training(env, q_table, start_episode, episodes, save_path, log_path):

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Reward', 'Step'])

    observation = env.reset()
    total_steps = 1

    # main loop
    for episode in range(start_episode,  start_episode + episodes):

        if env.has_terminal_tag:
            observation = env.reset()

        terminate = False

        step_counter = 0

        episode_losses = []
        episode_rewards = []

        while not terminate:

            env.render()

            action = q_table.choose_action(observation)

            observation_, reward, terminate, info = env.step(action)

            loss = q_table.learn(observation, action, reward, observation_)
            # record loss and reward
            episode_losses.append(loss)
            episode_rewards.append(reward)

            observation = observation_
            step_counter += 1
            total_steps += 1

            if not env.has_terminal_tag and total_steps % 2000 == 0:
                terminate = True

        episode_loss = np.mean(episode_losses)
        episode_reward = np.sum(episode_rewards) + 1

        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        file.flush()

        print(f'Episode: {episode} | Loss: {episode_loss: .4f} | Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        q_table.save(start_episode + episodes, save_path)

    file.close()
    env.close()