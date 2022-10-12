import csv
import numpy as np

def dqn_training(env, model, start_episode, episodes, save_path, log_path):

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Reward', 'Step'])

    total_steps = 1
    memory_size = model.memory_size

    observation = env.reset()

    for episode in range(start_episode, start_episode + episodes):

        if env.has_terminal_tag:
            observation = env.reset()

        terminate = False
        step_counter = 0

        episode_losses = []
        episode_reward = 0

        while not terminate:

            env.render()

            action = model.choose_action(observation)

            observation_, reward, terminate, info = env.step(action)

            model.store_transition(observation, action, reward, observation_)

            # training when steps greater than memory size
            if total_steps > memory_size:
                episode_losses.append(model.learn())
            episode_reward += reward

            # for no terminal tag environment, denote an episode as memory_size
            if not env.has_terminal_tag and total_steps > memory_size and total_steps % memory_size == 0:
                terminate = True

            observation = observation_
            total_steps += 1
            step_counter += 1

            if terminate and total_steps <= memory_size:
                step_counter = 1
                episode_losses = []
                episode_reward = 0
                terminate = False
                if env.has_terminal_tag:
                    observation = env.reset()

        episode_loss = np.mean(episode_losses)
        episode_reward += 1

        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        file.flush()
        print(f'Episode: {episode} | Loss: {episode_loss: .4f}| Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        model.save(start_episode + episodes, save_path)

    file.close()
    env.close()