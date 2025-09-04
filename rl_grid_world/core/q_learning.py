"""
VERSION 1.0.0
"""

import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha = 0.1, gamma = 0.99):
        """
        alpha: learning rate
        gamma: discount factor
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((env.grid_size * env.grid_size, 4))

    def state_to_index(self, state):
        row, column = state
        return row * self.env.grid_size + column
    
    def choose_action(self, state, epsilon = 0.1):
        """
        epsilon-greedy tragedy
        """
        state_idx = self.state_to_index(state)
        if np.random.random() < epsilon:
            return np.random.choice(self.env.action_space)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state, done):
        """
        update Q-table
        """
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        current_q = self.q_table[state_idx, action]

        if(done):
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx, action] = current_q + self.alpha * (target - current_q)
    
    def train(self, episodes = 1000, initial_epsilon = 0.8, epsilon_decay = 0.998, minimum_epsilon = 0.01):
        """
        episodes: number of trainning epoch
        epsilon: exploration rate
        epsilon_decay: exploration rate decay speed
        """
        print("\n=== Trainning begin ===\n")
        epsilon = initial_epsilon
        episode_rewards = []
        episode_steps = []

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_rewards = 0
            steps = 0

            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done, info = self.env.move(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_rewards += reward
                steps += 1
            
            epsilon = max(epsilon * epsilon_decay, minimum_epsilon)

            episode_rewards.append(total_rewards)
            episode_steps.append(steps)

            if (episode + 1) % 200 == 0:
                avg_reward = np.mean(episode_rewards[-200:])
                print(f"episode: {episode + 1}")
                print(f"average reward: {avg_reward:.2f}")
                print(f"exploration rate: {epsilon:.3f}")
        
        print("\n=== Trainning is complete ===\n")
        return episode_rewards, episode_steps

if __name__ == "__main__":
    from environment import GridWorld

    env = GridWorld()
    agent = QLearningAgent(env)

    agent.train()

    print("Current tragedy:")
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # choose action greedly
        action = agent.choose_action(state, epsilon = 0)
        next_state, reward, done, info = env.move(action)

        print(f"from state{state} to next_state{next_state}")
        print(f"reward: {reward}\n")

        state = next_state
        total_reward += reward
    
    print(f"total reward: {total_reward}")