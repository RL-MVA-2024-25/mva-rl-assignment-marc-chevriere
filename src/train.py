from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import time

from gymnasium.wrappers import TimeLimit

import xgboost as xgb
import joblib



class ProjectAgent:
    """
    A simple Fitted Q-Iteration (FQI) agent using XGBoost for Q-value approximation.
    """

    def __init__(self):

        self.gamma = 0.99          
        self.n_actions = 4           
        self.model = None
        self.replay_buffer = []
        self.state_dim = 6
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.n_iterations = 5

        self.all_actions_one_hot = np.eye(self.n_actions)
        self.X = np.array([])
        self.rewards_all = np.array([])
        self.dones_all = np.array([])
        self.states_next_all = np.array([])


    def act(self, observation, use_random=False):
        if (self.model is None) or (use_random and (random.random() < self.epsilon)):
            return np.random.randint(0, self.n_actions)
        
        else:
            duplicated_observation = np.tile(observation, (self.n_actions, 1))
            x_inputs = np.hstack((duplicated_observation, self.all_actions_one_hot))
            q_values = self.model.predict(x_inputs)
            return int(np.argmax(q_values))


    def save(self, path):
        if self.model is not None:
            joblib.dump(self.model, path)
        else:
            print("No model found to save.")


    def load(self):
        try:
            self.model = joblib.load("fqi_model_best.xgb")
        except:
            print("No saved model found at 'fqi_model.xgb'. Initialize a fresh model.")
            self.model = None


    def add_transition(self, s, a, r, s_next, done):
        self.replay_buffer.append((s, a, r, s_next, done))


    def do_fqi_training(self):
        if len(self.replay_buffer) == 0:
            return  

        states, actions, rewards, states_next, dones = zip(*self.replay_buffer)
        self.replay_buffer = []

        states = np.array(states)
        actions = np.array(actions)
        actions_one_hot = np.array([self.all_actions_one_hot[a] for a in actions])
        rewards = np.array(rewards)
        states_next = np.array(states_next)
        dones = np.array(dones)

        if self.X.size == 0:
            self.X = np.hstack((states, actions_one_hot))
            self.rewards_all = rewards
            self.dones_all = dones
            self.states_next_all = states_next
        else:
            self.X = np.vstack([self.X, np.hstack((states, actions_one_hot))])
            self.rewards_all = np.hstack([self.rewards_all, rewards])
            self.dones_all = np.hstack([self.dones_all, dones])
            self.states_next_all = np.vstack([self.states_next_all, states_next])

        if self.model is None:
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=0
            )
            self.model.fit(self.X, rewards, verbose=False)

        for _ in range(self.n_iterations):
            repeated_states = np.repeat(self.states_next_all, self.n_actions, axis=0)
            repeated_actions = np.tile(self.all_actions_one_hot, (self.states_next_all.shape[0], 1))
            SA = np.hstack((repeated_states, repeated_actions))
            q_values = self.model.predict(SA)
            q_values_reshaped = q_values.reshape(-1, 4)  
            best_q_values = np.max(q_values_reshaped, axis=1)
            targets = self.rewards_all + self.gamma * best_q_values * (1 - self.dones_all)
            self.model.fit(self.X, targets, verbose=False)


def train_fqi_agent(num_episodes, max_steps):

    agent = ProjectAgent()
    all_rewards = []

    for ep in range(num_episodes):
        s, _ = env.reset()
        ep_reward = 0

        for _ in range(max_steps):
            a = agent.act(s, use_random=True)
            s_next, r, done, truncated, _ = env.step(a)
            ep_reward += r
            agent.add_transition(s, a, r, s_next, done)
            s = s_next
            if done or truncated:
                break

        all_rewards.append(ep_reward)
        agent.do_fqi_training()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward} | Epsilon: {agent.epsilon}")

    agent.save("fqi_model.xgb")

    # Plot training curve
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('FQI Training Performance')
    plt.show()
    print("Training completed:", np.mean(all_rewards[-5:]))


if __name__ == "__main__":
    train_fqi_agent(num_episodes=600, max_steps=200)


# class ProjectAgent:
#     def act(self, observation, use_random=False):
#         return 0

#     def save(self, path):
#         pass

#     def load(self):
#         pass
