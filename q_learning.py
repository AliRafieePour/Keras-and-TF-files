import gym
import numpy as np
from matplotlib import pyplot as plt
env = gym.make("MountainCar-v0")


done = False

learning_rate = 0.1
discount = 0.95
episodes = 75000
show_every = 7500

epsilon = 0.5
start = 1
end = episodes // 2
decay = epsilon / (end - start)

discrete_os_size = [20] * len(env.observation_space.low)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

def get_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


render = False
rewards = []
Q = []
for i in range(0, episodes):
    reward_= 0
    discrete_state = get_state(env.reset())
    
    if i % show_every == 0:
        render = True
    else:
        render = False
        
    
    done = False
    while not done:
        if np.random.random() >= epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        reward_ += reward
        new_discrete_state = get_state(new_state)
        if render:
            env.render()
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print("episode number {}".format(i))
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    if i % 250 == 0:
        rewards.append(reward_)
        Q.append(q_table)
    if end>= i >= start:
        epsilon -= decay
env.close()



