import gym
import numpy as np

ENV = 'Game-v3'
EPISODES = 5000 
RENDER = False 

q_hist= []
obs_hist = []
count = 0

for i in range(EPISODES):
    env = gym.make(ENV)
    obs = env.reset()
    done = False
    reward = 0    
    if RENDER:
        env.render()
    while not done:
        obs_hist.append(obs.transpose((2, 0, 1)))
        obs=obs.squeeze()
        q = 10*np.sum(obs[::2][:, 2:]==obs[1::2][:, 2:], axis=1, dtype='float32') # whether the class is the same as the target class
        q -= 0.2*np.sum(abs(obs[::2][:, :2]-obs[1::2][:, :2]), axis=1)
        q[q<0] = 0
        q = np.hstack([q/np.sum(q),[0]]) if np.sum(q)!=0 else np.hstack([np.zeros_like(q), [1]])
        a = np.argmax(q)
        obs, r, done, _ = env.step(a)
        reward += r
        q_hist.append(np.argmax(q))
        if RENDER:
            env.render()
    if reward <= 100.0:
        count +=1

print count
np.savez(open('experience.npz', 'wb'), obs=obs_hist, act=q_hist)
