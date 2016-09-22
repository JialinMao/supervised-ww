import mxnet as mx
import numpy as np
import gym

prefix = 'model/game-0'
num_round = 10
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

env = gym.make('Game-v3')
obs = env.reset()
env.render()
obs = obs.transpose((2, 0, 1))
obs = np.expand_dims(obs, axis=0)
total_r = 0

prob = model.predict(obs)[0]
act = np.argsort(prob)[::-1][0]
obs, reward, done, _ = env.step(act)
while not done:
    total_r += reward
    obs = obs.transpose((2, 0, 1))
    obs = np.expand_dims(obs, axis=0)
    prob = model.predict(obs)[0]
    act = np.argsort(prob)[::-1][0]      
    obs, reward, done, _ = env.step(act)
    env.render()
total_r += reward
print total_r
