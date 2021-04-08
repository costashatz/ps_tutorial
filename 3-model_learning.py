import GPy
import gym
import numpy as np
import math
import copy
import time
import mountain_car

# Create the environment
env = gym.make("MountainCarContinuous-v0")
max_episode_steps = 999
max_steps = 1500

# Collect some data for learning the model
state = env.reset()

data = []
steps = 0
total_steps = 0
while True:
    action = env.action_space.sample() # sample random actions
    next_state, reward, done, _ = env.step(action)
    if (total_steps % 10) == 0:  # do not collect all data to avoid too slow learning
        data += [(state.copy(), action.copy(), next_state.copy())]
    state = next_state.copy()
    # env.render()
    steps = steps + 1
    total_steps = total_steps + 1
    if total_steps >= max_steps:
        break
    if done or steps >= max_episode_steps:
        state = env.reset()
        steps = 0

print(len(data), "data points")
N = len(data)
D = state.shape[0]
A = action.shape[0]
# Let's create the dataset
X = np.zeros((N, D+A))
Y = np.zeros((N, D))

for i in range(N):
    state, action, next_state = data[i]
    X[i, :D] = state.copy()
    X[i, D:] = action.copy()
    Y[i, :] = (next_state - state)

# Learn the model
print("Learning GP models...")
# we need one GP per output dimension
start_time = time.time()
models = []
for i in range(D):
    # we use a simple RBF kernel
    kernel = GPy.kern.RBF(input_dim=D+A, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X, Y[:, i].reshape((N, 1)), kernel)
    # m.optimize(messages=True) # we could optimize for the hyper-parameters also (highly recommended, takes some time!)
    models += [m]

total_time = time.time() - start_time
print("Models learnt in " + str(round(total_time, 2)) + "s")

# Check the performance of the learned models
preds = np.zeros((N, D))
for i in range(D):
    mu, _ = models[i].predict(X)
    preds[:, i] = mu.reshape((N,))

# MSE
print("MSE:", np.square(preds-Y).sum()/N)

# Plot predictions of models
mountain_car.plot(models)

# Plot predictions of "real" system
mountain_car.plot(None)
