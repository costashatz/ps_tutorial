import gym
import numpy as np
import copy

from es import run_es

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib

from palettable.colorbrewer.qualitative import Set2_7
colors = Set2_7.mpl_colors

# First let's create the environment
env = gym.make("MountainCarContinuous-v0")
max_episode_steps = 999
D = 2
A = 1

# Create the objective function to optimize (1 episode)
def f(x, display=False):
    # We use a simple linear policy of the form A*x+b
    X = x.copy()
    X = X.reshape((1, D+A))
    M = X[0, :D]
    b = X[0, -A]

    state = env.reset()
    steps = 0
    total_reward = 0.
    while True:
        action = np.array([M @ state + b])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        if display:
            env.render()
        steps = steps + 1
        if done or steps >= max_episode_steps:
            break
        state = next_state.copy()

    return -total_reward

N = 100  # population
l = 5 # num of elites
iterations = 40 # iterations

# Run the optimization 5 indepedent times to gather statistics
all_vals = []
all_mus = []
N_runs = 5
for _ in range(N_runs):
    # initial guess
    mu = np.ones((D+A, 1))
    # initial exploration
    sigma = np.ones((D+A, 1))*5.

    means, _, vals, _ = run_es(f, mu, sigma, N, l, iterations, verbose=True, noisy=True)

    all_vals += [vals]
    all_mus += [means]

# Plot the results
def plot_one(all_vals, index, ax):
    med = -np.median(all_vals.T, axis=1)
    high = -np.percentile(all_vals.T, 75, axis=1)
    low = -np.percentile(all_vals.T, 25, axis=1)

    ii = np.arange(all_vals.shape[1]) + 1
    ax.fill_between(ii, low, high, color=colors[index], alpha=0.25, linewidth=0)
    ax.plot(ii, med, color=colors[index])

all_vals = np.array(all_vals)
all_mus = np.array(all_mus)

fig, ax = plt.subplots(figsize=[10.24, 7.68])
plot_one(all_vals, 0, ax)
# plt.ylim([-500, 50])
legend = ax.legend(["MountainCar"], loc=1)
frame = legend.get_frame()
frame.set_facecolor('1.0')
frame.set_edgecolor('1.0')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
for spine in ax.spines.values():
    spine.set_position(('outward', 5))
ax.set_axisbelow(True)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_ylabel("Total Reward")
ax.set_xlabel("Optimization Generations")
plt.show()
