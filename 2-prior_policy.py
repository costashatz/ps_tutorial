import numpy as np
import math
import copy
import time
import pickle

from es import run_es

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib

from palettable.colorbrewer.qualitative import Set2_7
colors = Set2_7.mpl_colors

# A simple 2D Particle environment
class ParticleEnv:
    def __init__(self):
        self.dt = 0.01
        self.m = 1.
        self.b = 0.1
        self.max_force = 2.
        self.max_vel = 5.
        self.target = np.array([2., 2.])

        self.reset()

    # Reset function
    def reset(self, random_initial=False):
        if random_initial:
            high = np.array([2., 2., 0.5, 0.])
            self.state = np.random.uniform(low=-high, high=high)
        else:
            self.state = np.array([0., 0., 0., 0.])  # pos, vel
        return self.state

    # Step function
    def step(self, u):
        u = np.clip(u, -self.max_force, self.max_force)

        p = self.state[:2]
        v = self.state[2:]

        acc = u/self.m
        n_skip = 5
        for _ in range(n_skip):
            v = v + acc * self.dt
            v = np.clip(v, -self.max_vel, self.max_vel)
            p = p + v * self.dt

        reward = -np.linalg.norm(p-self.target)

        self.state = np.array([p[0], p[1], v[0], v[1]])

        return self.state, reward

# Create Objective function to optimize (1 episode with given policy)
def func(policy, display=False, random_initial=False, initial_state=None):
    env = ParticleEnv()
    max_episode_steps = 200
    if initial_state is None:
        state = env.reset(random_initial)
    else:
        state = initial_state
        env.state = np.copy(initial_state)
    steps = 0
    total_reward = 0.
    while True:
        action = policy(state)
        next_state, reward = env.step(action)
        total_reward += reward
        if display:
            print(steps, next_state)
        steps = steps + 1
        if steps >= max_episode_steps:
            break
        state = next_state.copy()

    return -total_reward

# A simple linear policy of the form: A*x+b
def unstructured_policy(x, display=False, random_initial=False, initial_state=None):
    X = x.copy()
    X = X.reshape((1, 4+2))
    M = X[0, :4]
    b = X[0, 4:]

    def pol(state):
        return np.array([M @ state + b]).reshape((2,))

    return func(pol, display, random_initial, initial_state)

# A stable linear policy of the form: x_dot = A(target-x), F = K*x_dot
def structured_policy(x, display=False, random_initial=False, initial_state=None):
    X = x.copy()
    X = X.reshape((1, 4+2))
    M = np.diag(np.exp(X[0, :4]))
    t = X[0, 4:]
    t = np.array([t[0], t[1], 0., 0.])  # we assume zero target velocity

    def pol(state):
        vel_commands = (M @ (t-state)).reshape((4, 1))
        Kp = 10.
        Kd = 10.
        u = Kp*vel_commands[:2] + Kd*vel_commands[2:]
        return u.reshape((2,))

    return func(pol, display, random_initial, initial_state)

# Run the optimizations 5 indepedent times to gather statistics
all_vals_structured = []
all_vals_unstructured = []
final_mus_structured = []
final_mus_unstructured = []
N_runs = 5

for _ in range(N_runs):
    mu = np.zeros((6, 1)) # initial estimate
    sigma = np.ones((6, 1))
    # run optimization and get result
    final_mu_structured, _, values_structured, _ = run_es(structured_policy, mu, sigma, 20, 5, 30, verbose=True)
    all_vals_structured += [values_structured]
    final_mus_structured += [final_mu_structured]

    mu = np.ones((6, 1)) # initial estimate
    sigma = np.ones((6, 1))
    # run optimization and get result
    final_mu_unstructured, _, values_unstructured, _ = run_es(unstructured_policy, mu, sigma, 20, 5, 30, verbose=True)
    all_vals_unstructured += [values_unstructured]
    final_mus_unstructured += [final_mu_unstructured]

all_vals_structured = np.array(all_vals_structured)
all_vals_unstructured = np.array(all_vals_unstructured)
final_mus_structured = np.array(final_mus_structured)
final_mus_unstructured = np.array(final_mus_unstructured)

# We could even save them for handling the data later
a_file = open("alldata.bin", "wb")
pickle.dump(all_vals_structured, a_file)
pickle.dump(all_vals_unstructured, a_file)
pickle.dump(final_mus_structured, a_file)
pickle.dump(final_mus_unstructured, a_file)
a_file.close()

# Load the data
# a_file = open("alldata.bin", "rb")
# all_vals_structured = pickle.load(a_file)
# all_vals_unstructured = pickle.load(a_file)
# final_mus_structured = pickle.load(a_file)
# final_mus_unstructured = pickle.load(a_file)
# a_file.close()

# First plot learning curves
def plot_one(all_vals, index, ax):
    med = -np.median(all_vals.T, axis=1)
    high = -np.percentile(all_vals.T, 75, axis=1)
    low = -np.percentile(all_vals.T, 25, axis=1)

    ii = np.arange(all_vals.shape[1]) + 1
    ax.fill_between(ii, low, high, color=colors[index], alpha=0.25, linewidth=0)
    ax.plot(ii, med, color=colors[index])


fig, ax = plt.subplots(figsize=[10.24, 7.68])
plot_one(all_vals_structured, 0, ax)
plot_one(all_vals_unstructured, 1, ax)
plt.ylim([-500, 50])
legend = ax.legend(["DS-Policy", "Agnostic-Policy"], loc=1)
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

# Plot something like a value function for one policy
def plot_policy(policy, mu, N_test=10):
    max_position = 3.
    min_position = -3.

    img = np.zeros((N_test, N_test))
    for i in range(N_test):
        x = i/float(N_test-1)*(max_position-min_position) + min_position
        for j in range(N_test):
            y = j/float(N_test-1)*(max_position-min_position) + min_position

            init_state = np.array([x, y, 0., 0.])

            img[j, i] = -policy(mu, initial_state=init_state)

    fig, ax = plt.subplots(figsize=[10.24, 7.68])
    im = ax.imshow(img, interpolation='bilinear', vmin=-1000., vmax=30.)
    fig.colorbar(im, ax=ax)
    plt.xticks(ticks=[i for i in range(0, N_test, 3)], labels=[np.round(float(i)/(N_test-1)*(max_position-min_position) + min_position, 2) for i in range(0, N_test, 3)])
    plt.yticks(ticks=[i for i in range(0, N_test, 3)], labels=[np.round(float(j)/(N_test-1)*(max_position-min_position) + min_position, 2) for j in range(0, N_test, 3)])

    i_index = (2-min_position)*float(N_test-1)/(max_position-min_position)
    plt.plot(i_index, i_index, 'r.')
    plt.show()


# Then plot "value-functions" for each policy
plot_policy(structured_policy, final_mus_structured[-1][-1])
plot_policy(unstructured_policy, final_mus_unstructured[-1][-1])
