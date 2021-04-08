import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import copy

# Helper functions for mountain car environment

min_position = -1.2
max_position = 0.6
min_action = -1.0
max_action = 1.0
max_speed = 0.07

def prior_model(x):
    position = copy.copy(x[0])
    velocity = copy.copy(x[1])
    force = min(max(copy.copy(x[-1]), min_action), max_action)

    velocity += force * 0.0015 - 0.0025 * math.cos(3 * position)
    if (velocity > max_speed):
        velocity = max_speed
    if (velocity < -max_speed):
        velocity = -max_speed
    position += velocity
    if (position > max_position):
        position = max_position
    if (position < min_position):
        position = min_position
    if (position == min_position and velocity < 0):
        velocity = 0
    next_state = np.array([position, velocity])

    return next_state

def plot(models = None, N_test = 50, plot_type = 'full'):
    D = 2
    A = 1
    # Let's do some plots
    X_test = np.zeros((N_test*N_test, D+A))

    for i in range(N_test):
        pos = float(i)/(N_test-1)*(max_position-min_position) + min_position
        for j in range(N_test):
            act = float(j)/(N_test-1)*(max_action-min_action) + min_action
            index = i*N_test + j
            X_test[index, 0] = pos
            # X_test[index, 1] = 0.07
            X_test[index, 2] = act

    Y_test = np.zeros((N_test*N_test, D))
    if models is None:
        for i in range(N_test):
            for j in range(N_test):
                index = i*N_test + j
                Y_test[index, :] = prior_model(X_test[index, :D]) - X_test[index, :D]
    else:
        for d in range(D):
            mu, _ = models[d].predict(X_test)
            Y_test[:, d] = mu.reshape((N_test*N_test,))

    img = np.zeros((N_test, N_test))
    for i in range(N_test):
        for j in range(N_test):
            index = i*N_test + j

            if plot_type == 'full':
                img[j, i] = X_test[index, 0] + Y_test[index, 0]
            else:
                img[j, i] = Y_test[index, 0]

    # Plot
    fig, ax = plt.subplots(figsize=[10.24, 7.68])
    if plot_type == 'full':
        im = ax.imshow(img, interpolation='bilinear', vmin=min_position, vmax=max_position)
    else:
        im = ax.imshow(img, interpolation='bilinear')
    ax.figure.colorbar(im, ax=ax)
    plt.xticks(ticks=[i for i in range(0, N_test, 3)], labels=[np.round(float(i)/(N_test-1)*(max_position-min_position) + min_position, 2) for i in range(0, N_test, 3)])
    plt.yticks(ticks=[i for i in range(0, N_test, 3)], labels=[np.round(float(j)/(N_test-1)*(max_action-min_action) + min_action, 2) for j in range(0, N_test, 3)])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_rotation(50)
    plt.xlabel('position, velocity=0')
    plt.ylabel('action')
    plt.show()
