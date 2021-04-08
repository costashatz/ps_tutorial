import numpy as np
import math
import copy

# Implementation of a simple Evolutionary Strategy

def generate_population(mu, sigma, N):
    pop = np.random.normal(size=(mu.shape[0], N))

    pop = np.copy(mu) + np.copy(sigma) * pop

    return pop


def evaluate_pop(pop, f):
    F = []
    for i in range(pop.shape[1]):
        F.append(f(pop[:, i]))

    return np.array([F])


def get_elites(pop, F, l):
    idx = np.argsort(F)
    idx_elites = idx[:, :l][0]

    return np.copy(pop[:, idx_elites]), idx_elites


def run_es(f, mu_init, sigma_init, N, l, iterations, alpha = 1., verbose=False, noisy=False):
    mu = mu_init
    sigma = sigma_init

    if verbose:
        print(0, 'mu:', mu.T)
        print('sigma:', sigma.T)
    
    means = [mu]
    sigmas = [sigma]
    vals = []

    total_evals = 0
    for it in range(iterations):
        pop = generate_population(mu, sigma, N)

        F = evaluate_pop(np.copy(pop), f)

        if noisy:
            vals += [np.mean(F)]
        else:
            vals += [f(mu)]

        best_pop, idx_elites = get_elites(np.copy(pop), np.copy(F), l)
        if verbose:
            print('elites:', np.mean(F[:, idx_elites]))
            print('mean:', vals[-1])
        mu_prev = np.copy(mu)

        mu = 1./float(l)*np.sum(best_pop, axis=1).reshape((mu.shape[0], 1))
        sigma = (1 - alpha) * sigma + alpha * np.sqrt(1./float(l)*np.sum(np.square(best_pop-mu_prev), axis=1).reshape((sigma.shape[0], 1)))

        means += [mu]
        sigmas += [sigma]

        total_evals += N
        if verbose:
            print(it+1, 'mu:', mu.T)
            print('sigma:', sigma.T)

    vals += [f(mu)]
    if verbose:
        print(total_evals, 'evaluations...')
    
    return means, sigmas, vals, total_evals
