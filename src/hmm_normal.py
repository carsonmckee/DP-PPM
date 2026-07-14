import math
import numpy as np 
from numba import njit, prange
from typing import List
from numba import njit, types, typed
import matplotlib.pyplot as plt
from numba.typed import Dict

# XXX new code below

@njit(fastmath=True)
def likelihood(y, mean, sigma2):
    return np.exp(-0.5 * (y-mean)*(y-mean) / sigma2 )/ np.sqrt(2*np.pi*sigma2)

@njit(fastmath=True)
def sample_categorical(probs):
    u = np.random.rand()
    cum = 0.0
    for k in range(len(probs)):
        cum += probs[k]
        if u < cum:
            return k
    return len(probs) - 1

@njit(fastmath=True)
def sample_states(y, pi, means, sigma2):

    T = len(y)
    K = len(means)

    alpha = 1.0/np.ones(K, dtype=np.float64)

    filtered = np.zeros((T, K))

    for t in range(T):

        # emission
        emission = likelihood(y[t], means, sigma2)
        # update
        alpha *= emission
        if np.sum(alpha) == 0:
            print('zero warning')
        alpha /= np.sum(alpha)

        filtered[t] = alpha

        # propagate
        alpha = alpha @ pi

    # --- Backward sampling ---
    z = np.zeros(T, dtype=np.int64)

    # sample final state
    z[T-1] = sample_categorical(filtered[T-1])

    # backward recursion
    for t in range(T-2, -1, -1):

        probs = filtered[t] * pi[:, z[t+1]]
        if np.sum(probs) == 0:
            print('zero warning')
        probs /= np.sum(probs)

        z[t] = sample_categorical(probs)

    return z

@njit(fastmath=True)
def get_transition_counts(states, K):

    counts = np.zeros((K, K), dtype=np.float64)

    for i in range(1, len(states)):
        counts[states[i-1], states[i]] += 1
    
    return counts

@njit(fastmath=True)
def sample_posterior_mean(xs, sigma2, mu0, tau0_2):
    xs = np.asarray(xs)
    n = len(xs)
    if n == 0:
        xbar = 0
    else:
        xbar = xs.mean()

    posterior_var = 1.0 / (1.0 / tau0_2 + n / sigma2)
    posterior_mean = posterior_var * (mu0 / tau0_2 + n * xbar / sigma2)

    return np.random.normal(posterior_mean, np.sqrt(posterior_var))

@njit(fastmath=True)
def sample(y: np.array, 
           n_samples: int, 
           burn_in: int,
           K: int,
           params: np.array, 
           a0: float=1.0, 
           b0: float=1.0, 
           c0: float=100.0, 
           d0: float=1.0,
           y_pred: float=0.0):
    
    T_ = y.size
    
    alpha, gamma = 1.0, 1.0

    pi = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                pi[i, j] = np.random.gamma(alpha, 1.0)
            else:
                pi[i, j] = np.random.gamma(alpha + gamma, 1.0)
            pi[i, :] /= np.sum(pi[i, :])
    
    means = np.random.normal(0, 1, K)

    states = np.ones(T_, dtype=np.int64)
    state_means = np.zeros(T_, dtype=np.float64)

    y_pred_mean = 0.0
    for it in range(n_samples):
        
        states = sample_states(y, pi, means, params[2]) 

        # update etas
        transition_counts = get_transition_counts(states, K)
        for k1 in range(K):
            for k2 in range(K):
                a_post = alpha + transition_counts[k1, k2] + (gamma if k1 == k2 else 0.0)
                pi[k1, k2] = np.random.gamma(a_post, 1.0)
            pi[k1, :] = pi[k1, :] / np.sum(pi[k1, :])

        # update means
        for k in range(K):
            means[k] = sample_posterior_mean(y[np.where(states == k)], params[2], params[0], params[1])

        n_states = len(set(states))
        alpha, gamma = update_alpha_gamma(K, states, alpha, gamma, a0=a0, b0=b0, c0=c0, d0=d0, step_alpha=1, step_gamma=1)

        # print(f'Iter: {it}, Number of states: {n_states}, alpha:', alpha, 'gamma:', gamma)

        if it >= burn_in:
            state_means += order_of_appearance(states)
            prev_state = states[-1]
            y_pred_mean += np.sum(likelihood(y_pred, means, params[2]) * pi[prev_state, :])

    state_means /= (n_samples - burn_in)
    y_pred_mean /= (n_samples - burn_in)
    
    return state_means, y_pred_mean

@njit(fastmath=True)
def compute_transition_counts(K, states):
    T = len(states)
    counts = np.zeros((K, K), dtype=np.int64)

    for t in range(T - 1):
        i = states[t]
        j = states[t + 1]
        counts[i, j] += 1

    return counts

@njit(fastmath=True)
def log_marginal_likelihood(K, counts, alpha, gamma):
    total = 0.0

    for i in range(K):
        row_sum = 0
        for j in range(K):
            row_sum += counts[i, j]

        total += math.lgamma(K * alpha + gamma)
        total -= math.lgamma(row_sum + K * alpha + gamma)

        for j in range(K):
            if j == i:
                total += math.lgamma(counts[i, j] + alpha + gamma)
                total -= math.lgamma(alpha + gamma)
            else:
                total += math.lgamma(counts[i, j] + alpha)
                total -= math.lgamma(alpha)

    return total

@njit(fastmath=True)
def log_gamma_prior(x, shape, rate):
    return (shape - 1.0) * math.log(x) - rate * x

@njit(fastmath=True)
def update_alpha_gamma(
    K,
    states,
    alpha,
    gamma,
    a0, b0,   # alpha prior: shape, rate
    c0, d0,   # gamma prior: shape, rate
    step_alpha,
    step_gamma
):
    counts = compute_transition_counts(K, states)

    nthin = 5

    for i in range(nthin):
        log_alpha = math.log(alpha)
        proposal_log_alpha = log_alpha + step_alpha * np.random.randn()
        proposal_alpha = math.exp(proposal_log_alpha)

        current_lp = (
            log_marginal_likelihood(K, counts, alpha, gamma)
            + log_gamma_prior(alpha, a0, b0)
            + log_gamma_prior(gamma, c0, d0)
            + log_alpha  # Jacobian
        )

        proposal_lp = (
            log_marginal_likelihood(K, counts, proposal_alpha, gamma)
            + log_gamma_prior(proposal_alpha, a0, b0)
            + log_gamma_prior(gamma, c0, d0)
            + proposal_log_alpha  # Jacobian
        )

        if math.log(np.random.rand()) < proposal_lp - current_lp:
            alpha = proposal_alpha
            log_alpha = proposal_log_alpha

    # ----- Update gamma -----
    for i in range(nthin):
        log_gamma = math.log(gamma)
        proposal_log_gamma = log_gamma + step_gamma * np.random.randn()
        proposal_gamma = math.exp(proposal_log_gamma)

        current_lp = (
            log_marginal_likelihood(K, counts, alpha, gamma)
            + log_gamma_prior(alpha, a0, b0)
            + log_gamma_prior(gamma, c0, d0)
            + log_gamma
        )

        proposal_lp = (
            log_marginal_likelihood(K, counts, alpha, gamma=proposal_gamma)
            + log_gamma_prior(alpha, a0, b0)
            + log_gamma_prior(proposal_gamma, c0, d0)
            + proposal_log_gamma
        )

        if math.log(np.random.rand()) < proposal_lp - current_lp:
            gamma = proposal_gamma

    return alpha, gamma

@njit(fastmath=True)
def order_of_appearance(clusters):

    map_to_ind = Dict.empty(types.int64, types.int64)

    counter = iter(range(len(clusters)))
    new_clusters = np.zeros(len(clusters), dtype=np.int64)
    for i in range(len(clusters)):
        if clusters[i] in map_to_ind:
            new_clusters[i] = map_to_ind[clusters[i]]
        else:
            map_to_ind[clusters[i]] = next(counter)
            new_clusters[i] = map_to_ind[clusters[i]]
    
    return new_clusters

def plot_data(y: np.array, clusters: np.array, max_cluster=6, true_clusters=None) -> None:
    T_ = len(clusters)
    if true_clusters is None:
        fig, axs = plt.subplots(1, 1, sharex=True)
        axs = [axs]
    else:
        fig, axs = plt.subplots(2, 1, sharex=True)
    
    x_, y_ = np.arange(1, T_+1), y[(y.size - T_):]
    for i in range(len(y_) - 1):
        color = plt.cm.tab10(clusters[i] / max_cluster)  # Normalize the value for colormap
        axs[0].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)
    
    if true_clusters is not None:
        for i in range(len(true_clusters) - 1):
            color = plt.cm.tab10(true_clusters[i] / max_cluster)  # Normalize the value for colormap
            axs[1].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)

    plt.show()

if __name__ == "__main__":
    from time import perf_counter

    RNG = np.random.default_rng(seed=1)
    y = np.concatenate([
        RNG.normal(0, 1, 50),
        RNG.normal(-2, 1, 50), 
        RNG.normal(2, 1, 50), 
        RNG.normal(4, 1, 50), 
        RNG.normal(0, 1, 50),
        RNG.normal(-2, 1, 50), 
        RNG.normal(2, 1, 50), 
        RNG.normal(4, 1, 50), 
    ])

    params = np.array([0, 4, 1], dtype=np.float64)
    
    K = 4
    # alpha = np.ones(K, dtype=np.float64)
    # gamma = 20
    
    # t1 = perf_counter()
    # log_preds = get_cum_log_pred(y, alpha, params, n_iter=150, burn_in=20, n_particles=75)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')

    # print(log_preds[-1])
    # plt.plot(log_preds)
    # plt.show()

    t1 = perf_counter()
    states, pred_density = sample(y, n_samples=15000, burn_in=1000, K=K, params=params, y_pred=4.0)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    
    print(pred_density)
    # print(true_clusters)
    # print(states)
    plot_data(y, states, max_cluster=K)
    # plot_data(y, states, max_cluster=max(max(true_clusters), int(max(states)) + 1), true_clusters=true_clusters)