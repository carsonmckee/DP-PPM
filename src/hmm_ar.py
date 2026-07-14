import math
import numpy as np 
from numba import njit, prange
from typing import List
from numba import njit, types, typed
import matplotlib.pyplot as plt
from numba.typed import Dict

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

    # ----- Update alpha -----
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

# XXX new code below

@njit(fastmath=True)
def likelihood(y, x, phi, sigma2):
    return np.exp(-0.5 * (y-phi*x)*(y-phi*x) / sigma2 )/ np.sqrt(2*np.pi*sigma2)

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
def sample_states(y, x, pi, phi, sigma2):

    T = len(y)
    K = len(phi)

    alpha = 1.0/np.ones(K, dtype=np.float64)

    filtered = np.zeros((T, K))

    for t in range(T):

        # emission
        diff = y[t] - phi * x[t]
        emission = np.exp(-0.5 * diff * diff / sigma2) / np.sqrt(2*np.pi*sigma2)
        
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
def sample_phi_sigma2_post(sum_yy, sum_xy, sum_xx, n, params):
    lam = params[0]
    a = params[1]
    b = params[2]
    
    a_n = a + 0.5*n
    lam_n = lam + sum_xx
    m_n = sum_xy / lam_n
    b_n = b + 0.5*(sum_yy - m_n*m_n * lam_n)
    
    sigma2 = 1.0/np.random.gamma(a_n, scale=1/b_n)
    phi = np.random.normal(m_n, np.sqrt(sigma2/lam_n))
    return phi, sigma2

@njit(fastmath=True)
def get_transition_counts(states, K):

    counts = np.zeros((K, K), dtype=np.float64)

    for i in range(1, len(states)):
        counts[states[i-1], states[i]] += 1
    
    return counts

@njit(fastmath=True)
def get_summary_states_for_state(y, x, states, k):
    
    T_ = y.size
    sum_yy, sum_xy, sum_xx, n = 0.0, 0.0, 0.0, 0.0

    for t in range(T_):
        if states[t] == k:
            sum_yy += y[t]*y[t]
            sum_xy += y[t]*x[t]
            sum_xx += x[t]*x[t]
            n += 1.0
    
    return sum_yy, sum_xy, sum_xx, n

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

    x = y[:y.size-1]
    y = y[1:]
    
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

    phis = np.random.uniform(-1, 1, size=K)
    sigma2s = np.random.gamma(0.5, 0.5, size=K)

    states = np.ones(T_, dtype=np.int64)
    state_means = np.zeros(T_, dtype=np.float64)

    y_pred_mean = 0.0
    for it in range(n_samples):
        
        states = sample_states(y, x, pi, phis, sigma2s) 

        # update etas
        transition_counts = get_transition_counts(states, K)
        for k1 in range(K):
            for k2 in range(K):
                a_post = alpha + transition_counts[k1, k2] + (gamma if k1 == k2 else 0.0)
                pi[k1, k2] = np.random.gamma(a_post, 1.0)
            pi[k1, :] = pi[k1, :] / np.sum(pi[k1, :])

        # update phis and sigma2s
        for k in range(K):
            sum_yy, sum_xy, sum_xx, n = get_summary_states_for_state(y, x, states, k)
            phi_, sigma2_ = sample_phi_sigma2_post(sum_yy, sum_xy, sum_xx, n, params)
            phis[k] = phi_ 
            sigma2s[k] = sigma2_

        n_states = len(set(states))
        alpha, gamma = update_alpha_gamma(K, states, alpha, gamma, a0=a0, b0=b0, c0=c0, d0=d0, step_alpha=1, step_gamma=1)

        # print(f'Iter: {it}, Number of states: {n_states}, alpha:', alpha, 'gamma:', gamma)

        if it >= burn_in:
            state_means += order_of_appearance(states)
            prev_state = states[-1]
            y_pred_mean += np.sum(likelihood(y_pred, y[-1], phis, sigma2s) * pi[prev_state, :])

    state_means /= (n_samples - burn_in)
    y_pred_mean /= (n_samples - burn_in)
    
    return state_means, y_pred_mean

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

def sim_ar_process(T_, phis, variances, cp_prob, cat_probs, RNG):
    
    out = np.zeros(T_+1)
    out[0] = RNG.normal(0, 1)
    phi = None
    variance = None
    clusters = np.zeros(T_, dtype=np.int64)
    for t in range(1, T_ + 1):
        if (RNG.uniform(0, 1) < cp_prob) or (t == 1):
            # change-point
            ind = RNG.choice(len(phis), p = cat_probs)
            phi = phis[ind]
            variance = variances[ind]
        clusters[t-1] = ind
        out[t] = RNG.normal(phi*out[t-1], np.sqrt(variance))

    return out[1:], order_of_appearance(clusters)

def plot_data(y: np.array, clusters: np.array, max_cluster=6, true_clusters=None) -> None:
    T_ = len(clusters)
    if true_clusters is None:
        fig, axs = plt.subplots(1, 1, sharex=True)
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
    # RNG = np.random.default_rng(seed=4)
    y, true_clusters = sim_ar_process(500, np.array([0.9, 0, -0.9, 0]), np.array([0.25, 1, 0.25, 3]), 1.0/75.0, np.ones(4)/4.0, RNG)
    
    params = np.array([1, 1, 1], dtype=np.float64)
    K = 4
    
    # t1 = perf_counter()
    # log_preds = get_cum_log_pred(y, alpha, params, n_iter=150, burn_in=20, n_particles=75)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')

    # print(log_preds[-1])
    # plt.plot(log_preds)
    # plt.show()
    
    t1 = perf_counter()
    states, pred_density = sample(y, n_samples=15000, burn_in=1000, K=K, params=params, y_pred=0.1)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    
    print(pred_density)
    # print(true_clusters)
    # print(states)
    plot_data(y, states, max_cluster=10, true_clusters=true_clusters)
    # plot_data(y, states, max_cluster=max(max(true_clusters), int(max(states)) + 1), true_clusters=true_clusters)