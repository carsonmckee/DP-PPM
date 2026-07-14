import math
import numpy as np 
from numba import njit, prange
from typing import List
from numba import njit, types, typed

@njit(fastmath=False, error_model="numpy")
def remove_col(arr, idx):
    if arr.ndim == 1:
        n = arr.shape[0]
        result = np.empty(n - 1, dtype=arr.dtype)
        new_i = 0
        for i in range(n):
            if i != idx:
                result[new_i] = arr[i]
                new_i += 1
        return result
    elif arr.ndim == 2:
        n_rows, n_cols = arr.shape
        result = np.empty((n_rows, n_cols - 1), dtype=arr.dtype)
        for i in range(n_rows):
            new_col = 0
            for j in range(n_cols):
                if j != idx:
                    result[i, new_col] = arr[i, j]
                    new_col += 1
        return result
    else:
        raise ValueError("Only 1D and 2D arrays are supported")

@njit(fastmath=False, error_model="numpy")
def remove_row(arr, row_idx):
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows - 1, n_cols), dtype=arr.dtype)
    
    new_row = 0
    for i in range(n_rows):
        if i != row_idx:
            for j in range(n_cols):
                result[new_row, j] = arr[i, j]
            new_row += 1
            
    return result

@njit(fastmath=False, error_model="numpy")
def likelihood(y, mu, sigma2):
    return np.exp(-0.5*(y-mu)*(y-mu)/sigma2) / np.sqrt(2*np.pi*sigma2)

@njit(fastmath=False, error_model="numpy")
def multinomial_sample(weights):
    """
    Generate a single multinomial sample (category index) given weights.
    
    Parameters
    ----------
    weights : 1D array of floats
        Probabilities for each category, must sum to 1.
    
    Returns
    -------
    int
        Index of the sampled category
    """
    # Compute cumulative sum
    # cum_weights = np.empty(weights.shape[0])
    cum_weights = [weights[0]]
    # cum_weights[0] = weights[0]
    for i in range(1, len(weights)):
        cum_weights.append(cum_weights[i-1] + weights[i])
    
    # Sample uniform [0,1)
    u = np.random.rand()
    
    # Find the first index where u < cumulative sum
    for i in range(len(cum_weights)):
        if u < cum_weights[i]:
            return i
    
    # Safety fallback (should not happen if weights sum to 1)
    return len(weights) - 1

@njit(fastmath=False, error_model="numpy")
def sample_states(y, u, pi, mus, sigma2, curr_states):

    K = pi.shape[0]
    T_ = len(y)
    dyn_prog = np.zeros((K, T_), dtype=np.float64)
    pi_ = pi[:K, :K]
    dyn_prog[:, 0] = (pi_[0, :] > u[0]) * likelihood(y[0], mus, sigma2)
    if np.sum(dyn_prog[:, 0]) < 10E-320:
        return curr_states
    dyn_prog[:, 0] /= np.sum(dyn_prog[:, 0])
    for t in range(1, T_):
        dyn_prog[:, t] = (dyn_prog[:, t-1] @ (pi_ > u[t]).astype(np.float64)) * likelihood(y[t], mus, sigma2)
        if np.sum(dyn_prog[:, t]) < 10E-320:
            return curr_states
        dyn_prog[:, t] /= np.sum(dyn_prog[:, t])
    
    states = np.zeros(T_, dtype=np.int64)
    states[T_-1] = multinomial_sample(dyn_prog[:, T_-1])
    for t in range(T_-2, -1, -1):
        r = dyn_prog[:, t] * (pi_[:, states[t+1]] > u[t+1]).astype(np.float64)
        if np.sum(r) < 10E-320:
            return curr_states
        r /= np.sum(r)
        states[t] = multinomial_sample(r)

    return states

@njit(fastmath=False, error_model="numpy")
def sample_hypers(states, betas, alpha0, gamma, alpha_a, alpha_b, gamma_a, gamma_b, num_i):
    
    K = betas.size - 1
    T_ = len(states)

    N = np.zeros((K, K), dtype=np.int64)
    N[0, states[0]] = 1
    for t in range(1, T_):
        N[states[t-1], states[t]] += 1

    M = np.zeros((K, K), dtype=np.int64)
    for j in range(K):
        for k in range(K):
            if N[j, k] == 0:
                M[j, k] = 0
            else:
                for l in range(N[j, k]): 
                    val = (alpha0 * betas[k]) / (alpha0 * betas[k] + l)
                    if np.random.uniform(0, 1) < val:
                        M[j, k] += 1
    
    alph_temp = np.append(np.sum(M, 0), np.array([gamma], dtype=np.float64))
    # print(alph_temp)
    betas = np.random.dirichlet(alph_temp)

    # resample alpha
    for i in range(num_i):
        row_sums = N.sum(axis=1)
        # w = np.random.beta((alpha0 + 1.0)*np.ones(K, dtype=np.float64), row_sums.astype(np.float64), size=K)
        w = np.ones(K, dtype=np.float64)
        for l in range(K):
            if row_sums[l] == 0:
                w[l] = 1
            else:
                w[l] = np.random.beta(alpha0+1.0, row_sums[l])
        p = row_sums / alpha0
        p = p / (p + 1.0)
        # s = np.random.binomial(1, p)
        s = np.zeros(K, dtype=np.float64)
        for l in range(K):
            s[l] = np.random.binomial(1, p[l])
        
        shape = alpha_a + M.sum() - s.sum()
        scale = 1.0 / (alpha_b - np.log(w).sum())
        if (shape <= 0.0) or (not np.isfinite(shape)):
            shape = 10E-100
        if (scale <= 0.0) or (not np.isfinite(scale)):
            scale = 10E-100
        alpha0 = np.random.gamma(shape, scale)
    
    # resample gamma
    k = len(betas)
    m = M.sum()

    for _ in range(num_i):
        mu = np.random.beta(gamma + 1.0, m)
        pi_mu = 1.0 / (1.0 + (m * (gamma_b - np.log(mu))) / (gamma_a + k - 1.0))
        if np.random.uniform(0, 1) < pi_mu:
            shape = gamma_a + k
        else:
            shape = gamma_a + k - 1.0
        
        scale = 1.0 / (gamma_b - np.log(mu))
        if (shape <= 0.0) or (not np.isfinite(shape)):
            shape = 10E-100
        if (scale <= 0.0) or (not np.isfinite(scale)):
            scale = 10E-100
        gamma = np.random.gamma(shape, scale)

    return betas, alpha0, gamma

@njit(fastmath=False, error_model="numpy")
def sample_transitions(S, H):
    K = H.size
    T_ = S.size
    pi = np.zeros((K-1, K), dtype=np.float64)
    N = np.zeros((K-1, K), dtype=np.float64)
    for t in range(1, T_):
        N[S[t-1], S[t]] += 1
    
    for k in range(K-1):
        is_nan = True
        c = 0
        while is_nan:
            if c > 5:
                pi[k, :] = np.zeros(K, dtype=np.float64)
                pi[k, np.argmax(N[k, :] + H)] = 1
                is_nan = False 
            else:
                pi[k, :] = np.random.dirichlet(N[k, :] + H)
                is_nan = np.isnan(pi[k, :].sum())
                c += 1
    
    return pi

@njit(fastmath=False, error_model="numpy")
def sample_posterior(y_sum, y2_sum, n, params):
    mu0, kappa0, alpha0, beta0 = params[0], params[1], params[2], params[3]

    if n == 0:
        # Sample from prior
        sigma2 = 1.0 / np.random.gamma(alpha0, 1.0 / beta0)
        mu = np.random.normal(mu0, np.sqrt(sigma2 / kappa0))
        return mu, sigma2

    # Posterior parameters
    kappa_n = kappa0 + n
    alpha_n = alpha0 + 0.5 * n

    mu_n = (kappa0 * mu0 + y_sum) / kappa_n

    term1 = y2_sum - (y_sum * y_sum) / n
    term2 = ((y_sum - n * mu0) ** 2) / n
    beta_n = beta0 + 0.5 * term1 + (kappa0 / (2.0 * kappa_n)) * term2
    
    # Sample sigma^2 from Inverse-Gamma
    sigma2 = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)

    # Sample mu conditional on sigma^2
    mu = np.random.normal(mu_n, np.sqrt(sigma2 / kappa_n))

    return mu, sigma2

@njit(fastmath=False, error_model="numpy")
def sample_mus_sigma2s(states, y,  params):
    K = len(set(states))

    mus = np.zeros(K, dtype=np.float64)
    sigma2s = np.zeros(K, dtype=np.float64)

    for k in range(K): 
        temp = np.where(states == k)
        y_temp = y[temp]

        mu, s2 = sample_posterior(np.sum(y_temp), np.sum(y_temp*y_temp), len(y_temp), params)
        mus[k] = mu
        sigma2s[k] = s2

    return mus, sigma2s

@njit(fastmath=False, error_model="numpy")
def break_sticks_numba(Pi, Beta, mus, sigma2s, alpha0, gamma, u, params, max_extra=10):
    """
    Numba-compatible stick-breaking update.
    Pi: K x L transition matrix (np.ndarray)
    Beta: length L vector of stick lengths
    Mus: length L vector of Mus
    alpha0, gamma: scalars
    u: threshold vector
    sigma2_0, mu_0: hyperparameters
    max_extra: maximum number of extra states we can add (preallocate)
    
    Returns updated Pi, Beta, Mus, K
    """

    added = 0
    while (np.max(Pi[:, -1]) > np.min(u)) and (added < max_extra):
        added += 1
        pl = Pi.shape[1]
        bl = len(Beta)
        assert bl == pl 

        new_Pi = np.zeros((Pi.shape[0]+1, Pi.shape[1]+1), dtype=np.float64)
        is_nan = True
        c = 0
        while is_nan:                
            if c > 5:
                new_row = np.zeros(len(Beta), dtype=np.float64)
                new_row[np.argmax(alpha0 * Beta)] = 1
                is_nan = False
            else:
                new_row = np.random.dirichlet(alpha0 * Beta)
                is_nan = np.isnan(new_row.sum())
                c += 1
        for i in range(Pi.shape[0]):
            for j in range(Pi.shape[1]):
                new_Pi[i, j] = Pi[i, j]
                new_Pi[-1, j] = new_row[j]
        Pi = new_Pi 
        new_mu, new_s2 = sample_posterior(0, 0, 0, params)
        mus = np.append(mus, np.array([new_mu], dtype=np.float64))
        sigma2s = np.append(sigma2s, np.array([new_s2], dtype=np.float64))

        # break beta stick
        be = Beta[-1]
        bg = np.random.beta(1.0, gamma)
        Beta = np.append(Beta, np.array([0.0], dtype=np.float64))
        Beta[-2] = bg*be
        Beta[-1] = (1-bg)*be

        a = alpha0 * Beta[-2]
        b = alpha0*(1-np.sum(Beta[:len(Beta)-1]))
        for k in range(bl):
            pe = Pi[k, -2]
            if (a == 0) and (b == 0):
                pg = np.random.uniform(0, 1)
            elif (a == 0) or (b == 0):
                pg = np.random.binomial(1, a/(a+b))
            else:
                pg = np.random.beta(a, b)
            Pi[k, -2] = pg*pe
            Pi[k, -1] = (1-pg)*pe
    K = Pi.shape[0]
    
    return Pi, Beta, mus, sigma2s, K

@njit(fastmath=False, error_model="numpy")
def pred_density(y, states, pi, mus, sigma2s, params):

    new_mu = np.random.normal(params[0], np.sqrt(params[1]))
    new_mu, new_s2 = sample_posterior(0, 0, 0, params)
    mu_temp = np.append(mus, np.array([new_mu], dtype=np.float64))
    sigma2_temp = np.append(sigma2s, np.array([new_s2], dtype=np.float64))
    # print(mu_temp)
    # print(sigma2_temp)
    pred_val = np.sum(pi[states[-1], :] * likelihood(y, mu_temp, sigma2_temp))
    return pred_val

@njit(fastmath=False, error_model="numpy")
def sample(y: np.array, 
           n_samples: int, 
           burn_in: int,
           params: np.array, 
           y_pred: np.array=0.0,
           alpha_a = 1.0, 
           alpha_b = 1.0, 
           gamma_a = 1.0, 
           gamma_b = 1.0, 
           verbose: bool=False):
    
    T_ = y.size
    
    state_means = np.zeros(T_, dtype=np.float64)

    states = np.zeros(T_, dtype=np.int64)
    K = len(set(states))

    alpha0, gamma = 1.0, 1.0
    
    for i in range(1):
        betas = np.ones(K+1, dtype=np.float64) / (K+1)
        betas, alpha0, gamma = sample_hypers(states, betas, alpha0, gamma, alpha_a, alpha_b, gamma_a, gamma_b, 5)

    mus, sigma2s = sample_mus_sigma2s(states, y, params)
    pi = sample_transitions(states, alpha0*betas)
    u = np.zeros(T_, dtype=np.float64)

    n_states = np.zeros(n_samples-burn_in, dtype=np.int64)


    y_pred_mean = 0.0
    for it in range(n_samples):
        
        # update slices
        u[0] = np.random.uniform(0, pi[0, states[0]])
        for t in range(1, T_):
            u[t] = np.random.uniform(0, pi[states[t-1], states[t]])
        
        # break sticks
        pi, betas, mus, sigma2s, K = break_sticks_numba(pi, betas, mus, sigma2s, alpha0, gamma, u, params)
        states = sample_states(y, u, pi, mus, sigma2s, states)
        # clean up
        counts = np.zeros(K, dtype=np.int64)
        for t in range(T_):
            counts[states[t]] += 1

        for k in range(len(counts)-1, -1 ,-1):
            if counts[k] == 0:
                betas[-1] += betas[k]
                betas = remove_col(betas, k)
                pi[:, -1] += pi[:, k]
                pi = remove_col(pi, k)
                pi = remove_row(pi, k)
                mus = remove_col(mus, k)
                sigma2s = remove_col(sigma2s, k)
                states[states > k] -= 1
        betas, alpha0, gamma = sample_hypers(states, betas, alpha0, gamma, alpha_a, alpha_b, gamma_a, gamma_b, 5)
        
        # sample mus and sigma2s
        mus, sigma2s = sample_mus_sigma2s(states, y, params)
        
        # sample transition matrix
        y_pred_ = pred_density(y_pred, states, pi, mus, sigma2s, params)
        pi = sample_transitions(states, alpha0*betas)

        if verbose:
            print('Iter: ', it, ", n states: ", mus.size, ", alpha: ", alpha0, ", gamma: ", gamma, ", y_pred: ", y_pred_)
        
        if it >= burn_in:
            state_means += states
            n_states[it-burn_in] = mus.size
            if (not np.isnan(y_pred_)) and np.isfinite(y_pred_):
                y_pred_mean += y_pred_
    
    state_means /= (n_samples - burn_in)
    y_pred_mean /= (n_samples - burn_in)

    # return n_states, state_means, y_pred_mean
    return n_states, states, y_pred_mean

def plot_data(y: np.array, clusters: np.array, max_cluster=6, true_clusters=None) -> None:
    T_ = len(clusters)
    if true_clusters is None:
        fig, axs = plt.subplots(1, 1, sharex=True)
    else:
        fig, axs = plt.subplots(2, 1, sharex=True)
    
    x_, y_ = np.arange(1, T_+1), y[(y.size - T_):]
    for i in range(len(y_) - 1):
        color = plt.cm.tab10(clusters[i] / max_cluster)  # Normalize the value for colormap
        if true_clusters is None:
            axs.plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)
        else:
            axs[0].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)
    
    if true_clusters is not None:
        for i in range(len(true_clusters) - 1):
            color = plt.cm.tab10(true_clusters[i] / max_cluster)  # Normalize the value for colormap
            axs[1].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)

    plt.show()


if __name__ == "__main__":
    import numba
    import pandas as pd
    from time import perf_counter
    import matplotlib.pyplot as plt

    # RNG = np.random.default_rng(seed=1)
    # y = np.concatenate([
    #     RNG.normal(0, 1, 50),
    #     RNG.normal(-2, 1, 50), 
    #     RNG.normal(2, 1, 50), 
    #     RNG.normal(4, 1, 50), 
    #     RNG.normal(0, 1, 50),
    #     RNG.normal(-2, 1, 50), 
    #     RNG.normal(2, 1, 50), 
    #     RNG.normal(4, 1, 50), 
    # ])
    
    path = "C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/"
    well_log = pd.read_csv(f'{path}/well_log_clean.csv').to_numpy().flatten()
    
    well_log = (well_log - np.mean(well_log)) / np.sqrt(np.var(well_log))

    params = np.array([0, 1, 1, 1], dtype=np.float64)

    t_pred = 3945

    t1 = perf_counter()
    n_states, states, pred = sample(well_log[:t_pred], n_samples=20000, burn_in=10000, params=params, y_pred=well_log[t_pred], verbose=True)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    
    np.savetxt("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/well_log_results/ihmm_single_state_sample.csv", states, delimiter=',')
    # np.savetxt("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/well_log_results/ihmm_n_states.csv", n_states, delimiter=',')
    print(pred)
    plot_data(well_log[:t_pred], states, max_cluster=np.max(states))