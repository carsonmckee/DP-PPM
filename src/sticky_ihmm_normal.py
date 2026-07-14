import math
import numpy as np 
from numba import njit, prange
from typing import List
from numba import njit, types, typed

# @njit(fastmath=False, error_model="numpy")
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

# @njit(fastmath=False, error_model="numpy")
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

# @njit(fastmath=False, error_model="numpy")
def likelihood(y, mu, sigma2):
    return np.exp(-0.5*(y-mu)*(y-mu)/sigma2) / np.sqrt(2*np.pi*sigma2)

# @njit(fastmath=False, error_model="numpy")
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

# @njit(fastmath=False, error_model="numpy")
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

# @njit(fastmath=False, error_model="numpy")
def sample_hypers(
    states,
    betas,
    tau,          # alpha + kappa
    gamma,
    rho,          # kappa / (alpha + kappa)
    tau_a,
    tau_b,
    gamma_a,
    gamma_b,
    rho_a,
    rho_b,
    num_i
):

    K = betas.size - 1
    T_ = len(states)

    # ------------------------------------------------------------
    # 1. Transition counts
    # ------------------------------------------------------------
    N = np.zeros((K, K), dtype=np.int64)

    N[0, states[0]] = 1
    for t in range(1, T_):
        N[states[t - 1], states[t]] += 1

    # ------------------------------------------------------------
    # 2. CRT table counts with sticky correction
    # ------------------------------------------------------------
    M = np.zeros((K, K), dtype=np.int64)

    for j in range(K):
        for k in range(K):

            if N[j, k] == 0:
                continue

            for l in range(N[j, k]):

                if j == k:
                    conc = (1.0 - rho) * tau * betas[k] + rho * tau
                else:
                    conc = (1.0 - rho) * tau * betas[k]

                val = conc / (conc + l)

                if np.random.uniform(0.0, 1.0) < val:
                    M[j, k] += 1

    # ------------------------------------------------------------
    # 3. Split diagonal tables (sticky vs non-sticky)
    # ------------------------------------------------------------
    W = np.zeros(K, dtype=np.int64)   # sticky tables

    for j in range(K):

        if M[j, j] > 0:

            p = (
                rho * tau
            ) / (
                rho * tau + (1.0 - rho) * tau * betas[j]
            )

            W[j] = np.random.binomial(M[j, j], p)

    M_bar = M.copy()
    for j in range(K):
        M_bar[j, j] -= W[j]

    # ------------------------------------------------------------
    # 4. Update beta (ONLY non-sticky tables)
    # ------------------------------------------------------------
    alph_temp = np.zeros(K + 1, dtype=np.float64)

    for k in range(K):
        alph_temp[k] = np.sum(M_bar[:, k]) + 1e-8

    alph_temp[K] = gamma

    betas = np.random.dirichlet(alph_temp)

    # ------------------------------------------------------------
    # 5. Update gamma (standard HDP update)
    # ------------------------------------------------------------
    m_bar = np.sum(M_bar) + 10e-10

    for _ in range(num_i):

        mu = np.random.beta(gamma + 1.0, m_bar)

        pi_mu = 1.0 / (
            1.0 + (m_bar * (gamma_b - np.log(mu))) /
            (gamma_a + K - 1.0)
        )

        if np.random.uniform() < pi_mu:
            shape = gamma_a + K
        else:
            shape = gamma_a + K - 1.0

        scale = 1.0 / (gamma_b - np.log(mu))

        if shape <= 0.0 or not np.isfinite(shape):
            shape = 1e-10
        if scale <= 0.0 or not np.isfinite(scale):
            scale = 1e-10

        gamma = np.random.gamma(shape, scale)

    # ------------------------------------------------------------
    # 6. Update tau = alpha + kappa
    # ------------------------------------------------------------
    row_sums = N.sum(axis=1)
    m_total = np.sum(M)

    for _ in range(num_i):

        # CRT augmentation
        s = np.zeros(K, dtype=np.float64)

        for k in range(K):
            p = row_sums[k] / (row_sums[k] + tau)
            s[k] = np.random.binomial(1, p)

        shape = tau_a + m_total - s.sum()
        scale = 1.0 / tau_b

        if shape <= 0.0 or not np.isfinite(shape):
            shape = 1e-10
        if scale <= 0.0 or not np.isfinite(scale):
            scale = 1e-10

        tau = np.random.gamma(shape, scale)

    # ------------------------------------------------------------
    # 7. Update rho (sticky proportion)
    # ------------------------------------------------------------
    W_sum = np.sum(W)
    M_sum = np.sum(M) - np.sum(W)   # non-sticky mass on diagonal

    for _ in range(num_i):

        rho = np.random.beta(
            rho_a + W_sum,
            rho_b + M_sum
        )

    # ------------------------------------------------------------
    # 8. Recover alpha and kappa
    # ------------------------------------------------------------
    alpha0 = (1.0 - rho) * tau
    kappa  = rho * tau

    return betas, alpha0, kappa, gamma

# @njit(fastmath=False, error_model="numpy")
def sample_transitions(S, H, kappa):
    K = H.size
    T_ = S.size
    pi = np.zeros((K-1, K), dtype=np.float64)
    N = np.zeros((K-1, K), dtype=np.float64)
    for t in range(1, T_):
        N[S[t-1], S[t]] += 1
    
    for k in range(K-1):
        N[k, k] += kappa
    
    for k in range(K-1):
        is_nan = True
        c = 0
        while is_nan:
            if c > 5:
                pi[k, :] = np.zeros(K, dtype=np.float64)
                pi[k, np.argmax(N[k, :] + H)] = 1
                is_nan = False 
            else:
                pi[k, :] = np.random.dirichlet(N[k, :] + H + 1e-8)
                is_nan = np.isnan(pi[k, :].sum())
                c += 1
    
    return pi

# @njit(fastmath=False, error_model="numpy")
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

# @njit(fastmath=False, error_model="numpy")
def sample_mus(states, y,  params):
    K = len(set(states))

    mus = np.zeros(K, dtype=np.float64)

    for k in range(K):
        temp = np.where(states == k)

        mu = sample_posterior_mean(y[temp], params[2], params[0], params[1])
        mus[k] = mu

    return mus

# @njit(fastmath=False, error_model="numpy")
def break_sticks_numba(
    Pi,
    Beta,
    mus,
    alpha0,
    gamma,
    kappa,
    u,
    params,
    max_extra=10,
):
    """
    Sticky HDP-IHMM beam-sampler state expansion.

    Parameters
    ----------
    Pi : (K,L) transition matrix
    Beta : (L,) global stick weights
    phis : emission parameters
    sigma2s : emission variances
    alpha0 : HDP concentration
    gamma : GEM concentration
    kappa : sticky parameter
    u : slice variables
    params : emission hyperparameters
    """

    added = 0

    while (np.max(Pi[:, -1]) > np.min(u)) and (added < max_extra):

        added += 1

        pl = Pi.shape[1]
        bl = len(Beta)

        assert bl == pl

        #
        # Expand Pi by one row and one column
        #
        new_Pi = np.zeros(
            (Pi.shape[0] + 1, Pi.shape[1] + 1),
            dtype=np.float64,
        )

        #
        # Sample transition row for the NEW state
        #
        prior = alpha0 * Beta.copy()

        # sticky self-transition mass
        prior[-1] += kappa

        is_nan = True
        c = 0

        while is_nan:

            if c > 5:
                new_row = np.zeros(len(Beta), dtype=np.float64)
                new_row[np.argmax(prior)] = 1.0
                is_nan = False
            else:
                new_row = np.random.dirichlet(prior + 1e-8)
                is_nan = np.isnan(new_row.sum())
                c += 1

        #
        # Copy old matrix
        #
        for i in range(Pi.shape[0]):
            for j in range(Pi.shape[1]):
                new_Pi[i, j] = Pi[i, j]

        #
        # Insert new state's transition row
        #
        for j in range(len(Beta)):
            new_Pi[-1, j] = new_row[j]

        Pi = new_Pi

        #
        # Sample emission parameters
        #
        new_mu = np.random.normal(params[0], np.sqrt(params[1]))
        mus = np.append(mus, np.array([new_mu], dtype=np.float64))

        #
        # Break beta stick
        #
        be = Beta[-1]

        bg = np.random.beta(1.0, gamma)

        Beta = np.append(
            Beta,
            np.array([0.0], dtype=np.float64),
        )

        Beta[-2] = bg * be
        Beta[-1] = (1.0 - bg) * be

        #
        # Split corresponding transition probabilities
        #
        for k in range(bl):

            pe = Pi[k, -2]

            #
            # Sticky HDP-HMM correction
            #
            a = alpha0 * Beta[-2]
            b = alpha0 * Beta[-1]

            #
            # New state's self-transition gets κ
            #
            if k == (bl - 1):
                a += kappa

            if (a == 0.0) and (b == 0.0):

                pg = np.random.uniform()

            elif (a == 0.0) or (b == 0.0):

                pg = np.random.binomial(
                    1,
                    a / (a + b),
                )

            else:

                pg = np.random.beta(a, b)

            Pi[k, -2] = pg * pe
            Pi[k, -1] = (1.0 - pg) * pe

    K = Pi.shape[0]

    return Pi, Beta, mus, K

# @njit(fastmath=False, error_model="numpy")
def pred_density(y, states, pi, mus, params):

    new_mu = np.random.normal(params[0], np.sqrt(params[1]))
    mu_temp = np.append(mus, np.array([new_mu], dtype=np.float64))

    pred_val = np.sum(pi[states[-1], :] * likelihood(y, mu_temp, params[2]))
    return pred_val

# @njit(fastmath=False, error_model="numpy")
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

    alpha0, gamma, kappa = 1.0, 1.0, 50.
    
    for i in range(1):
        betas = np.ones(K+1, dtype=np.float64) / (K+1)
        # betas, alpha0, gamma = sample_hypers(states, betas, alpha0, gamma, alpha_a, alpha_b, gamma_a, gamma_b, 5)
        betas, alpha0, kappa, gamma = sample_hypers(states, betas, alpha0+kappa, gamma, kappa / (alpha0 + kappa), 1, 1, 1, 1, 10, 1, 5)

    mus = sample_mus(states, y, params)
    pi = sample_transitions(states, alpha0*betas, kappa)
    u = np.zeros(T_, dtype=np.float64)

    y_pred_mean = 0.0
    for it in range(n_samples):
        
        # update slices
        # print('here1')
        u[0] = np.random.uniform(0, pi[0, states[0]])
        for t in range(1, T_):
            u[t] = np.random.uniform(0, pi[states[t-1], states[t]])
        
        # break sticks
        # print('here2')
        pi, betas, mus, K = break_sticks_numba(pi, betas, mus, alpha0, gamma, kappa, u, params)
        states = sample_states(y, u, pi, mus, params[2], states)
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
                states[states > k] -= 1
        # print('here3')
        # betas, alpha0, gamma = sample_hypers(states, betas, alpha0, gamma, alpha_a, alpha_b, gamma_a, gamma_b, 5)
        betas, alpha0, kappa, gamma = sample_hypers(states, betas, alpha0+kappa, gamma, kappa / (alpha0 + kappa), 1, 1, 1, 1, 10, 1, 5)
        
        # sample mus
        mus = sample_mus(states, y, params)
        # sample transition matrix
        # print('here4')
        pi = sample_transitions(states, alpha0*betas, kappa)
        if verbose and (it % 500 == 0):
            print('Iter: ', it, ", n states: ", mus.size, ", alpha: ", alpha0, ", gamma: ", gamma, ", kappa: ", kappa)
        if it >= burn_in:
            state_means += states
            y_pred_ = pred_density(y_pred, states, pi, mus, params)
            if (not np.isnan(y_pred_)) and np.isfinite(y_pred_):
                y_pred_mean += y_pred_
    
    state_means /= (n_samples - burn_in)
    y_pred_mean /= (n_samples - burn_in)

    return states, y_pred_mean

def multivariate_run(y: np.array, 
           n_samples: int, 
           burn_in: int,
           params: np.array, 
           y_pred: np.array):
    
    d = y.shape[0]
    
    preds = None

    for j in range(d):
        error = True
        while error:
            try:
                _, pred = sample(y[j, :], n_samples=n_samples, burn_in=burn_in, params=params, y_pred=y_pred[j])
                error = False
            except Exception as ex:
                print(ex)
                print('retrying')

        if preds is None:
            preds = pred 
        else:
            preds *= pred

    return np.mean(preds)

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
    from time import perf_counter
    import matplotlib.pyplot as plt

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

    # y = y[:150]
    
    # t1 = perf_counter()
    # log_preds = get_cum_log_pred(y, alpha, beta, gamma, params, n_iter=150, burn_in=20, n_particles=75)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')
    
    # print(log_preds[-1])
    # plt.plot(log_preds)
    # plt.show()

    t1 = perf_counter()
    states, pred = sample(y, n_samples=50000, burn_in=4000,params=params, y_pred=0.1, verbose=True)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    
    print(pred)
    plot_data(y, states, max_cluster=6)