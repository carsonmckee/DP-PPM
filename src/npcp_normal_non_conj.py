import math
import numpy as np 
from numba import njit, prange
from typing import List
from numba import njit, types, typed

@njit(fastmath=True)
def log_g(sum_y, sum_y2, n, params):

    mu0 = params[0]
    tau2 = params[1]
    sigma2 = params[2]
    
    if n == 0:
        return 0

    # sample mean
    ybar = sum_y / n

    # sum of squared deviations
    sse = sum_y2 - (sum_y ** 2) / n

    # components
    term1 = -0.5 * n * math.log(2 * math.pi * sigma2)
    term2 = -0.5 * sse / sigma2
    term3 = -0.5 * math.log(1 + n * tau2 / sigma2)
    term4 = -0.5 * (ybar - mu0) ** 2 / (sigma2 / n + tau2)

    return term1 + term2 + term3 + term4

@njit(fastmath=True)
def likelihood(y, phi, params):
    sigma2 = params[2]
    return np.exp(-0.5*(y-phi)*(y-phi)/sigma2) / np.sqrt(2*np.pi*sigma2)

@njit(fastmath=True)
def multinomial_sample(weights):
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

@njit(fastmath=True)
def multinomial_sample_n(weights, N):
    n = len(weights)
    
    # Precompute cumulative weights (Numba-friendly)
    cum_weights = np.empty(n)
    cum_weights[0] = weights[0]
    for i in range(1, n):
        cum_weights[i] = cum_weights[i-1] + weights[i]
    
    # Generate N uniform samples
    us = np.random.rand(N)
    
    # Output indices
    samples = np.empty(N, dtype=np.int64)
    
    # Use binary search instead of linear scan
    for j in range(N):
        samples[j] = np.searchsorted(cum_weights, us[j])
    
    return samples

key_type = types.int64
val_type = types.float64
item_type = types.Tuple((
    types.int64, # run length 
    types.int64, # cluster allocation
    types.float64[:], # cluster counts 
    types.float64[:], # phis
    types.float64 # theta_t
))

@njit(fastmath=True)
def transition_probs(alpha: float, n: np.array) -> np.array:
    #DP polya urn scheme

    probs = np.zeros(n.size + 1, dtype=np.float64)
    probs[:n.size] = n
    probs[-1] = alpha
    
    probs /= np.sum(probs)

    return probs

@njit(fastmath=True)
def create_new_part(prev_part, r, cluster, theta):
    if r == 1:
        if cluster < prev_part[2].shape[0]:
            new_part = (r, cluster, prev_part[2].copy(), prev_part[3].copy(), theta)
            new_part[2][cluster] += 1
        else:
            prev_n = np.append(prev_part[2], np.ones(1, dtype=np.float64))
            prev_phis = np.append(prev_part[3], np.array([theta], dtype=np.float64))
            new_part = (r, cluster, prev_n, prev_phis, theta)
    else:
        new_part = (r, cluster, prev_part[2].copy(), prev_part[3].copy(), theta)

    return new_part

@njit(fastmath=True)
def resize_arr(arr, N):
    if len(arr) < N:
        for i in range(N-len(arr)):
            arr.append(0)
    else:
        for i in range(len(arr)-N):
            arr.pop()

@njit(fastmath=True, parallel=True)
def filter_forward(n_particles: int, 
                   M: int,
                   y: np.array, 
                   alpha,
                   params: np.array,
                   trajectory: List[int],
                   cond_parts: List,
                   cp_prob, 
                   first_run: bool):
    
    T_ = y.size

    particles = []
    weights = []
    ancestors = []

    new_parts = typed.List.empty_list(item_type)
    new_weights = []
    new_ancestors = []
    w_sum = 0.0
    for j in range(n_particles):
        if (j == trajectory[0]) and (not first_run):
            new_parts.append(cond_parts[0])
            w = likelihood(y[0], cond_parts[0][4], params)
            new_weights.append(w)
            w_sum += w 
        else:
            phi = np.random.normal(params[0], np.sqrt(params[1]))
            new_parts.append((1, 0, np.ones(1, dtype=np.float64), np.array([phi], dtype=np.float64), phi))
            w = likelihood(y[0], phi, params)
            new_weights.append(w)
            w_sum += w
        new_ancestors.append(0)
    
    for j in prange(n_particles):
        new_weights[j] /= w_sum 
    
    particles.append(new_parts)
    weights.append(new_weights)
    ancestors.append(new_ancestors)
    
    for t in range(1, T_):
        
        new_weights = []
        new_ancestors = []
        new_clusters = []
        new_rs = []
        new_phis = []
        
        prev_parts = particles[-1]
        prev_weights = weights[-1]

        weight_sum = 0.0
        # bulk_phis = np.random.normal(params[0], np.sqrt(params[1]), size = n_particles*M)
        for ind in range(n_particles):
            prev_part = prev_parts[ind]
            prev_weight = prev_weights[ind]
            prev_cluster = prev_part[1]
            prev_r = prev_part[0]
            K = prev_part[2].size
            prev_phis = prev_part[3]

            # K + 1 + m outcomes: 

            # no cp 
            new_rs.append(prev_r + 1)
            new_clusters.append(prev_cluster)
            new_ancestors.append(ind)
            new_phis.append(prev_phis[prev_cluster])
            new_weight = prev_weight * likelihood(y[t], prev_phis[prev_cluster], params) * (1 - cp_prob)
            new_weights.append(new_weight)
            weight_sum += new_weight 

            # a cp
            probs = transition_probs(alpha, prev_part[2])
            for k in range(K):
                new_rs.append(1)
                new_clusters.append(k)
                new_ancestors.append(ind)
                new_phis.append(prev_phis[k])
                
                new_weight = prev_weight * cp_prob * probs[k] * likelihood(y[t], prev_phis[k], params)
                new_weights.append(new_weight)
                weight_sum += new_weight 
            
            # K+1 cluster 
            for m in range(M):
                new_rs.append(1)
                new_clusters.append(K)
                new_ancestors.append(ind)
                new_phi = np.random.normal(params[0], np.sqrt(params[1]))
                # new_phi = bulk_phis[ind*M + m]
                new_phis.append(new_phi)        
                new_weight = prev_weight * cp_prob * probs[K] * likelihood(y[t], new_phi, params) / M
                new_weights.append(new_weight)
                weight_sum += new_weight 
        
        # normalize weights 
        for l in range(len(new_weights)):
            new_weights[l] /= weight_sum

        new_parts = typed.List.empty_list(item_type)

        # resample
        resampled_ancestors = []
        resampled_weights = []
        resampled_inds = multinomial_sample_n(new_weights, n_particles)
        for j in range(n_particles):
            
            if (not first_run) and (j == trajectory[t]):
                # set equal to conditioned particle
                new_parts.append(cond_parts[t])
                resampled_ancestors.append(trajectory[t-1])
                resampled_weights.append(1.0/n_particles)
            else:
                ind = resampled_inds[j]
                ancestor_ind = new_ancestors[ind]
                prev_part = prev_parts[ancestor_ind]
                cluster = new_clusters[ind]
                r = new_rs[ind]
                theta = new_phis[ind]
                
                new_part = create_new_part(prev_part, r, cluster, theta)
                new_parts.append(new_part)
                resampled_ancestors.append(ancestor_ind)
                resampled_weights.append(1.0/n_particles)

        particles.append(new_parts)
        weights.append(resampled_weights)
        ancestors.append(resampled_ancestors)

    return particles, weights, ancestors

@njit
def reverse_inplace(lst):
    i = 0
    j = len(lst) - 1
    
    while i < j:
        tmp = lst[i]
        lst[i] = lst[j]
        lst[j] = tmp
        
        i += 1
        j -= 1

@njit(fastmath=True)
def sample_ancestors(particle_history, weight_history, ancestor_history):
    T_ = len(particle_history)
    trajectory = [0]*T_

    clusters_out = [0]*T_
    runs_out = [0]*T_
    particles = typed.List.empty_list(item_type)

    for t in range(T_-1, -1, -1):
        
        if t == (T_-1):
            # ind = np.argmax(weights[-1])
            ind = multinomial_sample(weight_history[t])
            # print(particle_history[-1][ind])
        else:
            ind = ancestor_history[t+1][ind]
        
        trajectory[t] = ind
        runs_out[t] = particle_history[t][ind][0]
        clusters_out[t] = particle_history[t][ind][1]
        particles.append(particle_history[t][ind])
    
    reverse_inplace(particles)

    return trajectory, runs_out, clusters_out, particles

@njit(fastmath=True)
def pred(y, alpha, cp_prob, curr_part, params):

    n = curr_part[2]
    K = n.size
    
    SS = curr_part[3]
    prev_cluster = curr_part[1]
    pred_est = 0.0

    # no cp
    likelihood = np.exp(log_g(SS[prev_cluster, 0] + y, SS[prev_cluster, 1] + y*y, SS[prev_cluster, 2] + 1, params) \
                            - log_g(SS[prev_cluster, 0], SS[prev_cluster, 1], SS[prev_cluster, 2], params))

    pred_est += (1 - cp_prob) * likelihood

    # cp
    probs = transition_probs(alpha, n)
    for k in range(K):
        likelihood = np.exp(log_g(SS[k, 0] + y, SS[k, 1] + y*y, SS[k, 2] + 1, params) \
                            - log_g(SS[k, 0], SS[k, 1], SS[k, 2], params))
        pred_est += likelihood * cp_prob * probs[k]
    
    likelihood = np.exp(log_g(y, y*y, 1, params))
    pred_est += likelihood * cp_prob * probs[K]

    return pred_est

@njit(fastmath=True)
def sample_p(U: np.array):

    T_ = U.size
    n = np.sum(U) - 1
    out = np.random.beta(n + 1, T_-n)
    while out > 0.1:
        out = np.random.beta(n + 1, T_-n)
    return out

@njit(fastmath=True)
def update_dp_alpha(alpha, n, K, a0, b0):

    # ---- Step 1: Sample auxiliary variable η ----
    eta = np.random.beta(alpha + 1.0, n)

    # ---- Step 2: Compute mixture weight ----
    # π_eta
    numerator = a0 + K - 1.0
    denominator = n * (b0 - math.log(eta)) + numerator
    pi_eta = numerator / denominator

    # ---- Step 3: Sample from mixture of Gammas ----
    if np.random.rand() < pi_eta:
        shape = a0 + K
    else:
        shape = a0 + K - 1.0

    rate = b0 - math.log(eta)

    # numpy uses shape–scale, so scale = 1/rate
    alpha_new = np.random.gamma(shape, 1.0 / rate)

    return alpha_new

@njit(fastmath=True)
def sample(y: np.array, 
           n_samples: int, 
           M: int,
           burn_in: int,
           n_particles: int, 
           params: np.array, 
           y_pred: float=0.0):
    
    T_ = y.size
    
    trajectory = [0]*T_
    clusters = [0]*T_
    
    runs = [1]*T_ 
    first_run = True
    curr_particles = typed.List.empty_list(item_type)

    alpha = 1.0
    cp_prob = 1.0/450.0

    U = np.zeros(T_, dtype=np.float64)
    cluster_means = np.zeros(T_, dtype=np.float64)

    y_pred_mean = 0.0
    pred_density = 0.0
    for it in range(n_samples):
        # print(it)

        particles, weights, ancestors = filter_forward(n_particles, M, y, alpha, params, trajectory, curr_particles, cp_prob, first_run)
        first_run = False

        trajectory, runs, clusters, curr_particles = sample_ancestors(particles, weights, ancestors)
        
        U_temp = (np.array(runs, dtype=np.float64) == 1)

        n_states = len(set(clusters))

        # update cp_prob and alpha
        cp_prob = sample_p(U_temp)

        alpha = update_dp_alpha(alpha, np.sum(U_temp), n_states, 2, 2.5)

        print(f'Iter: {it}, Number of states: {n_states}, cp_prob:', round(cp_prob, 4), ', alpha: ', round(alpha, 4))

        c_temp = np.array(clusters, dtype=np.float64)
        if it >= burn_in:
            # U += U_temp
            U[1:] += (c_temp[0:len(c_temp)-1] != c_temp[1:])
            cluster_means += c_temp

            # pred_density = pred(y_pred, alpha, cp_prob, particles[-1][trajectory[-1]], params)
            # y_pred_mean += pred_density

    U /= (n_samples - burn_in)
    cluster_means /= (n_samples - burn_in)
    y_pred_mean /= (n_samples - burn_in)

    return U, cluster_means, y_pred_mean

@njit(fastmath=True)
def sim_DP_PPM_states(T_, alpha, cp_prob):
    
    states = [0]
    n0 = np.ones(1, dtype=np.float64)
    for i in range(T_):
        prev_state = states[-1]
        K = n0.size
        probs = transition_probs(alpha, n0)

        cp = False
        if np.random.uniform(0, 1) < cp_prob:
            cp = True 
            new_state = multinomial_sample(probs)
        else:
            cp = False
            new_state = prev_state 

        if new_state == K:
            n0 = np.append(n0, np.zeros(1, dtype=np.float64))
        
        if cp:
            n0[new_state] += 1
        
        states.append(new_state)
    # print(n)
    return states

def plot_data(y: np.array, U, clusters: np.array, max_cluster=6, true_clusters=None) -> None:
    T_ = len(clusters)
    if true_clusters is None:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig, axs = plt.subplots(3, 1, sharex=True)
    
    x_, y_ = np.arange(1, T_+1), y[(y.size - T_):]
    for i in range(len(y_) - 1):
        color = plt.cm.tab20(clusters[i] / max_cluster)  # Normalize the value for colormap
        axs[0].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)
    
    axs[1].bar(x_, U, width=0.8, color='black')

    if true_clusters is not None:
        for i in range(len(true_clusters) - 1):
            color = plt.cm.tab10(true_clusters[i] / max_cluster)  # Normalize the value for colormap
            axs[2].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)

    plt.show()

@njit
def distribute_load(arr: list, threads: int = 8):
    
    buckets = []
    for i in range(threads):
        buckets.append(typed.List.empty_list(types.int64))
    
    i = 0
    bucket_number = int(i % threads)
    while len(arr) > 0:
        ind = arr.pop()
        buckets[bucket_number].append(ind)
        i += 1
        bucket_number = int(i % threads)
    
    out = []
    for bucket in buckets:
        out.extend(bucket)
    
    return out

@njit(fastmath=True, parallel=True)
def get_cum_log_pred(y, params, n_iter=1000, burn_in=100, n_particles=100):
    
    log_preds = np.zeros(len(y)-1, dtype=np.float64)
    
    inds = distribute_load(list(range(1, len(y))))

    for i in prange(len(y)-1):
        # print(i)
        ind = inds[i]
        _, _, pred_density = sample(y[:ind], 300, 50, 200, params, y_pred=y[ind])
        
        log_preds[ind-1] = np.log(pred_density)

    return np.cumsum(log_preds)

if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    from time import perf_counter
    import pandas as pd 
    
    # path = "C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/"
    # well_log = pd.read_csv(f'{path}/well_log_clean.csv').to_numpy().flatten()
    
    # params = np.array([0, 5, 1.5*np.var(well_log[20:150])], dtype=np.float64)

    # t1 = perf_counter()
    # U, clusters, pred_density = sample(well_log, n_samples=1, burn_in=0, n_particles=10, params=params, y_pred = 0.1)
    # t2 = perf_counter()
    # print('compile done')

    # t1 = perf_counter()
    # U, clusters, pred_density = sample(well_log, n_samples=200, burn_in=50, n_particles=150, params=params, y_pred = 0.1)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')
    # print(pred_density)

    # plot_data(well_log, U, clusters, max_cluster=15)

    # fig, axs = plt.subplots(2, 1, sharex=True) 
    # axs[0].plot(well_log)
    # axs[1].bar(np.arange(len(U)), U, color='black', width=4)
    # plt.show()

    RNG = np.random.default_rng(seed=1)
    
    params = np.array([0, 8, 1], dtype=np.float64)
    
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
        RNG.normal(0, 1, 50),
        RNG.normal(-2, 1, 50), 
        RNG.normal(2, 1, 50), 
        RNG.normal(4, 1, 50), 
        RNG.normal(0, 1, 50),
        RNG.normal(-2, 1, 50), 
        RNG.normal(2, 1, 50), 
        RNG.normal(4, 1, 50),
        RNG.normal(0, 1, 50),
        RNG.normal(-2, 1, 50), 
        RNG.normal(2, 1, 50), 
        RNG.normal(4, 1, 50),
    ])

    # y = y[:100]

    # t1 = perf_counter()
    # log_preds = get_cum_log_pred(y[:400], params, n_iter=150, burn_in=20, n_particles=75)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')
    # print(log_preds[-1])
    # plt.plot(log_preds)
    # plt.show()

    t1 = perf_counter()
    U, clusters, pred_density = sample(y, n_samples=10, M=30, burn_in=0, n_particles=4000, params=params, y_pred = 0.1)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    print(pred_density)

    plot_data(y, U, clusters, max_cluster=10)
