import math
import numpy as np 
from numba import njit, prange, types, typed
from typing import List
from numba import njit


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
def resample(n_particles: int, weights: np.array, keep_particle: int, first_run):
    EPS = 10e-300
    weights = np.asarray(weights, dtype=np.float64)
    weights[weights <= 10e-300] = EPS
    resampled_parts = []
    resampled_weights = []
    
    A_parts = []
    A_weights = []
    B_parts = []
    B_weights = []
    
    in_B = False
    
    sorted_indices = np.argsort(weights)
    N = len(weights)
    left_sum = 0.0
    right_sum = float(N)
    for i in range(N-n_particles):
        left_sum += weights[sorted_indices[i]]
        right_sum -= 1
        B_parts.append(sorted_indices[i])
        B_weights.append(weights[sorted_indices[i]])
        
        if sorted_indices[i] == keep_particle: # XXX
            in_B = True
    
    for i in range(N-n_particles, N):
        left_sum += weights[sorted_indices[i]]
        B_parts.append(sorted_indices[i])
        B_weights.append(weights[sorted_indices[i]])
        right_sum -= 1
        if sorted_indices[i] == keep_particle: # XXX
            in_B = True

        if (len(B_parts) == len(weights)):
            break
        
        if ((left_sum + weights[sorted_indices[i+1]]) + weights[sorted_indices[i+1]]*(right_sum - 1)) <= weights[sorted_indices[i+1]]*n_particles:
            break
    A_weight_sum = 0.0
    for i in range(len(B_parts), N):
        A_weight_sum += weights[sorted_indices[i]]
        A_parts.append(sorted_indices[i])
        A_weights.append(weights[sorted_indices[i]])
    K = len(A_parts)
    
    resampled_parts.extend(A_parts)
    resampled_weights.extend(A_weights)
    
    if in_B and (not first_run):
        resampled_B_parts = conditional_SOR(B_parts, B_weights, n_particles-K, keep_particle)
    else:
        resampled_B_parts = SOR(B_parts, B_weights, n_particles-K)
    resampled_parts.extend(resampled_B_parts)
    resampled_weights.extend([(1-A_weight_sum)/(n_particles-K)]*(n_particles-K))
    
    return resampled_parts, resampled_weights

@njit(fastmath=True)
def conditional_SOR(part_inds, weights, N, keep_particle):
    
    resampled = []
    w_sum = 0.0
    for i in range(len(weights)):
        w_sum += weights[i]
    
    keep_ind = 0
    cum_sum = [0.0]
    for i in range(len(weights)):
        cum_sum.append(cum_sum[-1] + weights[i] / w_sum)
        if part_inds[i] == keep_particle:
            keep_ind = i
    
    ind = 0
    V_ = np.random.uniform(cum_sum[keep_ind], cum_sum[keep_ind+1])
    Vp = V_ - math.floor(N*V_) / N
    r = 0
    while r < N:
        if (cum_sum[ind] < Vp) and (Vp < cum_sum[ind + 1]):
            resampled.append(part_inds[ind])
            r += 1
            Vp += (1.0/float(N))
        ind += 1
    
    return resampled

@njit(fastmath=True)
def SOR(part_inds, weights, N):
    resampled = []
    w_sum = 0.0
    for i in range(len(weights)):
        w_sum += weights[i]
    
    cum_sum = [0.0]
    for i in range(len(weights)):
        cum_sum.append(cum_sum[-1] + weights[i] / w_sum)
    ind = 0
    r = 0
    Vp = np.random.uniform(0, 1) * (1.0/float(N))
    while r < N:
        if (cum_sum[ind] < Vp) and (Vp <= cum_sum[ind + 1]):
            resampled.append(part_inds[ind])
            Vp += (1.0/float(N))
            r += 1
        ind += 1
    return resampled

@njit(fastmath=True)
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

@njit(fastmath=True)
def filter_forward(n_particles: int, 
                   y_sums: np.array, 
                   yy_sums: np.array, 
                   params: np.array,
                   trajectory: List[int],
                   cp_prob, 
                   first_run: bool):
    
    T_ = len(trajectory)

    particles = [[1]]
    weights = [[1.0]]

    for t in range(1, T_):
        prev_parts = particles[-1]
        prev_weights = weights[-1]

        new_parts = [1]
        new_weights = [0.0]

        weight_sum = 0.0
        for ind in range(len(prev_parts)):
            prev_part = prev_parts[ind]
            prev_weight = prev_weights[ind]
            
            # change-point
            y = y_sums[t+1] - y_sums[t+1-1]
            yy = yy_sums[t+1] - yy_sums[t+1-1]
            new_weight = prev_weight * cp_prob * np.exp(log_g(y, yy, 1, params))
            new_weights[0] += new_weight 
            weight_sum += new_weight

            # no change-point
            new_part = prev_part + 1
            y1 = y_sums[t+1] - y_sums[t+1-new_part]
            yy1 = yy_sums[t+1] - yy_sums[t+1-new_part]

            y2 = y_sums[t] - y_sums[t-prev_part]
            yy2 = yy_sums[t] - yy_sums[t-prev_part]

            new_parts.append(new_part)
            new_weight = prev_weight * (1-cp_prob) * np.exp(log_g(y1, yy1, new_part, params) - log_g(y2, yy2, prev_part, params))
            new_weights.append(new_weight)
            weight_sum += new_weight

        keep_ind = None
        # normalize weights 
        for l in range(len(new_weights)):
            new_weights[l] /= weight_sum
            if trajectory[t] == new_parts[l]:
                    keep_ind = l

        if len(new_weights) > n_particles:
            resampled_inds, resampled_weights = resample(n_particles, new_weights, keep_ind, first_run)
            resampled_parts = []
            for l in range(n_particles):
                ind = resampled_inds[l]
                resampled_parts.append(new_parts[ind])
            
            particles.append(resampled_parts)
            weights.append(resampled_weights)

        else:
            
            particles.append(new_parts)
            weights.append(new_weights)
            
    return particles, weights

@njit(fastmath=True)
def sample_backwards(particle_history, weight_history):
    T_ = len(particle_history)
    trajectory = [0]*T_

    for t in range(T_-1, -1, -1):
        
        if t == (T_-1):
            # ind = np.argmax(weights[-1])
            ind = multinomial_sample(weight_history[t])
            trajectory[t] = particle_history[t][ind]
            continue 

        next_part = trajectory[t+1]
        if next_part != 1:
            trajectory[t] = next_part - 1
        else:
            ind = multinomial_sample(weight_history[t])
            trajectory[t] = particle_history[t][ind]
    
    return trajectory

@njit(fastmath=True)
def pred(y_pred, y_sums, yy_sums, cp_prob, curr_part, params):

    T_ = len(yy_sums) - 1

    pred_est = 0.0

    y = y_sums[T_] - y_sums[T_ - curr_part]
    yy = yy_sums[T_] - yy_sums[T_ - curr_part]

    # no cp
    likelihood = np.exp(log_g(y + y_pred, yy + y_pred*y_pred, curr_part+1, params) \
                      - log_g(y, yy, curr_part, params))

    pred_est += (1 - cp_prob) * likelihood

    # cp
    likelihood = np.exp(log_g(y_pred, y_pred*y_pred, 1, params))
    pred_est += cp_prob * likelihood

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
def sample(y: np.array, 
           n_samples: int, 
           burn_in: int,
           n_particles: int, 
           params: np.array, 
           y_pred: float=0.0):
        
    T_ = y.size
            
    runs = [1]*T_ 
    first_run = True

    cp_prob = 1.0/450.0

    y_sums = [0.0]
    yy_sums = [0.0]

    for t in range(T_):
        y_sums.append(y_sums[-1] + y[t])
        yy_sums.append(yy_sums[-1] + y[t]*y[t])

    U = np.zeros(T_, dtype=np.float64)
    cluster_means = np.zeros(T_, dtype=np.float64)

    y_pred_mean = 0.0
    pred_density = 0.0
    for it in range(n_samples):

        error = True
        while error:
            try:
                particles, weights = filter_forward(n_particles, y_sums, yy_sums, params, runs, cp_prob, first_run)
                first_run = False
                error = False
            except Exception:
                print('retrying filtering step')
        
        runs = sample_backwards(particles, weights)
        
        U_temp = (np.array(runs, dtype=np.float64) == 1)

        # print(np.sum(U_temp))

        # update cp_prob and alpha
        cp_prob = sample_p(U_temp)
        
        # print(f'Iter: {it}, Number of CPs:', np.sum(U_temp), ', cp_prob:', round(cp_prob, 4))

        c_temp = np.cumsum(U_temp)
        if it >= burn_in:
            U += U_temp
            cluster_means += c_temp

            pred_density = pred(y_pred, y_sums, yy_sums, cp_prob, runs[-1], params)
            y_pred_mean += pred_density
    
    U /= (n_samples - burn_in)
    cluster_means /= (n_samples - burn_in)
    y_pred_mean /= (n_samples - burn_in)

    return U, cluster_means, y_pred_mean

def plot_data(y: np.array, U, clusters: np.array, max_cluster=6, true_clusters=None) -> None:
    T_ = len(clusters)
    if true_clusters is None:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig, axs = plt.subplots(3, 1, sharex=True)
    
    x_, y_ = np.arange(1, T_+1), y[(y.size - T_):]
    for i in range(len(y_) - 1):
        color = plt.cm.tab10(clusters[i] / max_cluster)  # Normalize the value for colormap
        axs[0].plot(x_[i:i+2], y_[i:i+2],lw=1.8, color=color)
    
    axs[1].bar(x_, U, 
            #    width=0.8, 
            width = 4,
               color='black')

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
    
    path = "C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/"
    well_log = pd.read_csv(f'{path}/well_log_clean.csv').to_numpy().flatten()

    print(well_log)

    plt.plot(well_log, color='black') 
    plt.show()

    v = 2*np.var(well_log[0:900])
    params = np.array([0, 6.0, v])
    t1 = perf_counter()
    U, clusters, pred_density = sample(well_log, n_samples=200, burn_in=50, n_particles=200, params=params, y_pred = 0.1)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    print(pred_density)

    plot_data(well_log, U, clusters, max_cluster=max(clusters))

    # t1 = perf_counter()
    # U, clusters, pred_density = sample(y, n_samples=200, burn_in=50, n_particles=150, alpha=0.15, params=params, y_pred = 0.1)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')
    # print(pred_density)

    RNG = np.random.default_rng(seed=1)
    
    params = np.array([0, 4, 1], dtype=np.float64)
    
    RNG = np.random.default_rng(seed=1)
    y = np.concatenate([
        RNG.normal(0, 1, 25),
        RNG.normal(-2, 1, 25), 
        RNG.normal(2, 1, 25), 
        RNG.normal(4, 1, 25), 
        RNG.normal(0, 1, 25),
        RNG.normal(-2, 1, 25), 
        RNG.normal(2, 1, 25), 
        RNG.normal(4, 1, 25), 
        RNG.normal(0, 1, 25),
        RNG.normal(-2, 1, 25), 
        RNG.normal(2, 1, 25), 
        RNG.normal(4, 1, 25), 
        RNG.normal(0, 1, 25),
        RNG.normal(-2, 1, 25), 
        RNG.normal(2, 1, 25), 
        RNG.normal(4, 1, 25),
    ])

    RNG = np.random.default_rng(seed=1)
    y = np.concatenate([
        RNG.normal(0, 1, 50),
        RNG.normal(-2, 1, 50), 
        RNG.normal(2, 1, 50), 
        RNG.normal(4, 1, 50), 

    ])

    # t1 = perf_counter()
    # log_preds = get_cum_log_pred(y[:400], params, n_iter=150, burn_in=20, n_particles=75)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')
    # print(log_preds[-1])
    # plt.plot(log_preds)
    # plt.show()

    # t1 = perf_counter()
    # U, clusters, pred_density = sample(y, n_samples=200, burn_in=100, n_particles=500, params=params, y_pred = 0.1)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')
    # print(pred_density)

    # plot_data(y, U, clusters, max_cluster=10)