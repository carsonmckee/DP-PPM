import math
import numpy as np 
import matplotlib.pyplot as plt
from numba import njit, prange
from typing import List
from numba import njit, types, typed

logpi = np.log(2*np.pi)

import numpy as np
import math
import random

# @njit(fastmath=False)
# def conditional_SOR(part_inds, weights, N, keep_particle):
#     w_sum = np.sum(weights)

#     cum_sum = np.empty(len(weights) + 1)
#     cum_sum[0] = 0.0

#     keep_ind = 0
#     for i in range(len(weights)):
#         cum_sum[i + 1] = cum_sum[i] + weights[i] / w_sum
#         if part_inds[i] == keep_particle:
#             keep_ind = i

#     V_ = random.uniform(cum_sum[keep_ind], cum_sum[keep_ind + 1])
#     Vp = V_ - math.floor(N * V_) / N

#     resampled = [0]*N

#     ind = 0
#     r = 0
#     while r < N:
#         if cum_sum[ind] < Vp < cum_sum[ind + 1]:
#             resampled[r] = part_inds[ind]
#             r += 1
#             Vp += 1.0 / N
#         else:
#             ind += 1

#     return resampled

# @njit(fastmath=False)
# def SOR(part_inds, weights, N):
#     w_sum = np.sum(weights)

#     cum_sum = np.empty(len(weights) + 1)
#     cum_sum[0] = 0.0

#     for i in range(len(weights)):
#         cum_sum[i + 1] = cum_sum[i] + weights[i] / w_sum

#     Vp = random.uniform(0.0, 1.0 / N)

#     resampled = [0]*N

#     ind = 0
#     r = 0
#     while r < N:
#         if cum_sum[ind] < Vp < cum_sum[ind + 1]:
#             resampled[r] = part_inds[ind]
#             r += 1
#             Vp += 1.0 / N
#         else:
#             ind += 1

#     return resampled

# @njit(fastmath=False)
# def resample(n_particles, weights, keep_particle, first_run):
#     EPS = 1e-299
#     N = len(weights)

#     weights = np.asarray(weights)
#     weights[weights == 0] = EPS

#     # Partial partition: top n_particles at end
#     partitioned = np.argpartition(weights, N - n_particles)

#     A = partitioned[N - n_particles:]

#     # Sort only top n_particles descending
#     A = A[np.argsort(weights[A])[::-1]]

#     B = partitioned[:N - n_particles]

#     A_parts = []
#     A_weights = []

#     B_parts = []
#     B_weights = []

#     B_sum = 0.0
#     A_sum = float(N)
#     in_B = False

#     for idx in B:
#         B_parts.append(idx)
#         B_weights.append(weights[idx])
#         B_sum += weights[idx]
#         A_sum -= 1

#         if idx == keep_particle:
#             in_B = True

#     A = list(A)

#     while len(A) > 1:
#         smallest_idx = A[-1]
#         w_current = weights[smallest_idx]
#         w_next = weights[A[-2]]

#         if ((B_sum + w_next) + w_next * (A_sum - 1)) <= w_next * n_particles:
#             break

#         B_parts.append(smallest_idx)
#         B_weights.append(w_current)
#         B_sum += w_current
#         A_sum -= 1

#         if smallest_idx == keep_particle:
#             in_B = True

#         A.pop()

#     A_weight_sum = 0.0
#     for idx in A:
#         A_parts.append(idx)
#         A_weights.append(weights[idx])
#         A_weight_sum += weights[idx]

#     K = len(A_parts)

#     resampled_parts = list(A_parts)
#     resampled_weights = list(A_weights)

#     remaining = n_particles - K

#     if remaining > 0:
#         if in_B and not first_run:
#             sampled = conditional_SOR(
#                 np.array(B_parts),
#                 np.array(B_weights),
#                 remaining,
#                 keep_particle,
#             )
#         else:
#             sampled = SOR(
#                 np.array(B_parts),
#                 np.array(B_weights),
#                 remaining,
#             )

#         resampled_parts.extend(sampled)

#         uniform_weight = (1.0 - A_weight_sum) / remaining
#         resampled_weights.extend([uniform_weight] * remaining)

#     return resampled_parts, resampled_weights

@njit(fastmath=False)
def log_g(sum_yy, sum_xy, sum_xx, n, params, loglam, lgammaa, logb):
    if n == 0:
        return 0.0
    
    lam = params[0]
    a = params[1]
    b = params[2]

    a_n = a + 0.5*n
    lam_n = lam + sum_xx
    m_n = sum_xy / lam_n
    b_n = b + 0.5*(sum_yy - m_n*m_n * lam_n)

    val = -0.5*n*logpi + 0.5*loglam - 0.5*np.log(lam_n)
    val += a*logb - lgammaa - a_n*np.log(b_n) + math.lgamma(a_n)
    
    return val

@njit(fastmath=False)
def resample(n_particles: int, weights: np.array, keep_particle: int, first_run):
    EPS = 10e-300
    weights = np.asarray(weights, dtype=np.float64)
    weights[weights == 0] = EPS
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

@njit(fastmath=False)
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

@njit(fastmath=False)
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
        if (cum_sum[ind] < Vp) and (Vp < cum_sum[ind + 1]):
            resampled.append(part_inds[ind])
            Vp += (1.0/float(N))
            r += 1
        ind += 1
    
    return resampled

@njit(fastmath=False)
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

key_type = types.int64
val_type = types.float64
item_type = types.Tuple((
    types.int64, # run length 
    types.int64, # cluster allocation
    types.float64[:], # cluster counts 
    types.float64[:, :] # summary stats 
))

key_type2 = types.UniTuple(types.float64, 3)

# value type: float64
value_type = types.float64

@njit(fastmath=False)
def transition_probs(alpha: float, n: np.array) -> np.array:
    #DP polya urn scheme

    probs = np.zeros(n.size + 1, dtype=np.float64)
    probs[:n.size] = n
    probs[-1] = alpha
    
    probs /= np.sum(probs)

    return probs

@njit(fastmath=False)
def create_new_part(prev_ss, prev_n, r, state, yjt, xjt):
    if r == 1:
        if state < prev_n.size:
            new_n = prev_n.copy()
            new_n[state] += 1

            new_ss = prev_ss.copy()
            new_ss[state, 0] += yjt*yjt
            new_ss[state, 1] += yjt*xjt
            new_ss[state, 2] += xjt*xjt
            new_ss[state, 3] += 1
            return new_n, new_ss
        else:
            new_n = np.append(prev_n, np.ones(1, dtype=np.float64))
            new_ss = np.append(prev_ss, np.array([[yjt*yjt, yjt*xjt, xjt*xjt, 1.0]], dtype=np.float64), axis=0)
            return new_n, new_ss
    else:
        new_ss = prev_ss.copy()
        new_ss[state, 0] += yjt*yjt
        new_ss[state, 1] += yjt*xjt
        new_ss[state, 2] += xjt*xjt
        new_ss[state, 3] += 1

        new_n = prev_n.copy()
        # new_n[state] += 1

        return new_n, new_ss

@njit
def resize(arr, n):
    if len(arr) < n:
        for _ in range(n - len(arr)):
            arr.append(arr[-1])
    else:
        for _ in range(len(arr) - n):
            arr.pop()

@njit(fastmath=False)
def filter_forward(n_particles: int, 
                   y: np.array, 
                   x: np.array,
                   alpha,
                   params: np.array,
                   trajectory: List[int],
                   runs: List[int], 
                   clusters: List[int],
                   cp_prob, 
                   first_run: bool, 
                   cache, 
                   y_pred):
    
    T_ = y.size

    run_particles = np.zeros((n_particles, T_), dtype=np.int64)
    state_particles = np.zeros((n_particles, T_), dtype=np.int64)
    ancestors = np.zeros((n_particles, T_), dtype=np.int64)

    curr_n = [np.zeros(1, dtype=np.float64) for i in range(n_particles)]
    prev_n = [np.zeros(1, dtype=np.float64) for i in range(n_particles)]

    curr_ss = [np.zeros((1, 1), dtype=np.float64) for i in range(n_particles)]
    prev_ss = [np.zeros((1, 1), dtype=np.float64) for i in range(n_particles)]

    curr_weights = [0.0]
    prev_weights = [0.0]

    total_inds = 0
    start_inds = [0]*n_particles
    num_prev_parts = 0

    lam = params[0]
    a = params[1]
    b = params[2]

    loglam = math.log(lam)
    lgammaa = math.lgamma(a)
    logb = math.log(b)

    for t in range(T_):
        
        if t == 0:
            # all parts in first cluster
            state_particles[0, t] = 0
            run_particles[0, t] = 1
            ancestors[0, t] = 0
            curr_n[0] = np.ones(1, dtype=np.float64)
            curr_ss[0] = np.array([[y[0]*y[0], y[0]*x[0], x[0]*x[0], 1]], dtype=np.float64)
            curr_weights = [1.0]
            num_prev_parts = 1
            total_inds = 3
            # curr_log_g[0] = log_g(y[0]*y[0], y[0]*x[0], x[0]*x[0], 1, params, loglam, lgammaa, logb)
            prev_n = curr_n.copy()
            prev_ss = curr_ss.copy()
            prev_weights = curr_weights
            continue

        new_weights = [0.0]*total_inds
        new_ancestors = [0]*total_inds
        new_clusters = [0]*total_inds
        new_rs = [0]*total_inds

        weight_sum = 0.0
        for ind in range(num_prev_parts):
            prev_weight = prev_weights[ind]
            prev_state = state_particles[ind, t-1]
            prev_run = run_particles[ind, t-1]

            ss = prev_ss[ind]
            n = prev_n[ind]
            K = n.size

            # K + 2 outcomes: 

            # no cp 
            likelihood = np.exp(log_g(ss[prev_state, 0] + y[t]*y[t], ss[prev_state, 1] + y[t]*x[t], ss[prev_state, 2] + x[t]*x[t],  ss[prev_state, 3]+1, params, loglam, lgammaa, logb) \
                        - log_g(ss[prev_state, 0], ss[prev_state, 1], ss[prev_state, 2], ss[prev_state, 3], params, loglam, lgammaa, logb))
            new_weight = prev_weight * likelihood * (1 - cp_prob)

            new_rs[start_inds[ind]] = prev_run + 1
            new_clusters[start_inds[ind]] = prev_state
            new_ancestors[start_inds[ind]] = ind
            new_weights[start_inds[ind]] = new_weight

            weight_sum += new_weight 

            # a cp
            # probs = transition_probs(alpha, n)
            norm_const = alpha + n.sum()
            for k in range(K):
                
                likelihood = np.exp(log_g(ss[k, 0]+y[t]*y[t], ss[k, 1]+y[t]*x[t], ss[k, 2]+x[t]*x[t], ss[k, 3]+1, params, loglam, lgammaa, logb) \
                        - log_g(ss[k, 0], ss[k, 1], ss[k, 2], ss[k, 3], params, loglam, lgammaa, logb))
                
                # new_weight = prev_weight * cp_prob * probs[k] * likelihood
                new_weight = prev_weight * cp_prob * (n[k]/norm_const) * likelihood

                new_rs[start_inds[ind]+k+1] = 1
                new_clusters[start_inds[ind]+k+1] = k
                new_ancestors[start_inds[ind]+k+1] = ind
                new_weights[start_inds[ind]+k+1] = new_weight

                weight_sum += new_weight 
            
            # K+1 cluster 
            likelihood = np.exp(log_g(y[t]*y[t], y[t]*x[t], x[t]*x[t], 1, params, loglam, lgammaa, logb))
            
            # new_weight = prev_weight * cp_prob * probs[K] * likelihood
            new_weight = prev_weight * cp_prob * (alpha/norm_const) * likelihood

            new_rs[start_inds[ind]+K+1] = 1
            new_clusters[start_inds[ind]+K+1] = K
            new_ancestors[start_inds[ind]+K+1] = ind
            new_weights[start_inds[ind]+K+1] = new_weight
            weight_sum += new_weight 
        
        keep_ind = None
        # normalize weights 
        for l in range(len(new_weights)):
            new_weights[l] /= weight_sum
            if (new_ancestors[l] == trajectory[t-1]) and (new_clusters[l] == clusters[t]) and (new_rs[l] == runs[t]):
                    keep_ind = l

        # new_parts = typed.List.empty_list(item_type)

        if len(new_weights) > n_particles:
            
            resampled_inds, resampled_weights = resample(n_particles, new_weights, keep_ind, first_run)
            for q in range(len(resampled_inds)):
                if resampled_inds[q] == keep_ind:
                    resampled_inds[q], resampled_inds[trajectory[t]] = resampled_inds[trajectory[t]], resampled_inds[q]
                    resampled_weights[q], resampled_weights[trajectory[t]] = resampled_weights[trajectory[t]], resampled_weights[q]
                    break
            
            for l in range(n_particles):
                ind = resampled_inds[l]
                ancestor_ind = new_ancestors[ind]
                state = new_clusters[ind]
                r = new_rs[ind]
                prev_ss_ = prev_ss[ancestor_ind]
                prev_n_ = prev_n[ancestor_ind]

                # print(l)
                new_n, new_ss = create_new_part(prev_ss_, prev_n_, r, state, y[t], x[t])
                
                curr_n[l] = new_n
                curr_ss[l] = new_ss

                ancestors[l, t] = ancestor_ind
                state_particles[l, t] = state 
                run_particles[l, t] = r

            num_prev_parts = n_particles
            total_inds = 0
            for l in range(n_particles):
                if l>=1:
                    start_inds[l] = total_inds 
                total_inds += curr_n[l].size + 2
            
            curr_weights = resampled_weights

        else:
            total_inds = 0
            num_prev_parts = len(new_weights)
            for l in range(len(new_weights)):
                ancestor_ind = new_ancestors[l]
                state = new_clusters[l]
                r = new_rs[l]
                prev_ss_ = prev_ss[ancestor_ind]
                prev_n_ = prev_n[ancestor_ind]

                new_n, new_ss = create_new_part(prev_ss_, prev_n_, r, state, y[t], x[t])
                
                curr_n[l] = new_n
                curr_ss[l] = new_ss

                ancestors[l, t] = ancestor_ind
                state_particles[l, t] = state 
                run_particles[l, t] = r

                if l>=1:
                    start_inds[l] = total_inds 
                total_inds += new_n.size + 2
                curr_weights = new_weights
        
        if t != (T_-1):
            curr_n, prev_n = prev_n, curr_n
            curr_ss, prev_ss = prev_ss, curr_ss
            curr_weights, prev_weights = prev_weights, curr_weights

    # pred estimate
    y_pred_est = 0.0
    for l in range(num_prev_parts):
        y_pred_est += curr_weights[l]*pred(y_pred, y[-1], alpha, cp_prob, state_particles[l, T_-1], curr_ss[l], curr_n[l], params)

    return state_particles, run_particles, ancestors, curr_weights, curr_n, curr_ss, y_pred_est

@njit(fastmath=False)
def sample_ancestors(state_particles, run_particles, ancestors, curr_weights):
    _, T_ = state_particles.shape
    trajectory = [0]*T_

    clusters_out = [0]*T_
    runs_out = [0]*T_

    for t in range(T_-1, -1, -1):
        
        if t == (T_-1):
            # ind = np.argmax(weights[-1])
            ind = multinomial_sample(curr_weights)
            # print(particle_history[-1][ind])
        else:
            ind = ancestors[ind, t+1]
        
        trajectory[t] = ind
        # runs_out[t] = particle_history[t][ind][0]
        runs_out[t] = run_particles[ind, t]
        # clusters_out[t] = particle_history[t][ind][1]
        clusters_out[t] = state_particles[ind, t]
    
    return trajectory, runs_out, clusters_out

@njit(fastmath=False)
def pred(y, x, alpha, cp_prob, prev_cluster, SS, n, params):

    # n = curr_part[2]
    K = n.size
    
    # SS = curr_part[3]
    # prev_cluster = curr_part[1]
    pred_est = 0.0

    lam = params[0]
    a = params[1]
    b = params[2]

    loglam = math.log(lam)
    lgammaa = math.lgamma(a)
    logb = math.log(b)

    # no cp
    likelihood = np.exp(log_g(SS[prev_cluster, 0] + y*y, SS[prev_cluster, 1] + y*x, SS[prev_cluster, 2] + x*x, SS[prev_cluster, 3]+1, params, loglam, lgammaa, logb) \
                            - log_g(SS[prev_cluster, 0], SS[prev_cluster, 1], SS[prev_cluster, 2], SS[prev_cluster, 3], params, loglam, lgammaa, logb))

    pred_est += (1 - cp_prob) * likelihood

    # cp
    probs = transition_probs(alpha, n)
    for k in range(K):
        likelihood = np.exp(log_g(SS[k, 0] + y*y, SS[k, 1] + y*x, SS[k, 2] + x*x, SS[k, 3]+1, params, loglam, lgammaa, logb) \
                            - log_g(SS[k, 0], SS[k, 1], SS[k, 2], SS[k, 3], params, loglam, lgammaa, logb))
        pred_est += likelihood * cp_prob * probs[k]
    
    likelihood = np.exp(log_g(y*y, y*x, x*x, 1, params, loglam, lgammaa, logb))
    pred_est += likelihood * cp_prob * probs[K]

    return pred_est

@njit(fastmath=False)
def sample_p(U: np.array):

    T_ = U.size
    n = np.sum(U) - 1
    out = np.random.beta(n + 1, T_-n)
    while out > 0.2:
        out = np.random.beta(n + 1, T_-n)
    return out

@njit(fastmath=False)
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

key_type = types.unicode_type
value_type = types.ListType(types.float64)

@njit(fastmath=False, error_model="numpy")
def sample_post(sum_yy, sum_xy, sum_xx, n, params):
    lam = params[0]
    a = params[1]
    b = params[2]
    
    a_n = a + 0.5*n
    lam_n = lam + sum_xx
    m_n = sum_xy / lam_n
    b_n = b + 0.5*(sum_yy - m_n*m_n * lam_n)

    sigma2 = 1.0/np.random.gamma(a_n, 1/b_n)
    mu = np.random.normal(m_n, np.sqrt(sigma2/lam_n))
    
    return mu, sigma2

@njit(fastmath=False, error_model="numpy")
def sample_phis_sigma2s(states, y, x, params):
    K = len(set(states))

    phis = np.zeros(K, dtype=np.float64)
    sigma2s = np.zeros(K, dtype=np.float64)

    for k in range(K):
        temp = np.where(states == k)
        y_temp = y[temp]
        x_temp = x[temp]
        yy = np.sum(y_temp*y_temp)
        yx = np.sum(y_temp*x_temp) 
        xx = np.sum(x_temp*x_temp)
        phi, s2 = sample_post(yy, yx, xx, len(y_temp), params)
        phis[k] = phi
        sigma2s[k] = s2
    
    return phis, sigma2s

@njit
def acf_lag(x, lag):
    n = len(x)
    
    # Compute mean
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n

    # Compute variance (denominator)
    var = 0.0
    for i in range(n):
        diff = x[i] - mean
        var += diff * diff

    # Compute covariance at lag
    cov = 0.0
    for i in range(n - lag):
        cov += (x[i] - mean) * (x[i + lag] - mean)

    # Normalize
    if var == 0.0:
        return 0.0
    
    return cov / var

@njit(fastmath=True)
def test_statistics(y_sim, HPPC):
        
    HPPC['acf1'].append(acf_lag(y_sim, 1))
    HPPC['acf2'].append(acf_lag(y_sim, 2))
    HPPC['var'].append(np.var(y_sim))

@njit(fastmath=True)
def get_hppc(h, HPPC, y, x, states, alpha, cp_prob, curr_n, params):
    
    n = curr_n.copy()

    y_sim = np.zeros(h, dtype=np.float64)
    phis, s2s = sample_phis_sigma2s(states, y, x, params)

    curr_state = int(states[-1])
    phi, sigma2 = phis[curr_state], s2s[curr_state]

    # simulate h steps forward
    for t in range(h):
        # first sim change-point
        u = np.random.uniform(0, 1)
        if u < cp_prob:
            # change-point
            probs = np.zeros(n.size + 1, dtype=np.float64)
            probs[:n.size] = n
            probs[-1] = alpha 
            probs /= np.sum(probs)
            curr_state = multinomial_sample(probs)
            if curr_state == n.size:
                new_phi, new_s2 = sample_post(0, 0, 0, 0, params)
                phis = np.append(phis, np.array([new_phi], dtype=np.float64))
                s2s = np.append(s2s, np.array([new_s2], dtype=np.float64))
                n = np.append(n, np.zeros(1, dtype=np.float64))
            phi, sigma2 = phis[curr_state], s2s[curr_state]
            n[curr_state] += 1
            
        else:
            # no change-point
            ... 
        
        if t == 0:
            # simulate data
            y_sim[t] = np.random.normal(phi*y[-1], np.sqrt(sigma2))
        else:
            y_sim[t] = np.random.normal(phi*y_sim[t-1], np.sqrt(sigma2))

    # compute test stats
    test_statistics(y_sim, HPPC)

@njit(fastmath=False, error_model="numpy")
def sample(y: np.array, 
           n_samples: int, 
           burn_in: int,
           n_particles: int, 
           params: np.array, 
           y_pred: float=0.0, 
           a: float =2.0,
           b: float =2.5,
           h_pred: int = 1,
           get_resids: bool=False,
           verbose: bool=False):
    
    x = y[:y.size-1]
    y = y[1:]
    
    T_ = y.size
    
    cache = typed.Dict.empty(
        key_type=key_type2,
        value_type=value_type,
        )

    trajectory = [0]*T_
    clusters = [0]*T_
    
    runs = [1]*T_ 
    first_run = True

    alpha = 1.0
    cp_prob = 1.0/50.0

    U = np.zeros(T_, dtype=np.float64)
    cluster_means = np.zeros(T_, dtype=np.float64)

    HPPC = typed.Dict.empty(
        key_type=key_type,
        value_type=value_type,
    )

    HPPC['acf1'] = typed.List.empty_list(types.float64)
    HPPC['acf2'] = typed.List.empty_list(types.float64)
    HPPC['var'] = typed.List.empty_list(types.float64)

    residuals = []

    n_state_store = np.zeros(n_samples-burn_in)
    pred_tot = (n_samples - burn_in)
    y_pred_mean = 0.0
    for it in range(n_samples):
        # print(it)
        
        state_particles, run_particles, ancestors, curr_weights, curr_n, curr_ss, pred_est = filter_forward(n_particles, y, x, alpha, params, trajectory, runs, clusters, cp_prob, first_run, cache, y_pred)
        first_run = False

        trajectory, runs, clusters = sample_ancestors(state_particles, run_particles, ancestors, curr_weights)
        
        n_states = len(set(clusters))

        U_temp = (np.array(runs, dtype=np.float64) == 1)

        # update cp_prob and alpha
        cp_prob = sample_p(U_temp)

        alpha = update_dp_alpha(alpha, np.sum(U_temp), n_states, a, b)

        c_temp = np.array(clusters, dtype=np.int64)
        if it >= burn_in:
            # U += U_temp
            U[1:] += (c_temp[0:len(c_temp)-1] != c_temp[1:])
            cluster_means += c_temp.astype(np.float64)
            n_state_store[it-burn_in] = curr_n[trajectory[-1]].size
            get_hppc(h_pred, HPPC, y, x, c_temp, alpha, cp_prob, curr_n[trajectory[-1]], params)
            if not np.isnan(pred_est):
                y_pred_mean += pred_est
            else:
                print('warning, nan')
                pred_tot -= 1
            
            if get_resids:
                resids = [0.0]*T_
                phis, sigma2s = sample_phis_sigma2s(c_temp, y, x, params)
                for t in range(T_):
                    resids[t] = (y[t] - x[t]*phis[c_temp[t]]) / np.sqrt(sigma2s[c_temp[t]])
                residuals.append(resids)
        
        if verbose:
            print(f'Iter: {it}, Number of states: {n_states}, cp_prob:', round(cp_prob, 4), ', alpha: ', round(alpha, 4), ', pred_density: ', pred_est)

    U /= (n_samples - burn_in)
    cluster_means /= (n_samples - burn_in)
    y_pred_mean /= pred_tot

    return HPPC, U, residuals, n_state_store, c_temp, y_pred_mean

@njit(fastmath=False)
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

def sim_ar_process(T_, phis, variances, cp_prob, cat_probs, RNG):
    
    out = np.zeros(T_+1)
    out[0] = RNG.normal(0, 1)
    phi = None
    variance = None
    clusters = np.zeros(T_)
    for t in range(1, T_ + 1):
        if (RNG.uniform(0, 1) < cp_prob) or (t == 1):
            # change-point
            ind = RNG.choice(len(phis), p = cat_probs)
            phi = phis[ind]
            variance = variances[ind]
        clusters[t-1] = ind
        out[t] = RNG.normal(phi*out[t-1], np.sqrt(variance))

    return out[1:], clusters

def plot_data(y: np.array, U, clusters: np.array, max_cluster=6, true_clusters=None) -> None:
    T_ = len(clusters)
    if true_clusters is None:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig, axs = plt.subplots(3, 1, sharex=True)
    
    x_, y_ = np.arange(1, T_+1), y[(y.size - T_):]
    for i in range(len(y_) - 1):
        color = plt.cm.tab10(clusters[i] / max_cluster)  # Normalize the value for colormap
        axs[0].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)
    
    axs[1].bar(x_, U, width=0.8, color='black')

    if true_clusters is not None:
        for i in range(len(true_clusters) - 1):
            color = plt.cm.tab10(true_clusters[i] / max_cluster)  # Normalize the value for colormap
            axs[2].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)

    plt.show()

def save_hppc(hppc, hppc_true, t, h, base=None):
    if base:
        path = f'{base}/t_bill_results/hppc/{t}_{h}'
    else:
        path = f't_bill_results/hppc/{t}_{h}'
    
    for key in hppc.keys():
        np.savetxt(f'{path}_{key}.csv', np.array(hppc[key]), delimiter=',')
        np.savetxt(f'{path}_{key}_true.csv', np.array(hppc_true[key]), delimiter=',')
    print('saved hppcs')

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
                _, _, pred = sample(y[j, :], n_samples=n_samples, burn_in=burn_in, params=params, y_pred=y_pred[j])
                error = False
            except Exception as ex:
                print(ex)
                print('retrying')

        if preds is None:
            preds = pred 
        else:
            preds *= pred

    return np.mean(preds)

if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    from time import perf_counter
    import pandas as pd 
    
    params = np.array([1, 1, 1], dtype=np.float64)

    # RNG = np.random.default_rng(seed=1)
    # y, true_clusters = sim_ar_process(500, np.array([0.9, 0, -0.9, 0]), np.array([0.25, 1, 0.25, 3]), 1.0/75.0, np.ones(4)/4.0, RNG)
    
    dat = pd.read_csv("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/WTB3MS.csv", header=None)
    y = dat[1].diff().to_numpy()[1:]
    y = (y - np.mean(y)) / np.sqrt(np.var(y))
    
    t_pred = len(y) - 2
    # t_pred = 1200
    h_pred = 25
    a, b = 2.0, 2.5
    
    print((a, b))
    
    # for t_pred in [600, 1200, 1800, 2400, 3000]:
    #     print(f't_pred: {t_pred}')
    #     t1 = perf_counter()
    #     HPPC, U, n_states, clusters, pred_density = sample(y[:t_pred], 
    #                                             n_samples=1500, 
    #                                             burn_in=100, 
    #                                             n_particles=400, 
    #                                             params=params, 
    #                                             y_pred = y[t_pred], 
    #                                             a=a, 
    #                                             b=b,
    #                                             h_pred=h_pred,
    #                                             verbose=True)
    #     t2 = perf_counter()
    #     print(f'Took {round(t2-t1, 2)}s')
    #     print(pred_density)

    #     save_hppc(HPPC, {'acf1':[], 'acf2':[], 'var':[]}, t_pred, h_pred, base="C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models")
    
    # plot_data(y, U, clusters, max_cluster=6)

    # np.savetxt("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/t_bill_results/npcp_single_state_sample.csv", clusters, delimiter=',')
    # np.savetxt("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/t_bill_results_clean/npcp_n_states.csv", n_states, delimiter=',')
    
    t_pred = len(y) - 2
    print(f't_pred: {t_pred}')
    t1 = perf_counter()
    HPPC, U, residuals, n_states, clusters, pred_density = sample(y[:t_pred], 
                                                                    n_samples=3000, 
                                                                    burn_in=200, 
                                                                    n_particles=1000, 
                                                                    params=params, 
                                                                    y_pred = y[t_pred], 
                                                                    a=a, 
                                                                    b=b,
                                                                    verbose=True, 
                                                                    get_resids=False)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    print(pred_density)
    
    # np.savetxt("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/t_bill_results/npcp_residuals.csv", np.array(residuals), delimiter=',')
    np.savetxt("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/t_bill_results_clean/npcp_n_states.csv", n_states, delimiter=',')
    
    print('done')

    