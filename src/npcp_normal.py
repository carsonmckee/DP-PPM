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
        if (cum_sum[ind] < Vp) and (Vp < cum_sum[ind + 1]):
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

key_type = types.int64
val_type = types.float64
item_type = types.Tuple((
    types.int64, # run length 
    types.int64, # cluster allocation
    types.float64[:], # cluster counts 
    types.float64[:, :] # summary stats 
))

@njit(fastmath=True)
def transition_probs(alpha: float, n: np.array) -> np.array:
    #DP polya urn scheme

    probs = np.zeros(n.size + 1, dtype=np.float64)
    probs[:n.size] = n
    probs[-1] = alpha
    
    probs /= np.sum(probs)

    return probs

# @njit(fastmath=True)
# def create_new_part(prev_part, r, cluster, yjt):
#     if r == 1:
#         if cluster < prev_part[2].shape[0]:
#             new_part = (r, cluster, prev_part[2].copy(), prev_part[3].copy())
#             new_part[2][cluster] += 1
#             new_part[3][cluster, 0] += yjt
#             new_part[3][cluster, 1] += yjt*yjt
#             new_part[3][cluster, 2] += 1
#         else:
#             prev_n = np.append(prev_part[2], np.ones(1, dtype=np.float64))
#             prev_ss = np.append(prev_part[3], np.array([[yjt, yjt*yjt, 1.0]], dtype=np.float64), axis=0)
#             new_part = (r, cluster, prev_n, prev_ss)
#     else:
#         new_part = (r, cluster, prev_part[2].copy(), prev_part[3].copy())
#         new_part[3][cluster, 0] += yjt
#         new_part[3][cluster, 1] += yjt*yjt
#         new_part[3][cluster, 2] += 1

#     return new_part

@njit(fastmath=False)
def create_new_part(prev_ss, prev_n, r, state, yjt):
    if r == 1:
        if state < prev_n.size:
            new_n = prev_n.copy()
            new_n[state] += 1

            new_ss = prev_ss.copy()
            new_ss[state, 0] += yjt
            new_ss[state, 1] += yjt*yjt
            new_ss[state, 2] += 1
            return new_n, new_ss
        else:
            new_n = np.append(prev_n, np.ones(1, dtype=np.float64))
            new_ss = np.append(prev_ss, np.array([[yjt, yjt*yjt, 1.0]], dtype=np.float64), axis=0)
            return new_n, new_ss
    else:
        new_ss = prev_ss.copy()
        new_ss[state, 0] += yjt
        new_ss[state, 1] += yjt*yjt
        new_ss[state, 2] += 1
        return prev_n.copy(), new_ss

@njit(fastmath=False)
def filter_forward(n_particles: int, 
                   y: np.array, 
                   alpha,
                   params: np.array,
                   trajectory: List[int],
                   runs: List[int], 
                   clusters: List[int],
                   cp_prob, 
                   first_run: bool, 
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
            curr_ss[0] = np.array([[y[0], y[0]*y[0], 1]], dtype=np.float64)
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
            likelihood = np.exp(log_g(ss[prev_state, 0] + y[t], ss[prev_state, 1] + y[t]*y[t],  ss[prev_state, 2]+1, params) \
                        - log_g(ss[prev_state, 0], ss[prev_state, 1], ss[prev_state, 2], params))
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
                
                likelihood = np.exp(log_g(ss[k, 0]+y[t], ss[k, 1]+y[t]*y[t], ss[k, 2]+1, params) \
                        - log_g(ss[k, 0], ss[k, 1], ss[k, 2], params))
                
                # new_weight = prev_weight * cp_prob * probs[k] * likelihood
                new_weight = prev_weight * cp_prob * (n[k]/norm_const) * likelihood

                new_rs[start_inds[ind]+k+1] = 1
                new_clusters[start_inds[ind]+k+1] = k
                new_ancestors[start_inds[ind]+k+1] = ind
                new_weights[start_inds[ind]+k+1] = new_weight

                weight_sum += new_weight 
            
            # K+1 cluster 
            likelihood = np.exp(log_g(y[t], y[t]*y[t], 1, params))
            
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
                new_n, new_ss = create_new_part(prev_ss_, prev_n_, r, state, y[t])
                
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

                new_n, new_ss = create_new_part(prev_ss_, prev_n_, r, state, y[t])
                
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
        y_pred_est += curr_weights[l]*pred(y_pred, alpha, cp_prob, state_particles[l, T_-1], curr_ss[l], curr_n[l], params)

    return state_particles, run_particles, ancestors, curr_weights, curr_n, curr_ss, y_pred_est

# @njit(fastmath=True, parallel=True)
# def filter_forward(n_particles: int, 
#                    y: np.array, 
#                    alpha,
#                    params: np.array,
#                    trajectory: List[int],
#                    runs: List[int], 
#                    clusters: List[int],
#                    cp_prob, 
#                    first_run: bool):
    
#     T_ = y.size

#     particles = []
#     weights = []
#     ancestors = []
#     total_inds = 0
#     start_inds = [0]*n_particles
#     for t in range(T_):
        
#         if t == 0:
#             # all parts in first cluster
#             new_parts = typed.List.empty_list(item_type)
#             new_parts.append((1, 0, np.ones(1, dtype=np.float64), np.array([[y[0], y[0]*y[0], 1]], dtype=np.float64)))
#             particles.append(new_parts)
#             weights.append([1.0])
#             ancestors.append([0])
#             total_inds = 3
#             continue
        
#         new_weights = [0.0]*total_inds
#         new_ancestors = [0]*total_inds
#         new_clusters = [0]*total_inds
#         new_rs = [0]*total_inds

#         prev_parts = particles[-1]
#         prev_weights = weights[-1]

#         weight_sum = 0.0
#         for ind in prange(len(prev_parts)):
#             prev_part = prev_parts[ind]
#             prev_weight = prev_weights[ind]
#             prev_cluster = prev_part[1]
#             prev_r = prev_part[0]
#             K = prev_part[2].size

#             # K + 2 outcomes: 

#             # no cp 
            
#             likelihood = np.exp(log_g(prev_part[3][prev_cluster, 0] + y[t], prev_part[3][prev_cluster, 1] + y[t]*y[t], prev_part[3][prev_cluster, 2] + 1, params) \
#                         - log_g(prev_part[3][prev_cluster, 0], prev_part[3][prev_cluster, 1], prev_part[3][prev_cluster, 2], params))
#             new_weight = prev_weight * likelihood * (1 - cp_prob)
#             # new_rs.append(prev_r + 1)
#             # new_clusters.append(prev_cluster)
#             # new_ancestors.append(ind)
#             # new_weights.append(new_weight)

#             new_rs[start_inds[ind]] = prev_r + 1
#             new_clusters[start_inds[ind]] = prev_cluster
#             new_ancestors[start_inds[ind]] = ind
#             new_weights[start_inds[ind]] = new_weight

#             weight_sum += new_weight 

#             # a cp
#             probs = transition_probs(alpha, prev_part[2])
#             for k in range(K):
                
#                 likelihood = np.exp(log_g(prev_part[3][k, 0] + y[t], prev_part[3][k, 1] + y[t]*y[t], prev_part[3][k, 2] + 1, params) \
#                         - log_g(prev_part[3][k, 0], prev_part[3][k, 1], prev_part[3][k, 2], params))
                
#                 new_weight = prev_weight * cp_prob * probs[k] * likelihood
#                 # new_rs.append(1)
#                 # new_clusters.append(k)
#                 # new_ancestors.append(ind)
#                 # new_weights.append(new_weight)

#                 new_rs[start_inds[ind]+k+1] = 1
#                 new_clusters[start_inds[ind]+k+1] = k
#                 new_ancestors[start_inds[ind]+k+1] = ind
#                 new_weights[start_inds[ind]+k+1] = new_weight

#                 weight_sum += new_weight 
            
#             # K+1 cluster 
#             likelihood = np.exp(log_g(y[t], y[t]*y[t], 1, params))
            
#             new_weight = prev_weight * cp_prob * probs[K] * likelihood
#             # new_rs.append(1)
#             # new_clusters.append(K)
#             # new_ancestors.append(ind)
#             # new_weights.append(new_weight)

#             new_rs[start_inds[ind]+K+1] = 1
#             new_clusters[start_inds[ind]+K+1] = K
#             new_ancestors[start_inds[ind]+K+1] = ind
#             new_weights[start_inds[ind]+K+1] = new_weight

#             weight_sum += new_weight 
        
#         keep_ind = None
#         # normalize weights 
#         for l in range(len(new_weights)):
#             new_weights[l] /= weight_sum
#             if (new_ancestors[l] == trajectory[t-1]) and (new_clusters[l] == clusters[t]) and (new_rs[l] == runs[t]):
#                     keep_ind = l

#         new_parts = typed.List.empty_list(item_type)

#         if len(new_weights) > n_particles:
            
#             resampled_inds, resampled_weights = resample(n_particles, new_weights, keep_ind, first_run)
#             for q in range(len(resampled_inds)):
#                 if resampled_inds[q] == keep_ind:
#                     resampled_inds[q], resampled_inds[trajectory[t]] = resampled_inds[trajectory[t]], resampled_inds[q]
#                     resampled_weights[q], resampled_weights[trajectory[t]] = resampled_weights[trajectory[t]], resampled_weights[q]
#                     break
            
#             resampled_ancestors = []
#             total_inds = 0
#             for l in range(n_particles):
#                 ind = resampled_inds[l]
#                 ancestor_ind = new_ancestors[ind]
#                 cluster = new_clusters[ind]
#                 r = new_rs[ind]
#                 prev_part = prev_parts[ancestor_ind]
                
#                 new_part = create_new_part(prev_part, r, cluster, y[t])
#                 new_parts.append(new_part)
#                 resampled_ancestors.append(ancestor_ind)
#                 if l >= 1:
#                     start_inds[l] = total_inds 
#                 total_inds += new_part[2].size + 2
            
#             particles.append(new_parts)
#             weights.append(resampled_weights)
#             ancestors.append(resampled_ancestors)

#         else:
#             total_inds = 0
#             for l in range(len(new_weights)):
#                 ancestor_ind = new_ancestors[l]
#                 cluster = new_clusters[l]
#                 r = new_rs[l]
#                 prev_part = prev_parts[ancestor_ind]
#                 new_part = create_new_part(prev_part, r, cluster, y[t])
#                 new_parts.append(new_part)
#                 if l >= 1:
#                     start_inds[l] = total_inds 
#                 total_inds += new_part[2].size + 2
            
#             particles.append(new_parts)
#             weights.append(new_weights)
#             ancestors.append(new_ancestors)

#     return particles, weights, ancestors

# @njit(fastmath=True)
# def sample_ancestors(particle_history, weight_history, ancestor_history):
    # T_ = len(particle_history)
    # trajectory = [0]*T_

    # clusters_out = [0]*T_
    # runs_out = [0]*T_

    # for t in range(T_-1, -1, -1):
        
    #     if t == (T_-1):
    #         # ind = np.argmax(weights[-1])
    #         ind = multinomial_sample(weight_history[t])
    #         # print(particle_history[-1][ind])
    #     else:
    #         ind = ancestor_history[t+1][ind]
        
    #     trajectory[t] = ind
    #     runs_out[t] = particle_history[t][ind][0]
    #     clusters_out[t] = particle_history[t][ind][1]
    
    # return trajectory, runs_out, clusters_out

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

@njit(fastmath=True)
def pred(y, alpha, cp_prob, prev_cluster, SS, n, params):

    K = n.size
    
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
           burn_in: int,
           n_particles: int, 
           params: np.array, 
           y_pred: float=0.0, 
           verbose=False):
        
    T_ = y.size
    
    trajectory = [0]*T_
    clusters = [0]*T_
    
    runs = [1]*T_ 
    first_run = True

    alpha = 1.0
    cp_prob = 1.0/450.0

    U = np.zeros(T_, dtype=np.float64)
    cluster_means = np.zeros(T_, dtype=np.float64)

    y_pred_mean = 0.0
    pred_density = 0.0
    for it in range(n_samples):
        # if verbose:
        #     print(it)

        state_particles, run_particles, ancestors, curr_weights, curr_n, curr_ss, pred_est = filter_forward(n_particles, y, alpha, params, trajectory, runs, clusters, cp_prob, first_run, y_pred)
        first_run = False

        trajectory, runs, clusters = sample_ancestors(state_particles, run_particles, ancestors, curr_weights)
        
        U_temp = (np.array(runs, dtype=np.float64) == 1)

        n_states = len(set(clusters))

        # update cp_prob and alpha
        cp_prob = sample_p(U_temp)

        alpha = update_dp_alpha(alpha, np.sum(U_temp), n_states, 2, 2.5)

        # pred_density = pred(y_pred, alpha, cp_prob, clusters[-1], curr_ss[trajectory[-1]], curr_n[trajectory[-1]], params)
        if verbose:
            print(f'Iter: {it}, Number of states: {n_states}, cp_prob:', round(cp_prob, 4), ', alpha: ', round(alpha, 4), ', pred density: ', pred_est)

        c_temp = np.array(clusters, dtype=np.float64)
        if it >= burn_in:
            # U += U_temp
            U[1:] += (c_temp[0:len(c_temp)-1] != c_temp[1:])
            cluster_means += c_temp
            y_pred_mean += pred_est

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
    
    path = "C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/"
    well_log = pd.read_csv(f'{path}/well_log_clean.csv').to_numpy().flatten()
    
    params = np.array([0, 10, np.var(well_log[:960])], dtype=np.float64)

    t1 = perf_counter()
    U, clusters, pred_density = sample(well_log, n_samples=1, burn_in=0, n_particles=10, params=params, y_pred = 0.1)
    t2 = perf_counter()
    print('compile done')
    
    t1 = perf_counter()
    U, clusters, pred_density = sample(well_log, n_samples=10, burn_in=0, n_particles=500, params=params, y_pred = 0.1, verbose=True)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    print(pred_density)

    plot_data(well_log, U, clusters, max_cluster=int(np.max(clusters)))

    fig, axs = plt.subplots(2, 1, sharex=True) 
    axs[0].plot(well_log)
    axs[1].bar(np.arange(len(U)), U, color='black', width=4)
    plt.show()

    # RNG = np.random.default_rng(seed=1)
    
    # params = np.array([0, 4, 1], dtype=np.float64)
    
    # RNG = np.random.default_rng(seed=1)
    # y = np.concatenate([
    #     RNG.normal(0, 1, 25),
    #     RNG.normal(-2, 1, 25), 
    #     RNG.normal(2, 1, 25), 
    #     RNG.normal(4, 1, 25), 
    #     RNG.normal(0, 1, 25),
    #     RNG.normal(-2, 1, 25), 
    #     RNG.normal(2, 1, 25), 
    #     RNG.normal(4, 1, 25), 
    #     RNG.normal(0, 1, 25),
    #     RNG.normal(-2, 1, 25), 
    #     RNG.normal(2, 1, 25), 
    #     RNG.normal(4, 1, 25), 
    #     RNG.normal(0, 1, 25),
    #     RNG.normal(-2, 1, 25), 
    #     RNG.normal(2, 1, 25), 
    #     RNG.normal(4, 1, 25),
    #     RNG.normal(0, 1, 25),
    #     RNG.normal(-2, 1, 25), 
    #     RNG.normal(2, 1, 25), 
    #     RNG.normal(4, 1, 25),
    # ])

    # # y = y[:100]

    # # t1 = perf_counter()
    # # log_preds = get_cum_log_pred(y[:400], params, n_iter=150, burn_in=20, n_particles=75)
    # # t2 = perf_counter()
    # # print(f'Took {round(t2-t1, 2)}s')
    # # print(log_preds[-1])
    # # plt.plot(log_preds)
    # # plt.show()

    # t1 = perf_counter()
    # U, clusters, pred_density = sample(y, n_samples=1000, burn_in=100, n_particles=100, params=params, y_pred = 0.1)
    # t2 = perf_counter()
    # print(f'Took {round(t2-t1, 2)}s')
    # print(pred_density)

    # plot_data(y, U, clusters, max_cluster=5)
