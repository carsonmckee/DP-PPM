import numpy as np 
from distributer import save_data

NORMAL_PARAM = np.array([0, 6, 0.5])
AR_PARAMS = np.array([1, 1, 1])

T_ = 850

def generate_yao_cps(RNG, cp_prob):
    
    out = (RNG.uniform(0, 1, size=T_+1) < cp_prob).astype(np.int64)
    out[0] = 1
    
    return out

def generate_HMM_states(RNG, P, inital_dist):
    
    K, _ = P.shape

    out = np.zeros(T_+1, dtype=np.int64)
    out[0] = RNG.choice(K, p=inital_dist)
    for t in range(1, T_+1):
        out[t] = RNG.choice(K, p=P[out[t-1], :])

    return out

def normal_one(RNG):
    # Yao + independent samples
    cp_prob = 1.0/50.0
    
    U = generate_yao_cps(RNG, cp_prob)
    out = np.zeros(T_)
    curr_mean = None
    for t in range(T_):
        if U[t] == 1:
            curr_mean = RNG.normal(NORMAL_PARAM[0], np.sqrt(NORMAL_PARAM[1]))
        out[t] = RNG.normal(curr_mean, np.sqrt(NORMAL_PARAM[2]))
    return out

def normal_two(RNG):
    # Yao + random switch between 10 states
    cp_prob = 1.0/50.0
    K = 10
    U = generate_yao_cps(RNG, cp_prob)
    out = np.zeros(T_)
    means = RNG.normal(NORMAL_PARAM[0], np.sqrt(NORMAL_PARAM[1]), size=K)
    for t in range(T_):
        if U[t] == 1:
            ind = RNG.choice(K)
            curr_mean = means[ind]
        out[t] = RNG.normal(curr_mean, np.sqrt(NORMAL_PARAM[2]))
    return out

def normal_three(RNG):
    # HMM with 4 states
    K = 4
    initial_dist = np.ones(K)/K
    P = np.zeros((K, K))
    for i in range(K):
        alpha = np.ones(K)
        alpha[i] += 100
        P[i, :] = RNG.dirichlet(alpha)

    states = generate_HMM_states(RNG, P, initial_dist)
    out = np.zeros(T_)
    means = RNG.normal(NORMAL_PARAM[0], np.sqrt(NORMAL_PARAM[1]), size=K)
    for t in range(T_):
        curr_mean = means[states[t]]
        out[t] = RNG.normal(curr_mean, np.sqrt(NORMAL_PARAM[2]))
    return out

def normal_four(RNG):
    # HMM with 5 states
    K = 5
    initial_dist = np.ones(K)/K
    P = np.zeros((K, K))
    for i in range(K):
        alpha = np.ones(K)
        alpha[i] += 100
        P[i, :] = RNG.dirichlet(alpha)

    states = generate_HMM_states(RNG, P, initial_dist)
    out = np.zeros(T_)
    means = RNG.normal(NORMAL_PARAM[0], np.sqrt(NORMAL_PARAM[1]), size=K)
    for t in range(T_):
        curr_mean = means[states[t]]
        out[t] = RNG.normal(curr_mean, np.sqrt(NORMAL_PARAM[2]))
    return out

def ar_one(RNG):
    # Yao + independent samples
    cp_prob = 1.0/50.0
    
    U = generate_yao_cps(RNG, cp_prob)
    out = np.zeros(T_+2)
    out[0] = RNG.normal(0, 1)
    curr_phi, curr_v = None, None
    for t in range(1, T_+2):
        if U[t-1] == 1:
            curr_phi = RNG.uniform(-1, 1)
            curr_v = RNG.gamma(1, 1)
        out[t] = RNG.normal(curr_phi*out[t-1], np.sqrt(curr_v))
    return out

def ar_two(RNG):
    # Yao + clustered samples
    cp_prob = 1.0/50.0
    
    U = generate_yao_cps(RNG, cp_prob)
    out = np.zeros(T_+2)
    out[0] = RNG.normal(0, 1)
    K = 10
    # phis = [0.9, -0.9, 0, 0.3, -0.3]
    # vs = [0.5, 0.5, 4, 0.5, 0.5]
    phis = np.random.uniform(-0.9, 0.9, K)
    vs = np.random.gamma(1, 1, K)
    for t in range(1, T_+2):
        if U[t-1] == 1:
            ind = RNG.choice(K)
            curr_phi = phis[ind]
            curr_v = vs[ind]
        out[t] = RNG.normal(curr_phi*out[t-1], np.sqrt(curr_v))
    return out

def ar_three(RNG):
    # HMM + with 4 states
    K = 4
    initial_dist = np.ones(K)/K
    P = np.zeros((K, K))
    for i in range(K):
        alpha = np.ones(K)
        alpha[i] += 100
        P[i, :] = RNG.dirichlet(alpha)

    states = generate_HMM_states(RNG, P, initial_dist)
    out = np.zeros(T_+2)
    out[0] = RNG.normal(0, 1)
    phis = [0.9, -0.9, 0, 0.3, -0.3]
    vs = [0.5, 0.5, 4, 0.5, 0.5]
    for t in range(1, T_+2):
        curr_phi = phis[states[t-1]]
        curr_v = vs[states[t-1]]
        out[t] = RNG.normal(curr_phi*out[t-1], np.sqrt(curr_v))
    
    return out

def ar_four(RNG):
    # HMM + with 4 states
    K = 5
    initial_dist = np.ones(K)/K
    P = np.zeros((K, K))
    for i in range(K):
        alpha = np.ones(K)
        alpha[i] += 100
        P[i, :] = RNG.dirichlet(alpha)

    states = generate_HMM_states(RNG, P, initial_dist)
    # print(states.tolist())
    out = np.zeros(T_+2)
    out[0] = RNG.normal(0, 1)
    phis = [0.9, -0.9, 0, 0.3, -0.3]
    vs = [0.5, 0.5, 4, 0.5, 0.5]
    for t in range(1, T_+2):
        curr_phi = phis[states[t-1]]
        curr_v = vs[states[t-1]]
        out[t] = RNG.normal(curr_phi*out[t-1], np.sqrt(curr_v))
    
    return out

normal_scenarios = [normal_one, normal_two, normal_three, normal_four]

ar_scenarios = [ar_one, ar_two, ar_three, ar_four]

def sim_normal_data(scenario: int, dataset: int):
    seed = scenario * 50 + dataset

    RNG = np.random.default_rng(seed=seed)
    
    f = normal_scenarios[scenario]

    return f(RNG)
    
def sim_ar_data(scenario: int, dataset: int):
    seed = 100000 + scenario * 50 + dataset

    RNG = np.random.default_rng(seed=seed)

    f = ar_scenarios[scenario]

    return f(RNG)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # base_path = "C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models"

    # XXX below simulates data and writes
    for scenario in range(4):
        for dataset in range(50):
            data = sim_normal_data(scenario, dataset)
            save_data(data, 'normal', scenario, dataset)
    
    for scenario in range(4):
        for dataset in range(50):
            data = sim_ar_data(scenario, dataset)    
            save_data(data, 'ar', scenario, dataset)

    # y = sim_normal_data(scenario=1, dataset=10)
    # # y = sim_ar_data(scenario=1, dataset=2)

    # # y = sim_normal_data(0, 0)

    # plt.plot(y, color='black')
    # plt.show()
    
    # initial_dist = np.array([0.4, 0.3, 0.3])

    # P = np.array([[0.8, 0.1, 0.1], 
    #               [0.1, 0.8, 0.1], 
    #               [0.1, 0.1, 0.8]])
    # print(generate_HMM_states(RNG, P, initial_dist))
    
    # print(generate_yao_cps(RNG, 0.05))
