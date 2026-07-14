import sys
import numpy as np 
import pandas as pd
from npcp_ar import sample as npcp 
from ihmm_ar import sample as ihmm 
from ppm_ar import sample as ppm, log_g
from sticky_ihmm_ar import sample as sticky_ihmm

MODEL_NAMES = ['sticky_ihmm', 'ihmm', 'ppm', 'ar1', 'npcp']
MODELS = [sticky_ihmm, ihmm, ppm, None, npcp]

T_ = 3769 - 2

# N_PARTS = 1000
# N_SAMPLES = 6000
# BURN_IN = 500
# VERBOSE = True

N_PARTS = 1000
N_SAMPLES = 6000
BURN_IN = 1000
VERBOSE = False

PARAMS = np.array([1, 1, 1], dtype=np.float64) 
# PARAMS = np.array([0.33*8, 4.0, 0.1], dtype=np.float64) # try these params?

if __name__ == "__main__":

    dat = pd.read_csv("WTB3MS.csv", header=None)
    y = dat[1].diff().to_numpy()[1:]
    y = (y - np.mean(y)) / np.sqrt(np.var(y))
    N = y.size
    
    task_id = int(sys.argv[1])
    model_id = task_id // T_
    model_id = 0 # Sticky ihmm
    model_name = MODEL_NAMES[model_id]
    sample = MODELS[model_id]
    
    t = (task_id % T_) + 2

    print(f'Task ID: {task_id}, model_name: {model_name}, t pred: {t}, dataset size: {N}, PARAMS = {PARAMS}')

    if sample is not None:
        if model_name in {'ppm', 'npcp'}:
            res = sample(y[:t], n_samples=N_SAMPLES, burn_in=BURN_IN, n_particles=N_PARTS, params=PARAMS, y_pred = y[t], verbose=VERBOSE)
        else:
            error = True
            while error:
                try:
                    res = sample(y[:t], n_samples=40000, burn_in=10000, params=PARAMS, y_pred=y[t], verbose=VERBOSE)
                    error  = False
                except Exception as ex:
                    print(ex)
                    print('retrying')
        state_means = res[-2]
        if model_name == 'sticky_ihmm':
            y_pred = np.mean(res[-1])
        else:
            y_pred = res[-1]
    else:
        # simple AR model
        y_temp = y[:t]
        x_ = y_temp[:len(y_temp)-1]
        y_ = y_temp[1:]

        yy = np.sum(y_*y_)
        yx = np.sum(y_*x_)
        xx = np.sum(x_*x_)
        
        state_means = np.zeros(t)
        y_pred = np.exp(log_g(yy+y[t]*y[t], yx+y[t]*y[t-1], xx+y[t-1]*y[t-1], len(y_)+1, PARAMS) - log_g(yy, yx, xx, len(y_), PARAMS))

    print(f"y pred est: {y_pred}")
    res_path = f't_bill_results/log_preds/{model_name}_{t-2}.csv'
    np.savetxt(res_path, np.array([y_pred]), delimiter=',')
    print(f"wrote results to: {res_path}")
    if (model_name in {'npcp', 'ihmm'}) and (t == 3768):
        states_path = f't_bill_results/{model_name}_states.csv'
        np.savetxt(states_path, state_means, delimiter=',')
        print(f'wrote states to {states_path}')