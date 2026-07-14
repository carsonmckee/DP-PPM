import sys
import numpy as np 
import pandas as pd
from npcp_normal_mean_var import sample as npcp 
from ihmm_normal_mean_var import sample as ihmm 
from ppm_normal_mean_var import sample as ppm
from sticky_ihmm_normal_mean_var import sample as sticky_ihmm


MODEL_NAMES = ['sticky_ihmm'
               'ihmm', 
               'npcp', 
               'ppm']
MODELS = [sticky_ihmm,
          ihmm, 
          npcp, 
          ppm]

N_PARTS = 350
N_SAMPLES = 5000
BURN_IN = 500
VERBOSE = False

PARAMS = np.array([0, 1, 1, 1], dtype=np.float64)
PARAMS = np.array([0, 0.02, 2.2, 0.12], dtype=np.float64)

if __name__ == "__main__":

    y = pd.read_csv(f'well_log_clean.csv').to_numpy().flatten()
    y = (y - np.mean(y)) / np.sqrt(np.var(y))
    T_ = y.size - 1
    
    task_id = int(sys.argv[1])
    # model_id = task_id // (T_)
    # model_name = MODEL_NAMES[model_id]
    # sample = MODELS[model_id]

    t = (task_id % T_) + 1

    # print(f'Task ID: {task_id}, model_name: {model_name}, t pred: {t}, dataset size: {T_}, PARAMS = {PARAMS}')
    print(f'Task ID: {task_id}, t pred: {t}, dataset size: {T_}, PARAMS = {PARAMS}')
    for model_id, model_name in enumerate(['sticky_ihmm']):
    # for model_id, model_name in enumerate(MODEL_NAMES):
        sample = MODELS[model_id]

        if sample is not None:
            if model_name in {'ppm', 'npcp'}:
                res = sample(y[:t], n_samples=N_SAMPLES, burn_in=BURN_IN, n_particles=N_PARTS, params=PARAMS, y_pred = y[t], verbose=VERBOSE)
            else:
                error = True
                while error:
                    try:
                        res = sample(y[:t], n_samples=20000, burn_in=10000, params=PARAMS, y_pred=y[t], verbose=VERBOSE)
                        error  = False
                    except Exception as ex:
                        print(ex)
                        print('retrying')
            n_states = res[-3]
            state_means = res[-2]
            if model_name == 'sticky_ihmm':
                y_pred = np.mean(res[-1])
            else:
                y_pred = res[-1]

        print(f"y pred est: {y_pred}")
        res_path = f'well_log_results/log_preds/{model_name}_{t-1}.csv'
        np.savetxt(res_path, np.array([y_pred]), delimiter=',')
        print(f"wrote results to: {res_path}")
        if (model_name in {'npcp', 'ihmm'}) and (t == 3945):
            states_path = f'well_log_results/{model_name}_states.csv'
            np.savetxt(states_path, state_means, delimiter=',')
            print(f'wrote states to {states_path}')
            
            nstates_path = f'well_log_results/{model_name}_nstates.csv'
            np.savetxt(nstates_path, n_states, delimiter=',')
            print(f'wrote nstates to {nstates_path}')