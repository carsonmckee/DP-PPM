import sys
import numpy as np

from numba import njit, prange
from distributer import task_id_to_likelihood_scenario_data_block_slice, read_data, save_results, BLOCK_SIZE

# XXX import models
from hmm_ar import sample as hmm_ar_sample
from ihmm_ar import sample as ihmm_ar_sample
from npcp_ar import sample as npcp_ar_sample
from ppm_ar import sample as ppm_ar_sample
from sticky_ihmm_ar import sample as sticky_ihmm_ar_sample

from hmm_normal import sample as hmm_normal_sample
from ihmm_normal import sample as ihmm_normal_sample
from npcp_normal import sample as npcp_normal_sample
from ppm_normal import sample as ppm_normal_sample
from sticky_ihmm_normal import sample as sticky_ihmm_normal_sample

from sim_data import NORMAL_PARAM, AR_PARAMS

N_SAMPLES = 5000
BURN_IN = 500
N_PARTICLES = 150
VERBOSE = True

def get_log_preds(data, likelihood, slice):
    slice_len = slice[1] - slice[0]
    
    ppm = [0.0]*slice_len
    hmm4 = [0.0]*slice_len
    hmm5 = [0.0]*slice_len
    ihmm = [0.0]*slice_len
    npcp = [0.0]*slice_len
    sticky_ihmm = [0.0]*slice_len
    
    print('start')
    for i in range(slice_len):
        print(i)
        upto = slice[0] + i + 1
        if upto >= 800:
            ppm.pop()
            hmm4.pop()
            hmm5.pop()
            ihmm.pop()
            npcp.pop()
            continue
            
        # upto = slice[i] + 1

        if likelihood == 'normal':
            
            # _, _, ppmi = ppm_normal_sample(data[:upto], n_samples=N_SAMPLES, burn_in=BURN_IN, n_particles=N_PARTICLES, params=NORMAL_PARAM, y_pred = data[upto])
            # print('ppm done')

            # _, hmm4i = hmm_normal_sample(data[:upto], n_samples=20000, burn_in=2000, K=4, params=NORMAL_PARAM, y_pred=data[upto])
            # print('hmm4 done')

            # _, hmm5i = hmm_normal_sample(data[:upto], n_samples=20000, burn_in=2000, K=5, params=NORMAL_PARAM, y_pred=data[upto])
            # print('hmm5 done')

            # error=True
            # while error:
            #     try:
            #         _, ihmmi = ihmm_normal_sample(data[:upto], n_samples=20000, burn_in=5000, params=NORMAL_PARAM, y_pred=data[upto])
            #         error = False
            #     except:
            #         ...
            # print('ihmm done')

            error=True
            while error:
                try:
                    _, sticky_ihmmi = sticky_ihmm_normal_sample(data[:upto], n_samples=12000, burn_in=2000, params=NORMAL_PARAM, y_pred=data[upto], verbose=VERBOSE)
                    sticky_ihmmi = np.mean(sticky_ihmmi)
                    error = False
                except Exception as ex:
                    print(ex)
                    print('retrying')
            print('sticky ihmm done')

            # _, _, npcpi = npcp_normal_sample(data[:upto], n_samples=N_SAMPLES, burn_in=BURN_IN, n_particles=N_PARTICLES, params=NORMAL_PARAM, y_pred=data[upto])
            # print('npcp done')
        else:
            
            # _, _, ppmi = ppm_ar_sample(data[:upto], n_samples=N_SAMPLES, burn_in=BURN_IN, n_particles=N_PARTICLES, params=AR_PARAMS, y_pred = data[upto])

            # print('ppm done')
            # _, hmm4i = hmm_ar_sample(data[:upto], n_samples=15000, burn_in=2000, K=4, params=AR_PARAMS, y_pred=data[upto])

            # print('hmm4 done')
            # _, hmm5i = hmm_ar_sample(data[:upto], n_samples=15000, burn_in=2000, K=5, params=AR_PARAMS, y_pred=data[upto])

            # print('hmm5 done')
            # error=True
            # while error:
            #     try:
            #         _, ihmmi = ihmm_ar_sample(data[:upto], n_samples=15000, burn_in=2000, params=AR_PARAMS, y_pred=data[upto])
            #         error = False                
            #     except:
            #         ...
            # print('ihmm done')

            error=True
            while error:
                try:
                    _, _, sticky_ihmmi = sticky_ihmm_ar_sample(data[:upto], n_samples=15000, burn_in=5000, params=AR_PARAMS, y_pred=data[upto], verbose=VERBOSE)
                    sticky_ihmmi = np.mean(sticky_ihmmi)
                    error = False
                except Exception as ex:
                    print(ex)
                    print('retrying')
            print('sticky ihmm done')

            # _, _, npcpi = npcp_ar_sample(data[:upto], n_samples=N_SAMPLES, burn_in=BURN_IN, n_particles=N_PARTICLES, params=AR_PARAMS, y_pred=data[upto])
            # print('npcp done')

        # ppm[i] = ppmi
        # hmm4[i] = hmm4i 
        # hmm5[i] = hmm5i 
        # ihmm[i] = ihmmi 
        # npcp[i] = npcpi
        sticky_ihmm[i] = sticky_ihmmi
    
    return ppm, hmm4, hmm5, ihmm, npcp, sticky_ihmm

if __name__ == "__main__":

    task_id = int(sys.argv[1])
    
    likelihood, scenario, dataset, block, slice = task_id_to_likelihood_scenario_data_block_slice(task_id)

    print(f'task_id: {task_id}, likelihood: {likelihood}, scenario: {scenario}, dataset: {dataset}, block: {block}, slice: {slice}, block size: {BLOCK_SIZE}')

    data = read_data(likelihood, scenario, dataset).flatten()

    print(data.size)
    
    # py run.py 0 "C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models"

    ppm, hmm4, hmm5, ihmm, npcp, sticky_ihmm = get_log_preds(data, likelihood, slice)

    print((ppm, hmm4, hmm5, ihmm, npcp, sticky_ihmm))
    
    # save_results(ppm, 'ppm', likelihood, scenario, dataset, block)
    # save_results(hmm4, 'hmm4', likelihood, scenario, dataset, block)
    # save_results(hmm5, 'hmm5', likelihood, scenario, dataset, block)
    # save_results(ihmm, 'ihmm', likelihood, scenario, dataset, block)
    # save_results(npcp, 'npcp', likelihood, scenario, dataset, block)
    save_results(sticky_ihmm, 'sticky_ihmm', likelihood, scenario, dataset, block)
    

