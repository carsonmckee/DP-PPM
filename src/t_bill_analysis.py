import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from npcp_ar import sample, plot_data
from time import perf_counter


if __name__ == "__main__":
    dat = pd.read_csv("C:/Users/k2259011/OneDrive - King's College London/Documents/univariate_models/WTB3MS.csv", header=None)
    y = dat[1].diff().to_numpy()[1:]
    y = (y - np.mean(y)) / np.sqrt(np.var(y))
    print(y.size)
    params = np.array([1, 1, 1], dtype=np.float64)

    t1 = perf_counter()
    U, clusters, cluster_means, pred_density = sample(y, n_samples=10, burn_in=0, n_particles=250, params=params, y_pred = 0.1, verbose=True)
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    print(pred_density)
    
    plot_data(y, U, clusters, max_cluster=np.max(clusters))