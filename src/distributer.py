import csv
import numpy as np
from math import floor

LIKELIHOODS = ['normal', 'ar']
SCENARIOS = [0, 1, 2, 3]
N_DATASETS = 50
T_ = 800
BLOCK_SIZE = 20

'''
1) task ids are ordered such that the first half are from the normal likelihood and second half the AR(1) likelihood
2) within each half, task ids are ordered according to the 4 scenarios
3) within each four scenario, then have the 50 datasets
4) finally within that each block
'''

def num_tasks():
    blocks_per_dataset = T_ / BLOCK_SIZE
    datasets_per_scenario = N_DATASETS
    scenarios_per_likelihood = 4
    return int(blocks_per_dataset * datasets_per_scenario * scenarios_per_likelihood * len(LIKELIHOODS))

def likelihood_scenario_data_block_to_id(likelihood: str, scenario: int, dataset: int, block: int) -> int:
    if block >= int(T_ / BLOCK_SIZE):
        raise ValueError(f'block : {block} is too large')
    blocks_per_dataset = T_ / BLOCK_SIZE
    datasets_per_scenario = N_DATASETS
    scenarios_per_likelihood = 4

    tasks_per_dataset = blocks_per_dataset
    tasks_per_scenario = datasets_per_scenario * tasks_per_dataset
    tasks_per_likelihood = scenarios_per_likelihood * tasks_per_scenario

    like = 0 if likelihood == 'normal' else 1

    task_id = like * tasks_per_likelihood + scenario * tasks_per_scenario + dataset * tasks_per_dataset + block

    return int(task_id)

def task_id_to_likelihood_scenario_data_block_slice(task_id):
    blocks_per_dataset = T_ / BLOCK_SIZE
    datasets_per_scenario = N_DATASETS
    scenarios_per_likelihood = 4

    tasks_per_dataset = blocks_per_dataset
    tasks_per_scenario = datasets_per_scenario * tasks_per_dataset
    tasks_per_likelihood = scenarios_per_likelihood * tasks_per_scenario
    if (task_id < 0) or (tasks_per_likelihood > len(LIKELIHOODS)*tasks_per_likelihood):
        raise ValueError(f'task_id: {task_id} is out of range')

    likelihood = floor(task_id / tasks_per_likelihood)
    rem_1 = task_id % tasks_per_likelihood
    scenario = floor(rem_1 / tasks_per_scenario)
    rem_2 = rem_1 % tasks_per_scenario
    dataset = floor(rem_2 / tasks_per_dataset)
    rem_3 = rem_2 % tasks_per_dataset
    block = int(rem_3) 

    like = 'normal' if likelihood == 0 else 'ar'

    if like == 'ar':
        slice = (block*BLOCK_SIZE+1, (block+1)*BLOCK_SIZE+1)
    else:
        slice = (block*BLOCK_SIZE, (block+1)*BLOCK_SIZE)
    return like, scenario, dataset, block, slice

def generate_task_ids():
    task_ids = []
    for likelihood in LIKELIHOODS:
        for scenario in SCENARIOS:
            for dataset in range(N_DATASETS):
                for block in range(int(T_ / BLOCK_SIZE)):
                    task_ids.append(likelihood_scenario_data_block_to_id(likelihood, scenario, dataset, block))

    return task_ids

def save_results(results: list, model: str, likelihood: str, scenario: int, dataset: int, block: int) -> None:
    
    full_path = f'results/{likelihood}/{model}_{scenario}_{dataset}_{block}.csv'
    
    with open(full_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(results)
    
    print(f"Wrote results to {full_path}")

def collect_results(base_path, models):
    
    results = dict()
    for likelihood in LIKELIHOODS:
        results[likelihood] = dict()
        for model in models:
            results[likelihood][model] = dict()
            for scenario in SCENARIOS:
                results[likelihood][model][scenario] = dict()
                for dataset in range(N_DATASETS):
                    results[likelihood][model][scenario][dataset] = []
                    for block in range(int(T_ / BLOCK_SIZE)):
                        full_path = f'{base_path}/{likelihood}/{model}_{scenario}_{dataset}_{block}.csv'
                        with open(full_path, 'r') as f:
                            reader = csv.reader(f)
                            for row in reader:
                                results[likelihood][model][scenario][dataset].extend([float(row_i) for row_i in row])
                    
                    results[likelihood][model][scenario][dataset] = sum(results[likelihood][model][scenario][dataset])
    
    return results

def save_data(data: list, likelihood, scenario, dataset, base_path=None):

    if isinstance(data, np.ndarray):
        data = data.tolist()
    if base_path:
        full_path = f'{base_path}/data/{likelihood}/{scenario}_{dataset}.csv'
    else:
        full_path = f'data/{likelihood}/{scenario}_{dataset}.csv'
    with open(full_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def read_data(likelihood, scenario, dataset) -> np.array:
    
    full_path = f'data/{likelihood}/{scenario}_{dataset}.csv'

    with open(full_path, 'r') as f:
        reader = csv.reader(f)
        output = []
        for row in reader:
            if len(row) > 0:
                output.append([float(ri) for ri in row])
    
    return np.array(output)

if __name__ == "__main__":

    likelihood = 'normal'
    scenario = 2
    dataset = 3
    block = 6

    task_id = likelihood_scenario_data_block_to_id(likelihood, scenario, dataset, block)
    print(task_id)

    print(task_id_to_likelihood_scenario_data_block_slice(task_id))

    task_ids = generate_task_ids()
    print(len(task_ids))
    print(num_tasks())