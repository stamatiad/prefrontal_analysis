import analysis_tools as analysis
import numpy as np
from pathlib import Path
import sys
from multiprocessing import Pool
import pandas as pd


simulations_dir = Path.cwd().joinpath('simulations')

no_of_conditions = 10
no_of_animals = 4
param_pairs = []
for animal_model in range(1, no_of_animals + 1):
    for learning_condition in range(1, no_of_conditions + 1):
        param_pairs.append([animal_model, learning_condition])

@analysis.time_it
def compute_nnmf_cv(inputs):
    '''
    Computes Cross Validation of NNMF.
    '''

    animal_model, learning_condition = inputs
    print(f'Running CV on Animal:{animal_model}, LC:{learning_condition}')
    NWBfile = analysis.load_nwb_file(
        animal_model=animal_model,
        learning_condition=learning_condition,
        experiment_config='structured',
        type='bn',
        data_path=simulations_dir
    )

    trial_len = analysis.get_acquisition_parameters(
        input_NWBfile=NWBfile,
        requested_parameters=['trial_len']
    )

    custom_range = (20, int(trial_len / 50))

    blah = analysis.determine_number_of_ensembles(
        NWBfile_array=[NWBfile],
        K_max=10,
        custom_range=custom_range,
        K_cv=20,
        rng_max_iters=1
    )

if __name__ == '__main__':
    cwd = Path.cwd()
    parallel = True

    # run in parallel:
    if parallel:
        print('Commencing parallel task!')
        with Pool(4) as p:
            results = p.map(compute_nnmf_cv, param_pairs)
    else:
        print('Commencing single task!')
        for params in param_pairs:
            compute_nnmf_cv(params)
    print('Done computing Cross Validation!')

    print('Tutto pronto!')
    sys.exit(0)
