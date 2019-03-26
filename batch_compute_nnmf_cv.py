import analysis_tools as analysis
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
import sys
from multiprocessing import Pool
import pandas as pd


simulations_dir = Path.cwd().joinpath('simulations')
glia_dir = Path(r'G:\Glia')
plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (15, 15)

# Do only figures that will probably not change much.
plt.rcParams.update({'font.family': 'Helvetica'})

no_of_conditions = 4
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
        max_clusters=20,
        custom_range=custom_range,
    )



if __name__ == '__main__':
    #cwd = Path(r'C:\Users\steve\Documents\analysis\Python')
    cwd = Path.cwd()
    compute = False
    parallel = False

    if compute:
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


    else:
        #compute_nnmf_cv(param_pairs[1])
        #TODO: Load the CV results an plot them to see if correct
        fig = plt.figure()
        for animal_model in range(3, no_of_animals + 1):
            for learning_condition in range(3, no_of_conditions + 1):
                print(f'NT:{animal_model}, LC:{learning_condition}')
                inputfile = cwd.joinpath(
                    f'cross_valid_errors_structured{animal_model}_{learning_condition}.hdf'
                )
                # Read CV results.
                try:
                    attribs = pd.read_hdf(inputfile, key='attributes').to_dict()
                    K = attribs['K'][0]
                    max_clusters = attribs['max_clusters'][0]
                    rng_max_iters = attribs['rng_max_iters'][0]
                    error_bar = pd.read_hdf(inputfile, key='error_bar') \
                        .values.reshape(max_clusters, K, rng_max_iters) \
                            .mean(axis=2)
                    error_test = pd.read_hdf(inputfile, key='error_test') \
                        .values.reshape(max_clusters, K, rng_max_iters) \
                            .mean(axis=2)
                    # Plot the errors to get the picture:
                    plt.plot(error_bar, color='C0', alpha=0.2)
                    plt.plot(error_test, color='C1', alpha=0.2)
                    plt.plot(error_bar.mean(axis=1), color='C0')
                    plt.plot(error_test.mean(axis=1), color='C1')
                    K_str_cv = np.argmin(error_test.mean(axis=0))
                    plt.title(f'NT:{animal_model}, LC:{learning_condition}, K*cv:{K_str_cv}')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    plt.savefig(str(cwd.joinpath(f'NT{animal_model} LC{learning_condition} K_cv{K_str_cv}_structured.png')), format='png')
                    plt.cla()
                    # Giati einai toso mikrotero to test error? Mipws ginetai kati me ta data
                    # kai to train blepei to test?


                except Exception as e:
                    print(f'Exception! {str(e)}')
                    pass

            print('ALL GOOD')




    print('Tutto pronto!')
    sys.exit(0)
