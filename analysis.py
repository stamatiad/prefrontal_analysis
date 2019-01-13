#from analysis_tools import quick_spikes
#from analysis_tools import generate_slices_g
import analysis_tools as at
from analysis_tools import from_zero_to, from_one_to
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pandas as pd
import h5py
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
from scipy import spatial



#===================================================================================================================
if __name__ == '__main__':
    # The file where network activity lies:
    network_activity_hdf5 = Path(fr'G:\Glia\all_data.hdf5')

    # Pass arguments from a dictionary to make it easier on the eyes.
    analysis_parameters = {
        'sim_stop': 3000,
        'q_size': 50,
        'total_qs': None, # this must be calculated later on.
        'ntrials': 10,
        'cellno': 333,
        'upper_threshold': 0,
        'lower_threshold': -10,
        'data_dim': 2
    }
    analysis_parameters['total_qs'] = int(np.floor(analysis_parameters['sim_stop'] / analysis_parameters['q_size']))

    # Parse voltage traces into network activity matrices once and save to file for later use:
    '''
    at.simulation_to_network_activity(
        tofile=network_activity_hdf5,
        animal_models=from_one_to(1),
        learning_conditions=from_one_to(10),
        ntrials=10,
        **analysis_parameters
    )
    '''

    # Load and plot network activity:
    # cellno, ntrials, total_qs
    network_activity = at.read_network_activity(
        fromfile=network_activity_hdf5,
        dataset=at.experiment_config_str(
            animal_model=1,
            learning_condition=1
        ),
        **analysis_parameters
    )
    #fig, ax = plt.subplots()
    #ax.imshow(data)
    #plt.show()

    t_L = at.pcaL2(
        data=network_activity,
        custom_range=range(20, 60),
        plot=False,
        **analysis_parameters
    )


    # k-means cluster network activity:
    at.determine_number_of_clusters(data=t_L, max_clusters=10, **analysis_parameters)

    #at.plot_pcaL2(data=t_L, klabels=klabels)

    print('breakpoint!')




