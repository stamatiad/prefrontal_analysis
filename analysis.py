#from analysis_tools import quick_spikes
#from analysis_tools import generate_slices_g
import analysis_tools as analysis
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
    #network_activity_hdf5 = Path(fr'G:\Glia\all_data.hdf5')
    network_activity_hdf5 = Path(fr'G:\Glia\str_data.hdf5')

    # Pass arguments from a dictionary to make it easier on the eyes.
    #TODO: these are inside each of the NWB files. Do I need them here?
    analysis_parameters = {
        'stim_start_offset': 50,
        'stim_stop_offset': 1050,
        'q_size': 50,
        'data_path': Path(r'G:\Glia')
    }

    # Lazy load the data as a NWB file. Easy to pass around and encapsulates info like trial length, stim times etc.
    nwbfile = analysis.load_nwb_file(
        animal_model=1,
        learning_condition=1,
        experiment_config='structured_nonmda'
    )


    # Parse voltage traces into network activity matrices once and save to file for later use:

    for lc in range(1, 11):
        # Load and plot network activity:
        #TODO: My goal is to replace this function with the NWB data type and load the data that I want.
        #TODO: Lazy read the data that I want (multiple trials, ONE learning condition, ONE animal) and create an
        # TimeSeries with the windowed activity (is the latter lies on a nice numpy arra? if not, use my own format)
        # I can pass around the TimeSeries with cells and qs as dims, having the trials as markers. Then reshape in each function
        # So the block below is reduced to the lazy load of the dataset. And AFTER that the pca will return a NDARRAY.
        binned_network_activity = analysis.bin_activity(
            nwbfile,
            q_size=50
        )

        fig = plt.figure()
        plt.imshow(binned_network_activity[:,3,:])
        plt.show

        #TODO: This function will load a timeseries object.
        t_L = analysis.pcaL2(
            data=binned_network_activity,
            custom_range=range(20, 60),
            plot=True,
            **analysis_parameters
        )


        # Determine the optimal number of clusters for all trials of a single animal model/learning condition.
        #TODO: pass a plot flag, so not needing an extra function call for that in the notebook.
        #y_array = np.linspace(0.01, 10, 1000)
        y_array = np.linspace(0.1, 100, 1000)
        K_star, K_labels = analysis.determine_number_of_clusters(data=t_L[:, :, -20:], max_clusters=10, y_array=y_array, **analysis_parameters)

        # Utilize only one y power, based on the dataset:
        y_i = 500

        #at.plot_pcaL2(data=t_L, smooth=True, **analysis_parameters)
        analysis.plot_pcaL2(data=t_L, klabels=K_labels[:, y_i].T, smooth=True, **analysis_parameters)

    print('breakpoint!')




