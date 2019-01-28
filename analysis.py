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

    # Lazy load the data as a NWB file. Easy to pass around and encapsulates info like trial length, stim times etc.
    nwbfile = analysis.load_nwb_file(
        animal_model=2,
        learning_condition=5,
        experiment_config='structured',
        data_path=Path(r'G:\Glia')
    )

    analysis.bin_activity(
        nwbfile,
        q_size=50
    )

    #binned_network_activity = nwbfile.acquisition['binned_activity'].data#[:, 1, :]
    #fig = plt.figure()
    #plt.imshow(binned_network_activity)
    #plt.show

    analysis.pcaL2(
        nwbfile,
        custom_range=(20, 60),
        smooth=True,
        plot=True
    )


    # Determine the optimal number of clusters for all trials of a single animal
    # model/learning condition.
    #TODO: clustering does not work. Is this due to overfitting?
    # I dought that, because some easy clusters are mislabeled.
    # Also are the labels correct (is this a label problem?)
    #y_array = np.linspace(0.01, 10, 1000)
    y_array = np.linspace(0.1, 100, 1000)
    K_star, K_labels = analysis.determine_number_of_clusters(
        nwbfile,
        max_clusters=10,
        y_array=y_array
    )

    # Utilize only one y power, based on the dataset:
    y_i = 500

    print(f'K* = {K_star[:, y_i]}')
    analysis.pcaL2(
        nwbfile, klabels=K_labels[:, y_i].T,
        custom_range=(20, 60),
        smooth=True, plot=True
    )

    print('breakpoint!')




