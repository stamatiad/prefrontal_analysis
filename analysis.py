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
from functools import partial
from collections import defaultdict
from scipy import stats
import sys

from pynwb import NWBHDF5IO

#===================================================================================================================
if __name__ == '__main__':


    animal_model = 4
    configuration = 'structured'

    NWBfiles = [
        analysis.load_nwb_file(
            animal_model=animal_model,
            learning_condition=learning_condition,
            experiment_config=configuration,
            data_path=Path(r'G:\Glia')
        )
        for learning_condition in range(1, 11)
    ]

    #data = nwbfile.acquisition['membrane_potential'].data
    #fig = plt.figure()
    #plt.ion()
    #plt.plot(data[10, :])
    #plt.show



    for nwbfile in NWBfiles:
        analysis.bin_activity(nwbfile, q_size=50)

    #binned_network_activity = \
    #    np.squeeze(nwbfile.acquisition['binned_activity'].data.data[:, :])
    #fig = plt.figure()
    #plt.ion()
    #plt.imshow(binned_network_activity)
    #plt.show


    trial_len = nwbfile.trials['stop_time'][0] - \
                nwbfile.trials['start_time'][0]  # in ms
    custom_range = (20, int(trial_len / 50))
    #analysis.pcaL2(
    #    NWBfiles, custom_range=custom_range, smooth=True, plot=True
    #)


    # Determine the optimal number of clusters for all trials of a single animal
    # model/learning condition.
    #y_array = np.linspace(0.01, 10, 1000)
    y_array = np.linspace(0.1, 100, 1000)
    K_star, K_labels = analysis.determine_number_of_clusters(
        NWBfiles,
        max_clusters=10,
        y_array=y_array,
        custom_range=custom_range
    )

    y_i = 500
    print(f'K* = {K_star[y_i, :]}')
    analysis.pcaL2(
        NWBfiles, klabels=K_labels[y_i, :].T,
        custom_range=custom_range,
        smooth=True, plot=True
    )
    plt.savefig(Path(fr'C:\Users\steve\Pictures\Animal_{animal_model}_{configuration}_dim_5.png'))
    print('Tutto pronto!')
    sys.exit()

    #========================================================================
    #========================================================================
    # FIGURE 3 A, B

    def separate_trials(input_NWBfile=None, acquisition_name=None):
        # Return an iterable of each trial acrivity.

        #TODO: check if wrapped and unwrap:
        raw_acquisition = input_NWBfile.acquisition[acquisition_name].data
        trials = input_NWBfile.trials
        #TODO: get samples_per_ms
        f = 10
        trial_activity = [
            raw_acquisition[:, int(trial_start_t*f):int(trial_end_t*f) - 1]
            for trialid, trial_start_t, trial_end_t in analysis.nwb_iter(trials)
        ]
        return trial_activity


    # Lazy load the data as a NWB file.
    input_NWBfile = Path(r'G:\Glia\excitatory_validation.nwb')
    nwbfile = NWBHDF5IO(str(input_NWBfile), 'r').read()
    per_trial_activity = {}
    per_trial_activity['normal_NMDA+AMPA'] = separate_trials(
        input_NWBfile=nwbfile, acquisition_name='normal_NMDA+AMPA'
    )
    per_trial_activity['normal_AMPA_only'] = separate_trials(
        input_NWBfile=nwbfile, acquisition_name='normal_AMPA_only'
    )
    per_trial_activity['noMg_NMDA+AMPA'] = separate_trials(
        input_NWBfile=nwbfile, acquisition_name='noMg_NMDA+AMPA'
    )
    fig, ax = plt.subplots()
    for trace in per_trial_activity['normal_NMDA+AMPA']:
        plt.plot(trace[0][500:5000], color='r')
    for trace in per_trial_activity['normal_AMPA_only']:
        plt.plot(trace[0][500:5000], color='b')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Somatic depolarization (mV)')
    fig, ax = plt.subplots()
    for trace in per_trial_activity['normal_NMDA+AMPA']:
        plt.plot(trace[0][500:5000], color='r')
    for trace in per_trial_activity['noMg_NMDA+AMPA']:
        plt.plot(trace[0][500:5000], color='b')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Somatic depolarization (mV)')

    #========================================================================
    #========================================================================
    # FIGURE 3 C


    def datasetName(id):
        return f'Animal {id}'

    def setBoxAttribtes(boxplot_handles, colors):
        for bplot in boxplot_handles:
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)
                patch.set_facecolor(color)
                patch.set_linewidth(2)
            for line in bplot['medians']:
                line.set_linewidth(2)
            for line in bplot['whiskers']:
                line.set_linewidth(2)
            for line in bplot['fliers']:
                line.set_linewidth(2)
            for line in bplot['caps']:
                line.set_linewidth(2)

    def statisticalAnnotation(columns=None, datamax=None, axobj=None):
        # statistical annotation
        x1, x2 = columns  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = datamax + datamax/10, datamax/10, 'k'
        axobj.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        axobj.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color=col)

    def is_significant(data=None, pthreshold=0.1):
        # Compare data in columns for significance.
        statistic, p = stats.ttest_ind(
            data[:, 0], data[:, 1],
            equal_var=False
        )
        print(f'p is {p}')
        if p < pthreshold:
            return True
        else:
            return False


    no_of_conditions = 10
    no_of_animals = 4
    optimal_clusters_of_group = defaultdict(partial(np.ndarray, 0))
    configurations = {1: 'structured', 2: 'random'}
    for animal_model in range(1, no_of_animals + 1):
        # Pool together no of clusters for one animal model:
        K_star_over_trials = np.zeros((no_of_conditions, 2))
        for config_id, config in configurations.items():
            for learning_condition in range(1, no_of_conditions + 1):
                try:
                    # Lazy load the data as a NWB file. Easy to pass around and encapsulates info like trial length, stim times etc.
                    nwbfile = analysis.load_nwb_file(
                        animal_model=animal_model,
                        learning_condition=learning_condition,
                        experiment_config=config,
                        data_path=Path(r'G:\Glia')
                    )

                    #data = nwbfile.acquisition['membrane_potential'].data
                    #fig = plt.figure()
                    #plt.ion()
                    #plt.plot(data[10, :])
                    #plt.show



                    analysis.bin_activity(nwbfile, q_size=50)

                    #binned_network_activity = \
                    #    np.squeeze(nwbfile.acquisition['binned_activity'].data.data[:, :])
                    #fig = plt.figure()
                    #plt.ion()
                    #plt.imshow(binned_network_activity)
                    #plt.show


                    trial_len = nwbfile.trials['stop_time'][0] - \
                                nwbfile.trials['start_time'][0]  # in ms
                    custom_range = (20, int(trial_len / 50))
                    #analysis.pcaL2(
                    #    nwbfile, custom_range=custom_range, smooth=True, plot=True
                    #)


                    # Determine the optimal number of clusters for all trials of a single animal
                    # model/learning condition.
                    #y_array = np.linspace(0.01, 10, 1000)
                    y_array = np.linspace(0.1, 100, 1000)
                    K_star, K_labels = analysis.determine_number_of_clusters(
                        nwbfile,
                        max_clusters=10,
                        y_array=y_array,
                        custom_range=custom_range
                    )

                    # Utilize only one y power, based on the dataset:
                    y_i = 500

                    #print(f'K* = {K_star[y_i, :]}')
                    #analysis.pcaL2(
                    #    nwbfile, klabels=K_labels[y_i, :].T,
                    #    custom_range=custom_range,
                    #    smooth=True, plot=True
                    #)

                    K_star_over_trials[learning_condition - 1, config_id - 1] = \
                        K_star[y_i]
                except Exception as e:
                    print(f'Got Exception during analysis {str(e)}')

        optimal_clusters_of_group[datasetName(animal_model)] = \
            K_star_over_trials

    print('breakpoint!')




    fig1, ax1 = plt.subplots()
    ax1.set_title('Optimal no of clusters per configuration')
    positions = [
        (position[0], position[0] + 1)
        for position in analysis.generate_slices(
                          size=3, number=no_of_animals,
                          start_from=1, to_slice=False
                      )
    ]
    bplots = []
    for animal, (pos_a, pos_b) in zip(range(1, no_of_animals + 1), positions):
        bp = ax1.boxplot(
            optimal_clusters_of_group[datasetName(animal)],
            positions=[pos_a, pos_b],
            widths=0.4,
            patch_artist=True
        )
        if is_significant(optimal_clusters_of_group[datasetName(animal)]):
            statisticalAnnotation(
                columns=(pos_a, pos_b),
                datamax=optimal_clusters_of_group[datasetName(animal)].max(),
                axobj=ax1
            )
        bplots.append(bp)
    setBoxAttribtes(boxplot_handles=bplots, colors=['blue', 'red'])
    ax1.set_xlim(0, 13)
    ax1.set_xticks([
        p + 0.5
        for p, _ in positions
    ])
    ax1.set_xticklabels([
        datasetName(i)
        for i in range(1, no_of_animals + 1)
    ])
    ax1.set_xlabel('Configurations')
    ax1.set_ylabel('K*')

