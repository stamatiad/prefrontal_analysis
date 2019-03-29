# <markdowncell>
# # # Generate Figure 2
# Network distinct response states, in response to different input. A.  Exemplar population activity (n=10 stimuli) reduced in two principal components (PCAL2) over time for n=3 different synaptic reconfigurations (learning conditions). Only delay period is plotted. Time is in seconds. B. Same procedure (population PCAL2 activity) for two structured network instances, with all learning conditions (n=10) pooled together, each responding with K* > n. Clusters identified as in A. C. Boxplot of optimal number of clusters (K*) after k-means (see Methods) for each n=4 structured network instances of n=10 synaptic reshufflings (learning conditions), for n=10 stimuli.
# <markdowncell>
# Import necessary modules:

# <codecell>
import notebook_module as nb
import analysis_tools as analysis
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from pathlib import Path
from pynwb import NWBHDF5IO
from itertools import chain
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
import seaborn as sb
import math

# <markdowncell>
# # Create figure 2.

# <codecell>
simulations_dir = Path.cwd().joinpath('simulations')
glia_dir = Path(r'G:\Glia')
plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (15, 15)

y_array = np.linspace(0.1, 100, 1000)
y_i = 500

axis_label_font_size = 10
no_of_conditions = 10
no_of_animals = 4


subplot_width = 3
subplot_height = 1
plt.ion()
figure4 = plt.figure(figsize=plt.figaspect(0.5))
figure4_axis = np.zeros((subplot_height, subplot_width), dtype=object)
dataset_name = lambda x : f'Network {x}'

# Fig 4B:
B_axis = figure4.add_subplot(
    subplot_height, subplot_width, 2
)
if False:
    # Plot what happens with no NMDA, no Mg:
    optimal_clusters_of_group = defaultdict(partial(np.ndarray, 0))
    configurations = ['structured_nonmda', 'structured_nomg']
    for animal_model in range(1, no_of_animals + 1):
        # Pool together no of clusters for one animal model:
        K_star_over_trials = np.ones((no_of_conditions, len(configurations)))
        for config_id, config in enumerate(configurations):
            for learning_condition in range(1, no_of_conditions + 1):
                try:
                    # Lazy load the data as a NWB file. Easy to pass around and encapsulates info like trial length, stim times etc.
                    NWBfile = analysis.load_nwb_file(
                        animal_model=animal_model,
                        learning_condition=learning_condition,
                        experiment_config=config,
                        type='bn',
                        data_path=simulations_dir
                    )
                    trial_len = analysis.get_acquisition_parameters(
                        input_NWBfile=NWBfile,
                        requested_parameters=['trial_len']
                    )
                    custom_range = (20, int(trial_len / 50))

                    # Determine the optimal number of clusters for all trials of a single animal
                    # model/learning condition.
                    K_star, K_labels, *_ = analysis.determine_number_of_clusters(
                        NWBfile_array=[NWBfile],
                        max_clusters=10,
                        y_array=y_array,
                        custom_range=custom_range
                    )

                    K_star_over_trials[learning_condition - 1, config_id] =                     K_star[y_i]
                except Exception as e:
                    print(f'Got Exception during analysis {str(e)}')

        optimal_clusters_of_group[dataset_name(animal_model)] =         K_star_over_trials

# For no NMDA case:
for animal_model in range(1, 1 + 1):
    try:
        # Pool together no of clusters for one animal model:
        # Lazy load the data as a NWB file. Easy to pass around and encapsulates info like trial length, stim times etc.
        NWBfiles = [
            analysis.load_nwb_file(
                animal_model=animal_model,
                learning_condition=learning_condition,
                experiment_config='structured_nonmda',
                type='bn',
                data_path=simulations_dir
            )
            for learning_condition in range(1, 5 + 1)
        ]

        trial_len, ntrials = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfiles[0],
            requested_parameters=['trial_len', 'ntrials']
        )
        custom_range = (20, int(trial_len / 50))

        # Plot the annotated clustering results:
        #TODO: are these correctly labeled?
        K_labels = np.matlib.repmat(np.arange(1, len(NWBfiles) + 1), ntrials, 1) \
            .T.reshape(ntrials, -1).reshape(1, -1)[0]
        analysis.pcaL2(
            NWBfile_array=NWBfiles,
            klabels=K_labels,
            custom_range=custom_range,
            smooth=True, plot_2d=True,
            plot_axes=B_axis
        )
    except Exception as e:
        print(f'Got Exception during analysis {str(e)}')




# Fig 4A
# Scatter plot PC VS clusters
A_axis = figure4.add_subplot(
    subplot_height, subplot_width, 1
)
A_axis.cla()
# Figure 3D:
optimal_clusters_of_group = defaultdict(partial(np.ndarray, 0))
configurations = ['structured', 'random']
for animal_model in range(1, no_of_animals + 1):
    # Pool together no of clusters for one animal model:
    K_star_over_trials = np.ones((no_of_conditions, len(configurations)))
    for config_id, config in enumerate(configurations):
        for learning_condition in range(1, no_of_conditions + 1):
            try:
                # Lazy load the data as a NWB file. Easy to pass around and encapsulates info like trial length, stim times etc.
                NWBfile = analysis.load_nwb_file(
                    animal_model=animal_model,
                    learning_condition=learning_condition,
                    experiment_config=config,
                    type='bn',
                    data_path=simulations_dir
                )

                #analysis.bin_activity(nwbfile, q_size=50)

                trial_len = analysis.get_acquisition_parameters(
                    input_NWBfile=NWBfile,
                    requested_parameters=['trial_len']
                )
                custom_range = (20, int(trial_len / 50))


                # Determine the optimal number of clusters for all trials of a single animal
                # model/learning condition.
                K_star, K_labels, *_ = analysis.determine_number_of_clusters(
                    NWBfile_array=[NWBfile],
                    max_clusters=10,
                    y_array=y_array,
                    custom_range=custom_range
                )

                K_star_over_trials[learning_condition - 1, config_id] =                     K_star[y_i]
            except Exception as e:
                print(f'Got Exception during analysis {str(e)}')

    optimal_clusters_of_group[dataset_name(animal_model)] =         K_star_over_trials

# Use histogram instead of boxplot:
tmp = [
    optimal_clusters_of_group[dataset_name(animal)][:, 0].tolist()
    for animal in range(1, no_of_animals + 1)
]
K_stars_structured = list(chain(*tmp))
tmp = [
    optimal_clusters_of_group[dataset_name(animal)][:, 1].tolist()
    for animal in range(1, no_of_animals + 1)
]
K_stars_random = list(chain(*tmp))

bins_str = np.arange(1, np.max(K_stars_structured) + 2, 1)
kstar_str_hist, *_ = np.histogram(K_stars_structured, bins=bins_str)
bins_rnd = np.arange(1, np.max(K_stars_random) + 2, 1)
kstar_rnd_hist, *_ = np.histogram(K_stars_random, bins=bins_rnd)

A_axis.plot(kstar_str_hist / len(K_stars_structured), color='C0')
A_axis.axvline(np.mean(K_stars_structured), color='C0', linestyle='--')
A_axis.plot(kstar_rnd_hist / len(K_stars_random), color='C1')
A_axis.axvline(np.mean(K_stars_random), color='C1', linestyle='--')
A_axis.set_xticks(range(bins_str.size + 1))
A_axis.set_xticklabels(np.round(bins_str, 1))
A_axis.set_xlabel('K*', fontsize=axis_label_font_size)
A_axis.set_ylabel('Relative Frequency', fontsize=axis_label_font_size)
nb.axis_normal_plot(axis=A_axis)
nb.adjust_spines(A_axis, ['left', 'bottom'])
nb.mark_figure_letter(A_axis, 'A')





nb.mark_figure_letter(B_axis, 'B')

#==============================================================================

plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.25)



# <codecell>
figure4.savefig('Figure_4.svg')
figure4.savefig('Figure_4.png')
print('Tutto pronto!')


#%%



