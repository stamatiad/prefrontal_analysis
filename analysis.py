import notebook_module as nb
import analysis_tools as analysis
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from pathlib import Path



# import the notebook_module here and write down the code to reproduce the
# figures (to debug and test). Then copy the code to the jupyter notebook.
# Do only figures that will probably not change much.
simulations_dir = Path.cwd().joinpath('simulations')
plt.rcParams.update({'font.family': 'Helvetica'})
#===============================================================================
#===============================================================================
# FIGURE 1 (PENDING)
#===============================================================================
#===============================================================================
# FIGURE 2
subplot_width = 3
subplot_height = 2
figure2 = plt.figure(figsize=plt.figaspect(0.5))
figure2_axis = np.zeros((subplot_height, subplot_width), dtype=object)
for idx in range(subplot_width):
    figure2_axis[0, idx] = figure2.add_subplot(
        subplot_height, subplot_width, idx + 1, projection='3d'
    )

y_array = np.linspace(0.1, 100, 1000)
y_i = 500
no_of_conditions = 2#10

# Plot same animal model, different learning conditions:
conditions = [1, 2, 3]
for idx, learning_condition in enumerate(conditions):
    NWBfile = analysis.load_nwb_file(
        animal_model=2,
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

    K_star, K_labels = analysis.determine_number_of_clusters(
        NWBfile_array=[NWBfile],
        max_clusters=no_of_conditions,
        y_array=y_array,
        custom_range=custom_range
    )

    # Plot the annotated clustering results:
    analysis.pcaL2(
        NWBfile_array=[NWBfile],
        klabels=K_labels[y_i, :].T,
        custom_range=custom_range,
        smooth=True, plot_3d=True,
        plot_axes=figure2_axis[0, idx]
    )


# Plot whole animal model state space:
for idx, animal_model in enumerate([1,2]):
    figure2_axis[1, idx] = figure2.add_subplot(
        subplot_height, subplot_width, 4 + idx
    )
    NWBfiles = [
        analysis.load_nwb_file(
            animal_model=animal_model,
            learning_condition=learning_condition,
            experiment_config='structured',
            type='bn',
            data_path=simulations_dir
        )
        for learning_condition in range(1, no_of_conditions + 1)
    ]

    trial_len, ntrials = analysis.get_acquisition_parameters(
        input_NWBfile=NWBfiles[0],
        requested_parameters=['trial_len', 'ntrials']
    )
    custom_range = (20, int(trial_len / 50))

    K_star, K_labels = analysis.determine_number_of_clusters(
        NWBfile_array=NWBfiles,
        max_clusters=no_of_conditions * ntrials,
        y_array=y_array,
        custom_range=custom_range
    )

    # Plot the annotated clustering results:
    analysis.pcaL2(
        NWBfile_array=NWBfiles,
        klabels=K_labels[y_i, :].T,
        custom_range=custom_range,
        smooth=True, plot_2d=True,
        plot_axes=figure2_axis[1, idx]
    )


# TODO: Plot number of clusters per animal/condition (na dw)
# Run for every learning condition and animal the k-means clustering:
figure2_axis[1, 2] = figure2.add_subplot(
    subplot_height, subplot_width, 6
)
no_of_animals = 4
optimal_clusters_of_group = defaultdict(partial(np.ndarray, 0))
for animal_model in range(1, no_of_animals + 1):
    # Pool together no of clusters for one animal model:
    K_star_over_trials = np.zeros((no_of_conditions, 1))
    for learning_condition in range(1, no_of_conditions + 1):
        # Lazy load the data as a NWB file. Easy to pass around and
        # encapsulates info like trial length, stim times etc.
        #TODO: this might raised some exceptions. Investigate!
        nwbfile = analysis.load_nwb_file(
            animal_model=animal_model,
            learning_condition=learning_condition,
            experiment_config='structured',
            type='bn',
            data_path=simulations_dir
        )

        trial_len = analysis.get_acquisition_parameters(
            input_NWBfile=nwbfile,
            requested_parameters=['trial_len']
        )

        # TODO: Where is custom range needed? determine a global way
        # of passing it around...
        custom_range = (20, int(trial_len / 50))

        K_star, K_labels = analysis.determine_number_of_clusters(
            NWBfile_array=[nwbfile],
            max_clusters=no_of_conditions,
            y_array=y_array,
            custom_range=custom_range
        )

        K_star_over_trials[learning_condition - 1, :] = \
            K_star[y_i]

    optimal_clusters_of_group[nb.datasetName(animal_model)] = \
        K_star_over_trials



figure2_axis[1, 2].set_title('Optimal no of clusters')
bplots = []
for pos, animal in enumerate(range(1, no_of_animals + 1)):
    bp = figure2_axis[1, 2].boxplot(
        optimal_clusters_of_group[nb.datasetName(animal)],
        positions=[pos],
        widths=0.4,
        patch_artist=True
    )
figure2_axis[1, 2].set_xlim(-1, 4)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.20)

figure2.savefig('Figure_2.svg')
print('Tutto pronto!')
sys.exit()
#===============================================================================
#===============================================================================
# FIGURE 2 Network capacity.
# Figure 2A:

# Figure2B einai to PCA clusters 2d, whole animals (ONLY STR)

#===============================================================================
#===============================================================================
# Run for every learning condition and animal the k-means clustering:
no_of_conditions = 10
no_of_animals = 1
optimal_clusters_of_group = defaultdict(partial(np.ndarray, 0))
configurations = {1: 'structured', 2: 'random'}
for animal_model in range(1, no_of_animals + 1):
    # Pool together no of clusters for one animal model:
    K_star_over_trials = np.zeros((no_of_conditions, 2))
    for config_id, config in configurations.items():
        for learning_condition in range(1, no_of_conditions + 1):
            # Lazy load the data as a NWB file. Easy to pass around and
            # encapsulates info like trial length, stim times etc.
            #TODO: set them up so data_path is local:
            #TODO: this might raised some exceptions. Investigate!
            nwbfile = analysis.load_nwb_file(
                animal_model=animal_model,
                learning_condition=learning_condition,
                experiment_config=config,
                type='bn',
                data_path=Path(r'G:\Glia')
            )

            trial_len, ncells = analysis.get_acquisition_parameters(
                input_NWBfile=nwbfile,
                requested_parameters=['trial_len', 'ncells']
            )

            try:
                # TODO: Where is custom range needed? determine a global way
                # of passing it around...
                custom_range = (20, int(trial_len / 50))
                # Determine the optimal number of clusters for all trials of a
                # single animal model/learning condition.
                y_array = np.linspace(0.1, 100, 1000)
                K_star, K_labels = analysis.determine_number_of_clusters(
                    input_NWBfiles=[nwbfile],
                    max_clusters=no_of_conditions,
                    y_array=y_array,
                    custom_range=custom_range
                )

                # Utilize only one y power, based on the dataset:
                y_i = 500
                K_star_over_trials[learning_condition - 1, config_id - 1] = \
                    K_star[y_i]
            except Exception as e:
                print(f'Got Exception during analysis {str(e)}')

    optimal_clusters_of_group[nb.datasetName(animal_model)] = \
        K_star_over_trials



# Figure 3C
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
        optimal_clusters_of_group[nb.datasetName(animal)],
        positions=[pos_a, pos_b],
        widths=0.4,
        patch_artist=True
    )
    if nb.is_significant(optimal_clusters_of_group[nb.datasetName(animal)]):
        nb.statisticalAnnotation(
            columns=(pos_a, pos_b),
            datamax=optimal_clusters_of_group[nb.datasetName(animal)].max(),
            axobj=ax1
        )
    bplots.append(bp)
nb.setBoxAttribtes(boxplot_handles=bplots, colors=['blue', 'red'])
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
# FIGURE 3



for animal_model in range(1, 4):
    for configuration in ['structured', 'random']:
        print(f'Config {configuration}, animal {animal_model}')
        NWBfiles = [
            analysis.load_nwb_file(
                animal_model=animal_model,
                learning_condition=learning_condition,
                experiment_config=configuration,
                type='bn',
                data_path=Path(r'G:\Glia')
            )
            for learning_condition in range(1, 2)
        ]

        #for nwbfile in NWBfiles:
        #    analysis.bin_activity(nwbfile, q_size=50)

        trial_len = NWBfiles[0].trials['stop_time'][0] - \
                    NWBfiles[0].trials['start_time'][0]  # in ms
        custom_range = (20, int(trial_len / 50))

        #analysis.pcaL2(
        #    NWBfiles,
        #    custom_range=custom_range,
        #    smooth=True, plot=True
        #)
        #print('blah')

        y_array = np.linspace(0.1, 100, 1000)
        K_star, K_labels = analysis.determine_number_of_clusters(
            NWBfiles,
            max_clusters=50,
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
        plt.savefig(Path(fr'C:\Users\steve\Pictures\Animal_{animal_model}_{configuration}_dim_5_nooverfitting.png'))
        #TODO: Na ta swzw kapou wste na mporw na ta xrhsimopoihsw meta..


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
                    type='bn',
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

