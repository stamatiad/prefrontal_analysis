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



# import the notebook_module here and write down the code to reproduce the
# figures (to debug and test). Then copy the code to the jupyter notebook.
# Do only figures that will probably not change much.
simulations_dir = Path.cwd().joinpath('simulations')
glia_dir = Path(r'G:\Glia')
plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (15, 15)

y_array = np.linspace(0.1, 100, 1000)
y_i = 500
#TODO: move it in notebook module:
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
#===============================================================================
#===============================================================================
# FIGURE 1 (PENDING)
NWBfile = analysis.load_nwb_file(
    animal_model=1,
    learning_condition=1,
    experiment_config='structured',
    type='mp',
    data_path=glia_dir
    #type='bn',
    #data_path=simulations_dir
)


#fig, plot_axis =plt.subplots(1,1)
#plt.ion()
#for cellid in range(250, 333):
#    vsoma = analysis.get_acquisition_potential(
#        NWBfile=NWBfile, cellid=cellid, trialid=7
#    )
#    plot_axis.plot(vsoma)
#    plt.title(f'cellid {cellid}')
#    plt.waitforbuttonpress()
#    plt.cla()
#G_axis_a = plt.subplot(1,1,1, projection='3d')
#plt.ion()


subplot_width = 4
subplot_height = 3
figure1 = plt.figure(figsize=plt.figaspect(subplot_height / subplot_width))
plt.ion()
gs1 = gridspec.GridSpec(2, 2, left=0.05, right=0.30, top=0.95, bottom=0.52, wspace=0.15, hspace=0.2)
#gs1.update(left=0.05, right=0.30, wspace=0.05)
A_axis_a = plt.subplot(gs1[0, 0])
A_axis_b = plt.subplot(gs1[1, 0])
B_axis_a = plt.subplot(gs1[0, 1])
B_axis_b = plt.subplot(gs1[1, 1])
gs2 = gridspec.GridSpec(6, 2, left=0.32, right=0.98, top=0.95, bottom=0.52, wspace=0.15, hspace=0.2)
#gs2.update(left=0.32, right=0.98, wspace=0.05)
C_axis_a = plt.subplot(gs2[0, 0])
C_axis_b = plt.subplot(gs2[1, 0])
C_axis_c = plt.subplot(gs2[2, 0])
C_axis_d = plt.subplot(gs2[3, 0])
C_axis_e = plt.subplot(gs2[4, 0])
C_axis_f = plt.subplot(gs2[5, 0])
D_axis_a = plt.subplot(gs2[:2, 1])
D_axis_b = plt.subplot(gs2[3:, 1])

gs3 = gridspec.GridSpec(2, 2, left=0.05, right=0.48, top=0.48, bottom=0.05, wspace=0.2, hspace=0.2)
#gs3.update(left=0.05, right=0.48, hspace=0.05)
E_axis_a = plt.subplot(gs3[0, 0])
E_axis_b = plt.subplot(gs3[1, 0])
E_axis_c = plt.subplot(gs3[0, 1])
E_axis_d = plt.subplot(gs3[1, 1])

gs4 = gridspec.GridSpec(2, 2, left=0.55, right=0.98, top=0.48, bottom=0.05, wspace=0.2, hspace=0.2)
#gs4.update(left=0.55, right=0.98, hspace=0.05)
F_axis_a = plt.subplot(gs4[0, 0])
F_axis_b = plt.subplot(gs4[1, 0])
G_axis_a = plt.subplot(gs4[0, 1], projection='3d')
G_axis_b = plt.subplot(gs4[1, 1])




sketch_structured = plt.imread('../Clustered_network_sketch.png')
A_axis_a.imshow(sketch_structured, interpolation="nearest")
nb.hide_axis_border(axis=A_axis_a)

sketch_random = plt.imread('../Random_network_sketch.png')
A_axis_b.imshow(sketch_random, interpolation="nearest")
nb.hide_axis_border(axis=A_axis_b)

sketch_pyramidal = plt.imread('../Pyramidal.png')
B_axis_a.imshow(sketch_pyramidal, interpolation="nearest")
nb.hide_axis_border(axis=B_axis_a)

sketch_interneuron = plt.imread('../Interneuron.png')
B_axis_b.imshow(sketch_interneuron, interpolation="nearest")
nb.hide_axis_border(axis=B_axis_b)

# Figure C, D
# Load a NWB file containing membrane potential:
#TODO: remove with interpolation the extra steps per ms of membrane potential
# in order to reduce NWB file size.
NWBfile = analysis.load_nwb_file(
    animal_model=1,
    learning_condition=1,
    experiment_config='structured',
    type='mp',
    data_path=glia_dir
    #type='bn',
    #data_path=simulations_dir
)

pyramidal_axes = [C_axis_a, C_axis_b, C_axis_c]
interneuron_axes = [C_axis_d, C_axis_e, C_axis_f]
exemplar_pyramidal_ids = [1, 6, 17]
exemplar_interneurons_ids = [252, 257, 268]
for id, axis_obj in zip(exemplar_pyramidal_ids, pyramidal_axes):
    vsoma = analysis.get_acquisition_potential(
        NWBfile=NWBfile, cellid=id, trialid=7
    )
    axis_obj.plot(vsoma, color='k')
    nb.hide_axis_border(axis=axis_obj)

for id, axis_obj in zip(exemplar_interneurons_ids, interneuron_axes):
    vsoma = analysis.get_acquisition_potential(
        NWBfile=NWBfile, cellid=id, trialid=7
    )
    axis_obj.plot(vsoma, color='k')
    nb.hide_axis_border(axis=axis_obj)

# Exemplar network rasterplot:
nb.plot_trial_spiketrains(NWBfile=NWBfile, trialid=7, plot_axis=D_axis_a)

# Dynamic network response:
#TODO: if trial 0 has no PA, why is saved/accessed? When do I remove them?
# Trials that have pa: 2, 6
trial_inst_ff = analysis.trial_instantaneous_frequencies(
    NWBfile=NWBfile, trialid=7, smooth=True
)
ff_threshold = 20  # Hz
#fig, plot_axis =plt.subplots(1,1)
#plt.ion()
for cellid, inst_ff in trial_inst_ff:
    if inst_ff.mean() > ff_threshold:
        D_axis_b.plot(inst_ff)
    #plt.title(f'cellid {cellid}')
    #plt.waitforbuttonpress()
    #plt.cla()
D_axis_b.set_xlim([0.0, 5000])
D_axis_b.set_ylim([0.0, 160])
D_axis_b.spines['left'].set_position('zero')
D_axis_b.spines['bottom'].set_position('zero')
D_axis_b.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
plt.show()




# Figure Ea
no_of_conditions = 3#10
no_of_animals = 2#4
stim_ISI_all = []
stim_ISI_CV_all = []
delay_ISI_all = []
delay_ISI_CV_all = []
for animal_model in range(1, no_of_animals + 1):
    for learning_condition in range(1, no_of_conditions + 1):
        NWBfile = analysis.load_nwb_file(
            animal_model=animal_model,
            learning_condition=learning_condition,
            experiment_config='structured',
            type='bn',
            data_path=simulations_dir
        )
        # Calculate ISI and its CV:
        stim_ISIs, stim_ISIs_CV = analysis.calculate_stimulus_isi(NWBfile)
        delay_ISIs, delay_ISIs_CV = analysis.calculate_delay_isi(NWBfile)

        stim_ISI_all.append(stim_ISIs)
        stim_ISI_CV_all.append(stim_ISIs_CV)
        delay_ISI_all.append(delay_ISIs)
        delay_ISI_CV_all.append(delay_ISIs_CV)

stim_ISI = list(chain(*stim_ISI_all))
delay_ISI = list(chain(*delay_ISI_all))
stim_ISI_CV = list(chain(*stim_ISI_CV_all))
delay_ISI_CV = list(chain(*delay_ISI_CV_all))
stim_isi_hist, *_ = np.histogram(stim_ISI, bins=np.arange(0, 200, 20))
delay_isi_hist, *_ = np.histogram(delay_ISI, bins=np.arange(0, 200, 20))
stim_isi_cv_hist, *_ = np.histogram(stim_ISI_CV, bins=np.arange(0, 2, 0.2))
delay_isi_cv_hist, *_ = np.histogram(delay_ISI_CV, bins=np.arange(0, 2, 0.2))

axis_label_font_size = 10
E_axis_a.plot(stim_isi_hist / len(stim_ISI), color='C0')
E_axis_a.plot(delay_isi_hist / len(delay_ISI), color='C1')
E_axis_a.set_xlabel('ISI length (ms)', fontsize=axis_label_font_size)
E_axis_a.set_ylabel('Relative Frequency', fontsize=axis_label_font_size)
nb.axis_normal_plot(axis=E_axis_a)
E_axis_b.plot(stim_isi_cv_hist / len(stim_ISI_CV), color='C0')
E_axis_b.plot(delay_isi_cv_hist / len(delay_ISI_CV), color='C1')
E_axis_b.set_xlabel('Coefficient of Variation', fontsize=axis_label_font_size)
E_axis_a.set_ylabel('Relative Frequency', fontsize=axis_label_font_size)
nb.axis_normal_plot(axis=E_axis_b)

# Figure 1E
# Lazy load the data as a NWB file.
input_NWBfile = simulations_dir.joinpath('excitatory_validation.nwb')
nwbfile = NWBHDF5IO(str(input_NWBfile), 'r').read()
per_trial_activity = {}
per_trial_activity['soma_NMDA+AMPA'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='normal_NMDA+AMPA'
)
per_trial_activity['dend_NMDA+AMPA'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='vdend_normal_NMDA+AMPA'
)

#TODO: why my data seems to be x4 times? This is also in the previous,somatic
# data that I have plotted successfully.. It seems to be a problem with the
# NWB file creation.
soma_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['soma_NMDA+AMPA']
]
dend_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['dend_NMDA+AMPA']
]

E_axis_c
E_axis_c.plot(soma_amplitude, color='C0')
E_axis_c.plot(dend_amplitude, color='C1')
E_axis_c.set_xlabel('Stimulus intensity', fontsize=axis_label_font_size)
E_axis_c.set_ylabel('Amplitude (mV)', fontsize=axis_label_font_size)
nb.axis_normal_plot(axis=E_axis_c)
#TODO: make ticks right!


sketch_amplitude = plt.imread('../Amplitude_Nevian.png')
E_axis_d.imshow(sketch_amplitude, interpolation="nearest")
E_axis_d.margins(0.0)
nb.hide_axis_border(axis=E_axis_d)

# Figure 1G
NWBfile = analysis.load_nwb_file(
    animal_model=1,
    learning_condition=2,
    experiment_config='structured',
    type='bn',
    data_path=simulations_dir
)
trial_len, pn_no, ntrials, trial_q_no = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len', 'pn_no', 'ntrials', 'trial_q_no']
)
custom_range = (0, int(trial_len / 50))

pca_net_activity = analysis.plot_pca_in_3d(
    NWBfile=NWBfile, custom_range=custom_range, smooth=True, plot_axes=G_axis_a
)
#azim, elev = G_axis_a.azim, G_axis_a.elev
G_axis_a.view_init(elev=-109.4, azim=53.2)

# Figure 1Gb:
# Plot velocity from raw network activity:
raw_net_activity = NWBfile. \
                              acquisition['binned_activity']. \
                              data[:pn_no, :]. \
    reshape(pn_no, ntrials, trial_q_no)
velocity = analysis.md_velocity(pca_data=raw_net_activity)
G_axis_b.plot(velocity.T, color='gray', alpha=0.2)
G_axis_b.plot(np.mean(velocity.T, axis=1), color='k', linewidth=2)
G_axis_b.set_ylabel('Energy Velocity (Hz/s)')
G_axis_b.set_xlabel('Time (ms)')
nb.axis_normal_plot(axis=G_axis_b)

print('blah')
# Figure 1F
# Load binned acquisition (all trials together)
binned_network_activity = NWBfile. \
      acquisition['binned_activity']. \
      data[:pn_no, :]. \
    reshape(pn_no, ntrials, trial_q_no)

# Perform correlation in each time bin state:
#TODO: giati ta trials einai 9 (pou shmainei oti anixneftikan only PA ones),
# alla to trial 0 den exei PA?
single_trial_activity = binned_network_activity[
    :pn_no, 2, custom_range[0]:custom_range[1]
]
duration = single_trial_activity.shape[1]
timelag_corr = np.zeros((duration, duration))
for ii in range(duration):
    for jj in range(duration):
        S = np.corrcoef(
            single_trial_activity[:, ii],
            single_trial_activity[:, jj]
        )
        timelag_corr[ii, jj] = S[0, 1]

#figure1, plot_axes = plt.subplots()
im = F_axis_a.imshow(timelag_corr)
F_axis_a.xaxis.tick_top()
for axis in ['top', 'bottom', 'left', 'right']:
    F_axis_a.spines[axis].set_linewidth(2)
F_axis_a.xaxis.set_tick_params(width=2)
F_axis_a.yaxis.set_tick_params(width=2)
time_axis_limits = (0, duration)
#TODO: change the 20 with a proper variable (do I have one?)
time_axis_ticks = np.linspace(0, duration, (duration / 20) + 1)
time_axis_ticklabels = analysis.q2sec(q_time=time_axis_ticks).astype(int)  #np.linspace(0, time_axis_limits[1], duration)
F_axis_a.set_xticks(time_axis_ticks)
F_axis_a.set_xticklabels(time_axis_ticklabels)
F_axis_a.set_yticks(time_axis_ticks)
F_axis_a.set_yticklabels(time_axis_ticklabels)
F_axis_a.set_ylabel('Time (s)')
F_axis_a.set_xlabel('Correlation')
figure1.colorbar(im, orientation='horizontal', fraction=0.05)

sketch_correlation = plt.imread('../Correlation_Murray.png')
F_axis_b.imshow(sketch_correlation, interpolation="nearest")
nb.hide_axis_border(axis=F_axis_b)



plt.show()

#===============================================================================
#===============================================================================
#print(f'Config {configuration}, animal {animal_model}')
#TODO: se afta to y prepei na einai daforetiko!
K_star_array = np.zeros((4, 10, 1000), dtype=int)
for animal_model in range(1, 5):
    for learning_condition in range(1, 11):
        print(f'am {animal_model}, lc {learning_condition}')
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

        #K_star = analysis.NNMF(
        #    NWBfile_array=[NWBfile],
        #    custom_range=custom_range,
        #    plot=False
        #)
        # Determine the optimal number of clusters for all trials of a single animal
        # model/learning condition.
        K_star = analysis.determine_number_of_ensembles(
            NWBfile_array=[NWBfile],
            max_clusters=50,
            y_array=y_array,
            custom_range=custom_range
        )
        K_star_array[animal_model - 1, learning_condition - 1, :] = K_star.T[0]
fig = plt.figure()
#plt.imshow(K_star_array[:, :, 50])
plt.plot(K_star_array[0,1,:])
plt.show()
print('blah')
# FIGURE 3 A, B



subplot_width = 3
subplot_height = 1
figure3 = plt.figure(figsize=plt.figaspect(subplot_height / subplot_width))
figure3_axis = np.zeros((subplot_height, subplot_width), dtype=object)
for idx in range(subplot_width):
    figure3_axis[0, idx] = figure3.add_subplot(
        subplot_height, subplot_width, idx + 1
    )

# Lazy load the data as a NWB file.
input_NWBfile = simulations_dir.joinpath('excitatory_validation.nwb')
nwbfile = NWBHDF5IO(str(input_NWBfile), 'r').read()
per_trial_activity = {}
per_trial_activity['normal_NMDA+AMPA'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='normal_NMDA+AMPA'
)
per_trial_activity['normal_AMPA_only'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='normal_AMPA_only'
)
per_trial_activity['noMg_NMDA+AMPA'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='noMg_NMDA+AMPA'
)

for trace in per_trial_activity['normal_NMDA+AMPA']:
    nmda_ampa_plot = figure3_axis[0, 0].plot(trace[0][500:5000], color='gray', label='NMDA+AMPA')
for trace in per_trial_activity['normal_AMPA_only']:
    ampa_only_plot = figure3_axis[0, 0].plot(trace[0][500:5000], color='C0', label='AMPA only')
figure3_axis[0, 0].set_xlabel('Time (ms)')
figure3_axis[0, 0].set_ylabel('Somatic depolarization (mV)')
figure3_axis[0, 0].legend((nmda_ampa_plot[0], ampa_only_plot[0]), ['NMDA+AMPA', 'AMPA only'], loc='upper right')

for trace in per_trial_activity['normal_NMDA+AMPA']:
    nmda_ampa_plot = figure3_axis[0, 1].plot(trace[0][500:5000], color='gray', label='NMDA+AMPA')
for trace in per_trial_activity['noMg_NMDA+AMPA']:
    nmda_nomg_plot = figure3_axis[0, 1].plot(trace[0][500:5000], color='C0', label='NMDA no Mg + AMPA')
figure3_axis[0, 1].set_xlabel('Time (ms)')
figure3_axis[0, 1].set_ylabel('Somatic depolarization (mV)')
figure3_axis[0, 1].legend((nmda_ampa_plot[0], nmda_nomg_plot[0]), ['NMDA+AMPA', 'NMDA no Mg + AMPA'], loc='upper right')


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



no_of_conditions = 3#10
no_of_animals = 4
optimal_clusters_of_group = defaultdict(partial(np.ndarray, 0))
configurations = ['structured', 'random']
for animal_model in range(1, no_of_animals + 1):
    # Pool together no of clusters for one animal model:
    K_star_over_trials = np.zeros((no_of_conditions, 2))
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
                K_star, K_labels = analysis.determine_number_of_clusters(
                    NWBfile_array=[NWBfile],
                    max_clusters=10,
                    y_array=y_array,
                    custom_range=custom_range
                )

                K_star_over_trials[learning_condition - 1, config_id] = \
                    K_star[y_i]
            except Exception as e:
                print(f'Got Exception during analysis {str(e)}')

    optimal_clusters_of_group[nb.datasetName(animal_model)] = \
        K_star_over_trials


figure3_axis[0, 2].set_title('Optimal no of clusters per configuration')
positions = [
    (position[0], position[0] + 1)
    for position in analysis.generate_slices(
        size=3, number=no_of_animals,
        start_from=1, to_slice=False
    )
]
bplots = []
for animal, (pos_a, pos_b) in zip(range(1, no_of_animals + 1), positions):
    bp = figure3_axis[0, 2].boxplot(
        optimal_clusters_of_group[nb.datasetName(animal)],
        positions=[pos_a, pos_b],
        widths=0.4,
        patch_artist=True,
        labels=configurations
    )
    '''
    if is_significant(optimal_clusters_of_group[datasetName(animal)]):
        statisticalAnnotation(
            columns=(pos_a, pos_b),
            datamax=optimal_clusters_of_group[datasetName(animal)].max(),
            axobj=ax1
        )
    '''
    bplots.append(bp)
setBoxAttribtes(boxplot_handles=bplots, colors=['C0', 'C2'])
figure3_axis[0, 2].set_xlim(0, 13)
figure3_axis[0, 2].set_xticks([
    p + 0.5
    for p, _ in positions
])
figure3_axis[0, 2].set_xticklabels([
    nb.datasetName(i)
    for i in range(1, no_of_animals + 1)
])
figure3_axis[0, 2].set_xlabel('Configurations')
figure3_axis[0, 2].set_ylabel('K*')
for tick in figure3_axis[0, 2].get_xticklabels():
    tick.set_rotation(45)

figure3_axis[0, 2].legend([bplots[0]['boxes'][0], bplots[0]['boxes'][1]], ['Structured', 'Random'], loc='upper right')

plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.30)

plt.show()
figure3.savefig('Figure_3.svg')
print('Tutto pronto!')
#===============================================================================
#===============================================================================
# FIGURE 2
subplot_width = 3
subplot_height = 2
figure2 = plt.figure(figsize=plt.figaspect(subplot_height / subplot_width))
figure2_axis = np.zeros((subplot_height, subplot_width), dtype=object)
for idx in range(subplot_width):
    figure2_axis[0, idx] = figure2.add_subplot(
        subplot_height, subplot_width, idx + 1, projection='3d'
    )

no_of_conditions = 10

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
models_list = range(1, no_of_animals + 1)
for pos, animal in enumerate(models_list):
    bp = figure2_axis[1, 2].boxplot(
        optimal_clusters_of_group[nb.datasetName(animal)],
        positions=[pos],
        widths=0.4,
        patch_artist=True
    )
figure2_axis[1, 2].set_xlim(-1, 4)
figure2_axis[1, 2].set_xticks(list(range(no_of_animals)))
figure2_axis[1, 2].set_xticklabels(['Model 1', 'Model 2', 'Model 3', 'Model 4'])
figure2_axis[1, 2].set_ylabel('K*')
for tick in figure2_axis[1, 2].get_xticklabels():
    tick.set_rotation(45)

plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.25)

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

