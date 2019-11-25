# <markdowncell>
# # # Generate Figure 3
# The catalytic role of NMDA nonlinearities in the creation of stable states. A. Removing the NMDA synaptic conductance and compensating by increasing AMPA conductance, results in loss of both the non-linear jump and the depolarizing plateau, for increasing synaptic drive. B. Same as A, but now after removal of the Mg blockage component (see Methods) only the non-linear jump is removed, with the depolarizing plateau retained. C. Removing either the non-linear jump or the depolarizing plateau from the network excitatory connections, eliminates  WM state space stable states number. D. This reduced states number is also produced if the structured connection configuration is replaced by a random one. 
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

# <markdowncell>
# ## Create figure 2
#TODO: This becomes Figure 2
#TODO: Na to network na einai to idio me to Fig 1 H!

# <codecell>
simulations_dir = Path.cwd().joinpath('simulations')
glia_dir = Path(r'G:\Glia')
plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
axis_label_font_size = 12
tick_label_font_size = 12
labelpad_x = 10
labelpad_y = 10

no_of_conditions = 10
no_of_animals = 4
plt.ion()
# FIGURE 2
subplot_width = 3
subplot_height = 2
figure2 = plt.figure(figsize=plt.figaspect(subplot_height / subplot_width))
figure2.patch.set_facecolor('white')
figure2_axis = np.zeros((subplot_height, subplot_width), dtype=object)

# The Network activity in two PC:
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
#TODO: yparxei kapoios logos pou dinw edw KAI to stimulus?
custom_range = (20, int(trial_len / 50))
K_star, K_labels, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=[NWBfile],
    max_clusters=no_of_conditions,
    custom_range=custom_range
)

# Figure 2A:
#I_axis_a = plt.subplot(FGHI_gs[0, 1:], projection='3d')
A_axis = figure2.add_subplot(
    subplot_height, subplot_width, 1, projection='3d'
)
nb.mark_figure_letter(A_axis, 'i')
custom_range = (0, int(trial_len / 50))
analysis.plot_pca_in_3d(
    NWBfile=NWBfile, custom_range=custom_range, smooth=True, plot_axes=A_axis,
    klabels=K_labels
)
#azim, elev = A_axis_a.azim, A_axis_a.elev
print((A_axis.azim, A_axis.elev))
A_axis.view_init(elev=14, azim=-135)
nb.mark_figure_letter(A_axis, 'a')

# Figure 2B:
B_axis = figure2.add_subplot(
    subplot_height, subplot_width, 4,
    projection='3d'
)


B_axis.cla()
analysis.pcaL2(
    NWBfile_array=[NWBfile],
    klabels=K_labels,
    custom_range=custom_range,
    smooth=True, plot_3d=True,
    plot_axes=B_axis,
    plot_stim_color=True
)
nb.mark_figure_letter(B_axis, 'b')

# Figure 2C:
C_axis = figure2.add_subplot(
    subplot_height, subplot_width, 2
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
    nmda_ampa_plot = C_axis.plot(trace[0][500:5000:10], color='gray', label='NMDA+AMPA')
for trace in per_trial_activity['normal_AMPA_only']:
    ampa_only_plot = C_axis.plot(trace[0][500:5000:10], color='C0', label='AMPA only')
C_axis.set_xlabel(
    'Time (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
C_axis.set_ylabel(
    'Somatic depolarization (mV)', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
C_axis.legend((nmda_ampa_plot[0], ampa_only_plot[0]), ['NMDA+AMPA', 'AMPA only'], loc='upper right')
nb.axis_normal_plot(C_axis)
nb.adjust_spines(C_axis, ['left', 'bottom'], blowout=2)
nb.mark_figure_letter(C_axis, 'c')

# Figure 2D:
D_axis = figure2.add_subplot(
    subplot_height, subplot_width, 3, projection='3d'
)
# The Network activity in two PC:
NWBfile = analysis.load_nwb_file(
    animal_model=1,
    learning_condition=1,
    experiment_config='structured_nonmda',
    type='bn',
    data_path=simulations_dir
)
trial_len, pn_no, ntrials, trial_q_no = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len', 'pn_no', 'ntrials', 'trial_q_no']
)
custom_range = (0, int(trial_len / 50))

K_star, K_labels, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=[NWBfile],
    max_clusters=no_of_conditions,
    custom_range=custom_range
)

TR_sp = analysis.sparsness(NWBfile, custom_range)
nb.report_value('Fig 2C: BIC', BIC_val)
nb.report_value('Fig 2C: Sparsness', TR_sp)

D_axis.cla()
analysis.pcaL2(
    NWBfile_array=[NWBfile],
    klabels=K_labels,
    custom_range=custom_range,
    smooth=True, plot_3d=True,
    plot_axes=D_axis,
    plot_stim_color=True
)
nb.mark_figure_letter(D_axis, 'd')


# Figure 2E:
E_axis = figure2.add_subplot(
    subplot_height, subplot_width, 5
)
for trace in per_trial_activity['normal_NMDA+AMPA']:
    nmda_ampa_plot = E_axis.plot(trace[0][500:5000:10], color='gray', label='NMDA+AMPA')
for trace in per_trial_activity['noMg_NMDA+AMPA']:
    nmda_nomg_plot = E_axis.plot(trace[0][500:5000:10], color='C0', label='NMDA no Mg + AMPA')
E_axis.set_xlabel(
    'Time (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
E_axis.set_ylabel(
    'Somatic depolarization (mV)', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
E_axis.legend((nmda_ampa_plot[0], nmda_nomg_plot[0]), ['NMDA+AMPA', 'NMDA no Mg + AMPA'], loc='upper right')
nb.axis_normal_plot(E_axis)
nb.adjust_spines(E_axis, ['left', 'bottom'], blowout=2)
nb.mark_figure_letter(E_axis, 'e')

# Figure 2F:
F_axis = figure2.add_subplot(
    subplot_height, subplot_width, 6, projection='3d'
)
# The Network activity in two PC:
NWBfile = analysis.load_nwb_file(
    animal_model=1,
    learning_condition=2,
    experiment_config='structured_nomg',
    type='bn',
    data_path=simulations_dir
)
trial_len, pn_no, ntrials, trial_q_no = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len', 'pn_no', 'ntrials', 'trial_q_no']
)
custom_range = (0, int(trial_len / 50))

K_star, K_labels, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=[NWBfile],
    max_clusters=no_of_conditions,
    custom_range=custom_range
)

TR_sp = analysis.sparsness(NWBfile, custom_range)
nb.report_value('Fig 2E: BIC', BIC_val)
nb.report_value('Fig 2E: Sparsness', TR_sp)

F_axis.cla()
analysis.pcaL2(
    NWBfile_array=[NWBfile],
    klabels=K_labels,
    custom_range=custom_range,
    smooth=True, plot_3d=True,
    plot_axes=F_axis,
    plot_stim_color=True
)
#F_axis.set_title(f'')
nb.mark_figure_letter(F_axis, 'f')


plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.30)

plt.show()


# <codecell>
figure2.savefig('Figure_2_final_right.png')
figure2.savefig('Figure_2_final_right.pdf')
print('Tutto pronto!')


#%%



