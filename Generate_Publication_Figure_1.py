# <markdowncell>
# # # Generate Figure 1
# Single neuron and network responses following stimulation. A. Top: Schematic representation of the network model. Excitatory (E) connectivity profile was based on experimental data. The excitatory population was reciprocally connected with the Inhibitory (I) in a feedback way. Bottom: Random network connectivity changes only excitatory connectivity. B. Top: Cartoon morphology of the pyramidal model. Bottom: Same for fast-spiking interneuron. C. Top: three exemplar responses of pyramidals in a single trial. Bottom: Same for interneurons. D. Top: Network response activity raster plot of pyramidal (blue) and interneurons (red) to a 1 sec stimulus. Bottom: Same trial’s instantaneous firing frequencies of each pyramidal (> 20Hz), showing its highly dynamic response during delay period. E. Histograms of inter spike interval length (top) and Coefficient of Variation (bottom) of all the structured trials for the stimulus period (blue) and delay period (red). F. Top: Non-linear NMDA responses are generated in the basal dendrites of the pyramidal neurons (top) as in (Nevian et al. 2007b) (bottom). Somatic (blue) and dendritic (red) depolarization from resting potential in response to increasing stimulus intensity. G. Overall network response energy (mean firing rate; top) and multidimensional velocity (bottom) aligned on stimulus period onset. H. Top: Cross correlation of network states between the stimulus period and the delay period over time (aligned on stimulus onset, 1 s stimulus). Bottom: Experimentally reported correlation from (Murray et al. 2017). I.  Network responses for 10 trials, under one learning condition, reduced to their first three principal components. Colormap denotes time.
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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

# <markdowncell>
# # Create figure 1.

# <codecell>
simulations_dir = Path.cwd().joinpath('simulations')
#Changes between computers.
glia_dir = Path(r'W:\taxidi\analysis\Python\simulations')
plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (15, 15)
axis_label_font_size = 12
tick_label_font_size = 12
labelpad_x = 10
labelpad_y = 10

plt.ion()
no_of_conditions = 10
no_of_animals = 4
#===============================================================================
#===============================================================================
# FIGURE 1 (PENDING)

# Create the figure objects:
subplot_width = 5
subplot_height = 4
figure_ratio = subplot_height / subplot_width
figure1 = plt.figure(figsize=plt.figaspect(figure_ratio))

# c is the size of subplot space/margin, for both h/w (in figure scale).
# If you are really OCD, you can use a second one, scaled by fig aspect ratio.
cw = 0.05
ch = cw / figure_ratio
main_gs = nb.split_gridspec(2, 1, ch, cw, left=0.05, right=0.95, top=0.99, bottom=0.08)
upper_gs = nb.split_gridspec(1, 3, ch, cw, gs=main_gs[0, :])
lower_gs = nb.split_gridspec(1, 4, ch, cw, gs=main_gs[1, :])
#upper_gs = main_gs[0, :].subgridspec(1, 3, wspace=0.15, hspace=0.15)
#lower_gs = main_gs[1, :].subgridspec(1, 2, wspace=0.15, hspace=0.15)
AB_gs = nb.split_gridspec(2, 2, ch, cw, gs=upper_gs[0, 0])
#AB_gs = upper_gs[0, 0].subgridspec(2, 2, wspace=0.15, hspace=new_p)
A_axis_a = plt.subplot(AB_gs[0, 0])
A_axis_b = plt.subplot(AB_gs[0, 1])
B_axis_a = plt.subplot(AB_gs[1, :])
nb.mark_figure_letter(A_axis_a, 'a')
nb.mark_figure_letter(B_axis_a, 'b')

CD_gs = nb.split_gridspec(6, 4, ch, cw, gs=upper_gs[0, 1:])
#CD_gs = upper_gs[0, 1:].subgridspec(6, 4, wspace=0.15, hspace=new_p)
C_gs = CD_gs[:, :3].subgridspec(6, 1, hspace=0.1)
C_axis_a = plt.subplot(C_gs[0, 0])
C_axis_b = plt.subplot(C_gs[1, 0])
C_axis_c = plt.subplot(C_gs[2, 0])
C_axis_d = plt.subplot(C_gs[3, 0])
C_axis_e = plt.subplot(C_gs[4, 0])
C_axis_f = plt.subplot(C_gs[5, 0])
nb.mark_figure_letter(C_axis_a, 'c')
D_axis_a = plt.subplot(CD_gs[:3, 3:])
D_axis_b = plt.subplot(CD_gs[3:, 3:])
nb.mark_figure_letter(D_axis_a, 'd')

E_gs = nb.split_gridspec(2, 1, ch, cw, gs=lower_gs[0, :2])
#E_gs = lower_gs[0, 0].subgridspec(2, 1, wspace=0.15, hspace=0.15*2)
E_axis_a = plt.subplot(E_gs[0, :])
E_axis_b = plt.subplot(E_gs[1, :])
nb.mark_figure_letter(E_axis_a, 'e')
FGH_gs = nb.split_gridspec(2, 3, ch, cw, gs=lower_gs[0, 2:])
#FGHI_gs = lower_gs[0, 1].subgridspec(1, 3, wspace=0.15, hspace=0.15)
#FGH_gs = nb.split_gridspec(2, 2, ch, cw, gs=FGHI_gs[0, 0])
#FGH_gs = FGHI_gs[0, 0].subgridspec(5, 1, wspace=0.15, hspace=0.15*5)
F_axis_a = plt.subplot(FGH_gs[:, :2])
nb.mark_figure_letter(F_axis_a, 'f')
G_axis_a = plt.subplot(FGH_gs[0, 2])
nb.mark_figure_letter(G_axis_a, 'g')
H_axis_a = plt.subplot(FGH_gs[1, 2])
nb.mark_figure_letter(H_axis_a, 'h')

plt.show()
figure1.savefig('FINAL.png')
print('Tutto pronto!')


sketch_pyramidal = plt.imread('Pyramidal.png')
A_axis_a.imshow(sketch_pyramidal, interpolation="nearest")
nb.hide_axis_border(axis=A_axis_a)

# Figure 1A
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
A_axis_b.plot(soma_amplitude[:5], color='C0')
A_axis_b.plot(dend_amplitude[:5], color='C1')
A_axis_b.set_xlabel(
    'Stimulus intensity', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
A_axis_b.set_ylabel(
    'Amplitude (mV)', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=A_axis_b)
nb.adjust_spines(A_axis_b, ['left', 'bottom'])

#TODO: Keep only the structured PLUS input vectors to show the simulation protocol.
#sketch_structured = plt.imread('Clustered_network_sketch.png')
#A_axis_b.imshow(sketch_structured, interpolation="nearest")
#nb.hide_axis_border(axis=A_axis_b)
#nb.mark_figure_letter(A_axis_b, 'b')


# Figure C
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
    axis_obj.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
    nb.hide_axis_border(axis=axis_obj)

for id, axis_obj in zip(exemplar_interneurons_ids, interneuron_axes):
    vsoma = analysis.get_acquisition_potential(
        NWBfile=NWBfile, cellid=id, trialid=7
    )
    axis_obj.plot(vsoma, color='k')
    axis_obj.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
    nb.hide_axis_border(axis=axis_obj)

nb.mark_figure_letter(C_axis_a, 'c')

#nb.set_horizontal_scalebar(
#    axis=C_axis_f,
#    label='1 second',
#    relativesize=1000,
#    distfromy=0.01,
#    distfromx=0.4
#)


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
step_isi = 20
step_cv = 0.2
bins_isi = np.arange(0, 200, step_isi)
bins_cv = np.arange(0, 2, step_cv)
stim_isi_hist, *_ = np.histogram(stim_ISI, bins=bins_isi)
delay_isi_hist, *_ = np.histogram(delay_ISI, bins=bins_isi)
stim_isi_cv_hist, *_ = np.histogram(stim_ISI_CV, bins=bins_cv)
delay_isi_cv_hist, *_ = np.histogram(delay_ISI_CV, bins=bins_cv)

# Do Kruskar Wallis test on distributions:
kruskal_result_cv = stats.kruskal(stim_ISI_CV, delay_ISI_CV, nan_policy='omit')
kruskal_result_isi = stats.kruskal(stim_ISI, delay_ISI, nan_policy='omit')

average_stim_isi = np.mean(stim_ISI)
average_delay_isi = np.mean(delay_ISI)
average_stim_cv = np.nanmean(stim_ISI_CV)
average_delay_cv = np.nanmean(delay_ISI_CV)

std_stim_isi = np.std(stim_ISI)
std_delay_isi = np.std(delay_ISI)
std_stim_cv = np.nanstd(stim_ISI_CV)
std_delay_cv = np.nanstd(delay_ISI_CV)

D_axis_a.plot(stim_isi_hist / len(stim_ISI), color='C0')
D_axis_a.axvline(np.mean(stim_ISI) / step_isi, color='C0', linestyle='--')
D_axis_a.plot(delay_isi_hist / len(delay_ISI), color='C1')
D_axis_a.axvline(np.mean(delay_ISI) / step_isi, color='C1', linestyle='--')
D_axis_a.set_xticks(range(0, bins_isi.size, 2))
D_axis_a.set_xticklabels(np.round(bins_isi * 2, 1), fontsize=tick_label_font_size)
D_axis_a.set_xlim([0.0, bins_isi.size])
D_axis_a.set_xlabel(
    'ISI length (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
D_axis_a.set_ylabel(
    'Relative Frequency', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=D_axis_a)
nb.adjust_spines(D_axis_a, ['left', 'bottom'])
#TODO: Why I have nans inside CV?
D_axis_b.plot(stim_isi_cv_hist / len(stim_ISI_CV), color='C0')
D_axis_b.axvline(np.nanmean(stim_ISI_CV) / step_cv, color='C0', linestyle='--')
D_axis_b.plot(delay_isi_cv_hist / len(delay_ISI_CV), color='C1')
D_axis_b.axvline(np.nanmean(delay_ISI_CV) / step_cv, color='C1', linestyle='--')
D_axis_b.set_xticks(range(0, bins_cv.size, 2))
D_axis_b.set_xticklabels(np.round(bins_cv * 2, 1), fontsize=tick_label_font_size)
D_axis_b.set_xlim([0.0, bins_cv.size])
D_axis_b.set_xlabel(
    'Coefficient of Variation', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
D_axis_b.set_ylabel(
    'Relative Frequency', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=D_axis_b)
nb.adjust_spines(D_axis_b, ['left', 'bottom'])
nb.mark_figure_letter(D_axis_a, 'd')



# Figure Ea
#TODO bale to idio animal me to Fig H1.
#TODO: opws einai to NWBfile einai ena tyxaio apo to loop pio panw!
# Exemplar network rasterplot:
# Trials that have pa: 2, 6. The 6 is quite nice!
nb.plot_trial_spiketrains(
    NWBfile=NWBfile, trialid=6, plot_axis=E_axis_a,
    axis_label_font_size=axis_label_font_size,
    tick_label_font_size=tick_label_font_size,
    labelpad_x=labelpad_x, labelpad_y=labelpad_y
)

#TODO: blah
TR_sp = analysis.sparsness(NWBfile)
nb.report_value(f'Fig 1E: Sparsness', TR_sp)


trial_len = \
    analysis.get_acquisition_parameters(
        input_NWBfile=NWBfile,
        requested_parameters=[
            'trial_len'
        ]
    )

# Dynamic network response:
trial_inst_ff = analysis.trial_instantaneous_frequencies(
    NWBfile=NWBfile, trialid=6, smooth=True
)
ff_threshold = 20  # Hz
#fig, plot_axis =plt.subplots(1,1)
#plt.ion()
for cellid, inst_ff in trial_inst_ff:
    if inst_ff.mean() > ff_threshold:
        E_axis_b.plot(inst_ff)
    #plt.title(f'cellid {cellid}')
    #plt.waitforbuttonpress()
    #plt.cla()
E_axis_b.set_xlim([0.0, 5000])
E_axis_b.set_ylim([0.0, 160])
nb.adjust_spines(E_axis_b, ['left', 'bottom'])
#E_axis_b.spines['left'].set_position('zero')
#E_axis_b.spines['bottom'].set_position('zero')
E_axis_b.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
E_axis_b.xaxis.set_ticks(np.arange(0, trial_len + 1000, 1000))
E_axis_b.xaxis.set_ticklabels(np.arange(0, 5, 1), fontsize=tick_label_font_size)
E_axis_b.set_xlabel(
    'Time (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
E_axis_b.set_ylabel(
    'Firing Frequency (Hz)', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=E_axis_b)
nb.adjust_spines(E_axis_b, ['left', 'bottom'])
nb.mark_figure_letter(E_axis_a, 'e')



# Figure G:
# Use the same file for the PCA 3d also:
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

# Plot velocity from raw network activity:
#TODO: all the data and how many Ls in PCA? Must be all!
# This is Hz/Sec.
net_activity = analysis.get_correct_trials(NWBfile)
# Filter only trials with PA
energy = analysis.energy(data=net_activity)
G_axis_a.cla()
G_axis_a.plot(energy.T, color='gray', alpha=0.2)
tmp = np.mean(energy.T, axis=1)
G_axis_a.plot(tmp[1:], color='k', linewidth=2)
duration = net_activity.shape[2]
time_axis_ticks = np.linspace(0, duration, (duration / 20) + 1)
time_axis_ticklabels = analysis.q2sec(q_time=time_axis_ticks).astype(int)  #np.linspace(0, time_axis_limits[1], duration)
G_axis_a.set_xticks(time_axis_ticks)
G_axis_a.set_xticklabels(time_axis_ticklabels, fontsize=tick_label_font_size)
G_axis_a.set_ylabel(
    'Energy (Hz/s)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
G_axis_a.set_xlabel(
    'Time (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
#TODO: use proper values, not hardcoded!
G_axis_a.axvspan(50.0 / 50, 1050.0 /50 , ymin=0, ymax=1, color='g', alpha=0.2)
nb.axis_normal_plot(axis=G_axis_a)
nb.adjust_spines(G_axis_a, ['left', 'bottom'])
nb.mark_figure_letter(G_axis_a, 'g')

# Figure 1I:
velocity = analysis.velocity(data=net_activity)
H_axis_a.cla()
H_axis_a.plot(velocity.T, color='gray', alpha=0.2)
H_axis_a.plot(np.mean(velocity.T, axis=1), color='k', linewidth=2)
H_axis_a.set_xticks(time_axis_ticks)
H_axis_a.set_xticklabels(time_axis_ticklabels, fontsize=tick_label_font_size)
H_axis_a.set_ylabel(
    'Velocity (Hz/s)', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
H_axis_a.set_xlabel(
    'Time (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
#TODO: use proper values, not hardcoded!
H_axis_a.axvspan(50.0 / 50, 1050.0 /50 , ymin=0, ymax=1, color='g', alpha=0.2)
nb.axis_normal_plot(axis=H_axis_a)
nb.adjust_spines(H_axis_a, ['left', 'bottom'])
nb.mark_figure_letter(H_axis_a, 'h')

# Figure 1F:
# Load binned acquisition (all trials together)
binned_network_activity = NWBfile.acquisition['binned_activity'] \
                              .data[:pn_no, :] \
    .reshape(pn_no, ntrials, trial_q_no)

# Perform correlation in each time bin state:
#TODO: giati ta trials einai 9 (pou shmainei oti anixneftikan only PA ones),
# alla to trial 0 den exei PA?
single_trial_activity = binned_network_activity[
                        :pn_no, 7, custom_range[0]:custom_range[1]
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
F_axis_a.set_xticklabels(time_axis_ticklabels, fontsize=tick_label_font_size)
F_axis_a.set_yticks(time_axis_ticks)
F_axis_a.set_yticklabels(time_axis_ticklabels, fontsize=tick_label_font_size)
F_axis_a.set_ylabel(
    'Time (s)', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
#F_axis_a.set_xlabel('')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(F_axis_a)
cax = divider.append_axes('bottom', size='5%', pad=0.05)
figure1.colorbar(im, orientation='horizontal', fraction=0.05,
                 cax=cax)
cax.set_xlabel(
    'Correlation', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
nb.mark_figure_letter(F_axis_a, 'f')

# Figure 1H:
K_star, K_labels, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=[NWBfile],
    max_clusters=no_of_conditions,
    custom_range=custom_range
)

TR_sp = analysis.sparsness(NWBfile, custom_range)
nb.report_value('Fig 1H: BIC', BIC_val)
nb.report_value('Fig 1H: Sparsness', TR_sp)

plt.draw()


plt.show()


# <codecell>
figure1.savefig('Figure_1.pdf')
figure1.savefig('Figure_1.svg')
figure1.savefig('Figure_1.png')
print('Tutto pronto!')


#%%



