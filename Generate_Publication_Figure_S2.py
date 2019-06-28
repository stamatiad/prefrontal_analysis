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
from pynwb import NWBFile
from pynwb import NWBHDF5IO
from datetime import datetime

# <markdowncell>
# # Create figure 1.

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

plt.ion()
axis_label_font_size = 10
no_of_conditions = 10
no_of_animals = 4
#===============================================================================
#===============================================================================
subplot_width = 2
subplot_height = 1
figure_ratio = subplot_height / subplot_width
figure1 = plt.figure(figsize=plt.figaspect(figure_ratio))
figure1.patch.set_facecolor('white')
#TODO: I tend to believe that the w/hspace is RELATIVE to the size of this grid.
# This asks for a absolute number, in order to have a visually pleasing grid.

# c is the size of subplot space/margin, for both h/w (in figure scale).
# If you are really OCD, you can use a second one, scaled by fig aspect ratio.
cw = 0.05
ch = cw / figure_ratio
a_gs = nb.split_gridspec(2, 4, ch, cw, left=0.05, right=0.95, top=0.99, bottom=0.08)
b_gs = nb.split_gridspec(3, 1, ch, cw, gs=a_gs[:, 1])
c_gs = nb.split_gridspec(3, 1, ch, cw, gs=a_gs[:, 2])
d_gs = nb.split_gridspec(3, 1, ch, cw, gs=a_gs[:, 3])

A_axis_a = plt.subplot(a_gs[0, 0])
A_axis_b = plt.subplot(a_gs[1, 0])
nb.mark_figure_letter(A_axis_a, 'a')

B_axis_a = plt.subplot(b_gs[0, :])
B_axis_b = plt.subplot(b_gs[1, :])
B_axis_c = plt.subplot(b_gs[2, :])
nb.mark_figure_letter(B_axis_a, 'b')

C_axis_a = plt.subplot(c_gs[0, :])
C_axis_b = plt.subplot(c_gs[1, :])
C_axis_c = plt.subplot(c_gs[2, :])
nb.mark_figure_letter(C_axis_a, 'c')

D_axis_a = plt.subplot(d_gs[0, :])
D_axis_b = plt.subplot(d_gs[1, :])
D_axis_c = plt.subplot(d_gs[2, :])
nb.mark_figure_letter(D_axis_a, 'd')

# Figure 1A
# Lazy load the data as a NWB file.
#TODO: Load the dense (single synapse steps) validation files.
def create_nwb_validation_dense_file(inputdir=None, outputdir=None, **kwargs):
    # Create a NWB file from the results of the validation routines.

    print('Creating NWBfile.')
    nwbfile = NWBFile(
        session_description='NEURON validation results.',
        identifier='excitatory_validation',
        session_start_time=datetime.now(),
        file_create_date=datetime.now()
    )

    # Partially automate the loading with the aid of a reading function;
    # use a generic reading function that you make specific with partial and
    # then just load all the trials:
    synapse_activation = list(range(1, 25, 1))
    basic_kwargs = {'ncells': 1, 'ntrials': len(synapse_activation), \
                    'stim_start_offset': 100, 'stim_stop_offset': 140,
                    'trial_len': 700, 'samples_per_ms': 10}


    # 'Freeze' some portion of the function, for a simplified one:
    read_somatic_potential = partial(
        analysis.read_validation_potential,
        inputdir=inputdir,
        synapse_activation=synapse_activation,
        location='vsoma'
    )

    # Load first batch:
    analysis.import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='normal',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='normal_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )

    # Rinse and repeat:
    analysis.import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='normal',
            currents='AMPA',
            nmda_bias=0.0,
            ampa_bias=50.0,
        ),
        timeseries_name='normal_AMPA_only',
        timeseries_description='Validation',
        **basic_kwargs
    )

    # Rinse and repeat:
    analysis.import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_somatic_potential,
            condition='noMg',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='noMg_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )


    # Use partial to remove some of the kwargs that are the same:
    read_dendritic_potential = partial(
        analysis.read_validation_potential,
        inputdir=inputdir,
        synapse_activation=synapse_activation,
        location='vdend'
    )

    # Load dendritic potential:
    analysis.import_recordings_to_nwb(
        nwbfile=nwbfile,
        read_function=partial(
            read_dendritic_potential,
            condition='normal',
            currents='NMDA+AMPA',
            nmda_bias=6.0,
            ampa_bias=1.0,
        ),
        timeseries_name='vdend_normal_NMDA+AMPA',
        timeseries_description='Validation',
        **basic_kwargs
    )


    # write to file:
    output_file = outputdir.joinpath(
        'excitatory_dense_validation.nwb'
    )
    print(f'Writing to NWBfile: {output_file}')
    with NWBHDF5IO(str(output_file), 'w') as io:
        io.write(nwbfile)


# Create the new validation file:
# for structured condition:
if False:
    inputdir = Path(r'W:\taxidi\analysis\Glia\publication_validation\excitatory_validation_dense')
    outputdir = Path(r'W:\taxidi\analysis\Python\simulations')
    try:
        create_nwb_validation_dense_file(
            inputdir=inputdir,
            outputdir=outputdir
        )
    except Exception as e:
        print(str(e))

    print('Done converting validation params and exiting.')

def plot_cross_correlation(NWBfile, plot_axis):
    # Add cross correlation:
    trial_len, pn_no, ntrials, trial_q_no = analysis.get_acquisition_parameters(
        input_NWBfile=NWBfile,
        requested_parameters=['trial_len', 'pn_no', 'ntrials', 'trial_q_no']
    )
    custom_range = (0, int(trial_len / 50))
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
    im = plot_axis.imshow(timelag_corr, vmin=0.7)
    plot_axis.xaxis.tick_top()
    for axis in ['top', 'bottom', 'left', 'right']:
        plot_axis.spines[axis].set_linewidth(2)
    plot_axis.xaxis.set_tick_params(width=2)
    plot_axis.yaxis.set_tick_params(width=2)
    time_axis_limits = (0, duration)
    #TODO: change the 20 with a proper variable (do I have one?)
    time_axis_ticks = np.linspace(0, duration, (duration / 20) + 1)
    time_axis_ticklabels = analysis.q2sec(q_time=time_axis_ticks).astype(int)  #np.linspace(0, time_axis_limits[1], duration)
    plot_axis.set_xticks(time_axis_ticks)
    plot_axis.set_xticklabels(time_axis_ticklabels, fontsize=tick_label_font_size)
    plot_axis.set_yticks(time_axis_ticks)
    plot_axis.set_yticklabels(time_axis_ticklabels, fontsize=tick_label_font_size)
    plot_axis.set_ylabel(
        'Time (s)', fontsize=axis_label_font_size,
        labelpad=labelpad_y
    )
    #plot_axis.set_xlabel('')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(plot_axis)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    figure1.colorbar(im, orientation='horizontal', fraction=0.05,
                     cax=cax)
    cax.set_xlabel(
        'Correlation', fontsize=axis_label_font_size,
        labelpad=labelpad_x
    )
# Call the analysis on it:
input_NWBfile = simulations_dir.joinpath('excitatory_dense_validation.nwb')
nwbfile = NWBHDF5IO(str(input_NWBfile), 'r').read()
per_trial_activity = {}
per_trial_activity['soma_NMDA+AMPA'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='normal_NMDA+AMPA'
)
per_trial_activity['normal_AMPA_only'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='normal_AMPA_only'
)
per_trial_activity['noMg_NMDA+AMPA'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='noMg_NMDA+AMPA'
)

normal_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['soma_NMDA+AMPA']
]
ampa_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['normal_AMPA_only']
]
mg_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['noMg_NMDA+AMPA']
]
A_axis_a.plot(normal_amplitude[:25], color='C0')
A_axis_a.plot(ampa_amplitude[:25], color='C1')
A_axis_a.set_xlabel('Stimulus intensity', fontsize=axis_label_font_size)
A_axis_a.set_ylabel('Amplitude (mV)', fontsize=axis_label_font_size)
A_axis_a.set_xticks(range(0, len(normal_amplitude[:25]), 5))
A_axis_a.set_xticklabels(range(1, len(normal_amplitude[:25]) + 1, 5))
nb.axis_normal_plot(axis=A_axis_a)
nb.adjust_spines(A_axis_a, ['left', 'bottom'])

A_axis_b.plot(normal_amplitude[:25], color='C0')
A_axis_b.plot(mg_amplitude[:25], color='C1')
A_axis_b.set_xlabel('Stimulus intensity', fontsize=axis_label_font_size)
A_axis_b.set_ylabel('Amplitude (mV)', fontsize=axis_label_font_size)
A_axis_b.set_xticks(range(0, len(normal_amplitude[:25]), 5))
A_axis_b.set_xticklabels(range(1, len(normal_amplitude[:25]) + 1, 5))
nb.axis_normal_plot(axis=A_axis_b)
nb.adjust_spines(A_axis_b, ['left', 'bottom'])


# Figure S2b
# Plot firing frequencies of non-NMDA, random configurations.
stim_ISI_all = []
stim_ISI_CV_all = []
delay_ISI_all = []
delay_ISI_CV_all = []
no_of_conditions = 5
for animal_model in range(1, no_of_animals + 1):
    for learning_condition in range(1, no_of_conditions + 1):
        NWBfile = analysis.load_nwb_file(
            animal_model=animal_model,
            learning_condition=learning_condition,
            experiment_config='structured_nonmda',
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

nb.report_value(f'Fig S2B: CV stim mean', average_stim_cv)
nb.report_value(f'Fig S2B: CV stim std', std_stim_cv)
nb.report_value(f'Fig S2B: CV delay mean', average_delay_cv)
nb.report_value(f'Fig S2B: CV delay std', std_delay_cv)

nb.report_value(f'Fig S2B: ISI stim mean', average_stim_isi)
nb.report_value(f'Fig S2B: ISI stim std', std_stim_isi)
nb.report_value(f'Fig S2B: ISI delay mean', average_delay_isi)
nb.report_value(f'Fig S2B: ISI delay std', std_delay_isi)

nb.report_value(f'Fig S2B: CV Kruskal p', kruskal_result_cv.pvalue)
nb.report_value(f'Fig S2B: ISI Kruskal p', kruskal_result_isi.pvalue)

B_axis_a.plot(stim_isi_hist / stim_isi_hist.sum(), color='C0')
B_axis_a.axvline(np.mean(stim_ISI) / step_isi, color='C0', linestyle='--')
B_axis_a.plot(delay_isi_hist / delay_isi_hist.sum(), color='C1')
B_axis_a.axvline(np.mean(delay_ISI) / step_isi, color='C1', linestyle='--')
B_axis_a.set_xticks(range(0, bins_isi.size, 2))
B_axis_a.set_xticklabels(np.round(bins_isi * 2, 1), fontsize=tick_label_font_size)
B_axis_a.set_xlim([0.0, bins_isi.size])
B_axis_a.set_xlabel(
    'ISI length (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
B_axis_a.set_ylabel(
    'Relative Frequency', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=B_axis_a)
nb.adjust_spines(B_axis_a, ['left', 'bottom'])
#TODO: Why I have nans inside CV?
B_axis_b.plot(stim_isi_cv_hist / stim_isi_cv_hist.sum(), color='C0')
B_axis_b.axvline(np.nanmean(stim_ISI_CV) / step_cv, color='C0', linestyle='--')
B_axis_b.plot(delay_isi_cv_hist / delay_isi_cv_hist.sum(), color='C1')
B_axis_b.axvline(np.nanmean(delay_ISI_CV) / step_cv, color='C1', linestyle='--')
B_axis_b.set_xticks(range(0, bins_cv.size, 2))
B_axis_b.set_xticklabels(np.round(bins_cv * 2, 1), fontsize=tick_label_font_size)
B_axis_b.set_xlim([0.0, bins_cv.size])
B_axis_b.set_xlabel(
    'Coefficient of Variation', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
B_axis_b.set_ylabel(
    'Relative Frequency', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=B_axis_b)
nb.adjust_spines(B_axis_b, ['left', 'bottom'])
nb.mark_figure_letter(B_axis_a, 'b')
plot_cross_correlation(NWBfile, B_axis_c)



# Figure S2c
# Plot firing frequencies of non-Mg, random configurations.
stim_ISI_all = []
stim_ISI_CV_all = []
delay_ISI_all = []
delay_ISI_CV_all = []
no_of_animals = 1
no_of_conditions = 5
for animal_model in range(1, no_of_animals + 1):
    for learning_condition in range(1, no_of_conditions + 1):
        NWBfile = analysis.load_nwb_file(
            animal_model=animal_model,
            learning_condition=learning_condition,
            experiment_config='structured_nomg',
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

nb.report_value(f'Fig S2C: CV stim mean', average_stim_cv)
nb.report_value(f'Fig S2C: CV stim std', std_stim_cv)
nb.report_value(f'Fig S2C: CV delay mean', average_delay_cv)
nb.report_value(f'Fig S2C: CV delay std', std_delay_cv)

nb.report_value(f'Fig S2C: ISI stim mean', average_stim_isi)
nb.report_value(f'Fig S2C: ISI stim std', std_stim_isi)
nb.report_value(f'Fig S2C: ISI delay mean', average_delay_isi)
nb.report_value(f'Fig S2C: ISI delay std', std_delay_isi)

nb.report_value(f'Fig S2C: CV Kruskal p', kruskal_result_cv.pvalue)
nb.report_value(f'Fig S2C: ISI Kruskal p', kruskal_result_isi.pvalue)

C_axis_a.plot(stim_isi_hist / stim_isi_hist.sum(), color='C0')
C_axis_a.axvline(np.mean(stim_ISI) / step_isi, color='C0', linestyle='--')
C_axis_a.plot(delay_isi_hist / delay_isi_hist.sum(), color='C1')
C_axis_a.axvline(np.mean(delay_ISI) / step_isi, color='C1', linestyle='--')
C_axis_a.set_xticks(range(0, bins_isi.size, 2))
C_axis_a.set_xticklabels(np.round(bins_isi * 2, 1), fontsize=tick_label_font_size)
C_axis_a.set_xlim([0.0, bins_isi.size])
C_axis_a.set_xlabel(
    'ISI length (ms)', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
C_axis_a.set_ylabel(
    'Relative Frequency', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=C_axis_a)
nb.adjust_spines(C_axis_a, ['left', 'bottom'])
#TODO: Why I have nans inside CV?
C_axis_b.plot(stim_isi_cv_hist / stim_isi_cv_hist.sum(), color='C0')
C_axis_b.axvline(np.nanmean(stim_ISI_CV) / step_cv, color='C0', linestyle='--')
C_axis_b.plot(delay_isi_cv_hist / delay_isi_cv_hist.sum(), color='C1')
C_axis_b.axvline(np.nanmean(delay_ISI_CV) / step_cv, color='C1', linestyle='--')
C_axis_b.set_xticks(range(0, bins_cv.size, 2))
C_axis_b.set_xticklabels(np.round(bins_cv * 2, 1), fontsize=tick_label_font_size)
C_axis_b.set_xlim([0.0, bins_cv.size])
C_axis_b.set_xlabel(
    'Coefficient of Variation', fontsize=axis_label_font_size,
    labelpad=labelpad_x
)
C_axis_b.set_ylabel(
    'Relative Frequency', fontsize=axis_label_font_size,
    labelpad=labelpad_y
)
nb.axis_normal_plot(axis=C_axis_b)
nb.adjust_spines(C_axis_b, ['left', 'bottom'])
nb.mark_figure_letter(C_axis_a, 'c')
plot_cross_correlation(NWBfile, C_axis_c)

# Figure S2d
# Plot firing frequencies of non-Mg, random configurations.
stim_ISI_all = []
stim_ISI_CV_all = []
delay_ISI_all = []
delay_ISI_CV_all = []
no_of_animals = 4
no_of_conditions = 10
for animal_model in range(1, no_of_animals + 1):
    for learning_condition in range(1, no_of_conditions + 1):
        NWBfile = analysis.load_nwb_file(
            animal_model=animal_model,
            learning_condition=learning_condition,
            experiment_config='random',
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

nb.report_value(f'Fig S2D: CV stim mean', average_stim_cv)
nb.report_value(f'Fig S2D: CV stim std', std_stim_cv)
nb.report_value(f'Fig S2D: CV delay mean', average_delay_cv)
nb.report_value(f'Fig S2D: CV delay std', std_delay_cv)

nb.report_value(f'Fig S2D: ISI stim mean', average_stim_isi)
nb.report_value(f'Fig S2D: ISI stim std', std_stim_isi)
nb.report_value(f'Fig S2D: ISI delay mean', average_delay_isi)
nb.report_value(f'Fig S2D: ISI delay std', std_delay_isi)

nb.report_value(f'Fig S2D: CV Kruskal p', kruskal_result_cv.pvalue)
nb.report_value(f'Fig S2D: ISI Kruskal p', kruskal_result_isi.pvalue)

D_axis_a.plot(stim_isi_hist / stim_isi_hist.sum(), color='C0')
D_axis_a.axvline(np.mean(stim_ISI) / step_isi, color='C0', linestyle='--')
D_axis_a.plot(delay_isi_hist / delay_isi_hist.sum(), color='C1')
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
plot_cross_correlation(NWBfile, D_axis_c)


plt.show()


# <codecell>
figure1.savefig('Figure_S2_new.png')
figure1.savefig('Figure_S2_new.pdf')
print('Tutto pronto!')


#%%



