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
figure1 = plt.figure(figsize=plt.figaspect(subplot_height / subplot_width))
figure1.patch.set_facecolor('white')
#TODO: I tend to believe that the w/hspace is RELATIVE to the size of this grid.
# This asks for a absolute number, in order to have a visually pleasing grid.
gs1 = gridspec.GridSpec(2, 4, left=0.05, right=0.95, top=0.95, bottom=0.10, wspace=0.35, hspace=0.2)
#gs1.update(left=0.05, right=0.30, wspace=0.05)
A_axis_a = plt.subplot(gs1[0, 0])
A_axis_b = plt.subplot(gs1[1, 0])
nb.mark_figure_letter(A_axis_a, 'a')

#gs1.update(left=0.05, right=0.30, wspace=0.05)
B_axis_a = plt.subplot(gs1[0, 1])
B_axis_b = plt.subplot(gs1[1, 1])
nb.mark_figure_letter(B_axis_a, 'b')

C_axis_a = plt.subplot(gs1[0, 2])
C_axis_b = plt.subplot(gs1[1, 2])
nb.mark_figure_letter(C_axis_a, 'c')

D_axis_a = plt.subplot(gs1[0, 3])
D_axis_b = plt.subplot(gs1[1, 3])
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
nb.axis_normal_plot(axis=A_axis_a)
nb.adjust_spines(A_axis_a, ['left', 'bottom'])

A_axis_b.plot(normal_amplitude[:25], color='C0')
A_axis_b.plot(mg_amplitude[:25], color='C1')
A_axis_b.set_xlabel('Stimulus intensity', fontsize=axis_label_font_size)
A_axis_b.set_ylabel('Amplitude (mV)', fontsize=axis_label_font_size)
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

B_axis_a.plot(stim_isi_hist / len(stim_ISI), color='C0')
B_axis_a.axvline(np.mean(stim_ISI) / step_isi, color='C0', linestyle='--')
B_axis_a.plot(delay_isi_hist / len(delay_ISI), color='C1')
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
B_axis_b.plot(stim_isi_cv_hist / len(stim_ISI_CV), color='C0')
B_axis_b.axvline(np.nanmean(stim_ISI_CV) / step_cv, color='C0', linestyle='--')
B_axis_b.plot(delay_isi_cv_hist / len(delay_ISI_CV), color='C1')
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

C_axis_a.plot(stim_isi_hist / len(stim_ISI), color='C0')
C_axis_a.axvline(np.mean(stim_ISI) / step_isi, color='C0', linestyle='--')
C_axis_a.plot(delay_isi_hist / len(delay_ISI), color='C1')
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
C_axis_b.plot(stim_isi_cv_hist / len(stim_ISI_CV), color='C0')
C_axis_b.axvline(np.nanmean(stim_ISI_CV) / step_cv, color='C0', linestyle='--')
C_axis_b.plot(delay_isi_cv_hist / len(delay_ISI_CV), color='C1')
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


plt.show()


# <codecell>
figure1.savefig('Figure_S2.png')
figure1.savefig('Figure_S2.pdf')
print('Tutto pronto!')


#%%



