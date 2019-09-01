import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import analysis_tools as analysis
import sys

# This is the file that creates the NWB files.
# This should be done right after NEURON ends, but due to constraints it is done at a
# later time.

# Pass arguments from a dictionary to make it easier on the eyes.
# These are the parameters that are unknown; used to parse NEURON output into NWB format.
analysis_parameters = {
    'stim_start_offset': 50,
    'stim_stop_offset': 1050,
    'q_size': 50,
    'total_qs': None,
    'ntrials': 10,
    'trial_len': 3000,
    'ncells': 333,
    'spike_upper_threshold': 0,
    'spike_lower_threshold': -10,
    'data_dim': 2,
    'samples_per_ms': 10,
    'q_size': 50
}
analysis_parameters['total_qs'] = \
    int(np.floor(analysis_parameters['trial_len'] / analysis_parameters['q_size']))

# Parse each of the NWB files and keep only the spiking activity. Can you
# keep the properties? (are the fields for the acquisition the same as the
# ones for the unit (where spikes lie)?
# Keep the binned activity and pass the parameters from the membrane potential
# to the binned activity.


parse_neuron = {}

def excitatory_validation():
    # for structured condition:
    inputdir = Path(r'G:\Glia\publication_validation\excitatory_validation')
    outputdir = Path(r'G:\Glia')
    try:
        analysis.create_nwb_validation_file(
            inputdir=inputdir,
            outputdir=outputdir
        )
    except Exception as e:
        print(str(e))

    print('Done converting validation params and exiting.')
    sys.exit()
parse_neuron['excitatory_validation'] = excitatory_validation

def excitatory_validation_ampatest():
    # for structured condition:
    inputdir = Path(r'G:\Glia\publication_validation\excitatory_validation_ampatest_1')
    outputdir = Path(r'G:\Glia\new_files')
    try:
        analysis.create_nwb_validation_file_ampatest(
            inputdir=inputdir,
            outputdir=outputdir
        )
    except Exception as e:
        print(str(e))

    print('Done converting validation params and exiting.')
    sys.exit()
parse_neuron['excitatory_validation_ampatest'] = excitatory_validation_ampatest


def structured_half_reciprocals():
    # for structured half reciprocals condition:
    inputdir = Path(r'\\139.91.162.90\cluster\stefanos\Documents\Glia')
    outputdir = inputdir
    for animal_model in range(1, 2):
        for learning_condition in range(1, 11):
            new_params = {
                **analysis_parameters,
                'excitation_bias': 1.75,
                'inhibition_bias': 1.5,
                'nmda_bias': 6.0,
                'ampa_bias': 1.0,
                'sim_duration': 5,
                'trial_len': 5000,
                'animal_model': animal_model,
                'learning_condition': learning_condition,
                'experiment_config': 'structured_half_reciprocals'
            }
            try:
                analysis.create_nwb_file(
                    inputdir=inputdir,
                    outputdir=outputdir,
                    **new_params
                )
            except Exception as e:
                print(str(e))
                pass

    print('Done and exiting.')
    sys.exit()
parse_neuron['structured_half_reciprocals'] = structured_half_reciprocals

def structured():
    # for structured condition:
    inputdir = Path(r'G:\Glia\structured')
    for animal_model in range(1, 2):
        for learning_condition in range(1, 2):
            new_params = {
                **analysis_parameters,
                'excitation_bias': 1.75,
                'inhibition_bias': 3.0,
                'nmda_bias': 6.0,
                'ampa_bias': 1.0,
                'sim_duration': 5,
                'trial_len': 5000,
                'animal_model': animal_model,
                'learning_condition': learning_condition,
                'experiment_config': 'structured'
            }
            try:
                analysis.create_nwb_file(
                    inputdir=inputdir,
                    outputdir=outputdir,
                    add_membrane_potential=True,
                    **new_params
                )
            except Exception as e:
                print(str(e))
                pass

    print('Done and exiting.')
    sys.exit()
parse_neuron['structured']= structured

def random():
    # for random condition:
    inputdir = Path(r'G:\Glia\random')
    for animal_model in range(1, 5):
        for learning_condition in range(1, 11):
            new_params = {
                **analysis_parameters,
                'excitation_bias': 1.75,
                'inhibition_bias': 1.5,
                'nmda_bias': 6.0,
                'ampa_bias': 1.0,
                'sim_duration': 5,
                'trial_len': 5000,
                'animal_model': animal_model,
                'learning_condition': learning_condition,
                'experiment_config': 'random'
            }
            try:
                analysis.create_nwb_file(
                    inputdir=inputdir,
                    outputdir=outputdir,
                    add_membrane_potential=False,
                    **new_params
                )
            except Exception as e:
                print(str(e))
                pass

    print('Done and exiting.')
    sys.exit()
parse_neuron['random'] = random

def structured_nonmda():
    # for no NMDA condition:
    inputdir = Path(r'G:\Glia\nonmda')
    for animal_model in range(1, 5):
        for learning_condition in range(1, 6):
            new_params = {
                **analysis_parameters,
                'excitation_bias': 1.75,
                'inhibition_bias': 1.0,
                'nmda_bias': 0.0,
                'ampa_bias': 50.0,
                'sim_duration': 5,
                'trial_len': 5000,
                'animal_model': animal_model,
                'learning_condition': learning_condition,
                'experiment_config': 'structured_nonmda'
            }
            analysis.create_nwb_file(
                inputdir=inputdir,
                outputdir=outputdir,
                add_membrane_potential=False,
                **new_params
            )

    print('Done and exiting.')
    sys.exit()
parse_neuron['structured_nonmda'] = structured_nonmda

def structured_nomg(
    inputdir=Path(r'G:\Glia\noMg_NEWER'),
    outputdir=Path(r'G:\Glia\noMg_NEWER')
    ):
    # for No Mg condition:
    for animal_model in range(1, 2):
        for learning_condition in range(1, 6):
            new_params = {
                **analysis_parameters,
                'excitation_bias': 1.75,
                'inhibition_bias': 3.0,
                'nmda_bias': 6.0,
                'ampa_bias': 1.0,
                'sim_duration': 5.0,
                'trial_len': 5000,
                'animal_model': animal_model,
                'learning_condition': learning_condition,
                'experiment_config': 'structured_nomg'
            }
            analysis.create_nwb_file(
                inputdir=inputdir,
                outputdir=outputdir,
                add_membrane_potential=False,
                fn_prefix='',
                **new_params
            )

    print('Done and exiting.')
    sys.exit()
parse_neuron['structured_nomg'] = structured_nomg

parse_neuron['structured_nomg']()
sys.exit()

# Plot AMPA saturation to test it:
#parse_neuron['excitatory_validation_ampatest']()
# Call the analysis on it:
simulations_dir = Path(r'G:\Glia\new_files')
input_NWBfile = simulations_dir.joinpath('excitatory_validation_ampatest.nwb')
nwbfile = NWBHDF5IO(str(input_NWBfile), 'r').read()
per_trial_activity = {}
per_trial_activity['soma_NMDA+AMPA'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='normal_NMDA+AMPA'
)
per_trial_activity['soma_AMPA_only'] = analysis.separate_trials(
    input_NWBfile=nwbfile, acquisition_name='normal_AMPA_only'
)

normal_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['soma_NMDA+AMPA']
]
ampa_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['soma_AMPA_only']
]

input_NWBfile2 = simulations_dir.joinpath('excitatory_dense_validation.nwb')
nwbfile2 = NWBHDF5IO(str(input_NWBfile2), 'r').read()
per_trial_activity = {}
per_trial_activity['soma_org_AMPA_only'] = analysis.separate_trials(
    input_NWBfile=nwbfile2, acquisition_name='normal_AMPA_only'
)

ampa_org_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['soma_org_AMPA_only']
]

input_NWBfile2 = simulations_dir.joinpath('excitatory_validation_ampatest_1.nwb')
nwbfile2 = NWBHDF5IO(str(input_NWBfile2), 'r').read()
per_trial_activity = {}
per_trial_activity['soma_unit_AMPA_only'] = analysis.separate_trials(
    input_NWBfile=nwbfile2, acquisition_name='normal_AMPA_only'
)

ampa_unit_amplitude = [
    trace[0][500:5000].max() - trace[0][400]
    for trace in per_trial_activity['soma_unit_AMPA_only']
]

fig, A_axis_a = plt.subplots()
A_axis_a.plot(normal_amplitude[:25], color='C0')
A_axis_a.plot(ampa_amplitude[:25], color='C1')
A_axis_a.plot(ampa_org_amplitude[:25], color='C2')
A_axis_a.plot(ampa_unit_amplitude[:25], color='C3')
A_axis_a.set_xlabel('Stimulus intensity')
A_axis_a.set_ylabel('Amplitude (mV)')

'''
parse_neuron['structured_nomg'](
    inputdir=Path(r'/mnt/g/Glia/noMg_new'),
    outputdir=Path(r'/mnt/g/Glia/new_files'),
    )
'''
print('Tutto pronto!')

if False:
    # Read back NWB format and check/plot output:
    input_file = outputdir.joinpath(
        analysis.experiment_config_filename(animal_model=1, learning_condition=1)
    )
    nwbfile = NWBHDF5IO(str(input_file), 'r').read()
    # Get somatic voltage membrane potential:
    network_voltage_traces = nwbfile.acquisition['membrane_potential'].data
    #TODO: Basika kane ena script pou 8a sou kanei parse ta parameters apo to NWB file se ena dict px.

    ncells = network_voltage_traces.shape[0]
    ntrials = len(nwbfile.trials)
    # This is how to get stimulus times for each trial:
    nwbfile.epochs['start_time'][0]
    nwbfile.epochs['tags'][0]
    # This is the current experiment identifier (structured, random etc):
    experiment_id = nwbfile.identifier
    # Here I am using the same trial length for all my trials, because its a simulation,
    # so I safely grab the first one only.
    trial_len = nwbfile.trials['stop_time'][0] - nwbfile.trials['start_time'][0]  # in ms
    samples_per_ms = nwbfile.acquisition['membrane_potential'].rate / 1000  # Sampling rate (Hz) / ms
    conversion_factor = nwbfile.acquisition['membrane_potential'].conversion

    cells_with_spikes = nwbfile.units['cell_id']
    cell = 1
    cellid = cells_with_spikes[cell]
    vt = network_voltage_traces[cellid, ::samples_per_ms]
    #spikes = nwbfile.units.get_unit_spike_times(1)
    spike_train = nwbfile.units['spike_times'][cell].reshape(1, -1)

    # Plot single cell response
    #TODO: include stimulus line (this should be included in the NWB file and be ploted here).
    plt.ion()
    fig, ax = plt.subplots(2, 1)
    ax[0].eventplot(spike_train)
    ax[1].plot(vt)
    ax[1].set(xlabel='time (ms)', ylabel='voltage (mV)')
    ax[0].set(title='Spike events')
    plt.show()

    # Plot network activity:
    network_spikes = nwbfile.units['spike_times']
    fig, ax = plt.subplots()
    for idx in range(len(network_spikes)):
        plt.eventplot(network_spikes[idx], lineoffsets=[cells_with_spikes[idx]])
    ax.set(xlabel='time (ms)', ylabel='Neurons ID', title='Network spike activity')
    plt.show()

