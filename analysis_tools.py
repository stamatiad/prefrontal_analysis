import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pandas as pd
from sklearn import decomposition
from scipy import spatial
import time
from functools import wraps
from collections import namedtuple
from scipy.signal import savgol_filter

from datetime import datetime
from pynwb import NWBFile
from pynwb import NWBHDF5IO
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from pynwb import TimeSeries
from collections import defaultdict
from functools import partial

def time_it(function):
    @wraps(function)
    def runandtime(*args, **kwargs):
        tic = time.perf_counter()
        result = function(*args, **kwargs)
        toc = time.perf_counter()
        print(f'{function.__name__} took {toc-tic} seconds.')
        return result
    return runandtime

experiment_config_filename = \
    'animal_model_{animal_model}_learning_condition_{learning_condition}_{experiment_config}.nwb'.format

simulation_template = (
    'SN{animal_model}'
    'LC{learning_condition}'
    'TR{trial}'
    '_EB{excitation_bias:.3f}'
    '_IB{inhibition_bias:.3f}'
    '_GBF2.000'
    '_NMDAb{nmda_bias:.3f}'
    '_AMPAb{ampa_bias:.3f}'
    '_{experiment_config}_simdur{sim_duration}').format

MD_params = namedtuple('MD_params', ['mu', 'S'])

def get_cell_type(cellid, pn_no):
    # simply determine the type, given that cells' ids are sorted with PN first:
    if cellid < pn_no:
        return 'PN'
    else:
        return 'PV'

def getargs(*argnames):
    '''getargs(*argnames, argdict)
    Convenience function to retrieve arguments from a dictionary in batch
    '''
    if len(argnames) < 2:
        raise ValueError('Must supply at least one key and a dict')
    if not isinstance(argnames[-1], dict):
        raise ValueError('last argument must be dict')
    kwargs = argnames[-1]
    if not argnames:
        raise ValueError('must provide keyword to get')
    if len(argnames) == 2:
        return kwargs.get(argnames[0])
    return [kwargs.get(arg) for arg in argnames[:-1]]

def nwb_iter(sequencelike):
    # This is a wrapper for NWB sequences. It returns something that you can iterate on.
    # Without this, iterating can be cumbersome (does not stop; you can't even create a list).
    # Although these types appear to support __len__ and __getitem__ :
    n = len(sequencelike)
    ctr = 0
    while ctr < n:
        yield sequencelike[ctr]
        ctr += 1

def load_nwb_file(**kwargs):
    animal_model, learning_condition, experiment_config, data_path = \
    getargs('animal_model', 'learning_condition', 'experiment_config', 'data_path', kwargs)

    filename = data_path.joinpath(experiment_config_filename(
        animal_model=animal_model,
        learning_condition=learning_condition,
        experiment_config=experiment_config
    ))
    nwbfile = NWBHDF5IO(str(filename), 'r').read()
    return nwbfile

def create_nwb_file(inputdir=None, outputdir=None, **kwargs):
    # Get parameters externally:
    experiment_config, animal_model, learning_condition, ntrials, trial_len, ncells, stim_start_offset, \
    stim_stop_offset, samples_per_ms, spike_upper_threshold, spike_lower_threshold, excitation_bias, \
        inhibition_bias, nmda_bias, ampa_bias, sim_duration = \
        getargs('experiment_config', 'animal_model', 'learning_condition', 'ntrials', 'trial_len', 'ncells', 'stim_start_offset', \
                   'stim_stop_offset', 'samples_per_ms', 'spike_upper_threshold', 'spike_lower_threshold', \
                'excitation_bias', 'inhibition_bias', 'nmda_bias', 'ampa_bias', 'sim_duration', kwargs)

    # the base unit of time is the ms:
    samples2ms_factor = 1 / samples_per_ms
    nsamples = trial_len * samples_per_ms

    # Expand the NEURON/experiment parameters in the acquisition dict:
    pn_no = 250
    pv_no = 83
    acquisition_description = {
        **kwargs,
        'pn_no': pn_no,
        'pv_no': pv_no
    }
    print('Creating NWBfile.')
    nwbfile = NWBFile(
        session_description='NEURON simulation results.',
        identifier=experiment_config,
        session_start_time=datetime.now(),
        file_create_date=datetime.now()
    )

    nwbfile.add_unit_column('cell_id', 'Id of the cell recorded')
    nwbfile.add_unit_column('cell_type', 'Type of the cell recorded (PN or PV)')
    nwbfile.add_trial_column('persistent_activity', 'If this trial has persistent activity')
    #nwbfile.add_epoch_column('stimulus')

    print('Loading files from NEURON output.')
    time_series_l = []
    spike_train_l = []
    spike_trains_d = defaultdict(partial(np.ndarray, 0))
    membrane_potential = np.zeros((ncells, nsamples * ntrials), dtype=float)
    for trial, (trial_start_t, trial_end_t) in enumerate(generate_slices(size=nsamples, number=ntrials)):
        # Search inputdir for files specified in the parameters
        inputfile = inputdir.joinpath(
            simulation_template(
                excitation_bias=excitation_bias,
                inhibition_bias=inhibition_bias,
                nmda_bias=nmda_bias,
                ampa_bias=ampa_bias,
                sim_duration=sim_duration,
                animal_model=animal_model,
                learning_condition=learning_condition,
                trial=trial,
                experiment_config=experiment_config
            )).joinpath('vsoma.hdf5')
        if inputfile.exists():
            # Convert dataframe to ndarray:
            voltage_traces = pd.read_hdf(inputfile, key='vsoma').values
            membrane_potential[:, trial_start_t:trial_end_t] = voltage_traces[:ncells, :nsamples]
            # Use a dict to save space:
            for cellid in range(ncells):
                #TODO: quick_spikes() appears to include spikes at the very last point in time. Is this a bug?
                spike_train = quick_spikes(
                    voltage_trace=voltage_traces[cellid],
                    upper_threshold=spike_upper_threshold,
                    lower_threshold=spike_lower_threshold,
                    plot=False
                )
                spike_trains_d[cellid] = np.append(
                    spike_trains_d[cellid], np.add(spike_train, trial_start_t * samples2ms_factor)
                )

            # Define the region of PA as the last 200 ms of the simulation:
            pa_stop = int(nsamples * samples2ms_factor) + trial_start_t * samples2ms_factor  # in ms
            pa_start = int(pa_stop - 200)
            has_persistent = False
            for cellid, spike_train in spike_trains_d.items():
                if any(spike_train > pa_start) and any(spike_train < pa_stop):
                    print(f'On trial:{trial}, cell:{cellid} has spikes, so PA.')
                    has_persistent = True
                    break
            #has_persistent = voltage_traces[:ncells, pa_start:pa_stop].max() > 0
            # Add trial:
            nwbfile.add_trial(
                start_time=trial_start_t * samples2ms_factor,
                stop_time=trial_end_t * samples2ms_factor,
                persistent_activity=has_persistent
            )
            # Add stimulus epoch for that trial:
            nwbfile.add_epoch(
                start_time=float(trial_start_t + stim_start_offset),
                stop_time=float(trial_start_t + stim_stop_offset),
                tags=f'trial {trial} stimulus'
            )
        else:
            #TODO: handle missing files!
            print('error!')
            pass
        print(f'Trial {trial}, processed.')

    # Chunk and compress the data:
    wrapped_data = H5DataIO(
        data=membrane_potential,
        chunks=True,  # <---- Enable chunking (although I'm not sure if it will do any good in my huge dataset.
        compression='gzip',
        compression_opts=9
    )
    # Add somatic voltage traces (all trials concatenated)
    vsoma_timeseries = TimeSeries(
        'membrane_potential',  # Name of the TimeSeries
        wrapped_data,  # Actual data
        'miliseconds',  # Base unit of the measurement
        starting_time=0.0,  # The timestamp of the first sample
        rate=10000.0,  # Sampling rate in Hz
        conversion=samples2ms_factor,  #  Scalar to multiply each element in data to convert it to the specified unit
        # Since we can only use strings, stringify the dict!
        description=str(acquisition_description)
    )
    nwbfile.add_acquisition(vsoma_timeseries)
    print('Time series acquired.')

    for cellid in range(ncells):
        if spike_trains_d[cellid].size > 0:
            # Get each trial start/end in seconds rather than ms:
            trial_intervals = [
                [trial_start, trial_end]
                for trial_start, trial_end in \
                generate_slices(
                    size=trial_len / 1000, number=ntrials, start_from=0
                )
            ]
            #TODO: save if unit is PN or PV
            nwbfile.add_unit(
                id=cellid,
                spike_times=spike_trains_d[cellid],
                obs_intervals=trial_intervals,
                cell_id=cellid,
                cell_type=get_cell_type(cellid, pn_no)
            )

    # write to file:
    output_file = outputdir.joinpath(
        experiment_config_filename(
            animal_model=animal_model, learning_condition=learning_condition,
            experiment_config=experiment_config
        )
    )
    print(f'Writing to NWBfile: {output_file}')
    with NWBHDF5IO(str(output_file), 'w') as io:
        io.write(nwbfile)

class open_hdf_dataframe():

    def __init__(self, filename=None, hdf_key=None):
        self.filename = filename
        self.hdf_key = hdf_key

    def __enter__(self):
        try:
            return pd.read_hdf(self.filename, key=self.hdf_key)
        except Exception as e:
            print(f'File ({self.filename}) or key ({self.hdf_key}) not found!')
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    pass

def bin_activity(input_NWBfile, **kwargs):
    q_size = getargs('q_size', kwargs)

    #Get ncells without load the whole aquisition dataset
    nwbfile_description_d = eval(
        input_NWBfile.acquisition['membrane_potential'].description
    )
    ncells = nwbfile_description_d['ncells']

    ntrials = len(input_NWBfile.trials)
    # This is the current experiment identifier (structured, random etc):
    experiment_id = input_NWBfile.identifier
    # Here I am using the same trial length for all my trials, because its a simulation,
    # so I safely grab the first one only.
    trial_len = input_NWBfile.trials['stop_time'][0] - input_NWBfile.trials['start_time'][0]  # in ms
    samples_per_ms = input_NWBfile.acquisition['membrane_potential'].rate / 1000  # Sampling rate (Hz) / ms
    conversion_factor = input_NWBfile.acquisition['membrane_potential'].conversion

    # CAUTION: these appear to return similar objects, where they dont. Be very careful on how you use them
    # together (e.g. zip() etc).
    # Also these appear to not behave like iterables. So create some out of them:
    cells_with_spikes = nwb_iter(input_NWBfile.units['cell_id'])
    spike_trains = nwb_iter(input_NWBfile.units['spike_times'])

    # Bin spiking activity for all trials/cells in total_qs bins of q_size size:
    #TODO: THIS REQUIRES that each trial is divided by q_size exactly! Impose that constraint!
    # How many qs in all trials?
    total_qs = int(np.floor(trial_len / q_size)) * ntrials
    trial_qs = int(np.floor(trial_len / q_size))
    binned_activity = np.zeros((ncells, total_qs), dtype=int)
    # This is essentially what we are doing, but since python is so slow, we refactor it with some optimized code.
    #for cellid, spike_train in zip(cells_with_spikes, iter(spike_trains)):
    #    for q, (q_start, q_end) in enumerate(generate_slices(size=q_size, number=total_qs)):
    #        binned_activity[cellid][q] = sum([1 for spike in spike_train if q_start <= spike and spike < q_end])
    #TODO: can you bypass the erroneous trials in the code below?
    try:
        for cellid, spike_train in zip(cells_with_spikes, spike_trains):
            #TODO: this is a serious bug!
            if spike_train.max() >= trial_len * ntrials:
                print('having spikes outside of trial! How is this possible?')
                spike_train = spike_train[:-1]
            bins = np.floor_divide(spike_train, q_size).astype(int)
            np.add.at(binned_activity[cellid][:], bins, 1)
    except Exception as e:
        print(str(e))

    # Chunk and compress the data:
    wrapped_data = H5DataIO(
        data=binned_activity,
        chunks=True,  # <---- Enable chunking (although I'm not sure if it will do any good in my huge dataset.
        compression='gzip',
        compression_opts=9
    )
    # Add binned activity:
    network_binned_activity = TimeSeries(
        'binned_activity',  # Name of the TimeSeries
        wrapped_data,  # Actual data
        'miliseconds',  # Base unit of the measurement
        starting_time=0.0,  # The timestamp of the first sample
        rate=20.0,  # Sampling rate in Hz
        conversion=float(q_size)  #  Scalar to multiply each element in data to convert it to the specified unit
    )
    input_NWBfile.add_acquisition(network_binned_activity)

    # Reshape before returning:
    return binned_activity.reshape(ncells, ntrials, trial_qs)

def simulation_to_network_activity(tofile=None, animal_models=None, learning_conditions=None, **kwargs):
    # Convert voltage traces to spike trains:
    configuration = kwargs.get('configuration', None)
    upper_threshold = kwargs.get('upper_threshold', None)
    lower_threshold = kwargs.get('lower_threshold', None)
    ntrials = kwargs.get('ntrials', None)
    total_qs = kwargs.get('total_qs', None)
    q_size = kwargs.get('q_size', None)
    cellno = kwargs.get('cellno', None)

    # For multiple runs:
    # Use panda dataframes to keep track across learning conditions and animal models.
    # Group multiple trials in the same dataframe, since these are our working unit.
    # Touch the output file:
    open(tofile, 'w').close()


    for animal_model in animal_models:
        for learning_condition in learning_conditions:
            windowed_activity = np.zeros((ntrials, total_qs, cellno), dtype=int)
            #data_key = f'SN{animal_model}LC{learning_condition}'
            dataset = experiment_config_filename(
                animal_model=animal_model,
                learning_condition=learning_condition
            )
            print(f'Handling: {dataset}')
            for trial in range(ntrials):
                inputfile = Path('G:\Glia') \
                    .joinpath(
                    simulation_template(animal_model=animal_model,
                                        learning_condition=learning_condition,
                                        trial=trial,
                                        configuration=configuration)) \
                    .joinpath('vsoma.hdf5')
                #with open_hdf_dataframe(filename=filename, hdf_key='vsoma') as df:
                if inputfile.exists():
                    # Convert dataframe to ndarray:
                    voltage_traces = pd.read_hdf(inputfile, key='vsoma').values
                    # Reduce each voltage trace to a list of spike times:
                    spike_trains = [
                        quick_spikes(voltage_trace=voltage_trace,
                                        upper_threshold=upper_threshold,
                                        lower_threshold=lower_threshold,
                                        plot=False)
                        for voltage_trace in voltage_traces
                    ]
                    # Sum spikes inside a window Q:
                    for cellid, spike_train in enumerate(spike_trains):
                        if len(spike_train) > 0:
                            for q, (q_start, q_end) in enumerate(generate_slices(size=q_size, number=total_qs)):
                                windowed_activity[trial][q][cellid] = sum([1 for spike in spike_train if q_start <= spike and spike < q_end])
            #fig, ax = plt.subplots()
            #ax.imshow(windowed_activity.reshape(total_qs * ntrials, cellno, order='C'))
            #plt.show()
            df = pd.DataFrame(windowed_activity.reshape(total_qs * ntrials, cellno, order='C'))
            df.to_hdf(tofile, key=dataset, mode='a')


#===================================================================================================================
def read_network_activity(fromfile=None, dataset=None, **kwargs):
    # Parse keyword arguments:
    ntrials = kwargs.get('ntrials', None)
    total_qs = kwargs.get('total_qs', None)
    cellno = kwargs.get('cellno', None)

    data = None
    # Read spiketrains and plot them
    df = pd.read_hdf(fromfile, key=dataset)
    #with open_hdf_dataframe(filename=filename, hdf_key=data_key) as df:
    if df is not None:
        data = df.values.T
        # Also reshape the data into a 3d array:
        data = data.reshape(data.shape[0], ntrials, total_qs, order='C')

    return data

def pcaL2(
        input_NWBfile, plot=False, custom_range=None, klabels=None,
        smooth=False, **kwargs
):
    #TODO: is this deterministic? Because some times I got an error in some
    # matrix.
    nwbfile_description_d = eval(input_NWBfile.acquisition['membrane_potential'].description)
    animal_model_id = nwbfile_description_d['animal_model']
    learning_condition_id = nwbfile_description_d['learning_condition']
    ncells = nwbfile_description_d['ncells']
    pn_no = nwbfile_description_d['pn_no']

    # Plot the  two first principal components of multiple trial binned activity.
    ntrials = len(input_NWBfile.trials)
    # Here I am using the same trial length for all my trials, because its a simulation,
    # so I safely grab the first one only.
    trial_len = input_NWBfile.trials['stop_time'][0] - input_NWBfile.trials['start_time'][0]  # in ms
    q_size = input_NWBfile.acquisition['binned_activity'].conversion
    trial_q_no = int(np.floor(trial_len / q_size))
    correct_trials_idx = list(
        nwb_iter(input_NWBfile.trials['persistent_activity'])
    )
    correct_trials_no = sum(correct_trials_idx)

    # Use custom_range to compute PCA only on a portion of the original data:
    if custom_range is not None:
        if not isinstance(custom_range, tuple):
            raise ValueError('Custom range must be a tuple!')
        trial_slice_start = int(custom_range[0])
        trial_slice_stop = int(custom_range[1])
        duration = trial_slice_stop - trial_slice_start
    else:
        duration = trial_q_no
    # Load binned acquisition (all trials together)
    binned_network_activity = input_NWBfile.acquisition['binned_activity'].data.data[:pn_no, :].reshape(pn_no, ntrials, trial_q_no)
    # Slice out non correct trials and unwanted trial periods:
    tmp = binned_network_activity[:, correct_trials_idx, trial_slice_start:trial_slice_stop]
    # Reshape in array with m=cells, n=time bins.
    tmp = tmp.reshape(pn_no, correct_trials_no * duration)

    # how many PCA components
    L = 2
    pca = decomposition.PCA(n_components=L)
    t_L = pca.fit_transform(tmp.T).T
    # Reshape PCA results into separate trials for plotting.
    t_L_per_trial = t_L.reshape(L, correct_trials_no, duration, order='C')
    if smooth:
        for trial in range(correct_trials_no):
            t_L_per_trial[0][trial][:] = savgol_filter(t_L_per_trial[0][trial][:], 11, 3)
            t_L_per_trial[1][trial][:] = savgol_filter(t_L_per_trial[1][trial][:], 11, 3)

    if plot:
        #Plots the t_L_r as 2d timeseries.
        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(111, projection='3d')
        plt.title(f'PCAL2 animal model {animal_model_id}, learning condition {learning_condition_id}')

        if klabels is not None:
            labels = klabels.tolist()
            nclusters = np.unique(klabels).size
            colors = cm.Set2(np.linspace(0, 1, nclusters))
            _, key_labels = np.unique(labels, return_index=True)
            handles = []
            for i, (trial, label) in enumerate(zip(range(correct_trials_no), labels)):
                x = t_L_per_trial[0][trial][:]
                y = t_L_per_trial[1][trial][:]
                handle, = ax.plot(x, y,
                                  range(duration),
                                  color=colors[label - 1],
                                  label=f'Cluster {label}'
                                  )
                if i in key_labels:
                    handles.append(handle)
            # Youmust group handles based on unique labels.
            plt.legend(handles)
        else:
            colors = cm.viridis(np.linspace(0, 1, duration - 1))
            for trial in range(correct_trials_no):
                for t, c in zip(range(duration - 1), colors):
                    ax.plot(
                        t_L_per_trial[0][trial][t:t+2],
                        t_L_per_trial[1][trial][t:t+2],
                        [t, t+1], color=c
                    )
        plt.show()

    return t_L_per_trial

def from_zero_to(x):
    '''
    Range wrapper, counting from zero to x excluded.
    :param x:
    :return:
    '''
    return range(x)

def from_one_to(x):
    '''
    Range wrapper, counting from one to x included.
    :param x:
    :return:
    '''
    return range(1, x + 1)

def quick_spikes(voltage_trace=None, upper_threshold=None, lower_threshold=None, samples_per_ms=10, plot=False):
    #   ADVANCED_SPIKE_COUNT(vt, lt, ht) find spikes in voltage trace vt ( that
    #   first cross high threshold and again low threshold).
    #   ADVANCED_SPIKE_COUNT(vt, lt, ht, 'threshold',true) will return spikes
    #   detected where NEURON with similar High threshold (ht) would (WIP).
    #   ADVANCED_SPIKE_COUNT(vt, lt, ht, 'plot',true) also plot results.
    #
    #   This updated function can handle:
    #   > voltage train without spikes
    #   > noisy (high freq) voltage train
    #
    #   author stamatiad.st@gmail.com

    #Find values above high threshold:
    upper_crossings = np.greater(voltage_trace, upper_threshold)
    lower_crossings = np.logical_not(upper_crossings)
    # You want the points where the vt crosses the upper threshold and again the lower one.
    # Simply detect crossing the upper threshold, across vt:
    ts_1 = np.add(upper_crossings.astype(int), np.roll(lower_crossings, 1).astype(int))
    spikes_start = np.nonzero(np.greater(ts_1, 1))[0]
    # Simply detect crossing the lower threshold, across vt:
    ts_2 = np.add(upper_crossings.astype(int), np.roll(lower_crossings.astype(int), -1))
    spikes_end = np.nonzero(np.greater(ts_2, 1))[0]
    # Make sure that we have the same amount of starts/ends:
    if spikes_start.size != spikes_end.size:
        raise ValueError('Check algorithm. Why is this happening?')
    # Then, get the maximum voltage in this region.
    spike_timings = []
    for start, stop in zip(spikes_start, spikes_end):
        spike_timings.append((np.argmax(voltage_trace[start:stop+1]) + start) / samples_per_ms)
    # Plot if requested.
    if plot:
        #vt_reduced = voltage_trace.loc[::samples_per_ms]
        vt_reduced = voltage_trace[::samples_per_ms]
        # Re index it:
        #vt_reduced.index = (range(vt_reduced.size))
        fig, ax = plt.subplots()
        ax.plot(vt_reduced)
        for st in spike_timings:
            plt.scatter(st, 50, s=10, marker='v', c='tab:red')

        ax.set(xlabel='time (ms)', ylabel='voltage (mV)',
               title='Spike events')

        #fig.savefig("test.png")
        plt.show()



    return spike_timings

def generate_slices(size=50, number=2, start_from=0, to_slice = True):
    '''
    Generate starting/ending positions of q_total windows q of size q_size.
    TODO: accomondate the user case starting from idx zero, rather than one.
    If the toslice is True, you got one extra included unit at the end, so the qs can be used to slice a array.
    :param q_size:
    :param q_total:
    :return:
    '''
    for q in range(number):
        q_start = q * size + start_from
        q_end = q_start + size + start_from
        # yield starting/ending positions of q (in ms)
        if to_slice:
            yield (q_start, q_end)
        else:
            yield (q_start, q_end - 1)

class NDPoint():
    def __init__(self, ndarray=None):
        assert ndarray.shape[0] == 1
        self.ndarray = ndarray
        self.m, self.n = ndarray.shape
        # this is point dimension
        self.dim = self.n


class MeanShiftCentroid(NDPoint):
    # This is specific to mean shift
    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = shift

    def __init__(self, ndarray=None):
        NDPoint.__init__(self, ndarray)
        self.shift = 100000

    def distance_from(self, pts_array):
        m, n = pts_array.shape
        assert self.dim == n, 'Arrays have different dimensions'
        distance = spatial.distance.pdist(
            np.concatenate((self.ndarray, pts_array), axis=0)
        )
        #TODO: do I start from 0 or 1: this is NOT squareform!
        return (distance[:m]).reshape(m, 1)

    def update(self, ndarray):
        # Updates/moves centroid. Also calculates norm of movement
        m, n = ndarray.shape
        assert m == 1, 'Nd array must be row vector!'
        self.shift = np.linalg.norm(self.ndarray - ndarray)
        self.ndarray = ndarray

    def to_point(self):
        return NDPoint(self.ndarray)
        pass





def K_rbf(distance=None, sigma=None):
    # Calculate K RBF for one or multiple points
    # distance is ||x-x`||
    return np.exp(-np.divide(np.power(distance, 2), 2 * (sigma ** 2)))

def mu_bar(k_rbf=None, xs=None):
    # calculate new mu_bar based on K_rbf:
    # K can be np.ndarray of one or many scalars
    m, n = xs.shape
    m2, n2 = k_rbf.shape
    assert m == m2, 'mu_bar: Arrays have different dims!'
    x_mu_bar = np.multiply(k_rbf, xs).sum(axis=0) / k_rbf.sum()
    return x_mu_bar.reshape(1,2)

def mean_shift(data=None, k=None, plot=False, **kwargs):
    #ntrials = kwargs.get('ntrials', None)
    ntrials = data.shape[1]
    #TODO: do I consider more than two dims?
    data_dim = 2  # kwargs.get('data_dim', None)
    #return [density_pts, sigma_hat]
    # k for the kmeans (how many clusters)
    # N are the number of trials (meanshift initial points)
    #N = properties.ntrials
    # Collapse data to ndim points (2d):
    new_len = data.shape[2]
    pts = data.reshape(data_dim, ntrials * new_len, order='C')
    # Take the average STD the cells in PCA space:
    sigma_hat = pts.std(axis=1)
    # TODO: make slices smaller, we need a rational size window (1sec?) to account for activity drifting.
    # TODO: to slicing den einai swsto gia C type arrays; prepei na to ftia3w, kai na bebaiw8w gia opou allou to xrisimopoiw!
    std_array = np.array([
        pts[:, slice_start:slice_end].std(axis=1)
        for slice_start, slice_end in generate_slices(size=new_len, number=ntrials)
    ])
    sigma_hat = std_array.mean(axis=0).mean()

    if plot:
        fig, ax = plt.subplots()
        plt.ion()
        ax.scatter(pts[0], pts[1], s=2, c='black')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.show()

    # Initialize k centroids:
    init_centroids = data.mean(axis=2)
    centroids = [
        MeanShiftCentroid(init_centroids[:, [x]].T)
        for x in range(ntrials)
    ]
    #rw_mus = squeeze(mean(X,1))'
    while any([centroid.shift for centroid in centroids] > sigma_hat/10):
        for centroid in centroids:
            k_rbf = K_rbf(
                distance=centroid.distance_from(pts.T),
                sigma=sigma_hat
            )
            mu_x_bar = mu_bar(k_rbf=k_rbf, xs=pts.T)
            #This centroid movement also updates the shift (movement distance):
            if plot:
                plt.plot(
                    # TypeError: only integer scalar arrays can be converted to a scalar index. Why I get this:
                    #np.concatenate(centroid.ndarray[0, [0]], mu_x_bar[0, [0]]),
                    [centroid.ndarray[0, 0], mu_x_bar[0, 0]],
                    [centroid.ndarray[0, 1], mu_x_bar[0, 1]],
                    c='lime'
                )
                plt.scatter(centroid.ndarray[:,0], centroid.ndarray[:,1], s=20, c='lime')
                plt.scatter(mu_x_bar[:,0], mu_x_bar[:,1], s=20, c='red', zorder=1000)
                plt.pause(0.0001)
            centroid.update(mu_x_bar)

    # Cast to a list of points:
    points = [
        centroid.to_point()
        for centroid in centroids
    ]

    return (points, sigma_hat)

def itriu(size=None, idx=None):
    # transform index (idx) to i,j pair in pdist, without getting its squareform:
    # Only for square (pdist)matrices of side size
    start = 1
    ctr = 0
    for i in range(size):
        for j in range(start, size):
            if ctr == idx:
                return (i, j)
            ctr += 1
        start += 1

def ndpoints2array(points=None, **kwargs):
    # TODO: works as expected??
    # convert a batch of same dimension points to an np.array, for handling:
    #Optionally keep only one dim (useful for plotting):
    assert len(points), 'Points array is empty!'
    only_dim = kwargs.get('only_dim', range(points[0].dim))
    pts_array = np.concatenate([
        point.ndarray[:, only_dim]
        for point in points
    ])
    return pts_array
    pass

def initialize_algorithm(data=None, k=None, plot=None, **kwargs):
    # TODO: remove the None default, so to have exceptions flying around in case of an error:
    # TODO: Do I consider more than two dimensions?
    data_dim = 2 #kwargs.get('data_dim', None)
    #ntrials = kwargs.get('ntrials', None)
    ntrials = data.shape[1]
    # return [J_k, label, dE_i_q, dM_i_q] = init_algo(X, m, plot_flag)

    mean_shift_points, sigma_hat = mean_shift(data=data, k=k, plot=False, **kwargs)
    # TODO: check if I get this error and handle it:
    #if any(isnan(density_pts))
    #    error('I got NaNs inside density points!');
    #end

    # Sort density points by their distance and merge them, pair-wise, until have only k of them:
    while len(mean_shift_points) > k:
        # Calculate distances between mean shift points:
        mean_shift_pts_inter_distance = spatial.distance.pdist(
            ndpoints2array(mean_shift_points)
        )
        # Locate the two points (i,j) having the smallest distance:
        idx = np.argsort(mean_shift_pts_inter_distance, axis=0)
        i, j = itriu(size=len(mean_shift_points), idx=idx[0])
        # Replace them with their mean:
        # TODO:this wont work
        tmp_point = NDPoint(
            np.mean(
                ndpoints2array([mean_shift_points[i], mean_shift_points[j]]),
            axis=0).reshape(1, 2)
        )
        mean_shift_points[i] = tmp_point
        mean_shift_points.pop(j)

    if plot:
        fig, ax = plt.subplots()
        plt.ion()
        ax.scatter(
            ndpoints2array(points=mean_shift_points, only_dim=0),
            ndpoints2array(points=mean_shift_points, only_dim=1),
        s=10, c='red')
        # Collapse data to ndim points (2d):
        new_len = data.shape[2]
        pts = data.reshape(data_dim, ntrials * new_len, order='C')
        ax.scatter(pts[0], pts[1], s=2, c='black')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.show()
        pass

    # Initialize the clusters with simple euclidean distance:
    ed_array = np.zeros((k, ntrials))
    for cluster in range(k):
        for trial in range(ntrials):
            ed_array[cluster, trial] = point2points_average_euclidean(
                point=mean_shift_points[cluster].ndarray, points=data[:, trial, :].T
            )
    klabels = ed_array.argmin(axis=0)
    return klabels

def create_agregate_dataset(klabels=None, k=None):
    # Group the trials in clusters, creating an aggregate dataset:
    aggregate_dataset = [
        np.nonzero(cluster == klabels)[0]
        for cluster in range(k)
    ]
    return aggregate_dataset

def mahalanobis_distance(idx_a=None, idx_b=None):
    # Compute the MD of a trial average (idx_b) from the cluster centroid (idx_a):
    # Return also the mu and S for later use.
    #TODO: check again how to handle e.g. mean to return a matrix not a vector. It really mess up the multiplications..
    dim, trials, t = idx_a.shape
    if any(np.array(idx_a.shape) < 1):
        return np.nan, MD_params(np.nan, np.nan)

    cluster_data = idx_a.reshape(dim, t * trials, order='C').T
    point_data = np.squeeze(idx_b).mean(axis=1).reshape(1, -1)
    mu = cluster_data.mean(axis=0).reshape(1, -1)
    S = np.cov(cluster_data.T)
    # Debug/scatter
    if False:
        fig, ax = plt.subplots(1,1)
        ax.scatter(point_data[:,0], point_data[:,1], s=50, c='r', marker='+')
        ax.scatter(np.squeeze(idx_b)[0,:], np.squeeze(idx_b)[1,:],s=20, c='r', marker='.')
        ax.scatter(cluster_data[:,0], cluster_data[:,1],s=5, c='k', marker='.')
        ax.scatter(mu[:,0], mu[:,1],s=50, c='k', marker='+')

    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError as e:
        raise e('Covariance matrix is not PD!')
    tmp = point_data - mu
    try:
        MD = np.sqrt(tmp @ np.linalg.inv(S) @ tmp.T)
    except np.linalg.LinAlgError as e:
        raise e
    return MD, MD_params(mu, S)

def point2points_average_euclidean(point=None, points=None):
    # return the average euclidean distance between the point a and points b
    # run dist of a single point against a list of points:
    m, n = points.shape
    distance = spatial.distance.pdist(
        np.concatenate((point, points), axis=0)
    )
    return np.mean(distance[:m])

@time_it
def kmeans_clustering(data=None, k=2, max_iterations=100, plot=False, **kwargs):
    #return (labels_final, J_k_final, S_i)
    # Perform kmeans clustering.
    # Input:
    #   X: n x d data matrix
    #   m: initialization parameter (k)
    # Adapted by Stefanos Stamatiadis (stamatiad.st@gmail.com).
    def rearrange_labels(klabels, distance_array, params):
        # Make the klabels unique and monotonically incrementing:
        # Change also the md_array row ordering to reflect that.
        key_labels = np.unique(klabels)
        # Labels start from 1:
        new_klabels = np.zeros(klabels.shape, dtype=int)
        new_params = {}
        for new_label, original_label in enumerate(key_labels):
            new_klabels[klabels == original_label] = new_label + 1
            # Swap distance array rows, utilizing slice copy and unpacking:
            distance_array[original_label, :],\
            distance_array[new_label, :] = \
                distance_array[new_label, :].copy(),\
                distance_array[original_label, :].copy()
            # Update parameters dictionary:
            new_params[new_label] = params[original_label]
        return new_klabels, distance_array, new_params

    def compare_arrays(arr1=None, arr2=None):
        arr_a = arr1.tolist()
        arr_b = arr2.tolist()
        assert len(arr_a) == len(arr_b), 'Arrays must have the same length!'
        for i, j in zip(arr_a, arr_b):
            if i != j:
                return False
        return True

    dims, ntrials, total_qs = data.shape
    # Initiallization step:
    klabels = initialize_algorithm(data=data, k=k, plot=plot, **kwargs)
    aggregate_dataset = create_agregate_dataset(klabels=klabels, k=k)
    for iteration in from_one_to(max_iterations):
        # Assignment step:
        md_array = np.zeros((k, ntrials))
        md_params_d = {}
        for cluster in range(k):
            for trial in range(ntrials):
                md_array[cluster, trial], md_params_d[cluster] = \
                    mahalanobis_distance(
                    idx_a=data[:, aggregate_dataset[cluster], :],
                    idx_b=data[:, [trial], :]
                )
        klabels_old = klabels
        klabels = np.nanargmin(md_array, axis=0)
        # Update step:
        aggregate_dataset = create_agregate_dataset(klabels=klabels, k=k)
        # Termination step:
        if k == 1:
            break
        #TODO: change name to something better:
        if compare_arrays(klabels_old, klabels):
            break
    # Calculate the within cluster distance:
    cumulative_dist = np.zeros((1, ntrials))
    for cluster_i in range(k):
        cumulative_dist[0, cluster_i] = np.nansum(
            md_array[cluster_i, [aggregate_dataset[cluster_i]]]
        )
    J_k = cumulative_dist.sum() / ntrials
    klabels, md_array, md_params_d = rearrange_labels(klabels, md_array, md_params_d)
    # Debug plot the results for quick inspection:
    if False:
        dim, trials, t = data.shape
        flatten_data = data.reshape(dim, t * trials, order='C').T
        fig, ax = plt.subplots(1,1)
        colors = cm.Set2(np.linspace(0, 1, len(klabels)))
        for i, l in enumerate(klabels):
            this_label = klabels == i
            len_label = sum(this_label)
            label_data = data[:, this_label, :].reshape(dim, t * len_label, order='C').T
            label_data_mean = label_data.mean(axis=0).reshape(1, -1)
            ax.scatter(label_data[:,0], label_data[:,1], s=20, c=colors[i,:], marker='.')
            ax.scatter(label_data_mean[:,0], label_data_mean[:,1], s=50, c=colors[i,:], marker='+')

        ax.scatter(flatten_data[:,0], flatten_data[:,1],s=5, c='k', marker='.')

    return klabels, J_k, md_array, md_params_d

def evaluate_clustering(klabels=None, md_array=None, md_params=None, **kwargs):
    # Calculate likelihood of each trial, given the cluster centroid:
    nclusters, ntrials = md_array.shape
    #TODO: do I use more than two dims?
    data_dim = 2  # kwargs.get('data_dim', None)

    ln_L = np.zeros((1, ntrials))
    ln_L.fill(np.nan)
    for trial in range(ntrials):
        #TODO: is the labels with the md_array values aligned? Or they change with the unique labeling code?
        #TODO: decide in using start from 0 or 1...
        cluster = klabels[trial] - 1
        mdist = md_array[cluster, trial]
        S = md_params[cluster].S
        # Remove clusters without any points:
        try:
            ln_L[0, trial] = \
                np.exp(-1/2 * mdist) / \
                np.sqrt((2*np.pi)**data_dim * np.linalg.det(S))
        except Exception as e:
            print('Something went wrong!')
            print(str(e))
    L_ln_hat = np.nansum(np.log(ln_L * 0.0001))

    if L_ln_hat > 0:
        print('L_ln_hat is wrong! Possibly near singular S!')
        BIC = np.nan
    else:
        BIC = np.log(ntrials)*nclusters - 2 * L_ln_hat
    return BIC


def test_for_overfit(klabels=None, data_pca=None, S=None, threshold=None):
    '''
    Test for overfit by computing the MD between centroids, but utilizing the
    overall data covariance. If k means overfits, return True.
    :param klabels:
    :param data_pca:
    :param S:
    :param threshold:
    :return:
    '''
    unique_labels = np.unique(klabels)
    k = len(unique_labels)
    if k == 1:
        print('We have only one cluster, so skipping test for overfit.')
        return False

    dim, ntrial, duration = data_pca.shape
    # Attention, labels must start from 0:
    aggregate_datasets = create_agregate_dataset(klabels=klabels-1, k=k)
    cluster_mu = []
    for dataset in aggregate_datasets:
        cluster_mu.append(data_pca[:, dataset, :].reshape(dim, -1).mean(axis=1))

    # I only need the triangular upper portion of the distance matrix:
    offset = 1
    for i in range(k):
        for j in range(offset, k):
            # Calculate the distance between clusters:
            xy_diff = cluster_mu[i] - cluster_mu[j]
            MD = np.sqrt(xy_diff @ np.linalg.inv(S) @ xy_diff.T)
            # If MD is less that the threshold, then k-means overfits
            if MD < threshold:
                return True
        offset += 1

    return False


def determine_number_of_clusters(
        input_NWBfile, max_clusters=None, y_array=None, custom_range=None,
        **kwargs
    ):
    '''
    Return the optimal number of clusters, as per BIC:

    :param input_NWBfile:
    :param max_clusters:
    :param y_array:
    :param custom_range:
    :param kwargs:
    :return:
    '''

    # Perform PCA to the binned network activity:
    nwbfile_description_d = eval(input_NWBfile.acquisition['membrane_potential'].description)
    animal_model_id = nwbfile_description_d['animal_model']
    learning_condition_id = nwbfile_description_d['learning_condition']
    trial_len = nwbfile_description_d['trial_len']
    #TODO: import properly the q_size:
    q_size = 50
    total_trial_qs = trial_len / q_size
    one_sec_qs = 1000 / q_size
    start_q = total_trial_qs - one_sec_qs
    # Analyze only each trial's last second:
    data_pca = pcaL2(
        input_NWBfile,
        custom_range=(start_q, total_trial_qs),
        plot=False
    )

    dims, ntrials, duration = data_pca.shape

    # move the means to the origin for each trial:
    all_datapoints = np.zeros((dims, duration, ntrials))
    for trial in range(data_pca.shape[1]):
        trial_datapoints = np.squeeze(data_pca[:, trial, :]).T
        mu = trial_datapoints.mean(axis=0)
        all_datapoints[:, :, trial] = np.transpose(trial_datapoints - mu)

    # Compute this 'average' S (covariance).
    S_all = np.cov(all_datapoints.reshape(dims, duration * ntrials))


    #assert max_clusters <= ntrials, 'Cannot run kmeans with greater k than the data_pcapoints!'
    if max_clusters > ntrials:
        print('Cannot run kmeans with greater k than the data_pcapoints!')
        max_clusters = ntrials

    kmeans_labels = np.zeros((max_clusters, ntrials), dtype=int)
    J_k_all = [0] * max_clusters
    BIC_all = [0] * max_clusters
    md_params_all = [0] * max_clusters
    # Calculate BIC for up to max_clusters:
    for i, k in enumerate(range(1, max_clusters + 1)):
        print(f'Clustering with {k} clusters.')
        klabels, J_k, md_array, md_params_d = kmeans_clustering(
            data=data_pca, k=k, max_iterations=100, **kwargs
        )
        # I need to check overfitting (clustering into multiple subclusters).
        #  Since
        # the BIC will BE better moving over greater k, we need to calculate a
        # maximum k - over that some k cluster centroids will be so close,
        # essentially belonging to the same cluster.
        # So after each k means I need to check if centroids are so close.
        #TODO: 8elw ena function pou 8a ypologizei ola me ola ta centroids
        # kai 8a entopizei afta pou exoun MD mikroterh apo ena threshold.
        # Afto 8a shmatodotei k to telos tou increase k, ka8ws einai quasi-
        # Deterministic to k-means init (mean-shift) kai oso kai na synexizw
        # den yparxei periptwsh na parw 'diaforetiko' syndyasmo apo labels.
        #TODO: afto pou me apasxolei einai, 8a einai to BIC kalo ean apla to
        # krathsw se afth th sta8erh timh?
        k_means_overfit = test_for_overfit(
            klabels=klabels, data_pca=data_pca, S=S_all, threshold=3.0
        )
        if k_means_overfit:
            # Stop searching for fit of greater ks.
            BIC_all[i:] = [BIC_all[i - 1]] * (max_clusters - i)
            kmeans_labels[i:, :] = kmeans_labels[i - 1, :]
            J_k_all[i:] = [J_k_all[i - 1]] * (max_clusters - i)
            md_params_all[i:] = [md_params_all[i - 1]] * (max_clusters - i)
            break
        else:
            BIC = evaluate_clustering(
                klabels=klabels, md_array=md_array, md_params=md_params_d, **kwargs
            )
            BIC_all[i] = BIC
            kmeans_labels[i, :] = klabels.T
            J_k_all[i] = J_k
            md_params_all[i] = md_params_d

    K_star = np.zeros((y_array.size, 1), dtype=int)
    K_labels = np.zeros((y_array.size, ntrials), dtype=int)
    for i, y in enumerate(y_array):
        if len(np.unique(BIC_all)) == 1:
            # If all BIC values are the same, we overfitted on k=2. So we get
            # the labels from k=1.
            K_s_labelidx = 0
            # and the K* is one:
            optimal_k = 1
        else:
            # Compute K* as a variant of the rate distortion function, utilizing BIC:
            K_s = np.argmax(np.diff(np.power(BIC_all, -y)))
            # The idx of the kmeans_labels array (starts from 0 = one cluster):
            K_s_labelidx = K_s + 1
            # This directly corresponds to how many clusters:
            optimal_k = K_s_labelidx + 1

        # Add 1 to start counting from 1, then another, since we diff above:
        # This is the optimal no of clusters:
        K_star[i, 0] = optimal_k
        # Store the klabels corresponding to each K*:
        K_labels[i, :] = kmeans_labels[K_s_labelidx, :]

    return K_star, K_labels


if __name__ == "__main__":

    print('success!')
