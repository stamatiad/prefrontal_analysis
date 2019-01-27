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

def create_nwb_file(inputdir=None, outputdir=None, **kwargs):
    # Get parameters externally:
    experiment_config, animal_model, learning_condition, ntrials, trial_len, ncells, stim_start_offset, \
    stim_stop_offset, samples_per_ms, spike_upper_threshold, spike_lower_threshold, excitation_bias, \
        inhibition_bias, nmda_bias, ampa_bias, sim_duration = \
        getargs('experiment_config', 'animal_model', 'learning_condition', 'ntrials', 'trial_len', 'ncells', 'stim_start_offset', \
                   'stim_stop_offset', 'samples_per_ms', 'spike_upper_threshold', 'spike_lower_threshold', \
                'excitation_bias', 'inhibition_bias', 'nmda_bias', 'ampa_bias', 'sim_duration', kwargs)

    # the base unit of time is the ms:
    conversion_factor = 1 / samples_per_ms
    nsamples = trial_len * samples_per_ms

    nwbfile = NWBFile(
        session_description='NEURON simulation results.',
        identifier=experiment_config,
        session_start_time=datetime.now(),
        file_create_date=datetime.now()
    )

    nwbfile.add_unit_column('cell_id', 'Id of the cell recorded')
    nwbfile.add_trial_column('persistent_activity', 'If this trial has persistent activity')
    #nwbfile.add_epoch_column('stimulus')

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
                spike_train = quick_spikes(
                    voltage_trace=voltage_traces[cellid],
                    upper_threshold=spike_upper_threshold,
                    lower_threshold=spike_lower_threshold,
                    plot=False
                )
                #TODO: to better name conversion factor!
                spike_trains_d[cellid] = np.append(
                    spike_trains_d[cellid], np.add(spike_train, trial_start_t * conversion_factor)
                )

            # Define the region of PA as the last 200 ms of the simulation:
            pa_stop = int(nsamples * conversion_factor)  # in ms
            pa_start = int(pa_stop - 200)
            has_persistent = voltage_traces[:ncells, pa_start:pa_stop].max() > 0
            # Add trial:
            nwbfile.add_trial(
                start_time=trial_start_t * conversion_factor,
                stop_time=trial_end_t * conversion_factor,
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
        conversion=conversion_factor  #  Scalar to multiply each element in data to convert it to the specified unit
    )
    nwbfile.add_acquisition(vsoma_timeseries)

    for cellid in range(ncells):
        if spike_trains_d[cellid].size > 0:
            #TODO: standardize these! You are using the format for that:
            # Get each trial start/end in no of samples, rather than ms:
            activity_intervals = [
                [q_start, q_end]
                for q_start, q_end in generate_slices(size=trial_len / 1000, number=ntrials, start_from=0)
            ]
            nwbfile.add_unit(
                id=cellid, spike_times=spike_trains_d[cellid],
                obs_intervals=activity_intervals,
                cell_id=cellid
            )

    # write to file:
    output_file = outputdir.joinpath(
        experiment_config_filename(
            animal_model=animal_model, learning_condition=learning_condition,
            experiment_config=experiment_config
        )
    )
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
            dataset = experiment_config_str(
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
                            for q, (q_start, q_end) in enumerate(generate_slices_g(size=q_size, number=total_qs)):
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
    #TODO: You can use something like that to get your arguments. It's more pythonic:
    # id, columns, desc, colnames = popargs('id', 'columns', 'description', 'colnames', kwargs)

    data = None
    # Read spiketrains and plot them
    df = pd.read_hdf(fromfile, key=dataset)
    #with open_hdf_dataframe(filename=filename, hdf_key=data_key) as df:
    if df is not None:
        data = df.values.T
        # Also reshape the data into a 3d array:
        data = data.reshape(data.shape[0], ntrials, total_qs, order='C')

    return data

def pcaL2(data=None, plot=False, custom_range=None, **kwargs):

    ntrials = kwargs.get('ntrials', None)

    #TODO: check that network activity have PA:
    #Can I move this to the NWB format and have it on creation?
    have_pa = [
        data[:, trial, -1].mean() > 0
        for trial in range(ntrials)
    ]
    data = data[:, have_pa, :]
    ntrials = sum(have_pa)
    # how many components
    L = 2
    pca = decomposition.PCA(n_components=L)
    # Use custom_range to compute PCA only on a portion of the original data:
    duration = len(custom_range)
    t_L = pca.fit_transform(
        #TODO: remove hardcoded nPC:
        data[:250, :, custom_range].reshape(250, ntrials * duration).T
    ).T
    t_L_reshaped = t_L.reshape(L, ntrials, duration, order='C')
    if plot:
        plot_pcaL2(data=t_L_reshaped)

    return t_L_reshaped

def plot_pcaL2(data=None, klabels=None, smooth=False, **kwargs):
    #Plots the data as 2d timeseries.
    # If labels are provided then it colors them also.
    dims, ntrials, duration = data.shape
    fig = plt.figure()
    plt.ion()
    ax = fig.add_subplot(111, projection='3d')

    if klabels is not None:
        labels = klabels.tolist()
        nclusters = np.unique(klabels).size
        colors = cm.Set2(np.linspace(0, 1, nclusters))
        _, key_labels = np.unique(labels, return_index=True)
        handles = []
        for i, (trial, label) in enumerate(zip(range(ntrials), labels)):
            if smooth:
                x = savgol_filter(data[0][trial][:], 11, 3)
                y = savgol_filter(data[1][trial][:], 11, 3)
            else:
                x = data[0][trial][:]
                y = data[1][trial][:]
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
        #TODO: incorporate smoothing here also:
        colors = cm.viridis(np.linspace(0, 1, duration - 1))
        for trial in range(ntrials):
            for t, c in zip(range(duration - 1), colors):
                ax.plot(data[0][trial][t:t+2], data[1][trial][t:t+2], [t, t+1], color=c)
    plt.show()

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
    data_dim = kwargs.get('data_dim', None)
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
        for slice_start, slice_end in generate_slices_g(size=new_len, number=ntrials)
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
    data_dim = kwargs.get('data_dim', None)
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
    cluster_data = idx_a.reshape(t * trials, -1, order='C')
    point_data = np.mean(idx_b.reshape(t, -1, order='C'), axis=0).reshape(1, -1)
    mu = np.mean(cluster_data, axis=0).reshape(1, -1)
    S = np.cov(cluster_data.T)
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
def kmeans_clustering(data=None, k=2, max_iterations=None, plot=False, **kwargs):
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
            distance_array[original_label, :], distance_array[new_label, :] = \
                distance_array[new_label, :].copy(), distance_array[original_label, :].copy()
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
                md_array[cluster, trial], md_params_d[cluster] = mahalanobis_distance(
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
    return klabels, J_k, md_array, md_params_d

def evaluate_clustering(klabels=None, md_array=None, md_params=None, **kwargs):
    # Calculate likelihood of each trial, given the cluster centroid:
    nclusters, ntrials = md_array.shape
    data_dim = kwargs.get('data_dim', None)

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
            ln_L[0, trial] = np.exp(-1/2 * mdist) / np.sqrt((2*np.pi)**data_dim * np.linalg.det(S))
        except Exception as e:
            raise e('Something went wrong!')
    L_ln_hat = np.nansum(np.log(ln_L * 0.0001))

    if L_ln_hat > 0:
        print('L_ln_hat is wrong! Possibly near singular S!')
        BIC = np.nan
    else:
        BIC = np.log(ntrials)*nclusters - 2 * L_ln_hat
    return BIC


def determine_number_of_clusters(data=None, max_clusters=None, y_array=None, **kwargs):
    # Return the optimal number of clusters, as per BIC:
    ntrials = data.shape[1]
    #assert max_clusters <= ntrials, 'Cannot run kmeans with greater k than the datapoints!'
    if max_clusters > ntrials:
        print('Cannot run kmeans with greater k than the datapoints!')
        max_clusters = ntrials
    dims, ntrials, duration = data.shape
    kmeans_labels = np.zeros((ntrials, max_clusters), dtype=int)
    J_k_all = list()
    BIC_all = list()
    md_params_all = list()
    for k in range(1, max_clusters + 1):
        print(f'Clustering with {k} clusters.')
        klabels, J_k, md_array, md_params_d = kmeans_clustering(data=data, k=k, max_iterations=100, plot=False, **kwargs)
        BIC = evaluate_clustering(klabels=klabels, md_array=md_array, md_params=md_params_d, **kwargs)
        BIC_all.append(BIC)
        kmeans_labels[:, k - 1] = klabels.T
        J_k_all.append(J_k)
        md_params_all.append(md_params_d)
    #TODO: den to kanw akomh, giati den fainetai na exw problhma me overfit (sto random pou me noiazei perissotero).
    '''
    # Calculate:
    run_average_translation = np.zeros((data.shape[:2]))
    for trial in range(ntrials):
        run_average_translation[:, trial] = np.diff(
            np.concatenate((data[:, trial, :].min(axis=1).reshape(1, dims),
            data[:, trial, :].max(axis=1).reshape(1,dims)), axis=0)
        ).T
    mean_intratrial_translation = run_average_translation.mean(axis=1)
    '''
    K_star = np.zeros((1, y_array.size), dtype=int)
    K_labels = np.zeros((ntrials, y_array.size), dtype=int)
    for i, y in enumerate(y_array):
        # Compute K* as a variant of the rate distortion function, utilizing BIC:
        K_s = np.argmax(np.diff(np.power(BIC_all, -y)))
        # The idx of the kmeans_labels array (starts from 0 = one cluster):
        K_s_labelidx = K_s + 1
        # This directly corresponds to how many clusters:
        K_s_trueidx = K_s_labelidx + 1
        # Add 1 to start counting from 1, then another, since we diff above:
        K_star[0, i] = K_s_trueidx
        # Store the klabels corresponding to each K*:
        K_labels[:, i] = kmeans_labels[:, K_s_labelidx]
        pass

    return K_star, K_labels


if __name__ == "__main__":

    print('success!')
