import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pandas as pd
import h5py
from scipy import spatial

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

def quick_spikes(voltage_trace=None, upper_threshold=None, lower_threshold=None, steps_per_ms=10, plot=False):
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
        spike_timings.append((np.argmax(voltage_trace[start:stop+1]) + start) / steps_per_ms)
    # Plot if requested.
    if plot:
        #vt_reduced = voltage_trace.loc[::steps_per_ms]
        vt_reduced = voltage_trace[::steps_per_ms]
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

def generate_slices_g(size=50, number=2):
    '''
    Generate starting/ending positions of q_total windows q of size q_size.
    :param q_size:
    :param q_total:
    :return:
    '''
    for q in from_one_to(number):
        q_start = (q - 1) * size + 1
        q_end = (q - 1) * size + size
        # yield starting/ending positions of q (in ms)
        yield (q_start, q_end)
        q += 1

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
    ntrials = kwargs.get('ntrials', None)
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
            ctr += 1
            if ctr == idx:
                return (i, j)
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
    ntrials = kwargs.get('ntrials', None)
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
                a=mean_shift_points[cluster].ndarray, b=data[:, trial, :].T
            )
    klabels = ed_array.argmin(axis=0)
    cluster_group_idx = [
        np.nonzero(cluster == klabels)[0]
        for cluster in range(k)
    ]
    md_array = np.zeros((k, ntrials))
    for cluster in range(k):
        for trial in range(ntrials):
            pass
            md_array[cluster, trial] = mahalanobis_distance(
                idx_a=data[:, cluster_group_idx[cluster], :],
                idx_b=data[:, [trial], :]
            )

    cumulative_dist = np.zeros((1, ntrials))
    for cluster_i in range(k):
        cumulative_dist[0, cluster_i] = np.nansum(
            md_array[cluster_i, [np.nonzero(cluster_i == klabels)[0]]]
        )
    J_k = cumulative_dist.sum() / ntrials
    return (klabels, J_k)

    print('done')

def mahalanobis_distance(idx_a=None, idx_b=None):
    # Compute the MD of a trial average (idx_b) from the cluster centroid (idx_a):
    #TODO: check again how to handle e.g. mean to return a matrix not a vector. It really mess up the multiplications..
    dim, trials, t = idx_a.shape
    if any(np.array(idx_a.shape) < 1):
        return np.nan
    cluster_data = idx_a.reshape(t * trials, -1, order='C')
    point_data = np.mean(idx_b.reshape(t, -1, order='C'), axis=0).reshape(1, -1)
    mu = np.mean(cluster_data, axis=0).reshape(1, -1)
    C = np.cov(cluster_data.T)
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError as e:
        raise e('Covariance matrix is not PD!')
    tmp = point_data - mu
    return np.sqrt(tmp @ np.linalg.inv(C) @ tmp.T)

def point2points_average_euclidean(a=None, b=None):
    # return the average euclidean distance between the point a and points b
    # run dist of a single point against a list of points:
    m, n = b.shape
    distance = spatial.distance.pdist(
        np.concatenate((a, b), axis=0)
    )
    return np.mean(distance[:m])

def kmeans_clustering(data=None, k=2, plot=False, **kwargs):
    #return (labels_final, J_k_final, S_i)
    # Perform kmeans clustering.
    # Input:
    #   X: n x d data matrix
    #   m: initialization parameter (k)
    # Adapted by Stefanos Stamatiadis (stamatiad.st@gmail.com).
    # mahal_dist_flag = true
    max_iter = 20
    # wsize = 20 # 4*50=200ms window!
    # n, d, N
    dims, ntrials, total_qs  = data.shape
    J_k, klabels = initialize_algorithm(data=data, k=k, plot=plot, **kwargs)
    pass


if __name__ == "__main__":

    print('success!')
