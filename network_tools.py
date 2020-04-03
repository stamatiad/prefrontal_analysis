# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:07:01 2018

@author: stefanos
"""
import warnings
import numpy as np
from scipy.spatial import distance
from scipy.special import comb as nchoosek
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict, defaultdict, namedtuple
import h5py
import time
import os
import sys
from itertools import chain
from functools import wraps
import pandas as pd
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
from contextlib import contextmanager
import copy
import notebook_module as nb
from scipy.signal import savgol_filter
import scipy.stats
from pathlib import Path
import pickle
import json
import seaborn as sb
import re

plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def check_requirements():
    # Check that user is running a supported python version and necessary packages:
    dependencies = [
        'h5py==2.8.0',
        'matplotlib==3.0.2',
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.2',
        'scipy==1.1.0',
        'sklearn==0.0',
        'tables==3.4.4',
        'wincertstore==0.2'
    ]
    py_major = sys.version_info[0]
    py_minor = sys.version_info[1]
    if py_major < 3 or py_minor < 7:
        raise Exception('You are running an unsupported python version ({}.{}). Please use 3.7 or newer!' \
                        .format(py_major, py_minor))

    # If a dependency is not met, a DistributionNotFound or VersionConflict is thrown.
    pkg_resources.require(dependencies)

def time_it(function):
    @wraps(function)
    def runandtime(*args, **kwargs):
        tic = time.perf_counter()
        result = function(*args, **kwargs)
        toc = time.perf_counter()
        print(f'{function.__name__} took {toc-tic} seconds.')
        return result
    return runandtime

def with_reproducible_rng(class_method):
    '''
    This is a function wrapper that calls rng.seed before every method call. Therefore user is expected to get the exact
    same results every time between different method calls of the same class instance.
    Multiple method calls in main file will produce the exact same results. This is of course if model parameters are
    the same between calls; changing e.g. cell no, will invoke different number of rand() called in each case.
    As seed, the network serial number is used.
    A function wrapper is used to dissociate different function calls: e.g. creating stimulus will be the same, even if
    user changed the number of rand() was called in a previous class function, resulting in the same stimulated cells.
    Multiple method calls in main file will produce the exact same results. This is of course if model parameters are
    the same between calls; changing e.g. cell no, will invoke different number of rand() called in each case.
    :param func:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        # this translates to self.serial_no ar runtime
        np.random.seed(args[0].serial_no)
        print(f'{class_method.__name__} reseeds the RNG.')
        return class_method(*args, **kwargs)
    return reset_rng

class UPair:
    # This is a tuple representing the unordered pairs of neurons in the network
    # Named tuples are immutable, yet this might prevent unwanted errors.
    # UPair = namedtuple('UPair', ['a', 'b', 'distance', 'type_cells', 'type_conn', 'prob_f'])
    @property
    def ucoord(self):
        return self._ucoord

    def __init__(self, a, b, distance, type_cells, type_conn=None, prob_f=None):
        self.a = a
        self.b = b
        self.distance = distance
        self.type_cells = type_cells
        self.type_conn = type_conn
        self.prob_f = prob_f
        # 'Privatly' set the ucoordinates:
        if a > b:
            self._ucoord = (b, a)
        else:
            self._ucoord = (a, b)

    def __repr__(self):
        return f'UPair({self.a}, {self.b}, {self.distance}, {self.type_cells})'

    def __eq__(self, other):
        # UPair equality only compares the LOCATION!
        if isinstance(other, UPair):
            return set([self.a, self.b]) == set([other.a, other.b])
        else:
            return False

    def __hash__(self):
        #return hash(self.__repr__())
        # A None hash should return TypeError, when accessed:
        return None


class Network:
    @property
    def serial_no(self):
        return self._serial_no

    @serial_no.setter
    def serial_no(self, serial_no):
        if serial_no < 1:
            raise ValueError("Only positive serial numbers allowed!")
        self._serial_no = serial_no

    @property
    def pv_no(self):
        return self._pv_no

    @pv_no.setter
    def pv_no(self, value):
        if value < 1:
            raise ValueError("Must have at least one PV!")
        self._pv_no = value
        #self._cell_no = self.pc_no + self.pv_no

    @property
    def pc_no(self):
        return self._pc_no

    @pc_no.setter
    def pc_no(self, value):
        if value < 1:
            raise ValueError("Must have at least one PC!")
        self._pc_no = value
        #self._cell_no = self.pc_no + self.pv_no

    @property
    def cell_no(self):
        self._cell_no = self.pc_no + self.pv_no
        return self._cell_no

    @cell_no.setter
    def cell_no(self, value):
        warnings.warn("Please set individually pc_no, pv_no!")

    @property
    def pc_somata(self):
        return self._pc_somata

    @pc_somata.setter
    def pc_somata(self, pos):
        if pos.size == 0:
            raise ValueError("Position is empty!")
        self._pc_somata = pos

    @property
    def pv_somata(self):
        return self._pv_somata

    @pv_somata.setter
    def pv_somata(self, pos):
        if pos.size == 0:
            raise ValueError("Position is empty!")
        self._pv_somata = pos

    @property
    def configurations(self):
        return self._configurations

    @configurations.setter
    def configurations(self, dict):
        self._configurations = dict

    @property
    def configuration_rnd(self):
        return self._configuration_rnd

    @configuration_rnd.setter
    def configuration_rnd(self, mat):
        if mat.size == 0:
            raise ValueError("Connectivity matrix is empty!")
        self._configuration_rnd = mat

    @property
    def weights_mat(self):
        return self._weights_mat

    @weights_mat.setter
    def weights_mat(self, mat):
        if mat.size == 0:
            raise ValueError("Weights matrix is empty!")
        self._weights_mat = mat

    @property
    def stim_groups_str(self):
        return self._stim_groups_str

    @stim_groups_str.setter
    def stim_groups_str(self, c_list):
        if not c_list:
            raise ValueError("Cell list is empty!")
        self._stim_groups_str = c_list

    @property
    def stim_groups_rnd(self):
        return self._stim_groups_rnd

    @stim_groups_rnd.setter
    def stim_groups_rnd(self, c_list):
        if not c_list:
            raise ValueError("Cell list is empty!")
        self._stim_groups_rnd = c_list

    @property
    def dist_mat(self):
        return self._dist_mat

    @dist_mat.setter
    def dist_mat(self, mat):
        if mat.size == 0:
            raise ValueError("Distance matrix is empty!")
        self._dist_mat = mat

    @property
    def trial_no(self):
        return self._trial_no

    @trial_no.setter
    def trial_no(self, t_no):
        if t_no < 1:
            raise ValueError('Trial number must be greater than one!')
        self._trial_no = t_no

    @property
    def stimulated_cells(self):
        return self._stimulated_cells

    @stimulated_cells.setter
    def stimulated_cells(self, mat):
        if mat.size < 1:
            raise ValueError('Stimulated cells matrix can not be empty!')
        self._stimulated_cells = mat

    # The upairs are acquired with copy, to avoid mutations. This is
    # communicated with a context manager, so the user will know how to
    # handle them.
    @property
    def upairs_d(self):
        # Return a copy of the dict, to avoid errors due to accidental mutation.
        # Thus one can use the util functions that connect upairs, choosing
        # whether or not the alterations will be retained by the network just
        # by using the context mananger.
        return copy.deepcopy(self._upairs_d)

    @upairs_d.setter
    def upairs_d(self, val):
        if not isinstance(val, dict):
            raise ValueError('upairs_d is not a dict!')
        # Copy again to avoid accidental mutation:
        self._upairs_d = copy.deepcopy(val)

    @contextmanager
    def get_upairs(self, cell_type=None):
        # This context manager is essentially non-necessary, yet in directly
        # communicates that there is some implicit acquire/release taking place
        # and also provides a handy function to select specific cell pair type.

        # Code to acquire resource, e.g.:
        #resource = acquire_resource(*args, **kwds)
        if cell_type:
            upairs = [
                upair
                for upair in self.upairs_d.values()
                if upair.type_cells == cell_type
            ]
        else:
            upairs = list(self.upairs_d.values())
        try:
            yield upairs
        finally:
            # Code to release resource, e.g.:
            #release_resource(resource)
            # Save the upairs list (above) inside the object, to always maintain
            # a current dict of all the upairs in the network.
            # Save the updated upairs to the network object:
            self.update_upairs(upair_list=upairs)

    def update_upairs(self, upair_list=None):
        '''
        Updates the mutated upairs to the network object.
        :param upair_list:
        :return:
        '''
        # Get a snapshot of the network upairs:
        tmp_network_upairs = self.upairs_d
        # Update the upair:
        for upair in upair_list:
            tmp_network_upairs[upair.ucoord] = upair
        # Save the updated upairs to the network object:
        self.upairs_d = tmp_network_upairs


    '''
    @connection_functions_d.setter
    def connection_functions_d(self, val):
        #TODO: implement that constraint:
        # In my code I only update the below keys, so only those are allowed to
        # be updated.
        pair_type, conn_type, conn_func = val
        # Update connection functions dictionary:
        self._connection_functions_d[pair_type][conn_type] = conn_func
    '''

    def get_connection_function(self, pair_type=None, conn_type=None):
        return self._connection_functions_d[pair_type][conn_type]

    def set_connection_function(self, pair_type=None, conn_type=None,
                                conn_func=None):
        self._connection_functions_d[pair_type][conn_type] = conn_func

    def __init__(self, serial_no, pc_no, pv_no):
        ''' Initialize network class.
        param serial_no Serial Number of the network, to recreate it no matter what.

        '''
        print('Initializing Network')
        self.serial_no = serial_no
        self.pc_no = pc_no
        self.pv_no = pv_no
        print('Network have %d PCs' % self.pc_no)
        print(self.pc_no)
        print(self.cell_no)
        # Initialize connectivity matrices:
        # configurations is a dictionary, since we can have multiple:
        #self.configurations = {} #np.full((self.cell_no, self.cell_no), False)
        #self.configuration_rnd = np.full((self.cell_no, self.cell_no), False)
        self.connectivity_mat = np.full((self.cell_no, self.cell_no), False)
        self.weights_mat = np.full((self.cell_no, self.cell_no), 0.0)
        self.upairs_d = {}
        self.configuration_alias = ''
        self.stats = {}
        # Define connectivity probability functions (distance dependent):
        # Neuronal pairs are unordered {A,B}, considering their type.
        # For each  pair we have three distinct, distance-dependend connection probabilities:
        # 1) A <-> B (reciprocal)
        # 2) A -> B
        # 3) B -> A
        # Other/custom connectivity functions can be used as well in the same framework, in order to connect different
        # network configurations i.e. random or just get the connection  probability of all pairs (vectorized = fast).

        # Average connectivity in Otsuka et al., 2009 is 0.25 (inside ~80um)
        # Average connectivity in Packer, Yuste, 2011 is 0.67 (inside ~200um)
        # We combine them, using an average, distance attenuating connectivity of 0.46
        pv_factor = 0.46/(0.154+0.019+0.077)
        # Use numpy array friendly lambdas. It is harder on the eyes, but can save hours (!) of computations:
        self._connection_functions_d = {}
        # PN with PN connectivity:
        self._connection_functions_d['PN_PN'] = {}
        self._connection_functions_d['PN_PN']['total'] \
            = lambda x: np.multiply(np.exp(np.multiply(-0.0052, x)), 0.22)
        self._connection_functions_d['PN_PN']['reciprocal'] \
            = lambda x: np.multiply(np.exp(np.multiply(-0.0085, x)), 0.12)
        self._connection_functions_d['PN_PN']['A2B'] \
            = lambda x: np.divide(np.subtract(self._connection_functions_d['PN_PN']['total'](x), self._connection_functions_d['PN_PN']['reciprocal'](x)), 2)
        self._connection_functions_d['PN_PN']['B2A'] \
            = lambda x: self._connection_functions_d['PN_PN']['A2B'](x)
        self._connection_functions_d['PN_PN']['unidirectional'] \
            = lambda x: np.add(self._connection_functions_d['PN_PN']['A2B'](x), self._connection_functions_d['PN_PN']['B2A'](x))
        # Random/uniform connection functions are refined later on, in order
        # to exactly balance its overall connection
        self._connection_functions_d['PN_PN']['uniform_reciprocal'] \
            = lambda x: np.multiply(np.ones(x.shape), 0)
        self._connection_functions_d['PN_PN']['uniform_A2B'] \
            = lambda x: np.multiply(np.ones(x.shape), 0)
        self._connection_functions_d['PN_PN']['uniform_B2A'] \
            = lambda x: np.multiply(np.ones(x.shape), 0)
        # Random/uniform connection functions are refined later on, in order
        # to exactly balance its overall connection (dd = distance dependend)
        self._connection_functions_d['PN_PN']['uniform_reciprocal_dd'] \
            = lambda x: np.multiply(np.ones(x.shape), 1)
        self._connection_functions_d['PN_PN']['uniform_A2B_dd'] \
            = lambda x: np.multiply(np.ones(x.shape), 1)
        self._connection_functions_d['PN_PN']['uniform_B2A_dd'] \
            = lambda x: np.multiply(np.ones(x.shape), 1)
        # probability with the structured configuration.
        # PN with PV connectivity:
        self._connection_functions_d['PN_PV'] = {}
        self._connection_functions_d['PN_PV']['total'] \
            = lambda x: np.exp(np.divide(x, -180))
        self._connection_functions_d['PN_PV']['reciprocal'] \
            = lambda x: np.multiply(self._connection_functions_d['PN_PV']['total'](x), 0.154 * pv_factor)
        self._connection_functions_d['PN_PV']['A2B'] \
            = lambda x: np.multiply(self._connection_functions_d['PN_PV']['total'](x), 0.019 * pv_factor)
        self._connection_functions_d['PN_PV']['B2A'] \
            = lambda x: np.multiply(self._connection_functions_d['PN_PV']['total'](x), 0.077 * pv_factor)
        # PV with PV connectivity:
        self._connection_functions_d['PV_PV'] = {}
        self._connection_functions_d['PV_PV']['reciprocal'] \
            = lambda x: np.multiply(np.ones(x.shape), 0.045)
        self._connection_functions_d['PV_PV']['A2B'] \
            = lambda x: np.multiply(np.ones(x.shape), 0.0675)
        self._connection_functions_d['PV_PV']['B2A'] \
            = lambda x: np.multiply(np.ones(x.shape), 0.0675)

    @with_reproducible_rng
    def populate_network(self, cube_side_len: int, plot=False):
        '''
        Disperses neurons uniformly/randomly in a cube of given side
        @param: cell_no number of network cells
        @param: cube_side_len cube side length in um
        '''
        def get_pair_type(i, j):
            a_type = ''
            b_type = ''
            if i < self.pc_no:
                a_type = 'PN'
            else:
                a_type = 'PV'
            if j < self.pc_no:
                b_type = 'PN'
            else:
                b_type = 'PV'
            # sort the types by a convention of list b, to simplify things:
            a = [a_type, b_type]
            b = ['PN', 'PV']
            a.sort(key=lambda v: b.index(v))
            return '{}_{}'.format(a[0], a[1])

        # Disperse cell_no points in a cube of cube_side_len side length:
        somata = np.multiply(np.random.rand(self.cell_no, 3), cube_side_len)

        # Create alternative/wrapped network configurations:
        X, Y, Z = np.meshgrid(np.array([1,2,3]), np.array([1,2,3]), np.array([1,2,3]) )
        cb_offest_pos = np.transpose(np.array([X.ravel(), Y.ravel(), Z.ravel()]))

        cb_offset = np.array([-cube_side_len, 0, cube_side_len])
        cb_offset_mat = cb_offset[np.subtract(cb_offest_pos,1)]

        dist_wrapped_vect = np.zeros(int(self.cell_no*(self.cell_no-1)/2))

        upairs_list = []
        ctr = 0
        start = 1
        for i in range(self.cell_no):
            for j in range(start, self.cell_no):
                tmp = np.zeros(cb_offset_mat.shape[0])
                for k in range(cb_offset_mat.shape[0]):
                    tmp[k] = np.sqrt((somata[i,0]-(somata[j,1]+cb_offset_mat[k,0]))**2 \
                        + (somata[i,1]-(somata[j,1]+cb_offset_mat[k,1]))**2 \
                        + (somata[i,2]-(somata[j,2]+cb_offset_mat[k,2]))**2)
                dist = np.amin(tmp)
                dist_wrapped_vect[ctr] = dist
                # Add also neuronal pair to list:
                #self.upairs.append(self.UPair(a=i, b=j, type_cells=get_pair_type(i, j), distance=dist, type_conn=None))
                # Maintain a dict with (upper triangular) coordinates of upairs (j always greater than i).
                upairs_list.append(
                    UPair(a=i, b=j, distance=dist, type_cells=get_pair_type(i, j))
                )
                ctr += 1
            start += 1
        # Write the upairs to the network:
        self.update_upairs(upair_list=upairs_list)

        # Set network intersomatic distance matrix:
        self.dist_mat = distance.squareform(dist_wrapped_vect)

        # Set PC/PV somata positions (Seperate PCs from PVs):
        self.pc_somata = somata[1:self.pc_no+1,:]
        self.pv_somata = somata[self.pc_no+1:,:]

        if plot:
            fig = plt.figure()
            plot_axis = fig.add_subplot(111, projection='3d')
            plot_axis.scatter(
                self.pc_somata[:, 0],
                self.pc_somata[:, 1],
                self.pc_somata[:, 2],
                c='b', marker='^'
            )
            plot_axis.scatter(
                self.pv_somata[:, 0],
                self.pv_somata[:, 1],
                self.pv_somata[:, 2],
                c='r', marker='o'
            )
            plot_axis.set_xlabel('X (um)')
            plot_axis.set_ylabel('Y (um)')
            plot_axis.set_zlabel('Z (um)')
            plot_axis.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            plot_axis.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            plot_axis.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            pca_axis_limits = (0, 180)
            plot_axis.set_xlim(pca_axis_limits)
            plot_axis.set_ylim(pca_axis_limits)
            plot_axis.set_zlim(pca_axis_limits)
            plot_axis.set_xticks(pca_axis_limits)
            plot_axis.set_yticks(pca_axis_limits)
            plot_axis.set_zticks(pca_axis_limits)
            plt.savefig('Network_sn{}_3d_positions.png'.format(self.serial_no))
            plt.savefig('Network_sn{}_3d_positions.svg'.format(self.serial_no))
            plt.savefig('Network_sn{}_3d_positions.pdf'.format(self.serial_no))
            plt.show()

        return


    @time_it
    @with_reproducible_rng
    def create_connections(self, alias='', rearrange_iterations=100, average_conn_prob=None,
                           plot=False):
        '''
        Create connectivity and weights of the network, based on the dictionary of connectivity functions and the
        configuration alias.
        :param rearrange_iterations:
        :param average_conn_prob:
        :param plot:
        :return:
        '''
        rnd = lambda: np.random.rand(1)[0]

        self.configuration_alias = alias


        @time_it
        def connect_all_pairs(upairs, connection_protocol=None, export_probabilities=False):
            '''
            Connect (or not) the given pair, based on connections probabilities.
            Second (optional) argument filters out only specific type of connections.
            :param i:
            :param j:
            :param dist:
            :return:
            '''
            # The initial concept was this function to be mappable to every pair, and connect it based on its type.
            # Unfortunately python's function call overhead is huge, so we need to become more array friendly
            # utilizing numpy efficient functions. Unfortunately this affects the readability of the code:
            #cpairs = [None]*len(upairs)
            #new_upairs = [None]*len(upairs)
            prev_i = 0
            query_pair_types = sorted(set(x.type_cells for x in upairs))
            #TODO: Sort the set (since its order is random!), to get
            # reproducible RNG results! IS THIS NECESSARY?
            for pair_type in query_pair_types:
                matching_upairs = [x for x in upairs if x.type_cells == pair_type]
                upairs_dist = np.asarray([p.distance for p in matching_upairs])
                prob_arr = np.full((len(connection_protocol)+1,len(matching_upairs)), 0.0)
                for i, conn_type in enumerate(connection_protocol.values(), 1):
                    prob_arr[i, :] = self.get_connection_function(
                        pair_type=pair_type,
                        conn_type=conn_type
                    )(upairs_dist)
                prob_arr = np.cumsum(prob_arr, axis=0)
                # Don't forget to include the (closed) last bin!
                prob_arr_flat = np.append(np.add(prob_arr, np.tile(np.arange(len(matching_upairs)),
                                                         (len(connection_protocol)+1, 1))).flatten('F'), len(matching_upairs))
                print(f'Length of matching pairs {len(matching_upairs)}')
                rnd_arr_flat = np.add(np.random.rand(1, len(matching_upairs)), np.arange(len(matching_upairs)))
                #DEBUG
                if False:
                    filename = Path.cwd().joinpath('debug_files',
                                                   f'network_debug_rnd_arr_flat_time_1_SN{self.serial_no}.hdf5')
                    df = pd.DataFrame({'serial_no': [self.serial_no], 'pc_no': [self.pc_no], 'pv_no': [self.pv_no],
                                       'configuration_alias': self.configuration_alias})
                    df.to_hdf(filename, key='attributes', mode='w')
                    df = pd.DataFrame(rnd_arr_flat)
                    df.to_hdf(filename, key='rnd_arr_flat')

                    filename = Path.cwd().joinpath('debug_files',
                                                   f'network_debug_prob_arr_flat_time_1_SN{self.serial_no}.hdf5')
                    df = pd.DataFrame({'serial_no': [self.serial_no], 'pc_no': [self.pc_no], 'pv_no': [self.pv_no],
                                       'configuration_alias': self.configuration_alias})
                    df.to_hdf(filename, key='attributes', mode='w')
                    df = pd.DataFrame(prob_arr_flat)
                    df.to_hdf(filename, key='prob_arr_flat')
                #END DEBUG
                blah = np.histogram(rnd_arr_flat, bins=prob_arr_flat)[0]
                blah2 = np.nonzero(blah)[0]
                blah3 = np.arange(0, prob_arr_flat.size, len(connection_protocol)+1)
                conn_code_arr = np.subtract(blah2, blah3[:-1])
                #TODO: now that upair is an object I can just update it:
                for i, (upair, conn_code) in enumerate(zip(matching_upairs, conn_code_arr)):
                    #tmp_d = upair._asdict()
                    #tmp_d['type_conn'] = connection_protocol.get(conn_code, 'none')
                    #if export_probabilities:
                    #    tmp_d['prob_f'] = np.append(prob_arr[:, i-prev_i], 1)
                    #new_upairs[i] = UPair(**tmp_d)
                    upair.type_conn = connection_protocol.get(conn_code, 'none')
                    if export_probabilities:
                        upair.prob_f = np.append(prob_arr[:, i-prev_i], 1)

                prev_i += len(matching_upairs)

            return upairs

        @time_it
        def connect_selected_pairs(upairs, connection_protocol=None,
                            export_probabilities=False):
            '''
            Connects i.e. recomputes connection probabilities and resample
            them to get a connection type only for the the given upairs.
            :param upairs:
            :return:
            '''
            # The initial concept was this function to be mappable to every pair, and connect it based on its type.
            # Unfortunately python's function call overhead is huge, so we need to become more array friendly
            # utilizing numpy efficient functions. Unfortunately this affects the readability of the code:
            prev_i = 0
            #TODO: You need to make sure these are only ONE TYPE OF CELLS
            pair_types = sorted(set(x.type_cells for x in upairs))
            try:
                if len(pair_types) > 1:
                    raise ValueError
                else:
                    pair_type = pair_types[0]
            except ValueError:
                print('You need to provide cells of the same type!')

            upairs_dist = np.asarray([p.distance for p in upairs])
            prob_arr = np.full((len(connection_protocol)+1,len(upairs)), 0.0)
            for i, conn_type in enumerate(connection_protocol.values(), 1):
                prob_arr[i, :] = self.get_connection_function(
                    pair_type=pair_type,
                    conn_type=conn_type
                )(upairs_dist)
            prob_arr = np.cumsum(prob_arr, axis=0)
            # Don't forget to include the (closed) last bin!
            prob_arr_flat = np.append(np.add(prob_arr, np.tile(np.arange(len(
                upairs)), (len(connection_protocol)+1, 1))).flatten('F'),
                                      len(upairs))
            print(f'Length of matching pairs {len(upairs)}')
            rnd_arr_flat = np.add(np.random.rand(1, len(upairs)), np.arange(len(upairs)))

            # DEBUG
            if False:
                filename = Path.cwd().joinpath('debug_files',
                                               f'network_sel_debug_rnd_arr_flat_SN{self.serial_no}.hdf5')
                df = pd.DataFrame(
                    {'serial_no': [self.serial_no], 'pc_no': [self.pc_no],
                     'pv_no': [self.pv_no],
                     'configuration_alias': self.configuration_alias})
                df.to_hdf(filename, key='attributes', mode='w')
                df = pd.DataFrame(rnd_arr_flat)
                df.to_hdf(filename, key='rnd_arr_flat')

                filename = Path.cwd().joinpath('debug_files',
                                               f'network_sel_debug_prob_arr_flat_S'
                                               f'N{self.serial_no}.hdf5')
                df = pd.DataFrame(
                    {'serial_no': [self.serial_no], 'pc_no': [self.pc_no],
                     'pv_no': [self.pv_no],
                     'configuration_alias': self.configuration_alias})
                df.to_hdf(filename, key='attributes', mode='w')
                df = pd.DataFrame(prob_arr_flat)
                df.to_hdf(filename, key='prob_arr_flat')
            # END DEBUG

            blah = np.histogram(rnd_arr_flat, bins=prob_arr_flat)[0]
            blah2 = np.nonzero(blah)[0]
            blah3 = np.arange(0, prob_arr_flat.size, len(connection_protocol)+1)
            conn_code_arr = np.subtract(blah2, blah3[:-1])
            #TODO: now that upair is an object I can just update it:
            for i, (upair, conn_code) in enumerate(zip(upairs, conn_code_arr)):
                upair.type_conn = connection_protocol.get(conn_code, 'none')
                if export_probabilities:
                    upair.prob_f = np.append(prob_arr[:, i-prev_i], 1)

            prev_i += len(upairs)

            return upairs

        @time_it
        def upairs2mat(upairs):
            '''
            Creates a connectivity matrix from upairs.
            Requires the upair_list to be pairs from a square matrix! Essentially user can update either PN to PN
            connectivity, or the whole matrix.
            :param upair_list:
            :return: A network connectivity matrix
            '''
            # The requirement of only square matrices stems from the connect_all_pairs() limitations. Because the
            # latter is only utilized array-wise, no independent pair connectivity can supported.
            # check first that the length is proper for a condenced matrix:
            s = len(upairs)
            # Grab the closest value to the square root of the number
            # of elements times 2 to see if the number of elements
            # is indeed a binomial coefficient.
            d = int(np.ceil(np.sqrt(s * 2)))
            # Check that cpair_list length is a binomial coefficient n choose 2 for some integer n >= 2.
            if d * (d - 1) != s * 2:
                raise ValueError('cpair_list is not from a square matrix!')

            # Allocate memory for the distance matrix.
            mat = np.asmatrix(np.full((d, d), False))
            for upair in upairs:
                if upair.type_conn == 'reciprocal':
                    mat[upair.a, upair.b] = True
                    mat[upair.b, upair.a] = True
                elif upair.type_conn == 'A2B':
                    mat[upair.a, upair.b] = True
                elif upair.type_conn == 'B2A':
                    mat[upair.b, upair.a] = True
                elif upair.type_conn == 'none':
                    pass # this needs some ironing.
                # This is the random/uniform configuration:
                elif upair.type_conn == 'uniform_reciprocal':
                    mat[upair.a, upair.b] = True
                    mat[upair.b, upair.a] = True
                elif upair.type_conn == 'uniform_A2B':
                    mat[upair.a, upair.b] = True
                elif upair.type_conn == 'uniform_B2A':
                    mat[upair.b, upair.a] = True
                # This is the random/uniform distance dependent:
                elif upair.type_conn == 'uniform_reciprocal_dd':
                    mat[upair.a, upair.b] = True
                    mat[upair.b, upair.a] = True
                elif upair.type_conn == 'uniform_A2B_dd':
                    mat[upair.a, upair.b] = True
                elif upair.type_conn == 'uniform_B2A_dd':
                    mat[upair.b, upair.a] = True
                elif upair.type_conn == 'total':
                    # Custom connectivity rules, need custom implementation. Lets leave it here for now:
                    # Total probability rule means that we return connection probability, rather than connectivity:
                    pass
                else:
                    raise ValueError('Connectivity rule not implemented for this type of connections!')
            return mat

        def perform_random_connectivity():
            # if a uniform connection probability is given, update the default.
            # Calculate unidirectional probability for the uniform / random network:
            # This is p = sqrt(overall + 1) - 1
            uniform_prob_unidirectional = np.sqrt(average_conn_prob + 1) - 1
            uniform_prob_reciprocal = uniform_prob_unidirectional**2
            #uniform_prob_reciprocal = average_conn_prob**2
            #uniform_prob_unidirectional = (average_conn_prob - (average_conn_prob**2))/2
            self.set_connection_function(
                pair_type='PN_PN',
                conn_type='uniform_reciprocal',
                conn_func=lambda x: \
                    np.multiply(np.ones(x.shape), uniform_prob_reciprocal)
            )
            self.set_connection_function(
                pair_type='PN_PN',
                conn_type='uniform_A2B',
                conn_func=lambda x: \
                    np.multiply(np.ones(x.shape), uniform_prob_unidirectional)
            )
            self.set_connection_function(
                pair_type='PN_PN',
                conn_type='uniform_B2A',
                conn_func=lambda x: \
                    np.multiply(np.ones(x.shape), uniform_prob_unidirectional)
            )
            # Utilize only the uniform type of connection:
            uniform = {0: 'uniform_reciprocal', 1: 'uniform_A2B', 2: 'uniform_B2A'}

            with self.get_upairs(cell_type='PN_PN') as pn_upairs:
                # Get the connectivity matrix:
                # Update the connectivity matrix only over PN to PN connections:
                connected_upairs = connect_all_pairs(
                    pn_upairs, connection_protocol=uniform
                )

            return upairs2mat(connected_upairs)

        # This function was deleted in commit:
        # dc0c1ba5ea3363c351b1130d74c7bb61185d34bf
        # I'm repurposing it here for intermediate connectivity cases.
        def perform_random_dd_connectivity(percentage=None):
            '''
            Performs random, yet distance dependend connectivity. It replaces
            some (percentage) of the already existing (structured)
            connections with random ones, weighting the overal iid connection
            probability based on distance.
            :param percentage: The percentage of the random, distance dependent.
            :return:
            '''

            # As in the random connectivity, use the same types of
            # connection, but weight them by each pair's intersomatic distance.
            # The iid probability of connection is p = sqrt(overall + 1) - 1
            # as previously, but I'm calculating it by recalculating the
            # overall term based on distance (x) of each pair.

            # derive the weighted probabilities from the already defined and
            # distance dependent total probability. I'm using the original
            # 'total' probability function for clarity on what i'm doing.
            self.set_connection_function(
                pair_type='PN_PN',
                conn_type='uniform_reciprocal_dd',
                conn_func=lambda x: np.power(np.sqrt(
                    self.get_connection_function(
                        pair_type='PN_PN',
                        conn_type='total',
                    )(x)
                    +1)-1,2)
            )
            self.set_connection_function(
                pair_type='PN_PN',
                conn_type='uniform_A2B_dd',
                conn_func=lambda x: np.sqrt(
                    self.get_connection_function(
                        pair_type='PN_PN',
                        conn_type='total',
                    )(x)
                    +1)-1
            )
            self.set_connection_function(
                    pair_type='PN_PN',
                    conn_type='uniform_B2A_dd',
                    conn_func=lambda x: np.sqrt(
                        self.get_connection_function(
                            pair_type='PN_PN',
                            conn_type='total',
                        )(x)
                        +1)-1
            )
            # Provide the strings for the connections to be performed:
            uniform_dd = {0: 'uniform_reciprocal_dd', 1: 'uniform_A2B_dd',
                       2: 'uniform_B2A_dd'}

            # now I need to selectively reapply this connectivity to only a
            # portion of the PN cells (need to recompute the probability as
            # well).

            # Get only pyramidal to pyramidal pairs:
            with self.get_upairs(cell_type='PN_PN') as pn_upairs:
                # Get a random percentage of these to alter their
                # connectivity into random and distance dependent:

                idx_pn = np.random.permutation(len(pn_upairs))
                # Replace % of the reciprocal pairs with non connected:
                total_upairs_reconnected = int(len(pn_upairs) * percentage)
                pn_upairs_affected = np.array(pn_upairs)[idx_pn[:total_upairs_reconnected]]

                # Update the connectivity matrix only for the affected cells:
                connected_upairs = connect_selected_pairs(
                    pn_upairs_affected.tolist(),
                    connection_protocol=uniform_dd
                )

            return upairs2mat(pn_upairs)


        def perform_structured_connectivity():
            #TODO: einai ta reciprocals sxedon ta misa?
            # Get a copy of the network's upairs to calculate overall
            # connection probabilities:
            # Filter out non PN cell pairs:
            pn_upairs = [pair for pair in self.upairs_d.values() if pair.type_cells == ('PN_PN')]

            if False:
                #TODO: DEBUG: json and save:
                pn_upairs_d_l = []
                for upair in pn_upairs:
                    upair_d = {'a':upair.a, 'b':upair.b, 'distance':upair.distance,
                               'type_cells':upair.type_cells,
                               'type_conn':upair.type_conn, 'prob_f':None}
                    pn_upairs_d_l.append(upair_d)


                #pn_upars_1:
                fn_p = Path.cwd().joinpath('debug_files', 'error', f'pn_upairs_1_SN{self.serial_no}.json')
                #pickle.dump(pn_upairs, open(fn_p, "wb"))
                with open(fn_p, 'w') as fp:
                    json.dump(pn_upairs_d_l, fp, indent=4)

            # use a dict to instruct what connections you need:
            # This is the distance dependent connection types from Perin et al., 2011:
            based_on_distance = {0: 'reciprocal', 1: 'A2B', 2: 'B2A'}
            # Get the connectivity matrix:
            #connectivity_mat[:self.pc_no, :self.pc_no] = upairs2mat(connect_all_pairs(pn_upairs, connection_protocol=based_on_distance))
            # Connect pairs based on total PN to PN probability:
            # use a dict to instruct what connections you need:
            # This is the overall (total) connection probability, also distance dependend (perin et al., 2011):
            overall_probability = {0: 'total'}
            # Get probabilities instead of the connectivity matrix:
            Pd_upairs = connect_all_pairs(
                pn_upairs,
                connection_protocol=overall_probability,
                export_probabilities=True
            )
            if False:
                #TODO: DEBUG: json and save:
                pd_upairs_d_l = []
                for upair in Pd_upairs:
                    upair_d = {'a':upair.a, 'b':upair.b, 'distance':upair.distance,
                               'type_cells':upair.type_cells,
                               'type_conn':upair.type_conn, 'prob_f':None}
                    pd_upairs_d_l.append(upair_d)


                #pn_upars_1:
                fn_p = Path.cwd().joinpath('debug_files', 'error', f'pd_upairs_1_SN{self.serial_no}.json')
                #pickle.dump(pn_upairs, open(fn_p, "wb"))
                with open(fn_p, 'w') as fp:
                    json.dump(pd_upairs_d_l, fp, indent=4)

            # Get type '0' connection probability for each pair (this is the total/overall probability):
            prob_f_list = [upair.prob_f[1] for upair in Pd_upairs]
            Pd = np.asmatrix(distance.squareform(prob_f_list))
            #DEBUG
            if False:
                filename = Path.cwd().joinpath('debug_files','error',f'network_debug_PdMat_SN{self.serial_no}.hdf5')
                df = pd.DataFrame({'serial_no': [self.serial_no], 'pc_no': [self.pc_no], 'pv_no': [self.pv_no],
                                   'configuration_alias': self.configuration_alias})
                df.to_hdf(filename, key='attributes', mode='w')
                df = pd.DataFrame(Pd)
                df.to_hdf(filename, key='Pd')

            sumPd = Pd.sum()
            # run iterative rearrangement:
            f_prob = np.zeros((self.pc_no, self.pc_no, rearrange_iterations))
            cn_prob = np.zeros((self.pc_no, self.pc_no, rearrange_iterations))
            cc = np.zeros((1,rearrange_iterations))

            #TODO: this is the connmat of the create_connections. MAKE IT RIGHT!
            #reformed_conn_mat = connectivity_mat[:self.pc_no, :self.pc_no]
            with self.get_upairs(cell_type='PN_PN') as pn_upairs:
                # Get network connectivity matrix:
                reformed_conn_mat = upairs2mat(pn_upairs)
                for t in range(1, rearrange_iterations):
                    print('@iteration {}'.format(t))
                    # TODO check clust_coeff function.
                    # giati to C bgainei olo mhden??
                    _, cc[0, t], _ = clust_coeff(reformed_conn_mat)
                    # compute common neighbors for each pair:
                    ncn = self.common_neighbors(reformed_conn_mat)
                    cn_prob = ncn / ncn.max()
                    # Scale the probabilities of connection by the common neighbor bias:
                    f_prob = np.multiply(cn_prob, Pd)
                    f_prob = f_prob * (sumPd / f_prob.sum())
                    # To allocate connections based on their relative type (unidirectional, reciprocals), compute a distance
                    # estimate and use it with the existing probability functions, to get the connectivity:
                    distance_estimate = (np.log(0.22) - np.log(f_prob)) / 0.0052
                    distance_estimate[distance_estimate < 0] = 0
                    distance_estimate = self.zero_diagonal(distance_estimate)
                    #TODO: this is not working. Why?
                    distance_estimate_vect = distance.squareform(distance_estimate)
                    # squareform works row-wise, also the Pd upairs, wo we can zip() them:
                    for upair, d in zip(pn_upairs, distance_estimate_vect):
                        upair.distance = d

                    # Now you can use this 'distance' estimate to reallocate the connections to unidirectional and reciprocal:
                    #TODO: why you need to assign to temp _next variable?
                    new_upairs = connect_all_pairs(
                        pn_upairs,
                        connection_protocol=based_on_distance
                    )
                    reformed_conn_mat = upairs2mat(new_upairs)
                    #TODO: DEBUG: save the connection matrix for each iteration
                    #to debug it againts the working commit: 6063a41
                    #filename = Path.cwd().joinpath('debug_files','d23a58a',f'iter_{t}_network_debug_SN{self.serial_no}.hdf5')
                    #df = pd.DataFrame({'serial_no': [self.serial_no], 'pc_no': [self.pc_no], 'pv_no': [self.pv_no],
                    #                   'configuration_alias': self.configuration_alias})
                    #df.to_hdf(filename, key='attributes', mode='w')
                    #df = pd.DataFrame(reformed_conn_mat)
                    #df.to_hdf(filename, key='reformed_conn_mat')


                #TODO: the context manager will update the upairs: are the
                # distances correct? Fix them?

            if False:
                # Save the clustering coefficient, for plotting it later.
                configuration_alias = 'structured'
                filename = Path(rf'C:\Users\steve\Documents\analysis\Python\new_files'). \
                    joinpath(f'{configuration_alias}_network_SN{self.serial_no}_clust_coeff.hdf5')
                df = pd.DataFrame({'serial_no': [self.serial_no], 'pc_no': [self.pc_no], 'pv_no': [self.pv_no],
                                   'configuration_alias': self.configuration_alias})
                df.to_hdf(filename, key='attributes', mode='w')
                df = pd.DataFrame(cc)
                df.to_hdf(filename, key='clust_coeff')

            # Plot
            if plot:
                savgol_window = int(cc.shape[1] / 10 + 1)
                savgol_polyorder = 3
                if savgol_window < savgol_polyorder:
                    print('!!!Use more rearrange_iterations or reduce savgol_window!')
                fig, ax = plt.subplots()
                ax.plot(cc[0, 1:], color='gray', alpha=0.2)
                ax.plot(
                    savgol_filter(cc[0, 1:], savgol_window, 3),
                    color='k', linewidth=2
                )
                ax.set_title('Clustering Coefficient')
                plt.xlabel('Iterations')
                plt.ylabel('Clustering Coefficient')
                nb.axis_normal_plot(axis=ax)
                nb.adjust_spines(ax, ['left', 'bottom'])
                plt.savefig(
                    f'Network_sn{self.serial_no}_clustering_coefficient.png'
                )
                plt.savefig(
                    f'Network_sn{self.serial_no}_clustering_coefficient.svg'
                )
                plt.savefig(
                    f'Network_sn{self.serial_no}_clustering_coefficient.pdf'
                )
                plt.show()

            # Save connectivity to the network:
            return reformed_conn_mat


        # Initially connect only non PN to PN pairs (but due to performance and code readability issues connect the PN
        # to PN pairs also; unordered pairs must lie on a square matrix!):
        print(f'Random initially {np.random.rand(1)[0]}')
        connectivity_mat = np.full((self.cell_no, self.cell_no), False)
        # use a dict to instruct what connections you need:
        # This is the distance dependent connection types from Perin et al., 2011:
        based_on_distance = {0: 'reciprocal', 1: 'A2B', 2: 'B2A'}
        # reuse the generated connectivity for PN_PV and PV_PV pairs that never changes again:
        with self.get_upairs() as all_upairs:
            # Get the connectivity matrix:
            connectivity_mat = upairs2mat(
                connect_all_pairs(
                    all_upairs, connection_protocol=based_on_distance
                )
            )
        #TODO: Why I do this again? Do the conn mats differ?
        connectivity_mat2 = upairs2mat(self.upairs_d.values())

        print(f'Random finally {np.random.rand(1)[0]}')
        #with open('connblah1.txt', 'w') as f:
        #    for element in connectivity_mat[-1, :]:
        #        f.write(f'{element}\n')

        # Plot to see the validated results of the connectivity routines:
        if plot:
            plt.ion()
            compute_reciprocal_across_distance(connectivity_mat[:self.pc_no, :self.pc_no],
                                            self.dist_mat[:self.pc_no, :self.pc_no],
                                               self.connection_functions_d['PN_PN']['reciprocal'],
                                               )
            compute_unidirectional_across_distance(connectivity_mat[:self.pc_no, :self.pc_no],
                                                self.dist_mat[:self.pc_no, :self.pc_no],
                                                   self.connection_functions_d['PN_PN']['unidirectional'],
                                                   )
            compute_pn2pv_unidirectional_across_distance(mat_pn_pv=connectivity_mat[:self.pc_no, self.pc_no:],
                                                         mat_pv_pn=connectivity_mat[self.pc_no:, :self.pc_no],
                                                         dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                         ground_truth=self.connection_functions_d['PN_PV']['A2B'],
                                                         )
            compute_pv2pn_unidirectional_across_distance(mat_pn_pv=connectivity_mat[:self.pc_no, self.pc_no:],
                                                         mat_pv_pn=connectivity_mat[self.pc_no:, :self.pc_no],
                                                         dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                         ground_truth=self.connection_functions_d['PN_PV']['B2A'],
                                                         )
            compute_pn_pv_reciprocal_across_distance(connectivity_mat[:self.pc_no, self.pc_no:],
                                                  connectivity_mat[self.pc_no:, :self.pc_no],
                                                  self.dist_mat[:self.pc_no, self.pc_no:],
                                                     self.connection_functions_d['PN_PV']['reciprocal']
                                                     )


        # Now if configuration is random, update the PN to PN connectivity matrix part and return:
        if self.configuration_alias is 'random':
            excitatory_conn_mat = perform_random_connectivity()

        # if configuration is structured,
        elif self.configuration_alias is 'structured':
            excitatory_conn_mat = perform_structured_connectivity()

        # If connectivity is intermediate:
        #TODO: check for the substring, because we need also the percentage
        elif 'intermediate' in self.configuration_alias:
            percentage = \
                int(re.search(r'\d+', self.configuration_alias).group())
            try:
                if percentage > 100:
                    raise ValueError
            except ValueError:
                print('Random configuration percentage given is not correct!')
            # Initially perform structured connectivity
            str_connmat = perform_structured_connectivity()
            fig, ax = plt.subplots()
            cax = ax.imshow(str_connmat, interpolation='nearest', cmap=cm.afmhot)
            ax.set_title(f'Network_sn{self.serial_no}_str_connmat_of '
                         f'{self.configuration_alias}')
            plt.savefig(f'Network_sn{self.serial_no}_str_conn_mat_of '
                         f'{self.configuration_alias}')

            excitatory_conn_mat = perform_random_dd_connectivity(percentage/100)
            fig, ax = plt.subplots()
            cax = ax.imshow(excitatory_conn_mat, interpolation='nearest', cmap=cm.afmhot)
            ax.set_title(f'Network_sn{self.serial_no}_sel_connmat_of '
                         f'{self.configuration_alias}')
            plt.savefig(f'Network_sn{self.serial_no}_sel_conn_mat_of '
                        f'{self.configuration_alias}')
        else:
            try:
                raise ValueError
            except ValueError:
                print('Can not interpret the connection configuration alias!')

        # Save resulting connectivity matrix to the network:
        connectivity_mat[:self.pc_no, :self.pc_no] = excitatory_conn_mat
        self.connectivity_mat = connectivity_mat
        # Plot if you want:
        if plot:
            fig, ax = plt.subplots()
            cax = ax.imshow(self.connectivity_mat, interpolation='nearest', cmap=cm.afmhot)
            ax.set_title('Network_sn{}_connectivity matrix of {}'.format(self.serial_no, self.configuration_alias))
            plt.savefig('Network_sn{}_connectivity_matrix_{}.png'.format(self.serial_no, self.configuration_alias))
            plt.show()

        return

    @with_reproducible_rng
    def initialize_trials(self, trial_no=0, stimulated_pn_no=50):
        # initialize stimulated cells in each trial:
        self.trial_no = trial_no
        self.stimulated_cells = np.full((self.trial_no, stimulated_pn_no), 0, dtype=int)
        for trial in range(self.trial_no):
            self.stimulated_cells[trial][:] = np.sort(np.random.permutation(np.arange(self.pc_no))[:stimulated_pn_no])

        ## Since round returns in [0,1), we are ok
        #self.stimulated_cells = np.asarray(
        #    np.round(np.multiply(np.random.rand(self.trial_no, stimulated_pn_no), self.pc_no)), dtype=int)
        ## Sort stimulated cells' list:
        #self.stimulated_cells = np.sort(self.stimulated_cells, axis=1)


    def common_neighbors(self, adjmat):
        #TODO: THIS SHOULD BE MATRIX MULTIPLICATION, WHY ITS A SIMPLE ONE?
        #DOES IT WORK IN THE ORIGINAL PAPER?
        adj = np.logical_or(adjmat, adjmat.T)
        adj = adj.astype(float) @ adj.T.astype(float)
        adj = self.zero_diagonal(adj)
        return adj


    def zero_diagonal(self, mat):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Matrix must be square!")
        for i in range(mat.shape[0]):
            mat[i,i] = 0.0
        return mat

    @with_reproducible_rng
    def create_weights(self):
        # Generate weights matrices:
        # Kayaguchi : more potent synapses in reciprocal pairs
        # Perin: more potent synapses with more clustering coeff:
        # We try to combine them
        # Every connected pair has weight of one:
        self.weights_mat = np.asarray(self.connectivity_mat, dtype=float)

        '''
        #need to create weights from common neighbors:
        scipy.stats.skewnorm.pdf()
        cn_range = np.arange(0, 8, 0.01)
        #TODO: get them from the CN matrix
        maxNeighbors = 20;
        alpha = 3 * ones(1, maxNeighbors); % linspace(3, 0.1, maxNeighbors) %.*exp( linspace(0, 0.5, maxNeighbors)). ^ -2; % [20, 10, 2, 1];
        maxSigma = 5;
        minSigma = 0.1;
        expClimb = (exp(linspace(0, 0.5, maxNeighbors)). ^ 5) - 1;
        expClimb = (expClimb / max(expClimb) * maxSigma) + minSigma;
        m = ones(1, maxNeighbors). * expClimb. * 2; % exp( linspace(0, 0.6, maxNeighbors)) - 1;
        m = m - m(1);
        sigma = 0.5 * ones(1, maxNeighbors). * expClimb %.*(1. / ( 1 + exp(-linspace(-2, 5, maxNeighbors)))) %.*exp( linspace(0, 0.5, maxNeighbors)). ^ -10; % [0.15, 0.2, 4, 6];
        normFactors = ones(1, maxNeighbors); % [0.0045, 0.006, 0.02, 0.02];
        % LogNormal PDF parameters:
        parmhat(1, 1: maxNeighbors) = linspace(0.1, 10, maxNeighbors);
        parmhat(2, 1: maxNeighbors) = 2.6;
        CNsnormal = [];
        % Probabilities changed to replicate mean EPSP amplitude per connections in
        % cluster(Perin et al, Fig .6 A):
        maxCDF = zeros(3, length(rang));
        myvar = [1, round(maxNeighbors / 2), maxNeighbors];
        for k=1:length(myvar)
        CNsnormal(k,:) = snormal(rang, m(myvar(k)), alpha(myvar(k)),
                                 sigma(myvar(k))) * normFactors(myvar(k));
        maxCDF(k,:) = cumsum(CNsnormal(k,:)) / max(cumsum(CNsnormal(k,:)));
        end
        figure;
        plot(CNsnormal(:,:)');hold on;
        set(gca, 'xtick', 1: 60:length(rang));
        set(gca, 'xticklabel', rang(1: 60:end));
        '''

        '''
        # Inhibitory weights are connection type dependent:
        # PV to PN weights must be 3 somatic 5 dendritic synapses.
        # These must result in ~90 pA current at the soma (reciprocal pairs; max)
        pv_pn_upairs = [pair for pair in self.upairs_d.values() if pair.type_cells == 'PN_PV' and pair.type_conn == 'reciprocal']
        for pair in pv_pn_upairs:
            a, b = pair.a, pair.b
            # EPSCs from PNs to PVs are 2.85 times greater if reciprocally connected:
            self.weights_mat[a, b] = 2.85
            # IPSCs from PVs to PNs are 2.2 times greater if reciprocally connected:
            # since A, B lie by convention on the upper half of the connectivity matrix, swap them to get to PV -> PN:
            self.weights_mat[b, a] = 2.2
        # At the end normalize the weight matrix (separately for each type to preserve validation)
        pv2pn_weights = self.weights_mat[self.pc_no:, :self.pc_no]
        pv2pn_weights = np.divide(pv2pn_weights, pv2pn_weights.max())
        self.weights_mat[self.pc_no:, :self.pc_no] = pv2pn_weights

        pn2pv_weights = self.weights_mat[:self.pc_no, self.pc_no:]
        pn2pv_weights = np.divide(pn2pv_weights, pn2pv_weights.max())
        self.weights_mat[:self.pc_no, self.pc_no:] = pn2pv_weights
        '''

    def create_network_stats(self):
        '''
        Connectivity stats ONLY for excitatory connections (PN to PN).
        :return:
        '''
        # TODO: check the correctness of network stats.
        # Gets statistics such no of reciprocal pairs, overall connection
        # probability etc for a square connectivity matrix.

        # make sure that connmat diagonal is zeros!
        # Get only excitatory connections:
        # this should be zero by default:
        connectivity_mat = self.zero_diagonal(self.connectivity_mat[:self.pc_no, :self.pc_no])

        N = connectivity_mat.shape[0]

        # compute stats:
        self.stats['nUniqueUnorderedPairs'] = (N**2 - N)/2
        self.stats['nUniqueOrderedPairs'] = (N**2 - N)

        connected = np.logical_or(connectivity_mat, connectivity_mat.T)
        reciprocal = np.logical_and(connectivity_mat, connectivity_mat.T)
        #unidirectional = np.logical_xor(connectivity_mat, reciprocal)
        unidirectional = np.logical_xor(connectivity_mat, connectivity_mat.T)
        # Get unordered pairs that are connected (not care about directionality):
        connected_up = np.logical_and(connected, np.asmatrix(np.triu(np.ones((N, N)), 1), dtype=bool))
        # The average connectivity is measured as the number of connected unordered pairs, out of total unordered pairs.
        self.stats['averageConnectivity'] = np.sum(connected_up) / self.stats['nUniqueUnorderedPairs']
        self.stats['nReciprocalUnorderedPairs'] = reciprocal.sum() / 2
        self.stats['nUnidirectionalUnorderedPairs'] = unidirectional.sum() / 2

        # UNVALIDATED VALUES TODO: validate values
        # overall connection probability
        #self.stats['connectedUnorderedPairsPercentage'] = np.sum(connected) / self.stats['nUniqueUnorderedPairs']
        #self.stats['unidirectionalUnorderedPairsPercentage'] = np.sum(unidirectional) / self.stats['nUniqueUnorderedPairs']
        #self.stats['bidirectionalUnorderedPairsPercentage'] = (np.sum(reciprocal)/2) / self.stats['nUniqueUnorderedPairs']

        _, self.stats['clust_coeff'], _ = clust_coeff(self.connectivity_mat)
        # degree distribution:
        #_, self.stats['indeg'], self.stats['outdeg'] = degrees(self.connectivity_mat);
        #self.stats['hrange'] = linspace(0,max([nstats.indeg,nstats.outdeg]),10);
        #nstats.histo_indeg = histcounts(nstats.indeg,nstats.hrange);
        #nstats.histo_outdeg = histcounts(nstats.outdeg,nstats.hrange);

    def export_network_parameters(self, export_path=os.getcwd(), postfix=''):
        '''
        Export hoc files with the network parameters for the NEURON simulation. This is done for the configuration
        alias of the network.
        :param export_path:
        :param postfix:
        :return:
        '''
        # Export Network Connectivity:
        # Export parameter matrices in .hoc file:
        with open(os.path.join(export_path,
                'importNetworkParameters_{}_SN{}_{}.hoc'.format(self.configuration_alias, self.serial_no, postfix)), 'w') as f:
            f.write('// This HOC file was generated with network_tools python module.\n')
            f.write('nPCcells={}\n'.format(self.pc_no))
            f.write('nPVcells={}\n'.format(self.pv_no))
            f.write('nAllCells={}\n'.format(self.cell_no))
            f.write('// Object decleration:\n')
            f.write('objref C, W\n')
            f.write('objref StimulatedPNs[{}]\n'.format(self.trial_no))
            f.write('C = new Matrix(nAllCells, nAllCells)\n')
            f.write('W = new Matrix(nAllCells, nAllCells)\n')
            f.write('\n// Import parameters: (long-long text following!)\n')
            # Network connectivity:
            pairs = [(i, j) for i in range(self.cell_no) for j in range(self.cell_no)]
            for (i, j) in pairs:
                f.write('C.x[{}][{}]={}\n'.format(i, j, int(self.connectivity_mat[i, j])))
            for (i, j) in pairs:
                f.write('W.x[{}][{}]={}\n'.format(i, j, self.weights_mat[i, j]))
            # Network stimulation:
            for trial in range(self.trial_no):
                f.write('StimulatedPNs[{}]=new Vector({})\n'.format(trial, self.stimulated_cells.shape[1]))
                for i in range(self.stimulated_cells.shape[1]):
                    f.write('StimulatedPNs[{}].x[{}]={}\n'.format(trial, i, self.stimulated_cells[trial][i]))
            f.write('//EOF\n')


    def save_data(self, export_path=os.getcwd(), filename_prefix='', filename_postfix=''):
        '''
        Store Network data to a HDF5 file
        :return:
        '''
        # Save using pandas for consistency:
        filename = os.path.join(export_path, '{}{}_network_SN{}{}.hdf5'
                                .format(filename_prefix, self.configuration_alias, self.serial_no, filename_postfix))
        df = pd.DataFrame({'serial_no': [self.serial_no], 'pc_no': [self.pc_no], 'pv_no': [self.pv_no],
                           'configuration_alias': self.configuration_alias})
        df.to_hdf(filename, key='attributes', mode='w')
        df = pd.DataFrame(self.connectivity_mat)
        df.to_hdf(filename, key='connectivity_mat')
        df = pd.DataFrame(self.weights_mat)
        df.to_hdf(filename, key='weights_mat')
        df = pd.DataFrame(self.dist_mat)
        df.to_hdf(filename, key='dist_mat')
        df = pd.DataFrame(self.upairs_d, index=[0])
        df.to_hdf(filename, key='upairs_d')
        df = pd.DataFrame(self.stats, index=[0])
        df.to_hdf(filename, key='stats')
        df = pd.DataFrame(self.pc_somata)
        df.to_hdf(filename, key='pc_somata')
        df = pd.DataFrame(self.pv_somata)
        df.to_hdf(filename, key='pv_somata')
        #blah = pd.read_hdf(filename, key='attributes')

def isdirected(adj):
    '''
    Check if adjucency matrix is directed.
    :param adj:
    :return:
    '''
    if adj.dtype is not np.dtype('bool'):
        raise ValueError('Adjucency matrix must be logical (bool)!')
    return not np.allclose(adj, adj.T, atol=1e-8)

def degrees(adj):
    '''
    :param adj:
    :return: 
    '''
    if adj.dtype is not np.dtype('bool'):
        raise ValueError('Adjucency matrix must be logical (bool)!')
    indeg = np.sum(adj, axis=0)
    outdeg = np.sum(adj, axis=1).T

    if isdirected(adj):
      deg = indeg + outdeg # total degree
    else:   # undirected graph: indeg=outdeg
      deg = indeg + np.diag(adj)  # add self-loops twice, if any
    return deg,indeg,outdeg

def kneighbors(adj,node_index,links_no):
    if adj.dtype is not np.dtype('bool'):
        raise ValueError('Adjucency matrix must be logical (bool)!')
    # here we need adj as float:
    adj = np.asmatrix(adj, dtype=float)
    adjk = adj
    for i in range(links_no-1): #=1:k-1;
        adjk = adjk * adj

    kneigh = np.nonzero(adjk[node_index,:]>0)
    # since is always a row vector, return only columns:
    return kneigh[1]

def subgraph(adj, subgraph_ind):
    if adj.dtype is not np.dtype('bool'):
        raise ValueError('Adjucency matrix must be logical (bool)!')
    return adj[subgraph_ind, subgraph_ind]

def selfloops(adj):
    if adj.dtype is not np.dtype('bool'):
        raise ValueError('Adjucency matrix must be logical (bool)!')
    return np.sum(np.diag(adj), axis=0)

def numedges(adj):
    if adj.dtype is not np.dtype('bool'):
        raise ValueError('Adjucency matrix must be logical (bool)!')
    sl = selfloops(adj) # counting the number of self-loops

    if not isdirected(adj): # sl==0    % undirected simple graph
        return adj.sum()/2
    elif not isdirected(adj) and sl>0 :
        sl=selfloops(adj)
        return (sum(sum(adj))-sl)/2+sl # counting the self-loops only once
    elif isdirected(adj):   # directed graph (not necessarily simple)
        return adj.sum()

def loops3(adj):
    # Implicitly we need again the adjacency matrix as float:
    adj = np.asmatrix(adj, dtype=float)
    return np.trace((adj*adj)*adj) / 6

def num_conn_triples(adj):
    c = 0  # initialize
    n = adj.shape[0]
    for i in range(n): #=1:length(adj)
        neigh = kneighbors(adj, node_index=i, links_no=1)
        if neigh.size < 2:
            continue  # handle leaves, no triple here
        c += nchoosek(neigh.size, 2)
    c -= 2*loops3(adj) # due to the symmetry triangles repeat 3 times in the nchoosek count
    return c

def clust_coeff(adj):
    '''
    :param adj:
    :return:
    '''
    if adj.dtype is not np.dtype('bool'):
        raise ValueError('Adjucency matrix must be logical (bool)!')
    if adj.shape[0] != adj.shape[1]:
        raise ValueError('Matrix adj is not square!')
    n = adj.shape[0]
    deg, indeg, outdeg = degrees(adj)
    C = np.full((n, 1), 0.0)
    if isdirected(adj):
        coeff = 1
    else:
        coeff = 2

    for i in range(n):
        if deg[0, i] == 1 | deg[0, i] == 0:
            C[i] = 0
            continue
        neigh = kneighbors(adj, node_index=i, links_no=1)
        edges_s = numedges(subgraph(adj, subgraph_ind=neigh))
        C[i] = coeff * edges_s / deg[0, i] / (deg[0, i] - 1)

    C1 = loops3(adj) / num_conn_triples(adj)
    C2 = np.sum(C, axis=0) / n
    return C, C1, C2

def get_reciprocal_number(mat):
    '''
    Return number of reciprocally connected pairs in connectivity matrix mat.
    :param mat:
    :return:
    '''
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square!')
    logical_mat = np.asmatrix(mat, dtype=bool)
    return np.logical_and(logical_mat,  logical_mat.getT()).sum() / 2

def get_unidirectional_number(mat):
    '''
    Return number of unidirectional connected pairs in connectivity matrix mat.
    :param mat:
    :return:
    '''
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square!')
    logical_mat = np.asmatrix(mat, dtype=bool)
    return np.logical_xor(logical_mat,  logical_mat.getT()).sum() / 2

def get_reciprocal_percentage(mat):
    '''
    Return percentage of reciprocally connected pairs in connectivity matrix mat.
    :param mat:
    :return:
    '''
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square!')
    n = mat.shape[0]
    return get_reciprocal_number(mat) / (n*(n-1)/2)

def get_unidirectional_percentage(mat):
    '''
    Return percentage of unidirectionaly connected pairs in connectivity matrix mat.
    :param mat:
    :return:
    '''
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square!')
    n = mat.shape[0]
    return get_unidirectional_number(mat) / (n*(n-1)/2)

def plot_connectivity_across_distance(
        conn_mat, distance_mat, conn_pair, conn_type, conn_functions_dict
    ):
    '''
    Plots a histogram with the relative frequency of reciprocal connections as a function of distance.
    If a ground truth function is provided, plot it for comparisson.
    :param mat:
    :return:
    '''
    bin_width = 10
    max_dist = 300
    # Make sure that arguments are correct:
    # If you have a non-square input mat, var is a list of two matrices:
    if isinstance(conn_mat, list):
        mat_pn_pv = conn_mat[0]
        mat_pv_pn = conn_mat[1]
        if mat_pn_pv.shape[0] != mat_pv_pn.shape[1]:
            raise ValueError('Matrices'' transpose must be of same dimensions!')
        # Also distance conn_mat is not square:
        if distance_mat.shape[0] < distance_mat.shape[1]:
            # we want dist_pn_pv, so transpose the matrix:
            distance_mat = distance_mat.getT()
        if mat_pn_pv.shape[0] != distance_mat.shape[0]:
            raise ValueError('Matrices must be of same size!')

        logical_mat_pn_pv = np.asmatrix(mat_pn_pv, dtype=bool)
        logical_mat_pv_pn = np.asmatrix(mat_pv_pn, dtype=bool)
        mat_unid = np.logical_and(logical_mat_pn_pv,  logical_mat_pv_pn.T)

        histo_bins = np.arange(0, max_dist, bin_width)
        histo_dist, bin_edges = np.histogram(distance_mat, bins=histo_bins)
        histo, bin_edges = np.histogram(distance_mat[mat_unid], bins=histo_bins)
    else:
        # Else:
        if conn_mat.shape[0] != conn_mat.shape[1]:
            raise ValueError('Matrix must be square!')
        if distance_mat.shape[0] != distance_mat.shape[1]:
            raise ValueError('Matrix must be square!')
        if conn_mat.shape[0] != distance_mat.shape[0]:
            raise ValueError('Matrices must be of same size!')

        n = conn_mat.shape[0]
        # get only upper triangular part:
        logical_mat = np.asmatrix(conn_mat, dtype=bool)
        mat_recip = np.logical_and(logical_mat,  logical_mat.getT())
        conn_vector = mat_recip[np.asmatrix(np.triu(np.ones((n, n)), 1), dtype=bool)]
        dist_vector = distance_mat[np.asmatrix(np.triu(np.ones((n, n)), 1), dtype=bool)]

        histo_bins = np.arange(0, max_dist, bin_width)
        histo_dist, bin_edges = np.histogram(dist_vector, bins=histo_bins)
        histo, bin_edges = np.histogram(dist_vector[np.asarray(conn_vector)[0]], bins=histo_bins)

    ground_truth = conn_functions_dict[conn_pair][conn_type]
    # Plot
    fig, ax = plt.subplots()
    # Normalize histogram of intra-neuronal distances to GT maximum, for
    # better visualization.
    cax = ax.plot(
        histo_bins[:-1], np.divide(histo, histo_dist), linewidth=2
    )
    cax = ax.plot(histo_bins, ground_truth(histo_bins), linewidth=2)
    cax = ax.bar(
        histo_bins[:-1], (histo / np.max(histo)) \
                         * np.max(ground_truth(histo_bins)),
        width=bin_width, color='gray', alpha=0.4
    )
    ax.set_title('Structured Configuration')
    plt.xlabel('Distance (um)')
    plt.ylabel(f'{conn_pair} {conn_type} Probability')
    nb.axis_normal_plot(axis=ax)
    nb.adjust_spines(ax, ['left', 'bottom'])

    plt.savefig(f'{conn_pair}_{conn_type}_across_distance.png')
    plt.show()

def plot_connectivity(
        histo, histo_bins, histo_dist, ground_truth, bin_width, ax=None
):
    if not ax:
        fig, ax = plt.subplots()
    # Normalize histogram of intra-neuronal distances to GT maximum, for
    # better visualization.
    cax = ax.plot(
        histo_bins[:-1], np.divide(histo, histo_dist), linewidth=2
    )
    if ground_truth:
        cax = ax.plot(histo_bins, ground_truth(histo_bins), linewidth=2)
        cax = ax.bar(
            histo_bins[:-1], (histo / np.max(histo)) \
                             * np.max(ground_truth(histo_bins)),
            width=bin_width, color='gray', alpha=0.4
        )
    ax.set_title('Structured Configuration')
    nb.axis_normal_plot(axis=ax)
    nb.adjust_spines(ax, ['left', 'bottom'])
    return ax

def plot_average_connectivity(
        histo, histo_bins, histo_dist, ground_truth, bin_width, ax=None
):
    if not ax:
        fig, ax = plt.subplots()
    #sb.regplot(x=list(chain(*K_s)), y=list(chain(*PC_no)), ax=ax, marker='.', color='C0')
    #TODO: prepei to histo na to kanw divide me to distance k meta na ta balw se DataFrame.
    flat_freq = histo.reshape((histo.size, 1))
    flat_dist = histo_dist.reshape((histo.size, 1))
    bins = np.concatenate((histo_bins[:-1], histo_bins[:-1], histo_bins[:-1], histo_bins[:-1]))
    flat_freq_per_dist = np.divide(flat_freq, flat_dist)
    flat_freq_per_dist[flat_freq_per_dist == 0] = np.nan
    df = pd.DataFrame({'Datapoints':flat_freq_per_dist[:, 0], 'bins': bins})
    sb.lineplot(x='bins', y='Datapoints', data=pd.DataFrame(df), ax=ax, dashes=False)

    if ground_truth:
        cax = ax.plot(histo_bins, ground_truth[0](histo_bins), linewidth=2)
        cax = ax.bar(
            histo_bins[:-1], (np.sum(histo, axis=0) / histo.sum()),
            width=bin_width, color='gray', alpha=0.4
        )
    ax.set_title('Structured Configuration')
    nb.axis_normal_plot(axis=ax)
    nb.adjust_spines(ax, ['left', 'bottom'])
    return ax

def compute_reciprocal_across_distance(
        mat, distance_mat, ground_truth=None, plot=True, plot_axis=None
):
    '''
    Plots a histogram with the relative frequency of reciprocal connections as a function of distance.
    If a ground truth function is provided, plot it for comparisson.
    :param mat:
    :return:
    '''
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square!')
    if distance_mat.shape[0] != distance_mat.shape[1]:
        raise ValueError('Matrix must be square!')
    if mat.shape[0] != distance_mat.shape[0]:
        raise ValueError('Matrices must be of same size!')

    n = mat.shape[0]
    # get only upper triangular part:
    logical_mat = np.asmatrix(mat, dtype=bool)
    mat_recip = np.logical_and(logical_mat,  logical_mat.getT())
    conn_vector = mat_recip[np.asmatrix(np.triu(np.ones((n, n)), 1), dtype=bool)]
    dist_vector = distance_mat[np.asmatrix(np.triu(np.ones((n, n)), 1), dtype=bool)]

    bin_width = 10
    histo_bins = np.arange(0, 160, bin_width)
    histo_dist, bin_edges = np.histogram(dist_vector, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_vector[np.asarray(conn_vector)[0]], bins=histo_bins)

    if plot:
        ax = plot_connectivity(
            histo, histo_bins, histo_dist, ground_truth, bin_width, ax=plot_axis
        )
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('PN-PN Reciprocal Probability')

        plt.savefig('PN-PN_reciprocal_across_distance.png')
        plt.savefig('PN-PN_reciprocal_across_distance.svg')
        plt.savefig('PN-PN_reciprocal_across_distance.pdf')
        plt.show()

    return (histo, histo_bins, histo_dist, ground_truth, bin_width)

def compute_unidirectional_across_distance(
        mat, distance_mat, ground_truth=None, plot=True, plot_axis=None
):
    '''
    Plots a histogram with the relative frequency of unidirectional connections as a function of distance.
    If a ground truth function is provided, plot it for comparisson.
    :param mat:
    :return:
    '''
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square!')
    if distance_mat.shape[0] != distance_mat.shape[1]:
        raise ValueError('Matrix must be square!')
    if mat.shape[0] != distance_mat.shape[0]:
        raise ValueError('Matrices must be of same size!')

    n = mat.shape[0]
    # get only upper triangular part:
    logical_mat = np.asmatrix(mat, dtype=bool)
    mat_unid = np.logical_xor(logical_mat,  logical_mat.getT())
    conn_vector = mat_unid[np.asmatrix(np.triu(np.ones((n, n)), 1), dtype=bool)]
    dist_vector = distance_mat[np.asmatrix(np.triu(np.ones((n, n)), 1), dtype=bool)]

    bin_width = 10
    histo_bins = np.arange(0, 160, bin_width)
    histo_dist, bin_edges = np.histogram(dist_vector, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_vector[np.asarray(conn_vector)[0]], bins=histo_bins)

    if plot:
        ax = plot_connectivity(
            histo, histo_bins, histo_dist, ground_truth, bin_width, ax=plot_axis
        )
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('PN-PN Unidirectional Probability')

        plt.savefig('PN-PN_unidirectional_across_distance.png')
        plt.savefig('PN-PN_unidirectional_across_distance.svg')
        plt.savefig('PN-PN_unidirectional_across_distance.pdf')
        plt.show()

    return (histo, histo_bins, histo_dist, ground_truth, bin_width)

def compute_pn_pv_reciprocal_across_distance(
        mat_pn_pv, mat_pv_pn, dist_mat, ground_truth=None, plot=True, plot_axis=None
):
    '''
    Plots a histogram with the relative frequency of pn to pv reciprocal connections as a function of distance.
    If a ground truth function is provided, plot it for comparisson.
    :param mat:
    :return:
    '''
    if mat_pn_pv.shape[0] != mat_pv_pn.shape[1]:
        raise ValueError('Matrices'' transpose must be of same dimensions!')
    if dist_mat.shape[0] < dist_mat.shape[1]:
        # we want dist_pn_pv, so transpose the matrix:
        dist_mat = dist_mat.getT()
    if mat_pn_pv.shape[0] != dist_mat.shape[0]:
        raise ValueError('Matrices must be of same size!')

    logical_mat_pn_pv = np.asmatrix(mat_pn_pv, dtype=bool)
    logical_mat_pv_pn = np.asmatrix(mat_pv_pn, dtype=bool)
    mat_unid = np.logical_and(logical_mat_pn_pv,  logical_mat_pv_pn.T)

    bin_width = 10
    histo_bins = np.arange(0, 160, bin_width)
    histo_dist, bin_edges = np.histogram(dist_mat, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_mat[mat_unid], bins=histo_bins)

    if plot:
        ax = plot_connectivity(
            histo, histo_bins, histo_dist, ground_truth, bin_width, ax=plot_axis
        )
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('PN2PV Reciprocal Probability')
        plt.savefig('PN-PV_reciprocal_across_distance.png')
        plt.savefig('PN-PV_reciprocal_across_distance.svg')
        plt.savefig('PN-PV_reciprocal_across_distance.pdf')
        plt.show()

    return (histo, histo_bins, histo_dist, ground_truth, bin_width)

def compute_pn2pv_unidirectional_across_distance(
        mat_pn_pv, mat_pv_pn, dist_mat, ground_truth=None, plot=True,
        plot_axis=None
):
    '''
    Plots a histogram with the relative frequency of pn to pv unidirectional connections as a function of distance.
    If a ground truth function is provided, plot it for comparisson.
    :param mat:
    :return:
    '''
    if mat_pn_pv.shape[0] != mat_pv_pn.shape[1]:
        raise ValueError('Matrices'' transpose must be of same dimensions!')
    if dist_mat.shape[0] < dist_mat.shape[1]:
        # we want dist_pn_pv, so transpose the matrix:
        dist_mat = dist_mat.getT()
    if mat_pn_pv.shape[0] != dist_mat.shape[0]:
        raise ValueError('Matrices must be of same size!')

    logical_mat_pn_pv = np.asmatrix(mat_pn_pv, dtype=bool)
    logical_mat_pv_pn = np.asmatrix(mat_pv_pn, dtype=bool)
    mat_unid = np.logical_and(np.logical_xor(logical_mat_pn_pv,  logical_mat_pv_pn.T), logical_mat_pn_pv)

    bin_width = 10
    histo_bins = np.arange(0, 160, bin_width)
    histo_dist, bin_edges = np.histogram(dist_mat, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_mat[mat_unid], bins=histo_bins)

    if plot:
        ax = plot_connectivity(
            histo, histo_bins, histo_dist, ground_truth, bin_width, ax=plot_axis
        )
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('PN2PV Unidirectional Probability')

        plt.savefig('plot_pn2pv_unidirectional_across_distance.png')
        plt.savefig('plot_pn2pv_unidirectional_across_distance.svg')
        plt.savefig('plot_pn2pv_unidirectional_across_distance.pdf')
        plt.show()

    return (histo, histo_bins, histo_dist, ground_truth, bin_width)

def compute_pv2pn_unidirectional_across_distance(
        mat_pn_pv, mat_pv_pn, dist_mat, ground_truth=None, plot=True,
        plot_axis=None
):
    '''
    Plots a histogram with the relative frequency of pn to pv unidirectional connections as a function of distance.
    If a ground truth function is provided, plot it for comparisson.
    :param mat:
    :return:
    '''
    if mat_pn_pv.shape[0] != mat_pv_pn.shape[1]:
        raise ValueError('Matrices'' transpose must be of same dimensions!')
    if dist_mat.shape[0] < dist_mat.shape[1]:
        # we want dist_pn_pv, so transpose the matrix:
        dist_mat = dist_mat.getT()
    if mat_pn_pv.shape[0] != dist_mat.shape[0]:
        raise ValueError('Matrices must be of same size!')

    logical_mat_pn_pv = np.asmatrix(mat_pn_pv, dtype=bool)
    logical_mat_pv_pn = np.asmatrix(mat_pv_pn, dtype=bool)
    mat_unid = np.logical_and(np.logical_xor(logical_mat_pn_pv,  logical_mat_pv_pn.T), logical_mat_pv_pn.T)

    bin_width = 10
    histo_bins = np.arange(0, 160, bin_width)
    histo_dist, bin_edges = np.histogram(dist_mat, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_mat[mat_unid], bins=histo_bins)

    if plot:
        ax = plot_connectivity(
            histo, histo_bins, histo_dist, ground_truth, bin_width, ax=plot_axis
        )
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('PV2PN Unidirectional Probability')

        plt.savefig('plot_pv2pn_unidirectional_across_distance.png')
        plt.savefig('plot_pv2pn_unidirectional_across_distance.svg')
        plt.savefig('plot_pv2pn_unidirectional_across_distance.pdf')
        plt.show()

    return (histo, histo_bins, histo_dist, ground_truth, bin_width)
