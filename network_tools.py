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
from functools import wraps

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
        return class_method(*args, **kwargs)
    return reset_rng

class Network:
    @property
    def serial_no(self):
        return self._serial_no
    
    @serial_no.setter
    def serial_no(self, serial_no):
        if serial_no < 1:
            raise ValueError("Only positive serian numbers allowed!")
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

    # This is a tuple representing the unordered pairs of neurons in the network
    UPair = namedtuple('UPair', ['a', 'b', 'distance', 'type_cells', 'type_conn', 'prob_f'])
    # This is a utility tuple, just to get probability functions also (when needed)
    #CPair = namedtuple('CPair', ['a', 'b', 'type_code', 'prob_f'])

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
        pass

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
        # Instead of a list of tuples, use a dict. This will make recalling much faster:
        #self.upairs = list()
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
                self.upairs_d[self.to_ucoord(i, j)] \
                    = self.UPair(a=i, b=j, distance=dist, type_cells=get_pair_type(i, j), type_conn=None, prob_f=None)
                ctr += 1
            start += 1

        # Set network intersomatic distance matrix:
        self.dist_mat = distance.squareform(dist_wrapped_vect)

        # Set PC/PV somata positions (Seperate PCs from PVs):
        self.pc_somata = somata[1:self.pc_no+1,:]
        self.pv_somata = somata[self.pc_no+1:,:]

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(somata[:,0], somata[:,1], somata[:,2], c='r', marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.savefig('Network_sn{}_3d_positions.png'.format(self.serial_no))
            plt.show()
        
        return
        

    @with_reproducible_rng
    def create_connections(self, alias='', rearrange_iterations=100, uniform_probability=None,
                           plot=False):
        '''
        Create connectivity and weights of the network, based on the dictionary of connectivity functions and the
        configuration alias.
        :param rearrange_iterations:
        :param uniform_probability:
        :param plot:
        :return:
        '''
        rnd = lambda: np.random.rand(1)[0]

        self.configuration_alias = alias
        # Define connectivity probability functions (distance dependent):
        # Neuronal pairs are unordered {A,B}, considering their type.
        # For each  pair we have three distinct, distance-dependend connection probabilities:
        # 1) reciprocal
        # 2) A -> B
        # 3) B -> A
        # Other/custom connectivity functions can be used as well in the same framework, in order to connect different
        # network configurations i.e. random or just get the connection  probability of all pairs (vectorized = fast).

        # Average connectivity in Otsuka et al., 2009 is 0.25 (inside ~80um)
        # Average connectivity in Packer, Yuste, 2011 is 0.67 (inside ~200um)
        # We combine them, using an average, distance attenuating connectivity of 0.46
        pv_factor = 0.46/(0.154+0.019+0.077)
        # Use numpy array friendly lambdas. It is harder on the eyes, but can save hours (!) of computations:
        connection_functions_d = {}
        # PN with PN connectivity:
        connection_functions_d['PN_PN'] = {}
        connection_functions_d['PN_PN']['total'] \
            = lambda x: np.multiply(np.exp(np.multiply(-0.0052, x)), 0.22)
        connection_functions_d['PN_PN']['reciprocal'] \
            = lambda x: np.multiply(np.exp(np.multiply(-0.0085, x)), 0.12)
        connection_functions_d['PN_PN']['A2B'] \
            = lambda x: np.divide(np.subtract(connection_functions_d['PN_PN']['total'](x), connection_functions_d['PN_PN']['reciprocal'](x)), 2)
        connection_functions_d['PN_PN']['B2A'] \
            = lambda x: connection_functions_d['PN_PN']['A2B'](x)
        connection_functions_d['PN_PN']['unidirectional'] \
            = lambda x: np.add(connection_functions_d['PN_PN']['A2B'](x), connection_functions_d['PN_PN']['B2A'](x))
        # Random/uniform connection functions are refined later on, in order to exactly balance its overall connection
        connection_functions_d['PN_PN']['uniform_reciprocal'] \
            = lambda x: np.multiply(np.ones(x.shape), 0)
        connection_functions_d['PN_PN']['uniform_A2B'] \
            = lambda x: np.multiply(np.ones(x.shape), 0)
        connection_functions_d['PN_PN']['uniform_B2A'] \
            = lambda x: np.multiply(np.ones(x.shape), 0)
        # probability with the structured configuration.
        # PN with PV connectivity:
        connection_functions_d['PN_PV'] = {}
        connection_functions_d['PN_PV']['total'] \
            = lambda x: np.exp(np.divide(x, -180))
        connection_functions_d['PN_PV']['reciprocal'] \
            = lambda x: np.multiply(connection_functions_d['PN_PV']['total'](x), 0.154 * pv_factor)
        connection_functions_d['PN_PV']['A2B'] \
            = lambda x: np.multiply(connection_functions_d['PN_PV']['total'](x), 0.019 * pv_factor)
        connection_functions_d['PN_PV']['B2A'] \
            = lambda x: np.multiply(connection_functions_d['PN_PV']['total'](x), 0.077 * pv_factor)
        # PV with PV connectivity:
        connection_functions_d['PV_PV'] = {}
        connection_functions_d['PV_PV']['reciprocal'] \
            = lambda x: np.multiply(np.ones(x.shape), 0.045)
        connection_functions_d['PV_PV']['A2B'] \
            = lambda x: np.multiply(np.ones(x.shape), 0.0675)
        connection_functions_d['PV_PV']['B2A'] \
            = lambda x: np.multiply(np.ones(x.shape), 0.0675)


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
            new_upairs = [None]*len(upairs)
            prev_i = 0
            query_pair_types = set(x.type_cells for x in upairs)
            for pair_type in query_pair_types:
                matching_upairs = [x for x in upairs if x.type_cells == pair_type]
                upairs_dist = np.asarray([p.distance for p in matching_upairs])
                prob_arr = np.full((len(connection_protocol)+1,len(matching_upairs)), 0.0)
                for i, conn_type in enumerate(connection_protocol.values(), 1):
                    prob_arr[i, :] = connection_functions_d[pair_type][conn_type](upairs_dist)
                prob_arr = np.cumsum(prob_arr, axis=0)
                # Don't forget to include the (closed) last bin!
                prob_arr_flat = np.append(np.add(prob_arr, np.tile(np.arange(len(matching_upairs)),
                                                         (len(connection_protocol)+1, 1))).flatten('F'), len(matching_upairs))
                rnd_arr_flat = np.add(np.random.rand(1, len(matching_upairs)), np.arange(len(matching_upairs)))
                blah = np.histogram(rnd_arr_flat, bins=prob_arr_flat)[0]
                blah2 = np.nonzero(blah)[0]
                blah3 = np.arange(0, prob_arr_flat.size, len(connection_protocol)+1)
                conn_code_arr = np.subtract(blah2, blah3[:-1])
                for i, (upair, conn_code) in enumerate(zip(matching_upairs, conn_code_arr), prev_i):
                    tmp_d = upair._asdict()
                    tmp_d['type_conn'] = connection_protocol.get(conn_code, 'none')
                    if export_probabilities:
                        tmp_d['prob_f'] = np.append(prob_arr[:, i-prev_i], 1)
                    new_upairs[i] = self.UPair(**tmp_d)
                prev_i += len(matching_upairs)
            return new_upairs

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
                elif upair.type_conn == 'total':
                    # Custom connectivity rules, need custom implementation. Lets leave it here for now:
                    # Total probability rule means that we return connection probability, rather than connectivity:
                    pass
                else:
                    raise ValueError('Connectivity rule not implemented for this type of connections!')
            return mat

        # Initially connect only non PN to PN pairs (but due to performance and code readability issues connect the PN
        # to PN pairs also; unordered pairs must lie on a square matrix!):
        connectivity_mat = np.full((self.cell_no, self.cell_no), False)
        # use a dict to instruct what connections you need:
        # This is the distance dependent connection types from Perin et al., 2011:
        # TODO: check that the network configuration will be the same, the second time that this function will run (for
        # the random config). Given that is the same, you can 'bake' the
        based_on_distance = {0: 'reciprocal', 1: 'A2B', 2: 'B2A'}
        # reuse the generated connectivity for PN_PV and PV_PV pairs that never changes again:
        all_upairs = [pair for pair in self.upairs_d.values()]
        connected_upairs = connect_all_pairs(all_upairs, connection_protocol=based_on_distance)
        # Get the connectivity matrix:
        connectivity_mat = upairs2mat(connected_upairs)
        # Update upairs in object. Then generate theconnectivity matrix
        for upair in connected_upairs:
            self.upairs_d[self.to_ucoord(upair.a, upair.b)] = upair

        # Now if configuration is random, update the PN to PN connectivity matrix part and return:
        if self.configuration_alias is 'random':
            # if a uniform connection probability is given, update the default.
            uniform_prob_reciprocal = uniform_probability**2
            uniform_prob_unidirectional = (uniform_probability - (uniform_probability**2))/2
            connection_functions_d['PN_PN']['uniform_reciprocal'] \
                = lambda x: np.multiply(np.ones(x.shape), uniform_prob_reciprocal)
            connection_functions_d['PN_PN']['uniform_A2B'] \
                = lambda x: np.multiply(np.ones(x.shape), uniform_prob_unidirectional)
            connection_functions_d['PN_PN']['uniform_B2A'] \
                = lambda x: np.multiply(np.ones(x.shape), uniform_prob_unidirectional)
            # Utilize only the uniform type of connection:
            uniform = {0: 'uniform_reciprocal', 1: 'uniform_A2B', 2: 'uniform_B2A'}
            pn_upairs = [pair for pair in self.upairs_d.values() if pair.type_cells == ('PN_PN')]
            # Update the connectivity matrix only over PN to PN connections:
            connected_upairs = connect_all_pairs(pn_upairs, connection_protocol=uniform)
            connectivity_mat[:self.pc_no, :self.pc_no] = upairs2mat(connected_upairs)
            # Update upairs in object. Then generate theconnectivity matrix
            for upair in connected_upairs:
                self.upairs_d[self.to_ucoord(upair.a, upair.b)] = upair

        # if configuration is structured,
        if self.configuration_alias is 'structured':
            #TODO: einai ta reciprocals sxedon ta misa?
            # Filter out non PN cell pairs:
            pn_upairs = [pair for pair in self.upairs_d.values() if pair.type_cells == ('PN_PN')]
            # use a dict to instruct what connections you need:
            # This is the distance dependent connection types from Perin et al., 2011:
            based_on_distance = {0: 'reciprocal', 1: 'A2B', 2: 'B2A'}
            # Get the connectivity matrix:
            #connectivity_mat[:self.pc_no, :self.pc_no] = upairs2mat(connect_all_pairs(pn_upairs, connection_protocol=based_on_distance))

            # Plot to see the validated results of the connectivity routines:
            plt.ion()
            plot_reciprocal_across_distance(connectivity_mat[:self.pc_no, :self.pc_no],
                                            self.dist_mat[:self.pc_no, :self.pc_no],
                                            connection_functions_d['PN_PN']['reciprocal'],
                                            plot=True)
            plot_unidirectional_across_distance(connectivity_mat[:self.pc_no, :self.pc_no],
                                            self.dist_mat[:self.pc_no, :self.pc_no],
                                            connection_functions_d['PN_PN']['unidirectional'],
                                                plot=True)
            plot_pn2pv_unidirectional_across_distance(mat_pn_pv=connectivity_mat[:self.pc_no, self.pc_no:],
                                                      mat_pv_pn=connectivity_mat[self.pc_no:, :self.pc_no],
                                                      dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                      ground_truth=connection_functions_d['PN_PV']['A2B'],
                                                      plot=True)
            plot_pv2pn_unidirectional_across_distance(mat_pn_pv=connectivity_mat[:self.pc_no, self.pc_no:],
                                                      mat_pv_pn=connectivity_mat[self.pc_no:, :self.pc_no],
                                                      dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                      ground_truth=connection_functions_d['PN_PV']['B2A'],
                                                      plot=True)
            plot_pn_pv_reciprocal_across_distance(connectivity_mat[:self.pc_no, self.pc_no:],
                                                      connectivity_mat[self.pc_no:, :self.pc_no],
                                                self.dist_mat[:self.pc_no, self.pc_no:],
                                                connection_functions_d['PN_PV']['reciprocal'])


            # Connect pairs based on total PN to PN probability:
            # use a dict to instruct what connections you need:
            # This is the overall (total) connection probability, also distance dependend (perin et al., 2011):
            overall_probability = {0: 'total'}
            # Get probabilities instead of the connectivity matrix:
            Pd_upairs = connect_all_pairs(pn_upairs, connection_protocol=overall_probability, export_probabilities=True)
            # Get type '0' connection probability for each pair (this is the total/overall probability):
            prob_f_list = [upair.prob_f[1] for upair in Pd_upairs]
            Pd = np.asmatrix(distance.squareform(prob_f_list))


            # run iterative rearrangement:
            f_prob = np.zeros((self.pc_no, self.pc_no, rearrange_iterations))
            cn_prob = np.zeros((self.pc_no, self.pc_no, rearrange_iterations))
            cc = np.zeros((1,rearrange_iterations))

            reformed_conn_mat = connectivity_mat[:self.pc_no, :self.pc_no]
            sumPd = Pd.sum()
            for t in range(1, rearrange_iterations):
                print('@iteration {}'.format(t))
                # TODO check clust_coeff function. Maby separate module?
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
                distance_estimate_vect = distance.squareform(distance_estimate)
                # squareform works row-wise, also the Pd upairs, wo we can zip() them:
                dist_estimate_upairs = [None]* len(Pd_upairs)
                for i, (upair, d) in enumerate(zip(Pd_upairs, distance_estimate_vect)):
                    tmp_d = upair._asdict()
                    tmp_d['distance'] = d
                    dist_estimate_upairs[i] = self.UPair(**tmp_d)

                # Now you can use this 'distance' estimate to reallocate the connections to unidirectional and reciprocal:
                reformed_conn_mat_next = upairs2mat(connect_all_pairs(dist_estimate_upairs, connection_protocol=based_on_distance))
                reformed_conn_mat = reformed_conn_mat_next

            # Save connectivity to the network:
            connectivity_mat[:self.pc_no, :self.pc_no] = reformed_conn_mat

            # Plot
            if plot:
                fig, ax = plt.subplots()
                cax = ax.plot(cc[0])
                ax.set_title('Clustering Coefficient')
                plt.xlabel('Iterations')
                plt.ylabel('Clustering Coefficient')
                plt.savefig('Network_sn{}_clustering_coefficient.png'.format(self.serial_no))
                plt.show()

        # Plot if you want:
        if plot:
            fig, ax = plt.subplots()
            cax = ax.imshow(connectivity_mat, interpolation='nearest', cmap=cm.afmhot)
            ax.set_title('Network_sn{}_connectivity matrix of {}'.format(self.serial_no, self.configuration_alias))
            plt.savefig('Network_sn{}_connectivity_matrix_{}.png'.format(self.serial_no, self.configuration_alias))
            plt.show()

        self.connectivity_mat = connectivity_mat
        return

    @with_reproducible_rng
    def initialize_trials(self, trial_no=0, stimulated_pn_no=50):
        # initialize stimulated cells in each trials:
        self.trial_no = trial_no
        self.stimulated_cells = np.full((self.trial_no, stimulated_pn_no), 0, dtype=int)
        for trial in range(self.trial_no):
            self.stimulated_cells[trial][:] = np.sort(np.random.permutation(np.arange(self.pc_no))[:stimulated_pn_no])

        ## Since round returns in [0,1), we are ok
        #self.stimulated_cells = np.asarray(
        #    np.round(np.multiply(np.random.rand(self.trial_no, stimulated_pn_no), self.pc_no)), dtype=int)
        ## Sort stimulated cells' list:
        #self.stimulated_cells = np.sort(self.stimulated_cells, axis=1)


    def to_ucoord(self, a, b):
        '''
        Returns the coordinates for a undirected pair. Only upper part of connectivity matrix.
        :param a:
        :param b:
        :return:
        '''
        if a > b:
            return (b, a)
        else:
            return (a, b)

    def common_neighbors(self, adjmat):
        adj = np.logical_or(adjmat, adjmat.getT())
        adj = adj.astype(float) * adj.getT().astype(float)
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
        unidirectional = np.logical_xor(connectivity_mat, reciprocal)
        # Get unordered pairs that are connected (not care about directionality):
        connected_up = np.logical_and(connected, np.asmatrix(np.triu(np.ones((N, N)), 1), dtype=bool))
        # The average connectivity is measured as the number of connected unordered pairs, out of total unordered pairs.
        self.stats['averageConnectivity'] = np.sum(connected_up) / self.stats['nUniqueUnorderedPairs']

        # UNVALIDATED VALUES TODO: validate values
        # overall connection probability
        self.stats['connectedUnorderedPairsPercentage'] = np.sum(connected) / self.stats['nUniqueUnorderedPairs']
        self.stats['unidirectionalUnorderedPairsPercentage'] = np.sum(unidirectional) / self.stats['nUniqueUnorderedPairs']
        self.stats['bidirectionalUnorderedPairsPercentage'] = (np.sum(reciprocal)/2) / self.stats['nUniqueUnorderedPairs']

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
        # TODO: check the Pandas lib!
        # Get attributes' names:
        #net_attrs = [a for a in dir(self) if not a.startswith('_') and not callable(getattr(self, a, None))]
        with h5py.File(os.path.join(export_path,
                            '{}Network_{}{}.hdf5'.format(filename_prefix, self.serial_no, filename_postfix)), 'w') as f:
            subgroup = f.create_group('configurations')
            subgroup.attrs['serial_no'] = self.serial_no
            subgroup.attrs['pc_no'] = self.pc_no
            subgroup.attrs['pv_no'] = self.pv_no
            subgroup['structured'] = self.configurations['structured']
            subgroup['dist_mat'] = self.dist_mat
            subgroup['stimulated_cells'] = self.stimulated_cells
            subgroup['pc_somata'] = self.pc_somata
            subgroup['pv_somata'] = self.pv_somata
            pass

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

def plot_reciprocal_across_distance(mat, distance_mat, ground_truth=None, plot=False):
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

    histo_bins = np.arange(0, 600, 10)
    histo_dist, bin_edges = np.histogram(dist_vector, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_vector[np.asarray(conn_vector)[0]], bins=histo_bins)

    # Plot
    if plot:
        fig, ax = plt.subplots()
        cax = ax.plot(histo_bins[:-1], np.divide(histo, histo_dist))
        if ground_truth:
            cax = ax.plot(histo_bins, ground_truth(histo_bins))
        ax.set_title('configuration str')
        plt.xlabel('Distance (um)')
        plt.ylabel('Reciprocal Probability')
        plt.savefig('plot_reciprocal_across_distance.png')
        plt.show()

def plot_unidirectional_across_distance(mat, distance_mat, ground_truth=None, plot=False):
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

    histo_bins = np.arange(0, 600, 10)
    histo_dist, bin_edges = np.histogram(dist_vector, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_vector[np.asarray(conn_vector)[0]], bins=histo_bins)

    # Plot
    if plot:
        fig, ax = plt.subplots()
        cax = ax.plot(histo_bins[:-1], np.divide(histo, histo_dist))
        if ground_truth:
            cax = ax.plot(histo_bins, ground_truth(histo_bins))
        ax.set_title('configuration str')
        plt.xlabel('Distance (um)')
        plt.ylabel('Unidirectional Probability')
        plt.savefig('plot_unidirectional_across_distance.png')
        plt.show()

def plot_pn_pv_reciprocal_across_distance(mat_pn_pv, mat_pv_pn, dist_mat, ground_truth=None):
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

    histo_bins = np.arange(0, 600, 10)
    histo_dist, bin_edges = np.histogram(dist_mat, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_mat[mat_unid], bins=histo_bins)
    # Plot
    fig, ax = plt.subplots()
    cax = ax.plot(histo_bins[:-1], np.divide(histo, histo_dist))
    if ground_truth:
        cax = ax.plot(histo_bins, ground_truth(histo_bins))
    ax.set_title('configuration str')
    plt.xlabel('Distance (um)')
    plt.ylabel('PN2PV Reciprocal Probability')
    plt.show()
def plot_pn2pv_unidirectional_across_distance(mat_pn_pv, mat_pv_pn, dist_mat, ground_truth=None, plot=False):
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

    histo_bins = np.arange(0, 600, 10)
    histo_dist, bin_edges = np.histogram(dist_mat, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_mat[mat_unid], bins=histo_bins)

    # Plot
    if plot:
        fig, ax = plt.subplots()
        cax = ax.plot(histo_bins[:-1], np.divide(histo, histo_dist))
        if ground_truth:
            cax = ax.plot(histo_bins, ground_truth(histo_bins))
        ax.set_title('configuration str')
        plt.xlabel('Distance (um)')
        plt.ylabel('PN2PV Unidirectional Probability')
        plt.savefig('plot_pn2pv_unidirectional_across_distance.png')
        plt.show()

def plot_pv2pn_unidirectional_across_distance(mat_pn_pv, mat_pv_pn, dist_mat, ground_truth=None, plot=False):
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

    histo_bins = np.arange(0, 600, 10)
    histo_dist, bin_edges = np.histogram(dist_mat, bins=histo_bins)
    histo, bin_edges = np.histogram(dist_mat[mat_unid], bins=histo_bins)

    # Plot
    if plot:
        fig, ax = plt.subplots()
        cax = ax.plot(histo_bins[:-1], np.divide(histo, histo_dist))
        if ground_truth:
            cax = ax.plot(histo_bins, ground_truth(histo_bins))
        ax.set_title('configuration str')
        plt.xlabel('Distance (um)')
        plt.ylabel('PV2PN Unidirectional Probability')
        plt.savefig('plot_pv2pn_unidirectional_across_distance.png')
        plt.show()
