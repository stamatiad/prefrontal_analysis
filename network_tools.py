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
    This is a function wrapper that calls rng.seed before everything else. Therefore user is expected to get the exact
    same results every time (given that no extra rand() calls are needed).
    As seed the network serial number is used.
    A function wrapper is used to dissociate different function calls: e.g. creating stimulus will be the same, even if
    user changed the number of rand() was called in a previous class function, resulting in the same stimulated cells.
    :param func:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        print(class_method.__name__ + " was called")
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
    def configuration_str(self):
        return self._configuration_str

    @configuration_str.setter
    def configuration_str(self, mat):
        if mat.size == 0:
            raise ValueError("Connectivity matrix is empty!")
        self._configuration_str = mat
    
    @property
    def configuration_rnd(self):
        return self._configuration_rnd
    
    @configuration_rnd.setter
    def configuration_rnd(self, mat):
        if mat.size == 0:
            raise ValueError("Connectivity matrix is empty!")
        self._configuration_rnd = mat

    @property
    def weights_str(self):
        return self._weights_str

    @weights_str.setter
    def weights_str(self, mat):
        if mat.size == 0:
            raise ValueError("Weights matrix is empty!")
        self._weights_str = mat

    @property
    def weights_rnd(self):
        return self._weights_rnd
    
    @weights_rnd.setter
    def weights_rnd(self, mat):
        if mat.size == 0:
            raise ValueError("Weights matrix is empty!")
        self._weights_rnd = mat
    
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

    UPair = namedtuple('UPair', ['a', 'b', 'type', 'distance'])
    CPair = namedtuple('CPair', ['a', 'b', 'type_code', 'prob_f'])

    @with_reproducible_rng
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
        self.configuration_str = np.full((self.cell_no, self.cell_no), False)
        self.configuration_rnd = np.full((self.cell_no, self.cell_no), False)
        self.weights_str = np.full((self.cell_no, self.cell_no), 0.0)
        self.weights_rnd = np.full((self.cell_no, self.cell_no), 0.0)
        pass

    @with_reproducible_rng
    def populate_network(self, cube_side_len: int, plot=False):
        '''
        Disperses neurons uniformly/randomly in a cube of given side
        @param: cell_no number of network cells
        @param: cube_side_len cube side length in um
        '''
        np.random.seed(self.serial_no)
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
            #tmp = ['{}_{}'.format(a_type, b_type), '{}_{}'.format(b_type, a_type)]
            return '{}_{}'.format(a[0], a[1])

        # Disperse cell_no points in a cube of cube_side_len side length:
        somata = np.multiply(np.random.rand(self.cell_no, 3), cube_side_len)

        # Create alternative/wrapped network configurations:
        X, Y, Z = np.meshgrid(np.array([1,2,3]), np.array([1,2,3]), np.array([1,2,3]) )
        cb_offest_pos = np.transpose(np.array([X.ravel(), Y.ravel(), Z.ravel()]))

        cb_offset = np.array([-cube_side_len, 0, cube_side_len])
        cb_offset_mat = cb_offset[np.subtract(cb_offest_pos,1)]

        dist_wrapped_vect = np.zeros(int(self.cell_no*(self.cell_no-1)/2))
        self.upairs = list()
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
                self.upairs.append(self.UPair(a=i, b=j, type=get_pair_type(i, j), distance=dist))
                ctr += 1
            start += 1

        # Set network intersomatic distance matrix:
        self.dist_mat = distance.squareform(dist_wrapped_vect)

        # Set unordered neuronal pairs:


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
            plt.show()
        
        return
        

    @with_reproducible_rng
    def create_connections(self):
        '''Connects neuronal pairs with given probability functions, based on their distance
            On PC2PC pairs also applies the common neighbor rule.
        '''
        np.random.seed(self.serial_no)
        #rnd_draws = np.random.rand(self.cell_no, self.cell_no)
        rnd = lambda: np.random.rand(1)[0]

        # Define connectivity probability functions (distance dependent):
        # Neuronal pairs are unordered {A,B}, considering their type. For each  pair we have three distinct, distance dependend connection probability
        # functions: reciprocal, A->B, B->A and of course the overall connection probability, total (their sum).
        pv_factor = 1/(0.154+0.019+0.077)
        # Use array friendly lambdas. It is harder on the eyes, but can save hours (!) of computations:
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



        # To make connections run a map function.
        def connect_pair(upair, connection_types=None):
            '''
            Connect (or not) the given pair, based on connections probabilities.
            Second (optional) argument filters out only specific type of connections.
            :param i:
            :param j:
            :param dist:
            :return:
            '''

            # Fetch from dictionary of connection functions the ones for that pair:
            conn_cum_sum = [0]
            for type, probability_function in connection_functions_d.items():
                if type.startswith(upair.type) and any(x in type for x in connection_types.values()):
                    # run the function and get the connection probability given the pair distance:
                    conn_cum_sum.append(probability_function(upair.distance))

            cumulative_probabilities = np.append(np.cumsum(np.asarray(conn_cum_sum)), [1])

            # Sample from the functions:
            myrnd = np.random.rand(1)
            tmp = np.nonzero(np.histogram(myrnd, bins=cumulative_probabilities)[0])[0][0]
            # the default connection is none (if non existent in the connection_types dictionary):
            conn_type = connection_types.get(tmp, 'none')
            return self.CPair(a=upair.a, b=upair.b, type_code=conn_type, prob_f=cumulative_probabilities)

        def connect_all_pairs(upairs, connection_types=None, export_probabilities=False):
            '''
            Connect (or not) the given pair, based on connections probabilities.
            Second (optional) argument filters out only specific type of connections.
            :param i:
            :param j:
            :param dist:
            :return:
            '''
            # Unfortunatelly python's function call overhead is huge, so we need to become more array friendly:
            # Unfortunatelly this affects the readability of the code:
            cpairs = [None]*len(upairs)
            prev_i = 0
            tic = time.perf_counter()
            query_pair_types = set(x.type for x in upairs)
            for pair_type in query_pair_types:
                matching_upairs = [x for x in upairs if x.type == pair_type]
                #print('Connecting: {}. Found {} pairs'.format(pair_type, len(matching_upairs)))
                upairs_dist = np.asarray([p.distance for p in matching_upairs])
                prob_arr = np.full((len(connection_types)+1,len(matching_upairs)), 0.0)
                for i, conn_type in enumerate(connection_types.values(), 1):
                    prob_arr[i, :] = connection_functions_d[pair_type][conn_type](upairs_dist)
                prob_arr = np.cumsum(prob_arr, axis=0)
                # prob_arr[len(connection_types)+1,:] = 1
                # Don't forget to include the (closed) last bin!
                prob_arr_flat = np.append(np.add(prob_arr, np.tile(np.arange(len(matching_upairs)),
                                                         (len(connection_types)+1, 1))).flatten('F'), len(matching_upairs))
                rnd_arr_flat = np.add(np.random.rand(1, len(matching_upairs)), np.arange(len(matching_upairs)))
                blah = np.histogram(rnd_arr_flat, bins=prob_arr_flat)[0]
                blah2 = np.nonzero(blah)[0]
                blah3 = np.arange(0, prob_arr_flat.size, len(connection_types)+1)
                conn_code_arr = np.subtract(blah2, blah3[:-1])
                #conn_type_arr = [connection_types.get(x, 'none') for x in conn_code_arr]
                for i, (upair, conn_code) in enumerate(zip(matching_upairs, conn_code_arr), prev_i):
                    if export_probabilities:
                        cpairs[i] = self.CPair(a=upair.a, b=upair.b, type_code=connection_types.get(conn_code, 'none'),
                                               prob_f=np.append(prob_arr[:, i-prev_i], 1))
                    else:
                        cpairs[i] = self.CPair(a=upair.a, b=upair.b, type_code=connection_types.get(conn_code, 'none'),
                                                 prob_f=0)
                #print('For pair type: {}, populated from {} to {}'.format(pair_type, prev_i, len(matching_upairs)))
                prev_i += len(matching_upairs)
            return cpairs

        def cpairs2mat(cpair_list):
            '''
            Creates a connectivity matrix from upairs
            :param upair_list:
            :return:
            '''
            # check first that the length is proper for a condenced matrix:
            s = len(cpair_list)
            # Grab the closest value to the square root of the number
            # of elements times 2 to see if the number of elements
            # is indeed a binomial coefficient.
            d = int(np.ceil(np.sqrt(s * 2)))

            # Check that v is of valid dimensions.
            if d * (d - 1) != s * 2:
                raise ValueError('Incompatible vector size. It must be a binomial '
                                 'coefficient n choose 2 for some integer n >= 2.')

            # Allocate memory for the distance matrix.
            mat = np.asmatrix(np.full((d, d), False))
            for cpair in cpair_list:
                if cpair.type_code == 'reciprocal':
                    mat[cpair.a, cpair.b] = True
                    mat[cpair.b, cpair.a] = True
                elif cpair.type_code == 'A2B':
                    mat[cpair.a, cpair.b] = True
                elif cpair.type_code == 'B2A':
                    mat[cpair.b, cpair.a] = True
                elif cpair.type_code == 'none':
                    pass # this needs some ironing.
                elif cpair.type_code == 'total':
                    # Custom connectivity rules, need custom implementation. Lets leave it here for now:
                    # Total probability rule means that we return connection probability, rather than connectivity:
                    pass
                else:
                    raise ValueError('Connectivity rule not implemented for this type of connections!')
            return mat


        # use a dict to instruct what connections you need:
        # This is the distance dependent connection types from Perin et al., 2011:
        connection_type_distance = {}
        connection_type_distance[0] = 'reciprocal'
        connection_type_distance[1] = 'A2B'
        connection_type_distance[2] = 'B2A'
        # Get the connectivity matrix:
        #self.configuration_str = cpairs2mat([connect_pair(upair, connection_type_distance) for upair in self.upairs])
        # Connect more efficiently:
        self.configuration_str = cpairs2mat(connect_all_pairs(self.upairs, connection_type_distance))

        # Plot to see the validated results of the connectivity routines:
        plt.ion()
        plot_reciprocal_across_distance(self.configuration_str[:self.pc_no, :self.pc_no],
                                        self.dist_mat[:self.pc_no, :self.pc_no],
                                        connection_functions_d['PN_PN']['reciprocal'],
                                        plot=True)
        plot_unidirectional_across_distance(self.configuration_str[:self.pc_no, :self.pc_no],
                                        self.dist_mat[:self.pc_no, :self.pc_no],
                                        connection_functions_d['PN_PN']['unidirectional'],
                                            plot=True)
        plot_pn2pv_unidirectional_across_distance(mat_pn_pv=self.configuration_str[:self.pc_no, self.pc_no:],
                                                  mat_pv_pn=self.configuration_str[self.pc_no:, :self.pc_no],
                                                  dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                  ground_truth=connection_functions_d['PN_PV']['A2B'],
                                                  plot=True)
        plot_pv2pn_unidirectional_across_distance(mat_pn_pv=self.configuration_str[:self.pc_no, self.pc_no:],
                                                  mat_pv_pn=self.configuration_str[self.pc_no:, :self.pc_no],
                                                  dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                  ground_truth=connection_functions_d['PN_PV']['B2A'],
                                                  plot=True)
        plot_pn_pv_reciprocal_across_distance(self.configuration_str[:self.pc_no, self.pc_no:],
                                                  self.configuration_str[self.pc_no:, :self.pc_no],
                                            self.dist_mat[:self.pc_no, self.pc_no:],
                                            connection_functions_d['PN_PV']['reciprocal'])
        
        # Plot if you want:
        fig, ax = plt.subplots()
        cax = ax.imshow(self.configuration_str, interpolation='nearest', cmap=cm.afmhot)
        ax.set_title('configuration str')
        plt.show()

        # Connect pairs based on total PN to PN probability:
        # Filter out non PN cell pairs:
        pn_upairs = [x for x in self.upairs if x.type.startswith('PN_PN')]
        # use a dict to instruct what connections you need:
        # This is the overall (total) connection probability, also distance dependend (perin et al., 2011):
        connection_type_total = {}
        connection_type_total[0] = 'total'
        # Get probabilities instead of the connectivity matrix:
        #Pd_pairs = [connect_pair(upair, connection_type_total) for upair in pn_upairs]
        Pd_pairs = connect_all_pairs(pn_upairs, connection_type_total, export_probabilities=True)
        prob_f_list = [cpair.prob_f[1] for cpair in Pd_pairs]
        Pd = np.asmatrix(distance.squareform(prob_f_list))

        E = cpairs2mat(connect_all_pairs(pn_upairs, connection_type_distance))

        # run iterative rearrangement:
        iters = 500
        f_prob = np.zeros((self.pc_no, self.pc_no, iters))
        cn_prob = np.zeros((self.pc_no, self.pc_no, iters))
        cc = np.zeros((1,iters))

        sumPd = Pd.sum()
        for t in range(1, iters):
            print(t)
            # giati to C bgainei olo mhden??
            pass
            _, cc[0,t], _ = clust_coeff(E)
            # compute common neighbors for each pair:
            ncn = self.common_neighbors(E)
            cn_prob = ncn / ncn.max()
            # Scale the probabilities of connection by the common neighbor bias:
            f_prob = np.multiply(cn_prob, Pd)
            f_prob = f_prob * (sumPd / f_prob.sum())
            # To allocate connections based on their relative type (unidirectional, reciprocals), compute a distance
            # estimate and use it with the existing probability functions, to get the connectivity:
            x = (np.log(0.22) - np.log(f_prob)) / 0.0052
            x[x < 0] = 0
            x = self.zero_diagonal(x)
            # Pass the distance estimates to upairs named tuple:
            dist_estimate_upairs = list()
            start = 1
            for i in range(self.pc_no):
                for j in range(start, self.pc_no):
                    dist_estimate_upairs.append(self.UPair(a=i, b=j, type='PN_PN', distance=x[i, j]))
                start += 1

            # Now you can use this 'distance' to reallocate the connections to unidirectional and reciprocal:
            nextE = cpairs2mat(connect_all_pairs(dist_estimate_upairs, connection_type_distance))
            E = nextE

        # Save connectivity to the network:
        self.configuration_str[:self.pc_no, :self.pc_no] = E

        # Plot
        fig, ax = plt.subplots()
        cax = ax.plot(cc[0])
        ax.set_title('Clustering Coefficient')
        plt.xlabel('Iterations')
        plt.ylabel('Clustering Coefficient')
        plt.savefig('clustering_coefficient.png')
        plt.show()

        print("A OK!")

    @with_reproducible_rng
    def initialize_trials(self, trial_no=0):
        np.random.seed(self.serial_no)
        # initialize stimulated cells in each trials:
        # Since round returns in [0,1), we are ok
        self.stimulated_cells = np.round(np.multiply(np.random.rand(1,self.trials_no), self.pc_no))




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
    def make_weights(self):
        np.random.seed(self.serial_no)
        # Create weight matrices:
        # I create weight distributions based on the fact that (Perin) the greater
        # the number of common neighbors, the more shifted the probability of the
        # post synaptic depolarization measured. This is a gross estimate, there
        # should be minor changes that might affect the results (e.g. reciprocal
        # connections have almost symmetric weights, neurons with greater out
        # degree distributions have smaller projecting weights (refs?)).

        # TODO:
        # Generate weights matrices:
        # Kayaguchi : more potent synapses in reciprocal pairs
        # Perin: more potent synapses with more clustering coeff:
        # We try to combine them

        # To manipulate the network better: distribute the weights based on number
        # of common neighbors, but change that distribution.

        # Extrapolating Perin's findings, more CN, more shift in distribution:
        # snormal = @(x,m,a,sigma) ((1+erf((a.*x)/sqrt(2)))/2) .* normpdf(x,m,sigma);
        # Y = lognpdf(X,mu,sigma)
        pass

    def export_network_parameters(self, configuration='', export_path=os.getcwd(), postfix=''):
        # Export Network Connectivity:
        # Export STRUCTURED parameter matrices in .hoc file:
        with open(os.path.join(export_path,
                'importNetworkParameters{}_SN{}_{}.hoc'.format(conf, self.serial_no, postfix)), 'w') as f:
            f.write('// This HOC file was generated with MATLAB\n')
            f.write('nPCcells={}\n'.format(self.pc_no))
            f.write('nPVcells={}\n'.format(self.pv_no))
            f.write('nAllCells={}\n'.format(self.cell_no))
            f.write('// Object decleration:\n')
            f.write('objref C, W\n')
            f.write('C = new Matrix(nAllCells, nAllCells)\n')
            f.write('W = new Matrix(nPCcells, nPCcells)\n')
            f.write('\n// Import parameters: (long-long text following!)\n')
            # network connectivity:
            pairs = [(i,j) for i in range(self.cell_no) for j in range(self.cell_no)]
            for (i,j) in pairs:
                f.write('C.x[{}][{}]={}\n'.format(i, j, self.configuration_str[i][j]))
            for (i,j) in pairs:
                f.write('W.x[{}][{}]={}\n'.format(i, j, self.weights_str[i][j]))
            f.write('//EOF\n')

    def export_stimulation_parameters(self, export_path=os.getcwd(), postfix=''):
        # Export stimulation parameters in .hoc file:
        with open(os.path.join(export_path,
                               'importStimulationParameters_SN{}_{}.hoc'.format(self.serial_no, postfix)), 'w') as f:
            f.write('// This HOC file was generated with MATLAB\n\n')
            f.write('// Object decleration:\n')
            f.write('objref PcellStimListSTR[{}]\n'.format(self.trials_no))
            f.write('objref PcellStimListRND[{}]\n'.format(self.trials_no))
            f.write('\n\n// Import parameters:\n\n')
            # Structured network stimulation:
            for trial in self.trials_no:
                f.write('PcellStimListSTR[{}]=new Vector({})\n'.format(trial, self.stimulated_cells.shape[0]))
                for i in range(self.stimulated_cells.shape[0]):
                    f.write('PcellStimListSTR[{}].x[{}]={}\n'.format(trial, i, self.stimulated_cells[trial][i]))
            # Random network stimulation:
            for trial in self.trials_no:
                f.write('PcellStimListRND[{}]=new Vector({})\n'.format(trial, self.stimulated_cells.shape[0]))
                for i in range(self.stimulated_cells.shape[0]):
                    f.write('PcellStimListRND[{}].x[{}]={}\n'.format(trial, i, self.stimulated_cells[trial][i]))
            f.write('//EOF\n')



    def save_data(self, export_path=os.getcwd(), filename_prefix='', filename_postfix=''):
        '''
        Store Network data to a HDF5 file
        :return:
        '''
        with h5py.File(os.path.join(export_path,
                            '{}Network_{}{}.hdf5'.format(filename_prefix, self.serial_no, filename_postfix)), 'w') as f:
            subgroup = f.create_group('configurations')
            subgroup.attrs['serial_no'] = self.serial_no
            subgroup.attrs['pc_no'] = self.pc_no
            subgroup.attrs['pv_no'] = self.pv_no
            subgroup['structured'] = self.configuration_str
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
