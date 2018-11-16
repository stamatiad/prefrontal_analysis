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
        
        
    def create_connections(self):
        '''Connects neuronal pairs with given probability functions, based on their distance
            On PC2PC pairs also applies the common neighbor rule.
        '''
        #rnd_draws = np.random.rand(self.cell_no, self.cell_no)
        rnd = lambda: np.random.rand(1)[0]

        # Define connectivity probability functions (distance dependent):
        # Neuronal pairs are unordered {A,B}, considering their type. For each  pair we have three distinct, distance dependend connection probability
        # functions: reciprocal, A->B, B->A and of course the overall connection probability, total (their sum).
        pv_factor = 1/(0.154+0.019+0.077)
        # Use ordered dictionary, because the key names are just for the reader (EDIT does'nt matter any more):
        connection_functions_d = OrderedDict()
        # PN with PN connectivity:
        connection_functions_d['PN_PN_total'] = lambda x: np.exp(-0.0052*x) * 0.22
        connection_functions_d['PN_PN_reciprocal'] = lambda x: np.exp(-0.0085*x) * 0.12
        connection_functions_d['PN_PN_A2B'] = lambda x: (connection_functions_d['PN_PN_total'](x) - connection_functions_d['PN_PN_reciprocal'](x))/2
        connection_functions_d['PN_PN_B2A'] = lambda x: connection_functions_d['PN_PN_A2B'](x)
        connection_functions_d['PN_PN_unidirectional'] = lambda x: connection_functions_d['PN_PN_A2B'](x) + connection_functions_d['PN_PN_B2A'](x)
        # PN with PV connectivity:
        connection_functions_d['PN_PV_total'] = lambda x: np.exp(x/-180)
        connection_functions_d['PN_PV_reciprocal'] = lambda x: connection_functions_d['PN_PV_total'](x) * 0.154 * pv_factor
        connection_functions_d['PN_PV_A2B'] = lambda x: connection_functions_d['PN_PV_total'](x) * 0.019 * pv_factor
        connection_functions_d['PN_PV_B2A'] = lambda x: connection_functions_d['PN_PV_total'](x) * 0.077 * pv_factor
        # PV with PV connectivity:
        connection_functions_d['PV_PV_reciprocal'] = lambda x: 0.045
        connection_functions_d['PV_PV_A2B'] = lambda x: 0.0675
        connection_functions_d['PV_PV_B2A'] = lambda x: 0.0675



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
                #if any(x in type for x in upair.type) and 'total' not in type:
                #if any(x in type for x in upair.type) and any(x in type for x in connection_types.values()):
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

        def connect_all_pairs(upairs, connection_types=None):
            '''
            Connect (or not) the given pair, based on connections probabilities.
            Second (optional) argument filters out only specific type of connections.
            :param i:
            :param j:
            :param dist:
            :return:
            '''
            cpairs = list()
            for upair in upairs:
                # Fetch from dictionary of connection functions the ones for that pair:
                conn_cum_sum = [0]
                # Since you assume that both the given pair type and connection type are a subset of connection types
                # dictionary, hashtable search them.
                for type, probability_function in connection_functions_d.items():
                    if type.startswith(upair.type) and any(x in type for x in connection_types.values()):
                        # run the function and get the connection probability given the pair distance:
                        conn_cum_sum.append(probability_function(upair.distance))

                #cumulative_probabilities = np.append(np.cumsum(np.asarray(conn_cum_sum)), [1])
                ## Sample from the functions:
                #conn_code = np.nonzero(np.histogram(np.random.rand(1), bins=cumulative_probabilities)[0])[0][0]
                ## the default connection is none (if non existent in the connection_types dictionary):
                #conn_type = connection_types.get(conn_code, 'none')
                #cpairs.append(self.CPair(a=upair.a, b=upair.b, type_code=conn_type, prob_f=cumulative_probabilities))
            return cpairs

        def cpairs2mat(cpair_list):
            '''
            Creates a connectivity matrix from upairs
            :param upair_list:
            :return:
            '''
            tic = time.perf_counter()
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
            toc = time.perf_counter()
            print('cpair time {}'.format(toc-tic))
            return mat


        # use a dict to instruct what connections you need:
        # This is the distance dependent connection types from Perin et al., 2011:
        connection_type_distance = {}
        connection_type_distance[0] = 'reciprocal'
        connection_type_distance[1] = 'A2B'
        connection_type_distance[2] = 'B2A'
        # Get the connectivity matrix:
        tic = time.perf_counter()
        self.configuration_str = cpairs2mat([connect_pair(upair, connection_type_distance) for upair in self.upairs])
        toc = time.perf_counter()
        print('OLD connection time {}'.format(toc-tic))
        # Connect more efficiently:
        tic = time.perf_counter()
        connect_all_pairs(self.upairs, connection_type_distance)
        toc = time.perf_counter()
        print('NEW connection time {}'.format(toc-tic))

        # Plot to see the validated results of the connectivity routines:
        plt.ion()
        plot_reciprocal_across_distance(self.configuration_str[:self.pc_no, :self.pc_no],
                                        self.dist_mat[:self.pc_no, :self.pc_no],
                                        connection_functions_d['PN_PN_reciprocal'],
                                        plot=True)
        plot_unidirectional_across_distance(self.configuration_str[:self.pc_no, :self.pc_no],
                                        self.dist_mat[:self.pc_no, :self.pc_no],
                                        connection_functions_d['PN_PN_unidirectional'],
                                            plot=True)
        plot_pn2pv_unidirectional_across_distance(mat_pn_pv=self.configuration_str[:self.pc_no, self.pc_no:],
                                                  mat_pv_pn=self.configuration_str[self.pc_no:, :self.pc_no],
                                                  dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                  ground_truth=connection_functions_d['PN_PV_A2B'],
                                                  plot=True)
        plot_pv2pn_unidirectional_across_distance(mat_pn_pv=self.configuration_str[:self.pc_no, self.pc_no:],
                                                  mat_pv_pn=self.configuration_str[self.pc_no:, :self.pc_no],
                                                  dist_mat=self.dist_mat[:self.pc_no, self.pc_no:],
                                                  ground_truth=connection_functions_d['PN_PV_B2A'],
                                                  plot=True)
        plot_pn_pv_reciprocal_across_distance(self.configuration_str[:self.pc_no, self.pc_no:],
                                                  self.configuration_str[self.pc_no:, :self.pc_no],
                                            self.dist_mat[:self.pc_no, self.pc_no:],
                                            connection_functions_d['PN_PV_reciprocal'])

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
        Pd_pairs = [connect_pair(upair, connection_type_total) for upair in pn_upairs]
        prob_f_list = [cpair.prob_f[1] for cpair in Pd_pairs]
        Pd = np.asmatrix(distance.squareform(prob_f_list))

        E = cpairs2mat([connect_pair(upair, connection_type_distance) for upair in pn_upairs])

        # run iterative rearrangement:
        iters = 1
        f_prob = np.zeros((self.pc_no, self.pc_no, iters))
        cn_prob = np.zeros((self.pc_no, self.pc_no, iters))
        cc = np.zeros((1,iters))

        sumPd = Pd.sum()
        for t in range(1, iters):
            print(t)
            # giati to C bgainei olo mhden??
            pass
            tic = time.perf_counter()
            _, cc[0,t], _ = clust_coeff(E)
            toc = time.perf_counter()
            print('clustcoef time {}'.format(toc-tic))
            # compute common neighbors for each pair:
            tic = time.perf_counter()
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
            toc = time.perf_counter()
            print('matrix time {}'.format(toc-tic))
            # Pass the distance estimates to upairs named tuple:
            tic = time.perf_counter()
            dist_estimate_upairs = list()
            start = 1
            for i in range(self.pc_no):
                for j in range(start, self.pc_no):
                    dist_estimate_upairs.append(self.UPair(a=i, b=j, type='PN_PN', distance=x[i, j]))
                start += 1
            toc = time.perf_counter()
            print('pair for loop time {}'.format(toc-tic))

            # Now you can use this 'distance' to reallocate the connections to unidirectional and reciprocal:
            tic = time.perf_counter()
            nextE = cpairs2mat([connect_pair(upair, connection_type_distance) for upair in dist_estimate_upairs])
            toc = time.perf_counter()
            print('map time {}'.format(toc-tic))
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

    def make_weights(self):
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

    def save_data(self, filename_prefix='', filename_postfix=''):
        '''
        Store Network data to a HDF5 file
        :return:
        '''
        with h5py.File('{}Network_{}{}.hdf5'.format(filename_prefix, self.serial_no, filename_postfix), 'w') as f:
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
