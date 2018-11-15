# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:07:01 2018

@author: stefanos
"""
import warnings
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict, defaultdict, namedtuple
import h5py

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
        rnd = lambda : np.random.rand(1)[0]

        # Define connectivity probability functions (distance dependent):
        # Neuronal pairs are unordered {A,B}, considering their type. For each  pair we have three distinct, distance dependend connection probability
        # functions: reciprocal, A->B, B->A and of course the overall connection probability, total (their sum).
        pv_factor = 1/(0.154+0.019+0.077)
        # Use ordered dictionary, because the key names are just for the reader:
        connection_functions_d = OrderedDict()
        # PN with PN connectivity:
        connection_functions_d['PN_PN_total_probability'] = lambda x: np.exp(-0.0052*x) * 0.22
        connection_functions_d['PN_PN_reciprocal'] = lambda x: np.exp(-0.0085*x) * 0.12
        connection_functions_d['PN_PN_A2B'] = lambda x: np.exp(-0.0052*x) * 0.22 - np.exp(-0.0085*x) * 0.12
        connection_functions_d['PN_PN_B2A'] = lambda x: np.exp(-0.0052*x) * 0.22 - np.exp(-0.0085*x) * 0.12
        # PN with PV connectivity:
        connection_functions_d['PN_PV_total'] = lambda x: np.exp(x/-180)
        connection_functions_d['PN_PV_reciprocal'] = lambda x: np.exp(x/-180) * 0.154 * pv_factor
        connection_functions_d['PN_PV_A2B'] = lambda x: np.exp(x/-180) * 0.019 * pv_factor
        connection_functions_d['PN_PV_B2A'] = lambda x: np.exp(x/-180) * 0.077 * pv_factor
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
        self.configuration_str = cpairs2mat([connect_pair(upair, connection_type_distance) for upair in self.upairs])
        plot_reciprocal_across_distance(self.configuration_str[:self.pc_no, :self.pc_no], self.dist_mat[:self.pc_no, :self.pc_no])

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
        iters = 1000
        f_prob = np.zeros((self.pc_no, self.pc_no, iters))
        cn_prob = np.zeros((self.pc_no, self.pc_no, iters))

        sumPd = Pd.sum()
        for t in range(1, iters):
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
            nextE = cpairs2mat([connect_pair(upair, connection_type_distance) for upair in dist_estimate_upairs])
            E = nextE

        # Save connectivity to the network:
        self.configuration_str[:self.pc_no, :self.pc_no] = E

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

def plot_reciprocal_across_distance(mat, distance_mat):
    '''
    Plots a histogram with the relative frequency of reciprocal connections as a function of distance.
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
    fig, ax = plt.subplots()
    cax = ax.plot(np.divide(histo, histo_dist))
    ax.set_title('configuration str')
    plt.show()

