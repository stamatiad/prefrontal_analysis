# %% Check the plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
from scipy.stats import invgamma
from pathlib import Path
import pandas as pd
import network_tools as nt
from itertools import chain
from load_synaptic_patterns import load_synaptic_patterns
import analysis_tools as analysis
from functools import partial, wraps

# ===%% Pycharm debug: %%===
import pydevd_pycharm
sys.path.append("pydevd-pycharm.egg")
DEBUG = True
if DEBUG:
    pydevd_pycharm.settrace(
        '79.167.48.178',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===

def reproducible_weights_rng(class_method):
    '''
    Reproducible weights matrix realization
    :param class_method:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        # this translates to self.serial_no ar runtime
        np.random.seed(kwargs['weights_realization'])
        print(f'{class_method.__name__} reseeds the RNG.')
        return class_method(*args, **kwargs)
    return reset_rng

def reproducible_learning_condition_rng(class_method):
    '''
    Reproducible weights matrix realization
    :param class_method:
    :return:
    '''
    @wraps(class_method)
    def reset_rng(*args, **kwargs):
        # this translates to self.serial_no ar runtime
        np.random.seed(kwargs['learning_condition'])
        print(f'{class_method.__name__} reseeds the RNG.')
        return class_method(*args, **kwargs)
    return reset_rng

def get_connectivity_matrix_pc():
    # load connectivity matrix:
    # Load original network dataset:
    export_path = Path('/home/cluster/stefanos/Documents/GitHub'
                       '/prefrontal_analysis/subset_files/')
    filename_prefix = ''
    configuration_alias = 'structured'
    serial_no = 1
    filename_postfix = ''
    filename = export_path.joinpath(
        f'{filename_prefix}{configuration_alias}_network_SN'
        f'{serial_no}{filename_postfix}.hdf5'
    )
    blah = pd.read_hdf(filename, key='attributes')
    serial_no = 1
    connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
    # weights_mat = pd.read_hdf(filename, key='weights_mat').values
    return connectivity_mat[:250, :250]


@reproducible_weights_rng
def create_weights(**kwargs):
    # THis is optional and more of a verbose error:
    weights_realization = kwargs.get('weights_realization', None)
    if not weights_realization:
        raise ValueError('You have not provided serial_no for RNG '
                     'reproducibility!')

    max_weight = 5 #mV

    connectivity_mat_pc = get_connectivity_matrix_pc()
    tmpnet = nt.Network(serial_no=1, pc_no=250, pv_no=83)
    ncn = tmpnet.common_neighbors(connectivity_mat_pc)

    b = 1.0
    a_step = 2.0 / (ncn.max() + 1)
    invg_array = []
    for i in range(int(ncn.max() + 1)):
        print(f"alpha = {2.0 - (i * a_step)}")
        invg_array.append(invgamma(2.0 - (i * a_step), scale=b))

    weights_mat_pc = np.zeros(ncn.shape)
    for i in range(250):
        for j in range(250):
            weights_mat_pc[i, j] = invg_array[int(ncn[i, j])].rvs()

    # Clip larger weights (I saw some):
    hist, *_ = np.histogram(weights_mat_pc,
                            np.arange(0, 100, 0.1))
    weights_mat_pc[weights_mat_pc > 10] = 10

    #plot the resulting distribution as debugging:
    # Check similarity with Brunel's data (figure 3D).
    ncn_bins = ncn.astype(int)
    max_somatic_depol = 10
    hist, *_ = np.histogram(weights_mat_pc,
                            np.arange(0, max_somatic_depol, 0.1))
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(0, hist.size), hist / hist.sum(), 'b-', lw=1)
    fig.savefig(f"WR{weights_realization}_Brunel_Fig3d.png")

    if False:
        #Plot all distributions from where we sample:
        fig, ax = plt.subplots(1, 1)
        for k in range(int(ncn.max() + 1)):
            x = np.linspace(0, max_somatic_depol, num=1000)
            ax.plot(x, invg_array[k].pdf(x), 'k-', lw=1)
        fig.savefig(f"WR{weights_realization}_validation_alldists.png")

        # validate the weights with respect to common neighbors:
        # This can be bypassed, by adding 1 and converting to int:
        # ncn_bins = np.digitize(ncn, np.arange(0,ncn.max()+1,1))
        ncn_bins = ncn.astype(int)
        max_somatic_depol = 10

        for k in range(int(ncn.max() + 1)):
            tmparr = []
            for i, idx in enumerate(tuple(ncn_bins)):
                tmparr.append(weights_mat_pc[i, idx == k])
            weights_valid = np.array(list(chain(*tmparr)))
            hist, *_ = np.histogram(weights_valid,
                                    np.arange(0, max_somatic_depol, 0.01))
            try:
                x = np.linspace(0, max_somatic_depol, num=1000)
                fig, ax = plt.subplots(1, 1)
                ax.plot(x, invg_array[k].pdf(x), 'k-', lw=1)
                ax.plot(x[:999], hist / hist.sum(), 'b-', lw=1)
                fig.savefig(f"WR{weights_realization}_validation_{k}.png")
            except Exception:
                # suck it up
                pass

        # Check similarity with Brunel's data (figure 3D).
        hist, *_ = np.histogram(weights_mat_pc,
                                np.arange(0, max_somatic_depol, 0.01))
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(0, hist.size), hist / hist.sum(), 'b-', lw=1)
        fig.savefig(f"Brunel_Fig3d.png")

    # Here you should check for reciprocal connections and increase the weights:
    # Kawaguchi & Kubota, elife, 2015 say that the weights are not significantly
    # larger in the reciprocal case. Also even if larger weights exist they were
    # not symmetrical. So I am not increasing any weights.
    return weights_mat_pc


@reproducible_learning_condition_rng
def generate_patterns(
        export_path=Path.cwd(), weights_mat_pc=None,
        inhomogeneous=False, **kwargs
):
    # THis is optional and more of a verbose error:
    learning_condition = kwargs.get('learning_condition', None)
    weights_realization = kwargs.get('weights_realization', None)
    if not kwargs.get('learning_condition', None):
        raise ValueError('You have not provided serial_no for RNG '
                     'reproducibility!')

    # Read NEURON pattern data and classify it in a dataframe:
    patterns_path = Path(r'/home/cluster/stefanos/Documents/GitHub/prefrontal-micro'
                         r'/experiment/network/synaptic_patterns')

    dendlen = kwargs['dendlen']
    dendno = kwargs['dendno']
    nseg = int(dendlen/30.0)
    iters = kwargs.get('iters',100000)
    max_depol_val = kwargs.get('max_depol_val',5)
    min_w = kwargs.get('min_w', -10)
    max_w = kwargs.get('max_w',10)
    patterns_df = load_synaptic_patterns(
        patterns_path,
        dendlen=dendlen,
        dendno=dendno,
        nseg=nseg,
        nsyn=5,
        max_depol_val=max_depol_val,
        min_w=min_w,
        max_w=max_w,
        iters=iters
    )

    # Change to inhomogeneous patterns if not found!
    synapse_location = kwargs.get('synapse_location', None)
    synapse_clustering = kwargs.get('synapse_clustering', None)
    dendritic_clustering = kwargs.get('dendritic_clustering', None)
    if (synapse_location is not None) and \
        (synapse_clustering is not None) and \
        (dendritic_clustering is not None):
        filters = {
            'synapse_location':synapse_location,
            'synapse_clustering': synapse_clustering,
            'dendritic_clustering': dendritic_clustering
        }

        # Provide the boilerplate to select class preset (e.g. distal)
        case_df = patterns_df.loc[(patterns_df[list(filters)] == pd.Series(
            filters)).all(axis=1)]
    else:
        inhomogeneous = True
        case_df = patterns_df

    # Generate NEURON arrays for each PC cell containing PID and W of each synapse:
    # This file will be addressible/identifiable by the filters above and used
    # along with the importNetworkParams files. Should change when the filters
    # change. Filters should exist as keywords in NEURON and saved in simulations
    # metadata.
    # I also need RNG seed here, as a SN of the pattern file.

    # If I provide a weights matrix for PCs use it. Else, choose a medial
    # value of somatic depolarization and assign 'uniform' patterns that
    # comply to the filters, yet are homegeneous in their depolarization.
    depol_bins = case_df['somatic_depolarization_bin'].values

    # TODO: compute the histogram to be sure that there are enough
    #  patterns for each depolarization bin:
    histo = np.histogram(depol_bins, np.arange(0,80,1))

    connectivity_mat_pc = get_connectivity_matrix_pc()
    depol_bin_mat_pc = np.zeros(connectivity_mat_pc.shape)
    if weights_mat_pc is None:
        real_weights = False
        #TODO: this depol bin should be GLOBAL to the df, not after the
        # filters! So the same patterns file, generates consistent no-real
        # weights across realizations.
        depol_bins_global = patterns_df['somatic_depolarization_bin'].values
        histo, *_ = np.histogram(
            depol_bins_global,
            np.arange(0, int(depol_bins_global.max()) + 1, 1)
        )
        depol_bin = np.argmax(histo) + 1

        # create a depol matrix, with 'bin' values:
        depol_bin_mat_pc[connectivity_mat_pc] = depol_bin

    else:
        real_weights = True
        # Scale down the weights as per the depolarization maximum:
        scaled_weights = (weights_mat_pc / weights_mat_pc.max()) * \
        depol_bins.max()
        scaled_weights = connectivity_mat_pc.astype(int) * scaled_weights
        depol_bin_mat_pc = np.ceil(scaled_weights)

    # sample the patterns:
    pair_d = {}
    for m in range(250):
        for n in range(250):
            if connectivity_mat_pc[m, n]:
                query_bin = int(depol_bin_mat_pc[m, n])
                # sample the depol values:
                subdf_idx, *_ = np.where(depol_bins == query_bin)
                if subdf_idx.size == 0:
                    # Logically this will be some border case. Print the
                    # query bin, just in case, to see what's happening.
                    print(f"No depol val for query bin: {query_bin}")
                    if np.abs(query_bin - depol_bins.min()) < np.abs(query_bin -
                                                         depol_bins.max()):
                        query_bin = depol_bins.min()
                    else:
                        query_bin = depol_bins.max()
                    print(f'\t replaced with {query_bin}')
                    subdf_idx, *_ = np.where(depol_bins == query_bin)
                max_ind = subdf_idx.size -1
                if max_ind == 0:
                    print(f"Single pattern meets criteria! {m},{n}, "
                          f"query_bin: {query_bin}")
                    rnd_idx = 0
                else:
                    # THis is equivalent to a random permutation
                    rnd_idx = np.random.randint(0, max_ind)
                case_idx = case_df.index.values
                pair_PID, pair_W, pair_D = case_df.loc[
                    case_idx[subdf_idx[rnd_idx]]
                ][['PID','W', 'D']]
                # Do a check that no PIDs are border cases 0/1:
                pair_PID[pair_PID==0] = 0.1
                pair_PID[pair_PID==1] = 0.99
                pair_d[(m,n)] = (pair_PID, pair_W, pair_D)


        # create a depol matrix, with 'bin' values:

        #depol_bin_mat_pc[connectivity_mat_pc] = 1

    #TODO: There is no serial number here: only learning condition i.e.
    # different reshuffling of the synapses. But I use serial_no, in order to
    # reuse the RNG wrapper. The SN is defined from the network I read/import.
    #CORRECTED I consideer the RW case as the only one
    if inhomogeneous:
        filename = export_path.joinpath(
            f"sp_inhomogeneous_dendno_{dendno}_dendlen_{dendlen}_"
            f"wr_{int(weights_realization)}_LC{learning_condition}.hoc"
        )
    else:
        filename = export_path.joinpath(
            f"sp_synloc_{synapse_location}_syncl_"
            f"{int(synapse_clustering)}_dendcl_{int(dendritic_clustering)}_"
            f"dendno_{dendno}_dendlen_{dendlen}_"
            f"wr_{int(weights_realization)}_LC{learning_condition}_norm.hoc"
        )

    with open(filename, 'w') as f:
        f.write(
            f'// This HOC file was generated with generate_patterns python '
            f'module.\n')
        f.write(f'objref synaptic_patterns[250][250]\n')
        f.write(f'objref synaptic_weights[250][250]\n')
        f.write(f'objref synaptic_dends[250][250]\n')

        #in each obj location that conn mat has a connection, create a vector:
        for afferent in range(250):
            for efferent in range(250):
                if connectivity_mat_pc[afferent, efferent]:
                    f.write(
                        f'synaptic_patterns[{afferent}][{efferent}]=new Vector'
                        f'(5)\n'
                    )
                    f.write(
                        f'synaptic_weights[{afferent}][{efferent}]=new Vector'
                        f'(5)\n'
                    )
                    f.write(
                        f'synaptic_dends[{afferent}][{efferent}]=new Vector'
                        f'(5)\n'
                    )

                    PID, W, D = pair_d[(afferent,efferent)]
                    for i, (pid, w, d) in enumerate(zip(PID,W, D)):
                        f.write(
                            f'synaptic_patterns[{afferent}][{efferent}].x[{i}]'
                            f'={pid}\n')
                        f.write(
                            f'synaptic_weights[{afferent}][{efferent}].x[{i}]'
                            f'={w}\n')
                        f.write(
                            f'synaptic_dends[{afferent}][{efferent}].x[{i}]'
                            f'={d}\n')

        # Network connectivity:
        f.write('//EOF\n')

    # Make changes in NEURON to run the new simulations:
    print("Pronto!")

if __name__ == '__main__':
    # test weights matrix RNG:
    #weights_mat_pc = create_weights(serial_no=1)
    #weights_mat_pc2 = create_weights(serial_no=1)
    #print('done')

    #I will keep the same weights matrix instantiation as this is a different
    # level of inhomogeneity
    # I will keep this 1 for now:
    for weights_realization in range(1, 2):
        # This is the synaptic pattern learning condition.
        for learning_condition in range(1, 2):

            gp = partial(
                generate_patterns,
                weights_mat_pc=create_weights(
                    weights_realization=weights_realization
                ),
                learning_condition=learning_condition,
                weights_realization = weights_realization,
                min_w=-10,
                max_w=10,
                iters=100000,
                dendlen=30,
                dendno=1,
                max_depol_val=5
            )

            params = {
                'synapse_location': ['proximal'],
                'synapse_clustering': [True],
                'dendritic_clustering': [True],
                'dendlen': [30],
                'dendno': [1]
            }
            '''
            # Inhomogeneous patterns!
            params = {
                'dendlen': [30, 150],
                'dendno': [2,3]
            }
            '''

            analysis.run_for_all_parameters(
                gp,
                **{'auto_param_array': params}
            )
