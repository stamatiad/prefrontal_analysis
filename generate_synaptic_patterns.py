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

# ===%% Pycharm debug: %%===
import pydevd_pycharm
sys.path.append("pydevd-pycharm.egg")
DEBUG = False
if DEBUG:
    pydevd_pycharm.settrace(
        '79.167.94.93',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===

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


@analysis.with_reproducible_rng
def create_weights(**kwargs):
    # THis is optional and more of a verbose error:
    if not kwargs.get('serial_no', None):
        raise ValueError('You have not provided serial_no for RNG '
                         'reproducibility!')

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
    weights_mat_pc[weights_mat_pc > 10] = 10

    if False:
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
                fig.savefig(f"weights_valid_{k}.png")
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


@analysis.with_reproducible_rng
def generate_patterns(
        filters, export_path=Path.cwd(), weights_mat_pc=None, **kwargs
):
    # THis is optional and more of a verbose error:
    if not kwargs.get('serial_no', None):
        raise ValueError('You have not provided serial_no for RNG '
                         'reproducibility!')

    # Read NEURON pattern data and classify it in a dataframe:
    patterns_path = Path(r'/home/cluster/stefanos/Documents/GitHub/prefrontal-micro'
                         r'/experiment/network/synaptic_patterns')

    patterns_df = load_synaptic_patterns(patterns_path)

    # Provide the boilerplate to select class preset (e.g. distal)
    case_df = patterns_df.loc[(patterns_df[list(filters)] == pd.Series(
        filters)).all(axis=1)]

    # Generate NEURON arrays for each PC cell containing PID and W of each synapse:
    # This file will be addressible/identifiable by the filters above and used
    # along with the importNetworkParams files. Should change when the filters
    # change. Filters should exist as keywords in NEURON and saved in simulations
    # metadata.
    # I also need RNG seed here, as a SN of the pattern file.

    # If I provide a weights matrix for PCs use it. Else, choose a medial
    # value of somatic depolarization and assign 'uniform' patterns that
    # comply to the filters, yet are homegeneous in their depolarization.
    depol_vals = case_df['somatic_depolarization_bin'].values
    connectivity_mat_pc = get_connectivity_matrix_pc()
    depol_bin_mat_pc = np.zeros(connectivity_mat_pc.shape)
    if weights_mat_pc is None:
        real_weights = False
        #TODO: this depol bin should be GLOBAL to the df, not after the
        # filters! So the same patterns file, generates consistent no-real
        # weights across realizations.
        depol_vals_global = patterns_df['somatic_depolarization_bin'].values
        histo, *_ = np.histogram(
            depol_vals_global,
            np.arange(0, int(depol_vals_global.max()) + 1, 1)
        )
        depol_bin = np.argmax(histo) + 1

        # create a depol matrix, with 'bin' values:
        depol_bin_mat_pc[connectivity_mat_pc] = depol_bin

    else:
        real_weights = True
        # Scale down the weights as per the depolarization maximum:
        scaled_weights = (weights_mat_pc * depol_vals.max()) / \
                          weights_mat_pc.max()
        scaled_weights = connectivity_mat_pc.astype(int) * scaled_weights
        depol_bin_mat_pc = np.ceil(scaled_weights)

    # sample the patterns:
    pair_d = {}
    for m in range(250):
        for n in range(250):
            if connectivity_mat_pc[m, n]:
                query_bin = depol_bin_mat_pc[m, n]
                # sample the depol values:
                subdf_idx, *_ = np.where(depol_vals == query_bin)
                max_ind = subdf_idx.size -1
                if max_ind == 0:
                    print(f"Single pattern meets criteria! {m},{n}, "
                          f"query_bin: {query_bin}")
                    rnd_idx = 0
                else:
                    # THis is equivalent to a random permutation
                    rnd_idx = np.random.randint(0, max_ind)
                case_idx = case_df.index.values
                pair_PID, pair_W = case_df.loc[
                    case_idx[subdf_idx[rnd_idx]]
                ][['PID','W']]
                # Do a check that no PIDs are border cases 0/1:
                pair_PID[pair_PID==0] = 0.1
                pair_PID[pair_PID==1] = 0.99
                pair_d[(m,n)] = (pair_PID, pair_W)


        # create a depol matrix, with 'bin' values:

        depol_bin_mat_pc[connectivity_mat_pc] = 1

    syn_loc = filters['synapse_location']
    syn_clust = filters['synapse_clustering']
    filename = export_path.joinpath(
        f"synaptic_patterns_location_{syn_loc}_clustering_"
        f"{int(syn_clust)}_real_weights_{int(real_weights)}_S"
        f"N{kwargs['serial_no']}.hoc"
    )
    with open(filename, 'w') as f:
        f.write(
            f'// This HOC file was generated with generate_patterns python '
            f'module.\n')
        f.write(f'objref synaptic_patterns[250][250]\n')
        f.write(f'objref synaptic_weights[250][250]\n')

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

                    PID,W = pair_d[(afferent,efferent)]
                    for i, (pid, w) in enumerate(zip(PID,W)):
                        f.write(
                            f'synaptic_patterns[{afferent}][{efferent}].x[{i}]'
                            f'={pid}\n')
                        f.write(
                            f'synaptic_weights[{afferent}][{efferent}].x[{i}]'
                            f'={w}\n')

        # Network connectivity:
        f.write('//EOF\n')

    # Make changes in NEURON to run the new simulations:
    print("Pronto!")

if __name__ == '__main__':
    filters = {
        'synapse_location': 'distal',
        'synapse_clustering': False,
    }
    serial_no = 1

    generate_patterns(
        filters,
        #weights_mat_pc=create_weights(serial_no=serial_no),
        serial_no=serial_no
    )
