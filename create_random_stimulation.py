from functools import wraps
import numpy as np
from  pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation as AP
import analysis_tools as analysis
import sys, os
from shutil import copyfile

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


def with_reproducible_rng(func):
    @wraps(func)
    def reset_rng(*args, **kwargs):
        # this is the first argument, that is the serial no that seeds the RNG.
        np.random.seed(args[0])
        print(f'{func.__name__} reseeds the RNG with {args[0]}.')
        return func(*args, **kwargs)
    return reset_rng

def zero_diagonal(mat):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square!")
    for i in range(mat.shape[0]):
        mat[i, i] = 0.0
    return mat

def common_neighbors(adjmat):
    adj = np.logical_or(adjmat, adjmat.T)
    adj = adj.astype(float) @ adj.T.astype(float)
    adj = zero_diagonal(adj)
    return adj

@with_reproducible_rng
def initialize_trials(serial_no, trial_no=10, stimulated_pn_no=50):
    # Initialize stimulated cells with the same RNG as the original code,
    # giving the flexibility to play with different size stimulation population.
    # initialize stimulated cells in each trials:
    stimulated_cells = np.full((trial_no, stimulated_pn_no), 0, dtype=int)
    for trial in range(trial_no):
        stimulated_cells[trial][:] = np.sort(np.random.permutation(np.arange(250))[:stimulated_pn_no])
    return stimulated_cells

def export_network_cluster_parameters(
    export_path=None,
    stimulated_cells=None,
    trials_per_cluster=10
    ):
    # Export Network Connectivity:
    # Export parameter matrices in .hoc file:
    pc_no = 250
    pv_no = 83
    cell_no = 333
    cluster_no = len(stimulated_cells)
    stim_cells_size = stimulated_cells.shape[1]
    total_trial_no = cluster_no * trials_per_cluster
    #SPS = Stimulated Population Size:
    import_hdf5_filename = export_path.joinpath(
        f'importNetworkParameters_{configuration_alias}_SN{serial_no}_'
        f'RI{stim_cells_size}.hoc'
        )
    with open( import_hdf5_filename, 'w') as f:
        f.write(f'// This HOC file was generated with network_tools python module.\n')
        f.write(f'nPCcells={pc_no}\n')
        f.write(f'nPVcells={pv_no}\n')
        f.write(f'nAllCells={cell_no}\n')
        f.write(f'// Object decleration:\n')
        f.write(f'objref C, W\n')
        f.write(f'objref StimulatedPNs[{total_trial_no}]\n')
        f.write(f'C = new Matrix(nAllCells, nAllCells)\n')
        f.write(f'W = new Matrix(nAllCells, nAllCells)\n')
        f.write(f'\n// Import parameters: (long-long text following!)\n')
        # Network connectivity:
        pairs = [(i, j) for i in range(cell_no) for j in range(cell_no)]
        for (i, j) in pairs:
            f.write(f'C.x[{i}][{j}]={int(connectivity_mat[i, j])}\n')
        for (i, j) in pairs:
            f.write(f'W.x[{i}][{j}]={weights_mat[i, j]}\n')
        # Network stimulation:
        trial_increment = 0
        for cluster in range(cluster_no):
            for trial in range(trial_increment, trials_per_cluster + trial_increment):
                f.write(f'StimulatedPNs[{trial}]=new Vector({stimulated_cells[cluster].size})\n')
                for i in range(stimulated_cells[cluster].size):
                    # stimcells is a Nx1 array:
                    f.write(f'StimulatedPNs[{trial}].x[{i}]={stimulated_cells[cluster][i]}\n')
            trial_increment += trials_per_cluster
        f.write('//EOF\n')

    for dendlen in ['small','medium','long']:
        for dendno in [1,2]:
            tmp_fn = export_path.joinpath(
                f'importNetworkParameters_{configuration_alias}_{dendno}'
                f'{dendlen}dend_SN{serial_no}_RI{stim_cells_size}.hoc'
            )
            copyfile(import_hdf5_filename, tmp_fn)

@with_reproducible_rng
def create_random_stimulation(sn, stim_size=50):
    '''
    If in any case one needs to generate more random inputs, just rerun the
    function with different size on the random array. The reproducible RNG
    will make the random arrays overlap.
    :param sn:
    :param stim_size:
    :return:
    '''
    stimulated_cells = np.random.randint(0, 249, size=(10, stim_size))
    return stimulated_cells

# Load original network dataset:
export_path = Path('/home/cluster/stefanos/Documents/GitHub'
                   '/prefrontal_analysis/correct_net_files')
filename_prefix = ''
configuration_alias = 'structured'
serial_no = 1
filename_postfix = ''
filename = os.path.join(export_path, '{}{}_network_SN{}{}.hdf5'
                        .format(filename_prefix, configuration_alias, serial_no, filename_postfix))
blah = pd.read_hdf(filename, key='attributes')
serial_no = 1
connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
weights_mat = pd.read_hdf(filename, key='weights_mat').values

#stimulated_cells = create_random_stimulation(serial_no, stim_size=50)
stimulated_cells = initialize_trials(
    serial_no,
    trial_no=10,
    stimulated_pn_no=int(250/5)
)

export_network_cluster_parameters(
    export_path=export_path,
    stimulated_cells=stimulated_cells,
    trials_per_cluster=10
)

print('Tutto pronto!')
