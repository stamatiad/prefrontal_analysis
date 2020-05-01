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
        'nestboxx.ddns.net',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===

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

def initialize_trials(serial_no, trial_no, stimulated_pn_no):
    #TODO:Do it better
    np.random.seed(serial_no)
    # Initialize stimulated cells with the same RNG as the original code,
    # giving the flexibility to play with different size stimulation population.
    # initialize stimulated cells in each trials:
    stimulated_cells = np.full((trial_no, stimulated_pn_no), 0, dtype=int)
    for trial in range(trial_no):
        stimulated_cells[trial][:]  = np.sort(np.random.permutation(np.arange(250))[:stimulated_pn_no])
    return stimulated_cells

# This is the subset per cluster: RNG also on cluster input per trial:
def export_network_cluster_parameters_per_trial(
        export_path=None,
        stimulated_cells=None,
        **kwargs
):
    # Export Network Connectivity:
    # Export parameter matrices in .hoc file:
    pc_no = 250
    pv_no = 83
    cell_no = 333
    cluster_no = len(stimulated_cells)
    trials_per_cluster = len(stimulated_cells[0])
    total_trial_no = cluster_no * trials_per_cluster
    #SPS = Stimulated Population Size:
    import_hdf5_filename = export_path.joinpath(
        f'importNetworkParameters_{configuration_alias}_SN{serial_no}_CP{cluster_no}.hoc'
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
            for trial, trial_global in enumerate(range(trial_increment,
                               trials_per_cluster + trial_increment)):
                f.write(f'StimulatedPNs[{trial_global}]=new Vector('
                        f'{stimulated_cells[cluster][trial].size})\n')
                for i in range(stimulated_cells[cluster][trial].size):
                    # stimcells is a Nx1 array:
                    f.write(f'StimulatedPNs[{trial_global}].x[{i}]='
                            f'{stimulated_cells[cluster][trial][i]}\n')
            trial_increment += trials_per_cluster
        f.write('//EOF\n')
    # also write the same file for all the dend configurations:
    #ui = input(f'Proceed in generating dend files for file'
    #           f' {import_hdf5_filename}?')
    #if 'yes' in ui:
    if False:
        for dendlen in ['small','medium','long']:
            for dendno in [1,2]:
                tmp_fn = export_path.joinpath(
                    f'importNetworkParameters_{configuration_alias}_{dendno}'
                    f'{dendlen}dend_SN{serial_no}_CP{cluster_no}.hoc'
                )
                copyfile(import_hdf5_filename, tmp_fn)



# This is for same stim in each cluster. RNG only on internal neuron model vars.
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
    total_trial_no = cluster_no * trials_per_cluster
    #SPS = Stimulated Population Size:
    import_hdf5_filename = export_path.joinpath(
        f'importNetworkParameters_{configuration_alias}_SN{serial_no}_CP{cluster_no}.hoc'
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
                    f.write(f'StimulatedPNs[{trial}].x[{i}]={stimulated_cells[cluster][i,0]}\n')
            trial_increment += trials_per_cluster
        f.write('//EOF\n')
    # also write the same file for all the dend configurations:
    #ui = input(f'Proceed in generating dend files for file'
    #           f' {import_hdf5_filename}?')
    #if 'yes' in ui:
    for dendlen in ['small','medium','long']:
        for dendno in [1,2]:
            tmp_fn = export_path.joinpath(
                f'importNetworkParameters_{configuration_alias}_{dendno}'
                f'{dendlen}dend_SN{serial_no}_CP{cluster_no}.hoc'
            )
            copyfile(import_hdf5_filename, tmp_fn)

# ==============================================================================
# DISCLAMER: Original clustering for revisions was not created with
# reproducible rng. So you might need to rerun this original clusterings.
# ==============================================================================

# Load original network dataset:
export_path = Path('/home/cluster/stefanos/Documents/GitHub'
                   '/prefrontal_analysis/subset_files/')
filename_prefix = ''
configuration_alias = 'structured'
serial_no = 1
filename_postfix = ''
filename = export_path.joinpath(f'{filename_prefix}{configuration_alias}_network_SN'
                 f'{serial_no}{filename_postfix}.hdf5'
)
blah = pd.read_hdf(filename, key='attributes')
serial_no = 1
connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
weights_mat = pd.read_hdf(filename, key='weights_mat').values

#TODO: perform clustering first!
# precompute common neighbor distances:
print('starting clustering!')
cn = common_neighbors(connectivity_mat[:250, :250])
for cluster_no in [2,3,4,5,6,7]:
    ap_labels = analysis.apclusterk(cn, cluster_no, prc=0, serial_no=serial_no)
    stimulated_cells = []
    for c in range(cluster_no):
        stimulated_cells.append(np.argwhere(ap_labels == c))

    # Keep only a random subset of each cluster's neurons on each trial.
    new_stim_cells = analysis.get_random_subset(
        stimulated_cells, max_stim_size=50, serial_no=serial_no
    )

    export_network_cluster_parameters_per_trial(
        export_path=export_path,
        stimulated_cells=new_stim_cells,
        trials_per_cluster=10
    )

print('Tutto pronto!')

