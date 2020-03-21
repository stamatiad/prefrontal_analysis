from functools import wraps
import numpy as np
import os
from  pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation as AP
import analysis_tools as analysis


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

# Load original network dataset:
export_path = Path(r'C:\Users\stefanos\Documents\GitHub\prefrontal_analysis\correct_net_files')
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

# Since AP does not work here, try matlab:
#np.savetxt("conn_mat.csv", connectivity_mat, delimiter=",")
#np.savetxt("common_neighbors.csv", cn, delimiter=",")

#TODO: perform clustering first!
# precompute common neighbor distances:
print('starting clustering!')
cn = common_neighbors(connectivity_mat[:250, :250])
cluster_no = 8
ap_labels = analysis.apclusterk(cn, cluster_no, prc=0)

#stimulated_cells = np.full((cluster_no, ), 0, dtype=int)
stimulated_cells = []
for c in range(cluster_no):
    stimulated_cells.append(np.argwhere(ap_labels == c))

export_network_cluster_parameters(
    export_path=export_path,
    stimulated_cells=stimulated_cells,
    trials_per_cluster=10
)
print('Tutto pronto!')


'''
ap = AP()
ap.affinity = 'precomputed'
ap.max_iter = 5000
ap.damping = 0.75
ap.verbose = True
#Compute AP preference range:
min_pref = -200
max_pref = 100
pref_range = np.arange(min_pref, max_pref, 10)
ap_pref_clusters = np.full((pref_range.size, 1), -1, dtype=int)
for i, pref in enumerate(pref_range):
    print(f'On i: {i}\n')
    ap.preference = pref
    ap_result = ap.fit(cn)
    if ap_result.n_iter_ < ap.max_iter:
        ap_pref_clusters[i, 0] = ap_result.labels_.max() + 1
print("tutto pronto")
plt.plot(pref_range, ap_pref_clusters)
# How do you take single dim nparray?
clust_pts = np.diff(ap_pref_clusters.T)
pref_range[np.argwhere(clust_pts.T>0)][:,0]
# Run AP for the selected pref value:
ap.preference = -140
ap_result = ap.fit(cn)
ap_labels = ap_result.labels_
'''

