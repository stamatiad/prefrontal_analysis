from functools import wraps
import numpy as np
import os
from  pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation as AP
import analysis_tools as analysis



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

# load a NWB, get its runs and locate the responding cells in PA period:
glia_dir = Path("\\\\139.91.162.90\\cluster\\stefanos\\Documents\\Glia\\")

cp_array = [2,3,4,5,6,7]
NWBfiles_1longdend = []
sparse_cp_trials = lambda cp: (cp - 1) * 10 + 1
cp_trials_len = 0
configuration_alias = 'structured_1longdend'
for cp in cp_array:
    cp_trials_len += cp

for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 2.0,
            'nmda_bias': 6.0,
            'ntrials': sparse_cp_trials(cp),
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_1longdend'
        }
    )
    if NWBfile:
        NWBfiles_1longdend.append(NWBfile)

# get from each NWB, for each trial the last bin
pooled_activity = np.array([])
for NWBfile in NWBfiles_1longdend:
    activity, *_ = analysis.get_binned_activity(
        NWBfile
    )
    if pooled_activity.size:
        pooled_activity = np.concatenate((pooled_activity, activity[:,:,-1]), axis=1)
    else:
        pooled_activity = activity[:,:,-1]

nstimuli = pooled_activity.shape[1]
stimulated_cells = []
for c in range(nstimuli):
    stimulated_cells.append(np.argwhere(pooled_activity[:,c] > 0))

export_network_cluster_parameters(
    export_path=export_path,
    stimulated_cells=stimulated_cells,
    trials_per_cluster=10
)
print('Tutto pronto!')



