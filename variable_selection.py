import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generate_synaptic_patterns import get_connectivity_matrix_pc
import network_tools as nt
from sklearn.cluster import AffinityPropagation as AP
from generate_synaptic_patterns import generate_patterns


# TODO: Affinity Propagation Network analysis:
# Calculate dendritic statistics for each cluster.
# Calculate common neighbor matrix:
dendno = 3
patterns_df = generate_patterns(
    export2file=False,
    weights_mat_pc=None,
    learning_condition=1,
    min_w=-10,
    max_w=10,
    iters=100000,
    max_depol_val=5,
    synapse_location='distal',
    synapse_clustering=False,
    dendritic_clustering=False,
    dendlen=150,
    dendno=dendno
)

# ta assemblies eiani poly fine tuned. Koita prwta sto diktyo synolika:
connectivity_mat_pc = get_connectivity_matrix_pc()

cell_sp_attrs = np.zeros((3, connectivity_mat_pc.shape[0]), dtype=float)
cell_sp_attrs[:, :] = np.nan
for cell in range(connectivity_mat_pc.shape[0]):
    src_rel = np.where(connectivity_mat_pc[:, cell])[0]
    if src_rel.size > 1:
        src = src_rel
        trg = cell
        assembly_idx = []  # pd.Series()
        for presyn in range(len(src)):
            filters = {'SRC': src[presyn], 'TRG': trg}
            idx_tmp = (patterns_df[list(filters)] == pd.Series(filters)).all(
                axis=1)
            assembly_idx.append(idx_tmp.values.astype(int).nonzero()[0][0])
        # assembly_idx.append(idx_tmp[idx_tmp ==
        # True])
        cell_df = patterns_df.iloc[assembly_idx]

        # compute statistics like how much clustering/weight/max weight location
        # per pair. Remember to publish all the data (like boxplot per cluster).
        # sto paradeigma pou exw: clustering sto compartment 4,
        # apo 2 presynaptics me total weithg, tade
        PID_vals = np.concatenate(cell_df['PID'].values)
        W_vals = np.concatenate(cell_df['W'].values)
        D_vals = np.concatenate(cell_df['D'].values)

        # den 8elw na exw kapoia discrete timh, alla float times pou na
        # mou lene:
        # 1) poso konta sto swma eimai (proximal/distal)
        # 2) poso dispersed sta segments eimai
        # 3) poso dispersed stous dends eimai.
        # SOS: since some of the weights are zero, remove those synapses:
        tmp_idx = np.where(W_vals != 0.0)[0]
        PID_vals = PID_vals[tmp_idx]
        W_vals = W_vals[tmp_idx]
        D_vals = D_vals[tmp_idx]

        # 1)Get average distance from soma:
        cell_sp_attrs[0, cell] = PID_vals.mean()

        # 2)Get std on segments:
        tmp_arr = []
        for i in range(dendno):
            syns_on_same_dend = np.where(D_vals == i)
            tmp_arr.append(PID_vals[syns_on_same_dend].std())
        cell_sp_attrs[1, cell] = np.array(tmp_arr).mean()

        # 3)Get std on dendrites:
        cell_sp_attrs[2, cell] = PID_vals.std()

# plot boxplots with the network statistics:
tmp_idx = np.isfinite(cell_sp_attrs[0, :])
fig = plt.figure()
plot_axes = fig.add_subplot(111)
plot_axes.boxplot(cell_sp_attrs[:, tmp_idx].T)
plt.savefig(f"Rev_5_sp_attrs_stats.png")

tmpnet = nt.Network(serial_no=1, pc_no=250, pv_no=83)
s = tmpnet.common_neighbors(connectivity_mat_pc)
N = s.shape[0]

S = s + (np.finfo(float).eps * s + np.finfo(float).tiny * 100) * np.random.rand(
    N, N)
ap = AP()
ap.affinity = 'precomputed'
ap.max_iter = 2000
ap.convergence_iter = 200
ap.damping = 0.9
ap.verbose = True
ap.preference = 0.5
ap_result = ap.fit(S)
idx = ap_result.labels_
# Maximum cluster no returned by AP:
tmpk = idx.max() + 1
for cluster in range(tmpk):
    mc_idx = np.where(idx == cluster)
    assembly = connectivity_mat_pc[mc_idx[0], :].T[mc_idx[0], :].T
    # get only postsynaptic cells with multiple presynaptics:
    if assembly.size > 1:
        for i in range(len(mc_idx)):
            src_rel = np.where(assembly[:, i])
            src = mc_idx[0][src_rel]
            trg = mc_idx[0][i]
            assembly_idx = []  # pd.Series()
            for presyn in range(len(src)):
                filters = {'SRC': src[presyn], 'TRG': trg}
                idx_tmp = (patterns_df[list(filters)] == pd.Series(
                    filters)).all(
                    axis=1)
                assembly_idx.append(idx_tmp.values.astype(int).nonzero()[0][0])
            # assembly_idx.append(idx_tmp[idx_tmp ==
            # True])
            cell_df = patterns_df.iloc[assembly_idx]

            # compute statistics like how much clustering/weight/max weight location
            # per pair. Remember to publish all the data (like boxplot per cluster).
            # sto paradeigma pou exw: clustering sto compartment 4,
            # apo 2 presynaptics me total weithg, tade
            PID_vals = np.concatenate(cell_df['PID'].values)
            W_vals = np.concatenate(cell_df['W'].values)
            D_vals = np.concatenate(cell_df['D'].values)

            # den 8elw na exw kapoia discrete timh, alla float times pou na
            # mou lene:
            # 1) poso konta sto swma eimai (proximal/distal)
            # 2) poso dispersed sta segments eimai
            # 3) poso dispersed stous dends eimai.
            # SOS: since some of the weights are zero, remove those synapses:
            tmp_idx = np.where(W_vals != 0.0)[0]
            PID_vals = PID_vals[tmp_idx]
            W_vals = W_vals[tmp_idx]
            D_vals = D_vals[tmp_idx]

            # 1)Get average distance from soma:
            PID_vals.mean()

            # 2)Get std on segments:
            tmp_arr = []
            for i in range(dendno):
                syns_on_same_dend = np.where(D_vals == i)
                tmp_arr.append(PID_vals[syns_on_same_dend].std())
            np.array(tmp_arr).mean()

            # 3)Get std on dendrites:
            PID_vals.std()

    # Plot them and check them for inconsistenceis.
