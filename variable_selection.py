import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from functools import wraps

from generate_synaptic_patterns import get_connectivity_matrix_pc
import network_tools as nt
import analysis_tools as analysis
from sklearn.cluster import AffinityPropagation as AP
from generate_synaptic_patterns import generate_patterns

# ===%% Pycharm debug: %%===
import pydevd_pycharm
sys.path.append("pydevd-pycharm.egg")
DEBUG = True
if DEBUG:
    pydevd_pycharm.settrace(
        '79.166.124.231',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===


# TODO: Affinity Propagation Network analysis:
# Calculate dendritic statistics for each cluster.
# Calculate common neighbor matrix:

data = []


def append_results_to_array(array=None):
    '''
    This function appends results AND corresponding run attributes to an array.
    This array can later on converted to a dataframe for ease of access.
    :param array:
    :return:
    '''
    def callable(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            result_stats_dict = function(*args, **kwargs)
            array.append(
                {
                    'results': result_stats_dict,
                    **kwargs
                }
            )
            #print(f'{function.__name__} took {toc-tic} seconds.')
            return result_stats_dict
        return wrapped
    return callable

@append_results_to_array(array=data)
def run_assembly_statistics(
    export2file=False,
    weights_mat_pc=None,
    learning_condition=None,
    min_w=-10,
    max_w=10,
    iters=100000,
    max_depol_val=5,
    synapse_location=None,
    clustering=None,
    dendlen=150,
    dendno=None,
    plot=False
):
    # Separate the clustering params:
    synapse_clustering = clustering
    dendritic_clustering = clustering

    patterns_df = generate_patterns(
        export2file=export2file,
        weights_mat_pc=weights_mat_pc,
        learning_condition=learning_condition,
        min_w=min_w,
        max_w=max_w,
        iters=iters,
        max_depol_val=max_depol_val,
        synapse_location=synapse_location,
        clustering=clustering,
        dendlen=dendlen,
        dendno=dendno,
        synapse_clustering=synapse_clustering,
        dendritic_clustering=dendritic_clustering
    )

    # ta assemblies eiani poly fine tuned. Koita prwta sto diktyo synolika:
    connectivity_mat_pc = get_connectivity_matrix_pc()

    # Per assembly (smaller clusters) statistics:
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

    allpref_assembly_sp_attrs = np.zeros((3, 100), dtype=object)
    allpref_assembly_sp_attrs[:, :] = np.nan
    cluster_no_arr = np.zeros((1, 200), dtype=int)
    cluster_pref_arr = np.zeros((1, 200), dtype=int)
    prev_cluster_no = 0
    pref_idx = 0
    for ap_preference in np.arange(-1000, 1000, 10):
        ap.preference = ap_preference
        ap_result = ap.fit(S)
        idx = ap_result.labels_
        # Maximum cluster no returned by AP:
        tmpk = idx.max() + 1
        if tmpk > prev_cluster_no:
            print(f"Found {tmpk} clusters!")
            prev_cluster_no = tmpk
            cluster_no_arr[0,pref_idx] = tmpk
            cluster_pref_arr[0,pref_idx] = ap_preference
            pref_idx += 1
        else:
            print(f"Continuing! {tmpk}")
            continue

        if tmpk < 250:
            assembly_sp_attrs = np.zeros((3, tmpk), dtype=float)
            assembly_sp_attrs[:, :] = np.nan
            for cluster in range(tmpk):
                mc_idx = np.where(idx == cluster)
                assembly = connectivity_mat_pc[mc_idx[0], :].T[mc_idx[0], :].T
                # get only postsynaptic cells with multiple presynaptics:
                if assembly.size > 1:
                    tmp_assembly_sp_attrs = np.zeros((3, len(mc_idx[0])), dtype=float)
                    tmp_assembly_sp_attrs[:, :] = np.nan
                    for cell in range(len(mc_idx[0])):
                        src_rel = np.where(assembly[:, cell])
                        if src_rel[0].size == 0:
                            continue
                        src = mc_idx[0][src_rel]
                        trg = mc_idx[0][cell]
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
                        tmp_assembly_sp_attrs[0, cell] = PID_vals.mean()

                        # 2)Get std on segments:
                        tmp_arr = []
                        for i in range(dendno):
                            syns_on_same_dend = np.where(D_vals == i)
                            tmp_arr.append(PID_vals[syns_on_same_dend].std())
                        tmp_assembly_sp_attrs[1, cell] = np.nanmean(np.array(tmp_arr))

                        # 3)Get std on dendrites:
                        #This std calculation is consistent because we are
                        # talking about the same dendrites. I.e. if the
                        # majority of synapses are located in a single
                        # dendrite, this will be the same dendrite,
                        # thus keeping the std in correct levels.
                        tmp_assembly_sp_attrs[2, cell] = D_vals.std()
                    assembly_sp_attrs[0,cluster] = np.nanmean(tmp_assembly_sp_attrs[0,
                                                              :])
                    assembly_sp_attrs[1,cluster] = np.nanmean(tmp_assembly_sp_attrs[1,
                                                              :])
                    assembly_sp_attrs[2,cluster] = np.nanmean(tmp_assembly_sp_attrs[2,
                                                              :])

            tmp_1 = np.isfinite(assembly_sp_attrs[0, :])
            tmp_2 = np.isfinite(assembly_sp_attrs[1, :])
            tmp_3 = np.isfinite(assembly_sp_attrs[2, :])


            allpref_assembly_sp_attrs[0,pref_idx] = \
                assembly_sp_attrs[0, tmp_1]
            allpref_assembly_sp_attrs[1,pref_idx] = \
                assembly_sp_attrs[1, tmp_2]
            allpref_assembly_sp_attrs[2,pref_idx] = \
                assembly_sp_attrs[2, tmp_3]

            '''
            allpref_assembly_sp_attrs[0:3,pref_idx] = \
                np.percentile(assembly_sp_attrs[0, tmp_1], [25,50,75]).T
            allpref_assembly_sp_attrs[3:6,pref_idx] = \
                np.percentile(assembly_sp_attrs[1, tmp_2], [25,50,75]).T
            allpref_assembly_sp_attrs[6:9,pref_idx] = \
                np.percentile(assembly_sp_attrs[2, tmp_3], [25,50,75]).T
            '''

        else:
            print("Found max number of assemblies. Terminating...")
            break

    if plot:
        # plot boxplots with the network statistics:
        '''
        fig = plt.figure()
        plot_axes = fig.add_subplot(111)
        # 1) Average distance from soma:
        plot_axes.plot(allpref_assembly_sp_attrs[0, :],'--r')
        plot_axes.plot(allpref_assembly_sp_attrs[1, :],'r')
        plot_axes.plot(allpref_assembly_sp_attrs[2, :],'--r')
        # 2) Average segment clustering:
        plot_axes.plot(allpref_assembly_sp_attrs[3, :],'--g')
        plot_axes.plot(allpref_assembly_sp_attrs[4, :],'g')
        plot_axes.plot(allpref_assembly_sp_attrs[5, :],'--g')
        # 3) Average dend clustering:
        plot_axes.plot(allpref_assembly_sp_attrs[6, :],'--b')
        plot_axes.plot(allpref_assembly_sp_attrs[7, :],'b')
        plot_axes.plot(allpref_assembly_sp_attrs[8, :],'--b')
        #plt.savefig(f"Rev_1_allpref_sp_attrs_stats_clustered.png")
        plt.savefig(
            f"Rev_1_LC{learning_condition}_"
            f"dendno{dendno}_"
            f"dendlen{dendlen}_"
            f"loc{synapse_location}_"
            f"SPCL{synapse_clustering}_U.png"
        )
        '''

    return \
        {'stats': allpref_assembly_sp_attrs,
         'cluster_no': cluster_no_arr,
         'preference': cluster_pref_arr
         }



# I need to generate patterns for ALL the combinations and get their respective:
# 1) detailed synaptic statistics.
# 2) number of attractors.
params={
    "export2file" : False,
    "weights_mat_pc" : None,
    "learning_condition" : [1,2],
    "min_w" : -10,
    "max_w" : 10,
    "iters" : 100000,
    "max_depol_val" : 5,
    "synapse_location" : ['proximal', 'distal'],
    "clustering": [False, True],
    "dendlen" : 150,
    "dendno" : [2,3],
    "plot": True
}

analysis.run_for_all_parameters( run_assembly_statistics,
    **{'auto_param_array': params}
)

print("Tutto pronto!")
df = pd.DataFrame(data)
df.to_hdf('data.hdf5', key='results', mode='w')

# plot each configuration:
for idx, index in enumerate(range(df.shape[0])):
    stats = df.iloc[index]['results']['stats']
    avr_location = np.zeros((3, 10), dtype=object)
    avr_seg = np.zeros((3, 10), dtype=object)
    avr_dend = np.zeros((3, 10), dtype=object)
    for i in range(10):
        avr_location[:, i] = np.percentile(stats[0, i+1], [25,50,75])
        avr_seg[:, i] = np.percentile(stats[1, i+1], [25,50,75])
        avr_dend[:, i] = np.percentile(stats[2, i+1], [25,50,75])

    fig = plt.figure()
    plot_axes = fig.add_subplot(111)
    # 1) Average distance from soma:
    plot_axes.plot(avr_location[0,:], '--r')
    plot_axes.plot(avr_location[1, :], 'r')
    plot_axes.plot(avr_location[2, :], '--r')
    # 2) Average segment clustering:
    plot_axes.plot(avr_seg[0, :], '--g')
    plot_axes.plot(avr_seg[1, :], 'g')
    plot_axes.plot(avr_seg[2, :], '--g')
    # 3) Average dend clustering:
    plot_axes.plot(avr_dend[0, :], '--b')
    plot_axes.plot(avr_dend[1, :], 'b')
    plot_axes.plot(avr_dend[2, :], '--b')
    plt.savefig(
        f"Rev_1_LC{df.iloc[index]['learning_condition']}_"
        f"dendno{df.iloc[index]['dendno']}_"
        f"dendlen{df.iloc[index]['dendlen']}_"
        f"loc{df.iloc[index]['synapse_location']}_"
        f"SPCL{df.iloc[index]['clustering']}_U.png"
    )

# Get the single/multi attractor configurations and compare their statistics.
multi_attr_idx = []
filter = { 'dendno': 2, 'clustering': False, 'synapse_location': 'distal', 'learning_condition': 1 }
multi_attr_idx.append(np.where((df[list(filter)] == pd.Series(
    filter)).all(axis=1))[0][0])
filter = { 'dendno': 2, 'clustering': False, 'synapse_location': 'distal', 'learning_condition': 2 }
multi_attr_idx.append(np.where((df[list(filter)] == pd.Series(
    filter)).all(axis=1))[0][0])
filter = { 'dendno': 2, 'clustering': True, 'synapse_location': 'distal', 'learning_condition': 1 }
multi_attr_idx.append(np.where((df[list(filter)] == pd.Series(
    filter)).all(axis=1))[0][0])
filter = { 'dendno': 3, 'clustering': False, 'synapse_location': 'distal', 'learning_condition': 1 }
multi_attr_idx.append(np.where((df[list(filter)] == pd.Series(
    filter)).all(axis=1))[0][0])
filter = { 'dendno': 3, 'clustering': False, 'synapse_location': 'proximal',
           'learning_condition': 2 }
multi_attr_idx.append(np.where((df[list(filter)] == pd.Series(
    filter)).all(axis=1))[0][0])
filter = { 'dendno': 3, 'clustering': True, 'synapse_location': 'distal',
           'learning_condition': 2 }
multi_attr_idx.append(np.where((df[list(filter)] == pd.Series(
    filter)).all(axis=1))[0][0])

#Plot the differences:
multi_attr_df = df.loc[multi_attr_idx,'results']
avr_location = np.zeros((len(multi_attr_idx),10),dtype=object)
avr_seg = np.zeros((len(multi_attr_idx),10),dtype=object)
avr_dend = np.zeros((len(multi_attr_idx),10),dtype=object)
for i, index in enumerate(range(multi_attr_df.shape[0])):
    stats = multi_attr_df.iloc[index]['stats']
    avr_location[i,:] = stats[0,1:11]
    avr_seg[i,:] = stats[1,1:11]
    avr_dend[i,:] = stats[2,1:11]

percentile_loc = np.zeros((3,avr_location.shape[1]))
percentile_seg = np.zeros((3,avr_location.shape[1]))
percentile_dend = np.zeros((3,avr_location.shape[1]))
for i in range(avr_location.shape[1]):
    percentile_loc[:,i] = \
        np.percentile(np.concatenate(avr_location[:,i], axis=0), [25,50,75])
    percentile_seg[:,i] = \
        np.percentile(np.concatenate(avr_seg[:,i], axis=0), [25,50,75])
    percentile_dend[:,i] = \
        np.percentile(np.concatenate(avr_dend[:,i], axis=0), [25,50,75])

fig = plt.figure()
plot_axes = fig.add_subplot(111)
# 1) Average distance from soma:
plot_axes.plot(percentile_loc[0, :], '--r')
plot_axes.plot(percentile_loc[1, :], 'r')
plot_axes.plot(percentile_loc[2, :], '--r')
# 2) Average segment clustering:
plot_axes.plot(percentile_seg[0, :], '--g')
plot_axes.plot(percentile_seg[1, :], 'g')
plot_axes.plot(percentile_seg[2, :], '--g')
# 3) Average dend clustering:
plot_axes.plot(percentile_dend[0, :], '--b')
plot_axes.plot(percentile_dend[1, :], 'b')
plot_axes.plot(percentile_dend[2, :], '--b')
# plt.savefig(f"Rev_1_allpref_sp_attrs_stats_clustered.png")
plt.savefig(
    f"Rev_1_multi_attractor_U.png"
)

#Plot the differences:
uni_attr_idx = np.ones((1,df.shape[0]), dtype=int)
uni_attr_idx[0,multi_attr_idx]=0
uni_attr_idx = np.where(uni_attr_idx)[1].tolist()
unit_attr_df = df.loc[uni_attr_idx,'results']
avr_location = np.zeros((len(uni_attr_idx),10),dtype=object)
avr_seg = np.zeros((len(uni_attr_idx),10),dtype=object)
avr_dend = np.zeros((len(uni_attr_idx),10),dtype=object)
for i, index in enumerate(range(unit_attr_df.shape[0])):
    stats = unit_attr_df.iloc[index]['stats']
    avr_location[i,:] = stats[0,1:11]
    avr_seg[i,:] = stats[1,1:11]
    avr_dend[i,:] = stats[2,1:11]

percentile_loc = np.zeros((3,avr_location.shape[1]))
percentile_seg = np.zeros((3,avr_location.shape[1]))
percentile_dend = np.zeros((3,avr_location.shape[1]))
for i in range(avr_location.shape[1]):
    percentile_loc[:,i] = \
        np.percentile(np.concatenate(avr_location[:,i], axis=0), [25,50,75])
    percentile_seg[:,i] = \
        np.percentile(np.concatenate(avr_seg[:,i], axis=0), [25,50,75])
    percentile_dend[:,i] = \
        np.percentile(np.concatenate(avr_dend[:,i], axis=0), [25,50,75])

fig = plt.figure()
plot_axes = fig.add_subplot(111)
# 1) Average distance from soma:
plot_axes.plot(percentile_loc[0, :], '--r')
plot_axes.plot(percentile_loc[1, :], 'r')
plot_axes.plot(percentile_loc[2, :], '--r')
# 2) Average segment clustering:
plot_axes.plot(percentile_seg[0, :], '--g')
plot_axes.plot(percentile_seg[1, :], 'g')
plot_axes.plot(percentile_seg[2, :], '--g')
# 3) Average dend clustering:
plot_axes.plot(percentile_dend[0, :], '--b')
plot_axes.plot(percentile_dend[1, :], 'b')
plot_axes.plot(percentile_dend[2, :], '--b')
# plt.savefig(f"Rev_1_allpref_sp_attrs_stats_clustered.png")
plt.savefig(
    f"Rev_1_uni_attractor_U.png"
)


# Network (one huge assembly) does not provide enough insight:
if False:
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
    plt.savefig(f"Rev_1_sp_attrs_stats.png")


