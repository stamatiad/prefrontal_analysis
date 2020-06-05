from pathlib import Path
import numpy as np
import sys, re
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


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

base_path = Path(r'/home/cluster/stefanos/Documents/GitHub/prefrontal'
                 r'-micro/experiment/network/synaptic_patterns')

def toFloat(s):
    result = np.NaN
    try:
        result = float(s)
    except Exception as e:
        # suck it up:
        pass
    return result

def toFloats(s):
    tmp = re.findall('[0-9]+\.[0-9]*', s)
    result = list(map(toFloat, tmp))
    return result

def load_synaptic_patterns(
        base_path,
        dendlen=None,
        dendno=None,
        iters=50000,
        min_w=0,
        max_w=4
):
    if not base_path.is_dir():
        print("Dir of voltage files non existent!")
        return 1

    file = base_path.joinpath(
        f'patterns_dendlen_{dendlen}_dendno_{dendno}_minW_'
        f'{min_w}_maxW_{max_w}_iters{iters}'
                              '.txt')
    with open(file, 'r') as fid:
        patterns = np.array(
            list(map(toFloats, fid.readlines()))
        )

    depol_histo, *_ = np.histogram(patterns[:,15], np.arange(0,50,0.1))
    depol_bins = np.digitize(patterns[:,15], np.arange(0,50,0.1))
    pid_mat = patterns[:,0:5]
    w_mat = patterns[:,5:10]
    dend_mat = patterns[:,10:15]

    # Calculate expected values:
    #TODO: I have confused no of segments and no of synapses.
    nseg = 5
    nsyn = 5
    nseg_step = 1/nseg
    segments = np.zeros((patterns.shape[0]*nseg*dendno, 2), dtype=float)
    # Get Weights pdf (uniform, but compute it nontheless):
    new_min_w = np.max([0, min_w])
    weights_bins = np.arange(new_min_w, max_w + 2, 1.0)
    w_histo, *_ = np.histogram(w_mat.flatten(), weights_bins)
    w_histo = np.divide(w_histo, patterns.shape[0] * nseg)
    for i, (PID_vals, W_vals, D_vals) in enumerate(zip(pid_mat, w_mat,
                                                      dend_mat)):
        #histo, *_ = np.histogram(PID_vals, np.arange(0,1.2,0.2))
        # Change the bins range, to accommodate multiple dendrites:
        dend_pid_vals = PID_vals + D_vals
        syn_idx = np.digitize(
            dend_pid_vals, np.arange(0,dendno+nseg_step,nseg_step)
        ) - 1
        for seg in range(nseg*dendno):
            #Ps_idx = np.digitize(W_vals[syn_idx == seg], weights_bins) - 1
            #Ps = w_histo[Ps_idx]
            segments[
                i*nseg*dendno + seg, :
            ] = \
            [
                np.where(syn_idx == seg)[0].size,
                W_vals[syn_idx == seg].sum()
            ]
    # Expected synapses per segment:
    total_syn_bins = np.arange(0, np.ceil(nsyn)+2 , 1.0)
    total_syn_histo, *_, = np.histogram(segments[:,0], total_syn_bins)
    total_syn_histo = np.divide(total_syn_histo, total_syn_histo.sum())
    E_syns = np.ceil(
        np.multiply(total_syn_bins[:-1], total_syn_histo).sum()
    )
    # Expected weight per segment
    total_w_max = segments[:,1].max()
    total_w_bins = np.arange(new_min_w, np.ceil(total_w_max)+2 , 1.0)
    total_w_histo, *_, = np.histogram(segments[:,1], total_w_bins)
    total_w_histo = np.divide(total_w_histo, total_w_histo.sum())
    E_W = np.multiply(total_w_bins[:-1] + 0.5, total_w_histo).sum()

    # Expected values per dendrite:
    dendrites = np.zeros((patterns.shape[0]*dendno, 1), dtype=float)
    dend_bins = np.arange(0,dendno + 2,1)
    for i, D_vals in enumerate(dend_mat):
        histo, *_ = np.histogram(D_vals, dend_bins)
        for dend in range(dendno):
            dendrites[i*dendno + dend, :] = [histo[dend]]

    # Expected synapses per dendrite:
    syn_per_dend_bins = np.arange(0, nsyn+2, 1)
    syn_per_dend_histo, *_, = np.histogram(dendrites[:,0], syn_per_dend_bins)
    syn_per_dend_histo = np.divide(syn_per_dend_histo, syn_per_dend_histo.sum())
    E_dends = np.ceil(
            np.multiply(syn_per_dend_bins[:-1], syn_per_dend_histo).sum()
    )

    if False:
        #This code is to compute statistics.
        #colors = cm.viridis(np.linspace(0, 1,depol_histo.shape[0]))
        #fig, ax = plt.subplots()

        array2 = np.array([])
        for depol_i in range(0,7):
            gid = np.where(depol_bins == depol_i)[0]
            if gid.size == 0:
                continue
            pid_raw = pid_mat[gid, :]
            w_raw = w_mat[gid, :]
            dend_raw = dend_mat[gid, :]
            data_idx = np.argsort(pid_raw, axis=1)

            pid_data = np.zeros(data_idx.shape)
            for i, idx in enumerate(tuple(data_idx)):
                pid_data[i, : ] = pid_raw[i, idx]
            w_data = np.zeros(data_idx.shape)
            for i, idx in enumerate(tuple(data_idx)):
                w_data[i, : ] = w_raw[i, idx]
            dend_data = np.zeros(data_idx.shape)
            for i, idx in enumerate(tuple(data_idx)):
                dend_data[i, : ] = dend_raw[i, idx]
            # I want the soma depol also:
            depol = patterns[depol_bins == depol_i, 10].reshape(-1,1)


            # Do individual histogram:
            array1 = np.full(pid_data.shape, 0)
            for i in range(pid_data.shape[0]):
                hist, *_ = np.histogram(pid_data[i, :], np.arange(0, 1.2, 0.2))
                array1[i, :] = hist


            if False:
                nseg = 5
                nsyn = 5
                syn_perc = np.full((nsyn+1, nseg), 0)
                for seg in range(nseg):
                    for syn in range(nsyn+1):
                        syn_perc[syn, seg] = np.sum(array1[:, seg] == syn)

                syn_perc_cumsum = np.cumsum(syn_perc, axis=0)
                labels = ['1', '2', '3', '4', '5']
                width = 0.35  # the width of the bars: can also be len(x) sequence
                fig1, ax1 = plt.subplots()
                ax1.bar(labels, syn_perc[0, :].tolist(), width, label=f'syn#0')
                for i in range(1,nsyn+1):
                    ax1.bar(labels, syn_perc[i,:].tolist(), width, label=f'syn#{i}',
                           bottom=syn_perc_cumsum[i-1,:].tolist())

                ax1.set_ylabel('# synapses')
                ax1.set_title('Synapse distribution per segment')
                ax1.legend()
                fig1.savefig(f'patterns_syn_dist{depol_i}.png')

                '''
                histo, *_ = np.histogram(pid_data, np.arange(0,1.2,0.2))
                if histo.sum():
                    ax.plot(np.arange(0.1,1.1,0.2), histo/histo.sum(), c=colors[depol_i])
                '''

            # add depol values:
            array1 = np.concatenate((gid.reshape(-1,1), depol, array1), axis=1)
            if array2.size == 0:
                array2 = array1
            else:
                array2 = np.concatenate((array1, array2))

        #plt.savefig(f"histo.png")
        print("TUTTO PRONTO!")

    # Now I need to classify the patterns with easy to use struct:
    #cols: bin (depol mV), location, clustering
    # each col is a list:
    PIDs = []
    Ws = []
    Ds = []
    depols = []
    bins = []
    locations = []
    clusterings = []
    dend_clusterings = []
    for PID_vals, W_vals, D_vals, depol_val, bin_val in zip(
            pid_mat, w_mat, dend_mat, patterns[:, 15], depol_bins):
        PIDs.append(PID_vals)
        Ws.append(W_vals)
        Ds.append(D_vals)
        depols.append(depol_val)
        bins.append(bin_val)

        step = 1/nseg

        histo, *_ = np.histogram(PID_vals, np.arange(0,1.0+step,step))
        #syn_idx = np.digitize(PID_vals, np.arange(0,1.0+step,step)) - 1

        # calculate location bias (proximal, medial distal):
        if histo[:2].sum() > histo[3:].sum():
            location = 'proximal'
        elif histo[:2].sum() < histo[3:].sum():
            location = 'distal'
        else:
            location = 'medial'
        locations.append(location)

        # calculate clustering bias (clustering or dispersed):
        dend_pid_vals = PID_vals + D_vals
        histo_dend_pid,*_ = np.histogram(
            dend_pid_vals, np.arange(0,dendno+nseg_step,nseg_step)
        )
        syn_idx = np.digitize(
            dend_pid_vals, np.arange(0,dendno+nseg_step,nseg_step)
        ) - 1
        # TODO: use expected value to calculate clustering.
        # Definition of clustering: synapses stacked on same segment.
        # The location where histo is > 2 (on syn_idx) must also have
        # max weights per segment greater than expected value
        max_w_per_seg = 0
        for seg in range(nseg*dendno):
            tmp_w = W_vals[np.where(syn_idx == seg)].sum()
            if tmp_w > max_w_per_seg:
                max_w_per_seg = tmp_w

        if histo_dend_pid.max() > E_syns and max_w_per_seg > E_W:
            clustering = True
        else:
            clustering = False
        clusterings.append(clustering)

        # TODO: use expected value to calculate dendritic clustering.
        # Definition of dend clustering: synapses existent on same dendrite:
        if dendno == 1:
            dend_clust = True
        else:
            histo, *_ = np.histogram(D_vals, dend_bins)
            if histo.max() > E_dends:
                dend_clust = True
            else:
                dend_clust = False
        dend_clusterings.append(dend_clust)




    # Combine into a dataframe:
    df = pd.DataFrame({
        'PID': PIDs,
        'W': Ws,
        'D': Ds,
        'somatic_depolarization': depols,
        'somatic_depolarization_bin': bins,
        'synapse_location': locations,
        'synapse_clustering': clusterings,
        'dendritic_clustering': dend_clusterings
    })

    return df



def visualize_vsoma():
    if not base_path.is_dir():
        print("Dir of voltage files non existent!")
        return 1

    files = list(base_path.glob('vsoma*.txt'))

    for i, file in enumerate(files):
        #sn = re.search('[0-9]*', str(file.name))
        sn = int(str(file.name)[6:11])
        if sn == 507:
            with open(file, 'r') as fid:
                vsoma = np.array(
                    list(map(toFloat, fid.readlines()))
                )

            fig, ax = plt.subplots()
            ax.plot(vsoma)
            plt.savefig(f"VSOMA_{sn}.png")

    print("TUTTO PRONTO!")

if __name__ == "__main__":
    load_synaptic_patterns(
        base_path,
        dendlen=30,
        dendno=1,
        iters=100000,
        min_w=-10,
        max_w=10
    )