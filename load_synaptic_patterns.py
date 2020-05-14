from pathlib import Path
import numpy as np
import sys, re
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


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

def load_synaptic_patterns(base_path):
    if not base_path.is_dir():
        print("Dir of voltage files non existent!")
        return 1

    file = base_path.joinpath('patterns_dendlen_150_minW_0_maxW_4_iters50000'
                              '.txt')
    with open(file, 'r') as fid:
        patterns = np.array(
            list(map(toFloats, fid.readlines()))
        )
    depol_histo, *_ = np.histogram(patterns[:,10], np.arange(0,20,1))
    depol_bins = np.digitize(patterns[:,10], np.arange(0,20,1))
    pid_mat = patterns[:,0:5]
    w_mat = patterns[:,5:10]

    colors = cm.viridis(np.linspace(0, 1,depol_histo.shape[0]))
    #fig, ax = plt.subplots()

    array2 = np.array([])
    for depol_i in range(0,7):
        gid = np.where(depol_bins == depol_i)[0]
        if gid.size == 0:
            continue
        pid_raw = pid_mat[gid, :]
        w_raw = w_mat[gid, :]
        data_idx = np.argsort(pid_raw, axis=1)

        pid_data = np.zeros(data_idx.shape)
        for i, idx in enumerate(tuple(data_idx)):
            pid_data[i, : ] = pid_raw[i, idx]
        w_data = np.zeros(data_idx.shape)
        for i, idx in enumerate(tuple(data_idx)):
            w_data[i, : ] = w_raw[i, idx]
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
    depols = []
    bins = []
    locations = []
    clusterings = []
    for PID_vals, W_vals, depol_val, bin_val in zip(
            patterns[:, :5], patterns[:, 5:10], patterns[:, 10], depol_bins):
        PIDs.append(PID_vals)
        Ws.append(W_vals)
        depols.append(depol_val)
        bins.append(bin_val)

        histo, *_ = np.histogram(PID_vals, np.arange(0,1.2,0.2))

        # calculate location bias (proximal, medial distal):
        if histo[:2].sum() > histo[3:].sum():
            location = 'proximal'
        elif histo[:2].sum() < histo[3:].sum():
            location = 'distal'
        else:
            location = 'medial'
        locations.append(location)

        # calculate clustering bias (clustering or dispersed):
        #TODO: calculate
        if histo.max() > 2:
            clustering = True
        else:
            clustering = False
        clusterings.append(clustering)

    # Combine into a dataframe:
    df = pd.DataFrame({
        'PID': PIDs,
        'W': Ws,
        'somatic_depolarization': depols,
        'somatic_depolarization_bin': bins,
        'synapse_location': locations,
        'synapse_clustering': clusterings
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
        if sn == 267:
            with open(file, 'r') as fid:
                vsoma = np.array(
                    list(map(toFloat, fid.readlines()))
                )

            fig, ax = plt.subplots()
            ax.plot(vsoma)
            plt.savefig(f"VSOMA_{sn}.png")

    print("TUTTO PRONTO!")

if __name__ == "__main__":
    load_synaptic_patterns(base_path)