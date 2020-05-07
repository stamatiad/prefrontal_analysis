from pathlib import Path
import re, sys
import numpy as np
import matplotlib.pyplot as plt
import analysis_tools as analysis
from functools import wraps, partial
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

data = []
base_path = Path(r'/home/cluster/stefanos/Documents/Glia')

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
            syn_histo = function(*args, **kwargs)
            array.append(
                {
                    'syn_histo': syn_histo,
                    **kwargs
                }
            )
            #print(f'{function.__name__} took {toc-tic} seconds.')
            return syn_histo
        return wrapped
    return callable

def synaptic_projections_histo(plot=False, **kwargs):
    if False:
        data_dir = base_path.joinpath(
            analysis.simulation_templates['load_iid_ri'](
            **kwargs)
        )
    if True:
        data_dir = base_path.joinpath(
            analysis.simulation_templates['load_ri'](
            **kwargs)
        )
    if not data_dir.is_dir():
        print("Dir non existent!")
        return 1

    # with a glob get all relative files:
    files = list(data_dir.glob('pid_pyramidal_*'))

    pyramidal_projections = np.full((250, 250),None, dtype=object)
    for file in files:
        with open(str(file), 'r') as fid:
            lines = fid.readlines()
            # Put in a sparse mat the list of connections:
            pid_list = []
            for line in lines:
                if 'src' in line:
                    if len(pid_list):
                        pyramidal_projections[int(src), int(trg)] = pid_list
                    _, src, trg = re.split('src=|trg=', line)
                    pid_list = []
                else:
                    pid_list.append(float(re.search('0\.[0-9]*', line)[0]))

    nseg = 2
    pid_histo = np.full((nseg,),0, dtype='int64')
    histo_array = np.empty((250,nseg))
    histo_array[:] = np.nan
    #TODO: THIS IS MADE FOR CLUSTBIAS VERSION!
    for n in range(250):
        for m in range(250):
            if pyramidal_projections[m, n]:
                tmp_histo = \
                    np.histogram(
                        pyramidal_projections[m, n],
                        bins=np.arange(0, 1.2, 1/nseg)
                    )[0]

                pid_histo = np.add(
                    pid_histo,
                    tmp_histo
                )
                histo_array[n,:] = tmp_histo
                break

    pid_histo = np.around(pid_histo/np.sum(pid_histo), decimals=2)
    syn_perc = np.full((11,nseg),0)
    for seg in range(nseg):
        for syn in range(11):
            syn_perc[syn, seg] = np.sum(histo_array[:, seg] == syn)

    syn_perc_cumsum = np.cumsum(syn_perc, axis=0)

    # 5 segment case:
    if False:
        labels = ['1', '2', '3', '4', '5']
        width = 0.35  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()

        ax.bar(labels, syn_perc[0, :].tolist(), width, label=f'syn#0')
        for i in range(1,11):
            ax.bar(labels, syn_perc[i,:].tolist(), width, label=f'syn#{i}',
                   bottom=syn_perc_cumsum[i-1,:].tolist())

        ax.set_ylabel('# synapses')
        ax.set_title('Synapse distribution per segment')
        ax.legend()
        dend_clust_seg = kwargs['dend_clust_seg']
        dend_clust_perc = kwargs['dend_clust_perc']
        plt.savefig(f'syn_dist_DCP{dend_clust_perc}_DCS{dend_clust_seg}.png')

    if True:
        labels = ['1', '2']
        width = 0.35  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()

        ax.bar(labels, syn_perc[0, :].tolist(), width, label=f'syn#0')
        for i in range(1,11):
            ax.bar(labels, syn_perc[i,:].tolist(), width, label=f'syn#{i}',
                   bottom=syn_perc_cumsum[i-1,:].tolist())

        dendno = kwargs['dendno']
        dendlen = kwargs['dendlen']
        learning_condition = kwargs['learning_condition']
        ax.set_ylabel('# synapses')
        ax.set_ylabel('segment id')
        ax.set_title(f'Synapse distribution ({dendno}'
                     f'{dendlen}dend, LC{learning_condition})')
        ax.legend()
        plt.savefig(
            f'syn_dist_{dendno}{dendlen}dend_LC{learning_condition}.png'
        )

    if plot:
        fig, ax = plt.subplots()
        ax.plot(pid_histo)
        plt.savefig('PID_HISTO_CP.PNG')
    return pid_histo


if False:
    # Heavy lifting Clustering:
    read_syn_projections = \
        append_results_to_array(array=data)(
            partial(synaptic_projections_histo,
                    animal_model=1,
                    learning_condition=1,
                    trial=0,
                    ampa_bias=1,
                    postfix='',
                    excitation_bias=1.75,
                    inhibition_bias=2.0,
                    nmda_bias=6.0,
                    sim_duration=5,
                    prefix='iid3_',
                    experiment_config='structured',
                    )
        )

    params = {
        'dend_clust_perc': [25, 50],
        'dend_clust_seg': [0, 2, 4],
        'ri': [50],
    }

if True:
    # MAIN COMPUTE: Morphological:
    read_syn_projections_2 = \
        append_results_to_array(array=data)(
            partial(synaptic_projections_histo,
                    animal_model=1,
                    learning_condition=2,
                    trial=0,
                    ampa_bias=1,
                    postfix='',
                    excitation_bias=1.75,
                    inhibition_bias=3.0,
                    nmda_bias=6.0,
                    sim_duration=5,
                    prefix='ds',
                    template_postfix='_ri',
                    )
        )

    params = {
        'dendlen': [ 'medium'],
        'dendno': [1],
        'connectivity_type': 'structured',
        'ri': [50],
        'ntrials': [1],
    }

analysis.run_for_all_parameters(
    read_syn_projections_2,
    **{'auto_param_array': params}
)

df = pd.DataFrame(data)

with pd.option_context(
        'display.max_rows', None, 'display.max_columns', None
):  # more options can be specified also
    print(df)

sys.exit(0)

fig, ax = plt.subplots()

for index in range(df.shape[0]):
    #NWB_array.append(df.loc[index, 'NWBfile'])
    syn_histo = df.loc[index, 'syn_histo']
    dend_clust_perc = df.loc[index, 'dend_clust_perc']
    dend_clust_seg = df.loc[index, 'dend_clust_seg']
    ax.plot(syn_histo, label=f'DCP{dend_clust_perc} DCS{dend_clust_seg}')

ax.legend()
plt.savefig(f"PID_HISTO_ALL.png")

sys.exit(0)
data_dir = Path(
    '/home/cluster/stefanos/Documents/Glia/'
    #'dsrpSN1LC1TR480_EB1.750_IB3.000_GBF2.000_NMDAb6.000_AMPAb1
    # .000_RI50_structured_1longdend_simdur5'
'SN1LC1TR18_EB1.750_IB3.000_GBF2.000_NMDAb6.000_AMPAb1.000_CP2_structured_allt_simdur5'
)
