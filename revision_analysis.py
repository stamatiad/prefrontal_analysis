# %% Imports

import analysis_tools as analysis
import numpy as np
from numpy import matlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
#import matplotlib
#matplotlib.use('Qt5Agg')
#plt = matplotlib.pyplot
from collections import defaultdict
from functools import partial
from pathlib import Path
from pynwb import NWBHDF5IO
from itertools import chain
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
from sklearn.linear_model import LinearRegression
import seaborn as sb
import math
import pandas as pd
from scipy import stats
from itertools import chain
import sys
from functools import wraps
from collections import OrderedDict
import copy

# ===%% Pycharm debug: %%===
import pydevd_pycharm
sys.path.append("pydevd-pycharm.egg")
DEBUG = True
if DEBUG:
    pydevd_pycharm.settrace(
        '79.167.94.93',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===

# %% Initialization
if DEBUG:
    matplotlib.interactive('on')
    matplotlib.use('Qt5Agg')
else:
    matplotlib.interactive('off')
print(f'Matplotlib is interactive: {matplotlib.is_interactive()}')
print(matplotlib.rcParams['backend'])

glia_dir = Path("/home/cluster/stefanos/Documents/Glia")

cp_array = [2, 3, 4, 5, 6, 7]
ri_array = [125, 83, 62, 41, 35]
sparse_cp_trials = lambda cp: (cp - 1) * 10 + 1
cp_trials_len = 0
for cp in cp_array:
    cp_trials_len += cp
NWBfiles_1smalldend = []
NWBfiles_2smalldend = []
NWBfiles_1mediumdend = []
NWBfiles_2mediumdend = []
NWBfiles_1longdend = []
NWBfiles_2longdend = []

data = []

dcp_array = [25, 50, 75, 100]
dcs_array = [0, 1, 2]


'''
def blah(*args, **kwargs):
    print(f'Here are my args:')
    for arg in args:
        print(f'arg: {arg}')
    print(f'Here are my kwargs:')
    for k,v in kwargs.items():
        print(f'Key: {k}, value {v}.')

params = {'param1':[1,2,3], 'param2':[4,5]}
analysis.run_for_all_parameters(blah, 2, **{'param0': 0, 'auto_param_array':
    params})
'''


# %% Load NEURON data
# this is exploratory to see where in the intermediate connectivity I need to
# change the E/I in order to get PA:
# This partial combines the simulations ran:
anatomical_cluster_iid_sims_intermediate = \
    analysis.append_results_to_array(array=data)(
        partial(analysis.load_nwb_from_neuron,
                glia_dir,
                excitation_bias=1.75,
                inhibition_bias=1.5,
                nmda_bias=6.0,
                sim_duration=2,
                prefix='iid2_',
                template_postfix='_iid'
                )
    )

# Load each run simulation:
dcp_array = [25, 100]
dcs_array = [0, 1, 2]
params = {'dend_clust_perc': dcp_array, 'dend_clust_seg': dcs_array,
          'cp': [2], 'ntrials': 11,
          'dendlen': ['small','medium','long'],
          'dendno': [1,2],
          'connectivity_type': [
             'intermediate_60',
              'intermediate_80',
          ]}

analysis.run_for_all_parameters(
    anatomical_cluster_iid_sims_intermediate,
    **{'auto_param_array': params}
)

def query_and_add_pa_column(dataframe, **kwargs):
    '''
    Queries the dataframe NWB objects and calculates the PA for each one (
    supports multiple runs on a single NWB file).
    Saves the results on a new column on the dataframe named 'PA'.
    ATTENTION: kwargs should be only the dataframe query terms! So the kwargs is
    the equivalent of the 'filter' dictionary for the dataframe!
    '''
    #TODO: implement something to filter out nonexistend keys/columns.
    #df_keys = set(df.columns.values.astype(str))
    #query_keys = set(filter_d.keys())
    ## Remove nonexistent query keys:
    #query_keys - df_keys

    #dataframe.loc[(dataframe[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]
    nwb_index = (dataframe[list(kwargs)] == pd.Series(kwargs)).all(axis=1)
    # this should be a single entry!
    if nwb_index.values.astype(int).nonzero()[0].size > 1:
        try:
            raise ValueError
        except ValueError:
            print('Multiple dataframe rows correspond to this query! PA '
                  'values will be wrong! Exiting...')
    df_index = nwb_index.values.astype(int).nonzero()[0][0]
    # Get the corresponding NWB file:
    NWBfile_array = dataframe[
        nwb_index
    ]['NWBfile'].values.tolist()
    # Get NWB's trials PA:
    trials_pa = list(analysis.nwb_iter(NWBfile_array[0].trials[
                                           'persistent_activity']))
    # Save back to dataframe the PA percentage:
    dataframe.at[df_index, 'PA'] = np.array(trials_pa).astype(int).mean()

df = pd.DataFrame(data)
# Use len for speed and float since we get a percentage:
df['PA'] = [-1.0] * len(df.index)

analysis.run_for_all_parameters(
    query_and_add_pa_column,
    df,
    **{'auto_param_array': params}
)

# make a function that will traverse the dataframe and add a PA column with
# the average PA on each NWB file.
'''
simulation = f'iid2_SN1LC1TR0_EB1.750_IB1' \
             f'.500_GBF2.000_NMDAb6.000_AMPAb1' \
             f'.000_CP2_DCP100_DCS2_intermediate_60_1smalldend_simdur2'
filter_d = {
    'dend_clust_perc': 100, 'dend_clust_seg': 2,
    'cp': 2, 'ntrials': 11,
    'dendlen': 'small',
    'dendno': 2,
    'connectivity_type': 'intermediate_60',
}
nwb_index = (df[list(filter_d)] == pd.Series(filter_d)).all(axis=1)
df.loc[nwb_index]
'''

dendno_array = [1,2]
dendlen_array = ['small', 'medium', 'long']

# Do this list a picture with named axis:
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
for dn, dendno in enumerate(dendno_array):
    for dl, dendlen in enumerate(dendlen_array):
        pa_array = np.zeros((len(dcp_array), len(dcs_array)), dtype=float)
        for i, dcp in enumerate(dcp_array):
            for j, dcs in enumerate(dcs_array):
                # This is the most anti-pythonic syntax ever:
                df_indices = \
                    (df['cp'] == 2) \
                    & (df['dend_clust_perc'] == dcp) \
                    & (df['dend_clust_seg'] == dcs) \
                    & (df['dendno'] == dendno) \
                    & (df['dendlen'] == dendlen) \
                    & (df['connectivity_type'] == 'intermediate_80')

                pa_array[i,j] = df.at[df_indices.values.astype(int).nonzero()[
                                       0][0], 'PA']
        axes[dn][dl].imshow(pa_array, cmap='Blues', vmin=0.0, vmax=1.0)
        print(f'on dend no {dn}, dend len {dl}:')
        print(pa_array)

for i, (row, dendno) in enumerate(zip(axes, dendno_array)):
    for j, (cell, dendlen) in enumerate(zip(row, dendlen_array)):
        if i == len(axes) - 1:
            cell.set_xlabel(f"Segment ({dendlen})")
        if j == 0:
            cell.set_ylabel(f"percent (#dend:{dendno})")

fig.savefig(f'Intermediate_80_CP2.png')
plt.close()
## add a big axis, hide frame
#fig.add_subplot(111, frameon=False)
## hide tick and tick label of the big axis
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
#                right=False)
#plt.xlabel('small medium long')
#plt.ylabel('number of dendrites')

print("Tutto pronto! Exiting now...")

sys.exit(0)
print('Done exploration')

# This partial combines the simulations ran:
anatomical_cluster_iid_sims_intermediate = \
    analysis.append_results_to_array(array=data)(
    partial(analysis.load_nwb_from_neuron,
            glia_dir,
            excitation_bias=1.0,
            inhibition_bias=2.3,
            nmda_bias=6.0,
            sim_duration=5,
            prefix='iid2_',
            template_postfix='_iid'
            )
)

# Load each run simulation:
dcp_array = [25, 100]
dcs_array = [0, 1, 2]
params = {'dend_clust_perc': dcp_array, 'dend_clust_seg': dcs_array,
          'cp': [2], 'ntrials': 11,
          'dendlen': ['small','medium','long'],
          'dendno': [1,2],
          'connectivity_type': [
              'intermediate_20',
              'intermediate_40',
              'intermediate_60',
              'intermediate_80',
          ]}

analysis.run_for_all_parameters(
    anatomical_cluster_iid_sims_intermediate,
    **{'auto_param_array': params}
)

df = pd.DataFrame(data)
list(df.columns.values)
# Use len for speed:
PA = [0.0] * len(df.index)
df['PA'] = PA

# make a function that will traverse the dataframe and add a PA column with
# the average PA on each NWB file.

dendno_array = [1,2]
dendlen_array = ['small', 'medium', 'long']
# Do this list a picture with named axis:
for p, cp in enumerate([2]):
    for dn, dendno in enumerate(dendno_array):
        for dl, dendlen in enumerate(dendlen_array):
            for i, dcp in enumerate(dcp_array):
                for j, dcs in enumerate(dcs_array):
                    # This is the most anti-pythonic syntax ever:
                    df_indices = \
                    (df['cp'] == cp) \
                    & (df['dend_clust_perc'] == dcp) \
                    & (df['dend_clust_seg'] == dcs) \
                    & (df['dendno'] == dendno) \
                    & (df['dendlen'] == dendlen)

                    NWBfile_array = df[
                            df_indices
                        ]['NWBfile'].values.tolist()
                    pa_values = list(analysis.nwb_iter(NWBfile_array[0].trials[
                                                           'persistent_activity']))
                    df.at[df_indices.values.astype(int).nonzero()[0][0],
                          'PA'] = np.array( pa_values).astype(int).mean()

print("Tutto pronto! Exiting now...")
sys.exit(0)

sys.exit(0)

# This partial combines the simulations ran:
anatomical_cluster_iid_simulations = \
    analysis.append_results_to_array(array=data)(
        partial(analysis.load_nwb_from_neuron,
                glia_dir,
                excitation_bias=1.0,
                inhibition_bias=2.0,
                nmda_bias=6.0,
                sim_duration=5,
                prefix='iid2_',
                template_postfix='_iid'
                )
    )

params = {'dend_clust_perc':dcp_array, 'dend_clust_seg':dcs_array,
          'cp': cp_array, 'ntrials': 71,
          'dendlen': ['small','medium','long'],
          'dendno': [1,2],
          'connectivity_type': ['structured']}

analysis.run_for_all_parameters(
    anatomical_cluster_iid_simulations,
    **{'auto_param_array': params}
)

df = pd.DataFrame(data)
list(df.columns.values)

sys.exit(0)

for dcp in (dcp_array):
    for dcs in (dcs_array):
        for p, cp in enumerate(cp_array):
            NWBfile = anatomical_cluster_iid_simulations(
                ntrials=sparse_cp_trials(cp),
                cp=cp,
                #experiment_config='structured_1smalldend',
                dend_clust_perc=dcp,
                dend_clust_seg=dcs
            )
            if NWBfile:
                # NWBfiles_1smalldend.append(NWBfile)
                data.append(
                    {'NWBfile': NWBfile, 'cp': cp, 'dend_clust_perc': dcp,
                     'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'small'}
                )


def load_runs_ri_search():
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.5,
                        'nmda_bias': 6.0,
                        'ntrials': 1,
                        'sim_duration': 2,
                        'ri': ri,
                        'experiment_config': 'structured_1smalldend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_1smalldend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'small'}
                    )

    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.5,
                        'nmda_bias': 6.0,
                        'ntrials': 1,
                        'sim_duration': 2,
                        'ri': ri,
                        'experiment_config': 'structured_2smalldend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_2smalldend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'small'}
                    )

    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.5,
                        'nmda_bias': 6.0,
                        'ntrials': 1,
                        'sim_duration': 2,
                        'ri': ri,
                        'experiment_config': 'structured_1mediumdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'medium'}
                    )

    #NWBfiles_2mediumdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.5,
                        'nmda_bias': 6.0,
                        'ntrials': 1,
                        'sim_duration': 2,
                        'ri': ri,
                        'experiment_config': 'structured_2mediumdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_2mediumdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'medium'}
                    )

    #NWBfiles_1longdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.5,
                        'nmda_bias': 6.0,
                        'ntrials': 1,
                        'sim_duration': 2,
                        'ri': ri,
                        'experiment_config': 'structured_1longdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_1longdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'long'}
                    )

    #NWBfiles_2longdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.5,
                        'nmda_bias': 6.0,
                        'ntrials': 1,
                        'sim_duration': 2,
                        'ri': ri,
                        'experiment_config': 'structured_2longdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_2longdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'long'}
                    )

def load_neuron_runs():
    #NWBfiles_1smalldend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
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
                        'experiment_config': 'structured_1smalldend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid'
                )
                if NWBfile:
                    #NWBfiles_1smalldend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'cp': cp, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'small'}
                    )

    # %%

    #NWBfiles_2smalldend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
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
                        'experiment_config': 'structured_2smalldend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid'
                )
                if NWBfile:
                    #NWBfiles_2smalldend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'cp': cp, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'small'}
                    )

    #NWBfiles_1mediumdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
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
                        'experiment_config': 'structured_1mediumdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid'
                )
                if NWBfile:
                    data.append(
                        {'NWBfile': NWBfile, 'cp': cp, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'medium'}
                    )

    #NWBfiles_2mediumdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
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
                        'experiment_config': 'structured_2mediumdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid'
                )
                if NWBfile:
                    #NWBfiles_2mediumdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'cp': cp, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'medium'}
                    )

    #NWBfiles_1longdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
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
                        'experiment_config': 'structured_1longdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid'
                )
                if NWBfile:
                    #NWBfiles_1longdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'cp': cp, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'long'}
                    )

    #NWBfiles_2longdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
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
                        'experiment_config': 'structured_2longdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid'
                )
                if NWBfile:
                    #NWBfiles_2longdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'cp': cp, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'long'}
                    )

def load_neuron_runs_ri():
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.0,
                        'nmda_bias': 6.0,
                        'ntrials': 11,
                        'sim_duration': 5,
                        'ri': ri,
                        'experiment_config': 'structured_1smalldend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_1smalldend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'small'}
                    )

    # %%

    #NWBfiles_2smalldend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.0,
                        'nmda_bias': 6.0,
                        'ntrials': 11,
                        'sim_duration': 5,
                        'ri': ri,
                        'experiment_config': 'structured_2smalldend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_2smalldend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'small'}
                    )

    #NWBfiles_1mediumdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.0,
                        'nmda_bias': 6.0,
                        'ntrials': 11,
                        'sim_duration': 5,
                        'ri': ri,
                        'experiment_config': 'structured_1mediumdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'medium'}
                    )

    #NWBfiles_2mediumdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.0,
                        'nmda_bias': 6.0,
                        'ntrials': 11,
                        'sim_duration': 5,
                        'ri': ri,
                        'experiment_config': 'structured_2mediumdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_2mediumdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'medium'}
                    )

    #NWBfiles_1longdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.0,
                        'nmda_bias': 6.0,
                        'ntrials': 11,
                        'sim_duration': 5,
                        'ri': ri,
                        'experiment_config': 'structured_1longdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_1longdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 1, 'dendlen': 'long'}
                    )

    #NWBfiles_2longdend = []
    for dcp in (dcp_array):
        for dcs in (dcs_array):
            for p, ri in enumerate(ri_array):
                NWBfile = analysis.load_nwb_from_neuron(
                    glia_dir,
                    reload_raw=False,
                    new_params={
                        'excitation_bias': 1.0,
                        'inhibition_bias': 2.0,
                        'nmda_bias': 6.0,
                        'ntrials': 11,
                        'sim_duration': 5,
                        'ri': ri,
                        'experiment_config': 'structured_2longdend',
                        'prefix': 'iid2_',
                        'dend_clust_perc': dcp,
                        'dend_clust_seg': dcs
                    },
                    template_postfix='_iid_ri'
                )
                if NWBfile:
                    #NWBfiles_2longdend.append(NWBfile)
                    data.append(
                        {'NWBfile': NWBfile, 'ri': ri, 'dend_clust_perc': dcp,
                         'dend_clust_seg': dcs, 'dendno': 2, 'dendlen': 'long'}
                    )

# %%

print("Commencing analysis")
# Load directly the NWB files:
load_neuron_runs()
#load_neuron_runs_ri()
#load_runs_ri_search()
#sys.exit(0)

# Check convergence attractors for each parameter:
df = pd.DataFrame(data)


dendno_array = [1,2]
dendlen_array = ['small', 'medium', 'long']
# Do this list a picture with named axis:
for p, cp in enumerate(cp_array):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    for dn, dendno in enumerate(dendno_array):
        for dl, dendlen in enumerate(dendlen_array):
            pa_array = np.zeros((len(dcp_array), len(dcs_array)), dtype=int)
            for i, dcp in enumerate(dcp_array):
                for j, dcs in enumerate(dcs_array):
                    # This is the most anti-pythonic syntax ever:
                    NWBfile_array = df[
                        (df['cp'] == cp)
                        & (df['dend_clust_perc'] == dcp)
                        & (df['dend_clust_seg'] == dcs)
                        & (df['dendno'] == dendno)
                        & (df['dendlen'] == dendlen)
                        ]['NWBfile'].values.tolist()
                    pa_values = list(analysis.nwb_iter(NWBfile_array[0].trials[
                                                           'persistent_activity']))
                    pa_array[i, j] = np.array(pa_values).astype(int).mean()
            axes[dn][dl].imshow(pa_array)

    for i, (row, dendno) in enumerate(zip(axes, dendno_array)):
        for j, (cell, dendlen) in enumerate(zip(row, dendlen_array)):
            if i == len(axes) - 1:
                cell.set_xlabel(f"Segment ({dendlen})")
            if j == 0:
                cell.set_ylabel(f"percent (#dend:{dendno})")

    fig.savefig(f'CP_{cp}.png')
    plt.close()

print("Tutto pronto! Exiting now...")
sys.exit(0)


dcp_array = [25,100]
# Do this list a picture with named axis:
for p, ri in enumerate(ri_array):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    for dn, dendno in enumerate(dendno_array):
        for dl, dendlen in enumerate(dendlen_array):
            pa_array = np.zeros((len(dcp_array), len(dcs_array)), dtype=int)
            for i, dcp in enumerate(dcp_array):
                for j, dcs in enumerate(dcs_array):
                    # This is the most anti-pythonic syntax ever:
                    NWBfile_array = df[
                               (df['ri'] == ri)
                               & (df['dend_clust_perc'] == dcp)
                               & (df['dend_clust_seg'] == dcs)
                               & (df['dendno'] == dendno)
                               & (df['dendlen'] == dendlen)
                               ]['NWBfile'].values.tolist()
                    pa_values = list(analysis.nwb_iter(NWBfile_array[0].trials[
                                                      'persistent_activity']))
                    pa_array[i, j] = pa_values[0]
            axes[dn][dl].imshow(pa_array)
            #ax = plt.subplot(2, 3, dn*3 + dl+1)
            #ax.imshow(pa_array)

    for i, (row, dendno) in enumerate(zip(axes, dendno_array)):
        for j, (cell, dendlen) in enumerate(zip(row, dendlen_array)):
            if i == len(axes) - 1:
                cell.set_xlabel(f"Segment ({dendlen})")
            if j == 0:
                cell.set_ylabel(f"percent (#dend:{dendno})")

    fig.savefig(f'CP_{ri}.png')
    plt.close()
    ## add a big axis, hide frame
    #fig.add_subplot(111, frameon=False)
    ## hide tick and tick label of the big axis
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
    #                right=False)
    #plt.xlabel('small medium long')
    #plt.ylabel('number of dendrites')

print("Tutto pronto! Exiting now...")
sys.exit(0)

print('done with that')

# =========================================================
#load individual runs.
# load the voltage traces to see if you miss something:
simulation = f'iid2_SN1LC1TR0_EB1.750_IB1' \
             f'.500_GBF2.000_NMDAb6.000_AMPAb1' \
             f'.000_CP2_DCP100_DCS2_intermediate_60_1smalldend_simdur2'
simulation2 = f'iid2_SN1LC1TR0-10_EB1.750_IB1' \
             f'.500_GBF2.000_NMDAb6.000_AMPAb1' \
             f'.000_CP2_DCP100_DCS2_intermediate_60_1smalldend_simdur2'
inputfile=f'/home/cluster/stefanos/Documents/Glia/{simulation}/vsoma.hdf5'
nwbfile=f'/home/cluster/stefanos/Documents/Glia/{simulation2}.nwb'
NWBfile = NWBHDF5IO(str(nwbfile), 'r').read()

voltage_traces = pd.read_hdf(inputfile, key='vsoma').values
fig, ax = plt.subplots()
ax.plot(voltage_traces[25,:])
plt.pause(1)


binned_network_activity = NWBfile. \
                              acquisition['binned_activity']. \
                              data[:250, :]. \
    reshape(250, 1, 40)

fig1,ax1 = plt.subplots()
ax1.imshow(binned_network_activity[:50,0,:])
plt.pause(1)
# =========================================================

analysis.compare_dend_params([
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 25)
        & (df['dend_clust_seg'] == 0) ]['NWBfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 25)
       & (df['dend_clust_seg'] == 1)]['NWBfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 25)
       & (df['dend_clust_seg'] == 2)]['NWBfile'].values.tolist(),
],[
    '25-proximal',
    '25-medial',
    '25-distal'
])



analysis.compare_dend_params([
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 50) & (df['dend_clust_seg'] == 0)]['nwbfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 50) & (df['dend_clust_seg'] == 1)]['NWBfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 50) & (df['dend_clust_seg'] == 2)]['NWBfile'].values.tolist()
],[
    '50-proximal',
    '50-medial',
    '50-distal'
])

analysis.compare_dend_params([
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 75) & (df['dend_clust_seg'] == 0)]['NWBfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 75) & (df['dend_clust_seg'] == 1)]['NWBfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 75) & (df['dend_clust_seg'] == 2)]['NWBfile'].values.tolist()
],[
    '75-proximal',
    '75-medial',
    '75-distal'
])

analysis.compare_dend_params([
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 100) & (df['dend_clust_seg'] == 0)]['NWBfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 100) & (df['dend_clust_seg'] == 1)]['NWBfile'].values.tolist(),
    df[(df['cp'] == 2) & (df['dend_clust_perc'] == 100) & (df['dend_clust_seg'] == 2)]['NWBfile'].values.tolist()
],[
    '100-proximal',
    '100-medial',
    '100-distal'
])

print("Tutto pronto!")
# Show the attractor landscape for different type of dendrites.
analysis.compare_dend_params([
    NWBfiles_1smalldend,
    NWBfiles_1mediumdend,
    NWBfiles_1longdend
],[
    'NWBfiles_1smalldend',
    'NWBfiles_1mediumdend',
    'NWBfiles_1longdend'
])

print("Tutto pronto!")
