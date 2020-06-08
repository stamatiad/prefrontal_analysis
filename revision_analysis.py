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
import time
from sklearn import linear_model

# ===%% Pycharm debug: %%===
import pydevd_pycharm
sys.path.append("pydevd-pycharm.egg")
DEBUG = False
if DEBUG:
    pydevd_pycharm.settrace(
        '79.167.48.178',
        port=12345,
        stdoutToServer=True,
        stderrToServer=True
    )
# ===%% -------------- %%===

# %% Initialization
if DEBUG:
    matplotlib.interactive('off')
    #matplotlib.interactive('on')
    #matplotlib.use('Qt5Agg')
else:
    matplotlib.interactive('off')

print(f'Matplotlib is interactive: {matplotlib.is_interactive()}')
print(matplotlib.rcParams['backend'])

glia_dir = Path("/home/cluster/stefanos/Documents/Glia")

cp_array = [2, 3, 4, 5, 6, 7]
ri_array = [125, 83, 62, 41, 35]

# You need a version with kwarg!
def sparse_cp_trials(cp=None, **kwargs):
    return (cp - 1) * 10 + 1

def all_cp_trials(cp=None, **kwargs):
    return cp * 10

# This is our main data array, later to be converted to dataframe:
data = []

dcp_array = [25, 50, 75, 100]
dcs_array = [0, 2, 4]

# LOAD Homogeneous data:runs
if True:
    homogeneous_analysis = \
        analysis.append_results_to_array(array=data)(
            partial(analysis.load_nwb_from_neuron,
                    glia_dir,
                    excitation_bias=1.75,
                    nmda_bias=6.0,
                    sim_duration=5,
                    prefix='sp',
                    template_postfix='_sp_ri',
                    connectivity_type='structured',
                    ri=50,
                    ntrials=200,
                    reload_raw=False
            )
        )

    params = {
        'inhibition_bias': [2.0], #np.arange(1.0, 3.5, 0.5).tolist(),
        'wr': [4], #np.arange(1,10,1).tolist(),
        'dendlen':['small'],
        'dendno': [2,3],
        'sploc': ['proximal'],
        'spcl': [ 1],
        'spdcl': [ 1],
        'learning_condition': [1,2,3,4,5],
    }

    analysis.run_for_all_parameters(
        homogeneous_analysis,
        **{'auto_param_array': params}
    )

    df = pd.DataFrame(data)
    df['PA'] = [-1.0] * len(df.index)
    df['sparsness'] = [-1.0] * len(df.index)
    df['attractors_len'] = [-1.0] * len(df.index)
    analysis.run_for_all_parameters(
        analysis.query_and_add_pa_column,
        df,
        **{'auto_param_array': params}
    )
    analysis.run_for_all_parameters(
        analysis.query_and_add_sparsness_column,
        df,
        **{'auto_param_array': params}
    )
    analysis.run_for_all_parameters(
        analysis.query_and_add_attractors_len_column,
        df,
        **{'auto_param_array': params}
    )
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):  # more options can be specified also
        print(df)

    #sys.exit(0)

    # Plot:
    for index in range(df.shape[0]):
        NWBfile = df.loc[index, 'NWBfile']
        if NWBfile:
            dendlen = df.loc[index, 'dendlen']
            dendno = df.loc[index, 'dendno']
            inhibition_bias = df.loc[index, 'inhibition_bias']
            WR = df.loc[index, 'wr']
            spcl = df.loc[index, 'spcl']
            spdcl = df.loc[index, 'spdcl']
            sploc = df.loc[index, 'sploc']
            PA = df.loc[index, 'PA']
            LC = df.loc[index, 'learning_condition']
            trial_len = analysis.get_acquisition_parameters(
                input_NWBfile=NWBfile,
                requested_parameters=['trial_len']
            )
            delay_range = (20, int(trial_len / 50))
            K_star, K_labels, *_ = analysis.determine_number_of_clusters(
                NWBfile_array=[NWBfile],
                max_clusters=20,
                custom_range=delay_range
            )
            fig = plt.figure()
            plot_axes = fig.add_subplot(111)
            try:
                analysis.pcaL2(
                    NWBfile_array=[NWBfile],
                    custom_range=delay_range,
                    klabels=K_labels,
                    smooth=True,
                    plot_2d=True,
                    plot_stim_color=True,
                    plot_axes=plot_axes,
                )
            except ValueError:
                pass

            '''
            plt.savefig(f"SPInhom_Attr_{dendno}{dendlen}dend_"
                        f"IB{inhibition_bias}_"
                        f"PA{PA}_LC_{LC}_rw.png")
            '''
            plt.savefig(f"SP_Attr_{dendno}{dendlen}dend_spcl{int(spcl)}"
                        f"_sploc{sploc}_IB{inhibition_bias}_"
                        f"PA{PA}_LC_{LC}_WR{WR}.png")
    sys.exit(0)

    # This is a simple boxplot with each case:
    # It is not very informative about the specific relations of
    # dependent / independent variables.
    case_l = [
        {'dendlen': 'small'},
        {'dendlen': 'original'},
        {'spcl': 0, 'spdcl': 0},
        {'spcl': 1, 'spdcl': 1},
        {'sploc': 'proximal'},
        {'sploc': 'distal'},
    ]
    case_data = []
    case_label = []
    for case in case_l:
        nwb_index = (df[list(case)] == pd.Series(case)).all(axis=1)
        attr_len = df.loc[nwb_index, 'attractors_len'].values.tolist()
        case_data.append(attr_len)
        case_label.append(
            f"{next(iter(case.keys()))}:"
            f"{next(iter(case.values()))}"
        )

    fig = plt.figure()
    plot_axes = fig.add_subplot(111)
    plot_axes.boxplot(case_data)
    plot_axes.set_xticklabels(case_label, rotation=45, fontsize=8)
    fig.tight_layout()
    plt.savefig(f"TEST_small_cases.png")

    # THis is the detailed/individual cases
    case_l = [
        [
            {'dendlen': 'small', 'dendno': 2, 'inhibition_bias':2.0,
             'spcl': 0, 'spdcl': 0, 'sploc': 'proximal'},
            {'dendlen': 'small', 'dendno': 3, 'inhibition_bias': 2.0,
             'spcl': 0, 'spdcl': 0, 'sploc': 'proximal'},
            {'dendlen': 'original', 'dendno': 2, 'inhibition_bias': 2.0,
             'spcl': 0, 'spdcl': 0, 'sploc': 'proximal'},
            {'dendlen': 'original', 'dendno': 3, 'inhibition_bias': 2.0,
             'spcl': 0, 'spdcl': 0, 'sploc': 'proximal'},
        ],
        [
            {'dendlen': 'small', 'dendno': 2, 'inhibition_bias': 1.8,
             'spcl': 0, 'spdcl': 0, 'sploc': 'distal'},
            {'dendlen': 'small', 'dendno': 3, 'inhibition_bias': 2.0,
             'spcl': 0, 'spdcl': 0, 'sploc': 'distal'},
            {'dendlen': 'original', 'dendno': 2, 'inhibition_bias': 2.0,
             'spcl': 0, 'spdcl': 0, 'sploc': 'distal'},
            {'dendlen': 'original', 'dendno': 3, 'inhibition_bias': 1.8,
             'spcl': 0, 'spdcl': 0, 'sploc': 'distal'},
        ],
        [
            {'dendlen': 'small', 'dendno': 2, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'proximal'},
            {'dendlen': 'small', 'dendno': 3, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'proximal'},
            {'dendlen': 'original', 'dendno': 2, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'proximal'},
            {'dendlen': 'original', 'dendno': 3, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'proximal'},
        ],
        [
            {'dendlen': 'small', 'dendno': 2, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'distal'},
            {'dendlen': 'small', 'dendno': 3, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'distal'},
            {'dendlen': 'original', 'dendno': 2, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'distal'},
            {'dendlen': 'original', 'dendno': 3, 'inhibition_bias': 2.0,
             'spcl': 1, 'spdcl': 1, 'sploc': 'distal'},
        ],
    ]
    case_data = []
    for i, subcases in enumerate(case_l):
        case_data.append([])
        for case in subcases:
            nwb_index = (df[list(case)] == pd.Series(case)).all(axis=1)
            attr_len = df.loc[nwb_index, 'attractors_len'].values.tolist()
            case_data[i].append(attr_len[0])

    fig = plt.figure()
    plot_axes = fig.add_subplot(111)

    x_labels = ['small, 2dend', 'small, 3dend', 'original, 2dend', 'original, '
                                                                 '3dend']
    bar_labels = ['proximal, no cluster', 'distal, no cluster',
                  'proximal, cluster', 'distal, cluster']
    fill_arr = [False, False, True, True]
    x = np.arange(len(x_labels))  # the label locations
    width = 0.5  # the width of the bars

    color_map = cm.get_cmap('Set1')
    colors = color_map(np.linspace(0,1,9))
    for i in range(4):
        _ = plot_axes.bar(x + i*(width / 4), case_data[i], width/4,
                     label=bar_labels[i], fill=fill_arr[i],
                          edgecolor=colors[i%2], facecolor=colors[i%2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plot_axes.set_ylabel('# of attractors')
    plot_axes.set_title('Attractors by dendritic configuration')
    plot_axes.set_xticks(x)
    plot_axes.set_xticklabels(x_labels, rotation=45, fontsize=14)
    plot_axes.legend()
    fig.tight_layout()
    plt.savefig(f"BAR_all_cases.png")

    # This is my attempt to a multiple linear regression analysis:
    # This might be nonlinear, I don't really know...
    proper_df = df[df['NWBfile'].notnull()]
    independent_var_dt =  proper_df\
            .drop('attractors_len', 1)\
            .drop('NWBfile', 1)\
            .drop('inhibition_bias', 1)\
            .drop('spdcl', 1) \
            .drop('PA', 1) \
            .drop('sparsness', 1)
    independent_var_dt['dendlen'] = \
        independent_var_dt['dendlen'].replace('small',0)
    independent_var_dt['dendlen'] = \
        independent_var_dt['dendlen'].replace('original',1)
    independent_var_dt['sploc'] = \
        independent_var_dt['sploc'].replace('proximal',0)
    independent_var_dt['sploc'] = \
        independent_var_dt['sploc'].replace('distal',1)
    dependent_var_df = pd.DataFrame(
        proper_df.attractors_len, columns=["attractors_len"]
    )
    lm = linear_model.LinearRegression()
    model = lm.fit(
        independent_var_dt,dependent_var_df["attractors_len"]
    )
    print(model.coef_)

    sys.exit(0)


# This is the inhomogeneous case:
inhomogeneous_analysis = \
    analysis.append_results_to_array(array=data)(
        partial(analysis.load_nwb_from_neuron,
                glia_dir,
                excitation_bias=1.75,
                nmda_bias=6.0,
                sim_duration=5,
                prefix='spi',
                template_postfix='_spi_ri',
                connectivity_type='structured',
                ri=50,
                ntrials=500,
                wr=1,
                reload_raw=False
                )
    )
'''
params = {
    'inhibition_bias': [2.0],#np.arange(1.0, 3.5, 0.5).tolist(),
    'dendlen':['original','small'],
    'dendno': [2,3],
    'sploc': ['proximal','distal'],
    'spcl': [0,1],
    'spdcl': [0,1],
}
params = {
    'inhibition_bias': [2.0],#np.arange(1.0, 3.5, 0.5).tolist(),
    'dendlen':['small'],
    'dendno': [1,2,3],
    'sploc': ['proximal'],
    'spcl': [1],
    'spdcl': [1],
    'learning_condition': [1,2,3,4],
}
'''
#These are the params for the inhomogenous case.
params = {
    'inhibition_bias': [1.8],#np.arange(1.0, 3.5, 0.5).tolist(),
    'dendlen':['small'],
    'dendno': [3],
    'learning_condition': list(range(1,11))
}

analysis.run_for_all_parameters(
    inhomogeneous_analysis,
    **{'auto_param_array': params}
)

df = pd.DataFrame(data)
df['PA'] = [-1.0] * len(df.index)
df['sparsness'] = [-1.0] * len(df.index)
analysis.run_for_all_parameters(
    analysis.query_and_add_pa_column,
    df,
    **{'auto_param_array': params}
)
analysis.run_for_all_parameters(
    analysis.query_and_add_sparsness_column,
    df,
    **{'auto_param_array': params}
)
with pd.option_context('display.max_rows', None, 'display.max_columns',
                       None):  # more options can be specified also
    print(df)

sys.exit(0)
# Plot:
for index in range(df.shape[0]):
    NWBfile = df.loc[index, 'NWBfile']
    if NWBfile:
        dendlen = df.loc[index, 'dendlen']
        dendno = df.loc[index, 'dendno']
        inhibition_bias = df.loc[index, 'inhibition_bias']
        '''
        spcl = df.loc[index, 'spcl']
        spdcl = df.loc[index, 'spdcl']
        sploc = df.loc[index, 'sploc']
        '''
        PA = df.loc[index, 'PA']
        LC = df.loc[index, 'learning_condition']
        trial_len = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len']
        )
        delay_range = (20, int(trial_len / 50))
        K_star, K_labels, *_ = analysis.determine_number_of_clusters(
            NWBfile_array=[NWBfile],
            max_clusters=20,
            custom_range=delay_range
        )
        fig = plt.figure()
        plot_axes = fig.add_subplot(111)
        try:
            analysis.pcaL2(
                NWBfile_array=[NWBfile],
                custom_range=delay_range,
                klabels=K_labels,
                smooth=True,
                plot_2d=True,
                plot_stim_color=True,
                plot_axes=plot_axes,
            )
        except ValueError:
            pass

        plt.savefig(f"SPInhom_Attr_{dendno}{dendlen}dend_"
                    f"IB{inhibition_bias}_"
                    f"PA{PA}_LC_{LC}_rw.png")
        '''
        plt.savefig(f"SP_Attr_{dendno}{dendlen}dend_spcl{int(spcl)}"
                    f"_sploc{sploc}_IB{inhibition_bias}_"
                    f"PA{PA}_LC_{LC}_rw.png")
        '''
sys.exit(0)

#Previous file load was the prefix: t3d2, dendno: 4
blah = \
    analysis.append_results_to_array(array=data)(
        partial(analysis.load_nwb_from_neuron,
                glia_dir,
                learning_condition=1,
                excitation_bias=1.75,
                nmda_bias=6.0,
                sim_duration=5,
                prefix='sp',
                template_postfix='_sp_ri',
                dendlen='original',
                dendno=1,
                connectivity_type='structured',
                ri=50,
                ntrials=1,
                sprw=1,
                )
    )

params = {
    'inhibition_bias': np.arange(0.5, 3.5, 0.5).tolist(),
    'sploc': 'distal',
    'spcl': [0],
}

analysis.run_for_all_parameters(
    blah,
    **{'auto_param_array': params}
)

df = pd.DataFrame(data)
df['PA'] = [-1.0] * len(df.index)
df['sparsness'] = [-1.0] * len(df.index)
analysis.run_for_all_parameters(
    analysis.query_and_add_pa_column,
    df,
    **{'auto_param_array': params}
)
analysis.run_for_all_parameters(
    analysis.query_and_add_sparsness_column,
    df,
    **{'auto_param_array': params}
)
with pd.option_context('display.max_rows', None, 'display.max_columns',
                       None):  # more options can be specified also
    print(df)

sys.exit(0)
# Plot:
for index in range(df.shape[0]):
    NWBfile = df.loc[index, 'NWBfile']
    trial_len = analysis.get_acquisition_parameters(
        input_NWBfile=NWBfile,
        requested_parameters=['trial_len']
    )
    delay_range = (20, int(trial_len / 50))
    K_star, K_labels, *_ = analysis.determine_number_of_clusters(
        NWBfile_array=[NWBfile],
        max_clusters=20,
        custom_range=delay_range
    )
    fig = plt.figure()
    plot_axes = fig.add_subplot(111)
    try:
        analysis.pcaL2(
            NWBfile_array=[NWBfile],
            custom_range=delay_range,
            klabels=K_labels,
            smooth=True,
            plot_2d=True,
            plot_stim_color=True,
            plot_axes=plot_axes,
        )
    except ValueError:
        pass
    plt.savefig(f"PCA_SP_Attractors_distal_cluster_rw.png")

sys.exit(0)

'''
NWBfile = analysis.load_nwb_from_neuron(
    glia_dir,
    excitation_bias=1.75,
    inhibition_bias=1.0,
    nmda_bias=6.0,
    sim_duration=5,
    prefix='t3d',
    template_postfix='_ri',
    dendlen='medium',
    dendno=3,
    connectivity_type='structured',
    ri=50,
    ntrials=1,
    learning_condition=1,
    reload_raw=True
)

print(f"NWB:{NWBfile}")
binned_network_activity = NWBfile. \
                              acquisition['binned_activity']. \
                              data[:250, :]
print(f"{binned_network_activity.shape}")

print('PA:'
      f'{analysis.get_nwb_list_valid_ntrials([NWBfile])}'
      )
fig1,ax1 = plt.subplots()
ax1.imshow(binned_network_activity[:50,0,:])
plt.savefig("BLAH.png")
print('Sparseness:'
      f'{analysis.sparsness(NWBfile)}'
      )
sys.exit(0)
'''

# %% Trimmed synapses simulations:
# %% Load NEURON data
if False:
    # This partial combines the simulations ran:
    dend_trimsyn_sims = \
        analysis.append_results_to_array(array=data)(
            partial(analysis.load_nwb_from_neuron,
                    glia_dir,
                    excitation_bias=1.75,
                    inhibition_bias=1.5,
                    nmda_bias=6.0,
                    sim_duration=5,
                    prefix='ts',
                    template_postfix='_ri',
                    experiment_config='structured',
                    )
        )

    params = {
        'ri': [50],
        'ntrials': [500],
    }

    analysis.run_for_all_parameters(
        dend_trimsyn_sims,
        **{'auto_param_array': params}
    )

    df = pd.DataFrame(data)
    df['PA'] = [-1.0] * len(df.index)
    df['sparsness'] = [-1.0] * len(df.index)
    params.pop('ntrials')
    analysis.run_for_all_parameters(
        analysis.query_and_add_pa_column,
        df,
        **{'auto_param_array': params}
    )

    analysis.run_for_all_parameters(
        analysis.query_and_add_sparsness_column,
        df,
        **{'auto_param_array': params}
    )
    '''
    df['attractors_len'] = [-1] * len(df.index)

    # Dont use the ntrials, since can vary between configurations:

    # Add the number of attractors found on the dataframe
    analysis.run_for_all_parameters(
        analysis.query_and_add_attractors_len_column,
        df,
        **{'auto_param_array': params}
    )

    '''

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    print("checkpoint")

    NWB_array = []
    for index in range(df.shape[0]):
        NWB_array.append(df.loc[index, 'NWBfile'])

    '''
    # out, because not enough colors for stimuli
    print("plotting Stimulus")
    for index, NWBfile in enumerate(NWB_array):
        #Manually plot different trials of the same stimuli:
        #This is a manual version of the compare_dend_params()
        trial_len, ntrials = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len', 'ntrials']
        )
        delay_range = (20, int(trial_len / 50))
        all_range = (0, int(trial_len / 50))

        # Assume 10 trials per stimulus:
        k_labels_arrays = [
            [i+1]
            for i in range(ntrials)
        ]
        K_labels = np.array(list(chain(*k_labels_arrays)))

        label_tags = [
            f'Stimulus {i+1}'
            for i in range(ntrials)
        ]

        fig = plt.figure()
        plot_axis = fig.add_subplot(111)

        analysis.pcaL2(
            NWBfile_array=[NWBfile],
            custom_range=delay_range,
            klabels=K_labels,
            smooth=True,
            plot_2d=True,
            plot_stim_color=True,
            plot_axes=plot_axis,
            legend_labels=label_tags
        )
        plt.savefig(f"PCA_TS_STIMULUS_IB1_index_{index}.png")
    '''

    print("plotting attractors")
    for index in range(df.shape[0]):
        NWBfile = df.loc[index, 'NWBfile']
        trial_len = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len']
        )
        delay_range = (20, int(trial_len / 50))
        K_star, K_labels, *_ = analysis.determine_number_of_clusters(
            NWBfile_array=[NWBfile],
            max_clusters=20,
            custom_range=delay_range
        )
        fig = plt.figure()
        plot_axes = fig.add_subplot(111)
        try:
            analysis.pcaL2(
                NWBfile_array=[NWBfile],
                custom_range=delay_range,
                klabels=K_labels,
                smooth=True,
                plot_2d=True,
                plot_stim_color=True,
                plot_axes=plot_axes,
            )
        except ValueError:
            pass
        plt.savefig(f"PCA_TS_STIMULUS_IB1.5_Attractors_index_{index}.png")



# %% Nassi meeting CLUSTER/Location:
# %% Load NEURON data
if False:
    # This partial combines the simulations ran:
    random_input_dend_cluster_sims = \
        analysis.append_results_to_array(array=data)(
            partial(analysis.load_nwb_from_neuron,
                    glia_dir,
                    excitation_bias=1.75,
                    inhibition_bias=2.0,
                    nmda_bias=6.0,
                    sim_duration=5,
                    prefix='iid3_',
                    template_postfix='_iid_ri',
                    experiment_config='structured',
                    )
        )

    params = {
        'dend_clust_perc': [25, 50],
        'dend_clust_seg': dcs_array,
        'ri': [50],
        'ntrials': [100],
    }

    analysis.run_for_all_parameters(
        random_input_dend_cluster_sims,
        **{'auto_param_array': params}
    )

    df = pd.DataFrame(data)
    '''
    df['PA'] = [-1.0] * len(df.index)
    df['attractors_len'] = [-1] * len(df.index)
    df['sparsness'] = [-1.0] * len(df.index)

    # Dont use the ntrials, since can vary between configurations:
    params.pop('ntrials')

    # Add the number of attractors found on the dataframe
    analysis.run_for_all_parameters(
        analysis.query_and_add_attractors_len_column,
        df,
        **{'auto_param_array': params}
    )

    analysis.run_for_all_parameters(
        analysis.query_and_add_pa_column,
        df,
        **{'auto_param_array': params}
    )

    analysis.run_for_all_parameters(
        analysis.query_and_add_sparsness_column,
        df,
        **{'auto_param_array': params}
    )
    '''

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    print("checkpoint")

    NWB_array = []
    for index in range(df.shape[0]):
        NWB_array.append(df.loc[index, 'NWBfile'])

    for index, NWBfile in enumerate(NWB_array):
        #Manually plot different trials of the same stimuli:
        #This is a manual version of the compare_dend_params()
        trial_len, ntrials = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len', 'ntrials']
        )
        delay_range = (20, int(trial_len / 50))
        all_range = (0, int(trial_len / 50))

        # Assume 10 trials per stimulus:
        k_labels_arrays = [
            [i+1] * 10
            for i in range(int(ntrials/10))
        ]
        K_labels = np.array(list(chain(*k_labels_arrays)))

        label_tags = [
            f'Stimulus {i+1}'
            for i in range(int(ntrials / 10))
        ]

        fig = plt.figure()
        plot_axis = fig.add_subplot(111)

        analysis.pcaL2(
            NWBfile_array=[NWBfile],
            custom_range=delay_range,
            klabels=K_labels,
            smooth=True,
            plot_2d=True,
            plot_stim_color=True,
            plot_axes=plot_axis,
            legend_labels=label_tags
        )
        plt.savefig(f"PCA_DS_CLUST_STIMULUS_index_{index}.png")
    sys.exit(0)

    fig = plt.figure()
    plot_axis = fig.add_subplot(111)
    analysis.compare_dend_params(
        NWBarray_of_arrays=NWB_array,
        dataset_names=[
            'Clustering 25% proximal',
            'Clustering 50% proximal',
            'Clustering 25% medial',
            'Clustering 50% medial',
            'Clustering 25% distal',
            'Clustering 50% distal',
        ],
        plot_axis=plot_axis
    )
    plt.savefig(f"PCA_DS_ALLCLUST.png")
    print("checkpoint")
    sys.exit(0)

    for index in range(df.shape[0]):
        NWBfile = df.loc[index, 'NWBfile']
        trial_len = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len']
        )
        delay_range = (20, int(trial_len / 50))
        K_star, K_labels, *_ = analysis.determine_number_of_clusters(
            NWBfile_array=[NWBfile],
            max_clusters=20,
            custom_range=delay_range
        )
        fig = plt.figure()
        plot_axes = fig.add_subplot(111)
        try:
            analysis.pcaL2(
                NWBfile_array=[NWBfile],
                custom_range=delay_range,
                klabels=K_labels,
                smooth=True,
                plot_2d=True,
                plot_stim_color=True,
                plot_axes=plot_axes,
            )
        except ValueError:
            pass
        fig.savefig(f"PCA_DATA_DS_IB2_CLUST_index_{index}.png")

    sys.exit(0)

    # Now run the analysis/plotting to check if you have a paper:
    # Print the correlation of free variables with attractor number:
    for param, vals in params.items():
        data = {}
        for i, val in enumerate(vals):
            query_d = { param: val }
            nwb_index = (df[list(query_d)] == pd.Series(query_d)).all(axis=1)
            df_index = nwb_index.values.astype(int).nonzero()[0]
            tmp_var = df.loc[df_index, 'attractors_len'].values
            # Since (yet) zero attractors means no data, make them nan:
            data[i] = tmp_var[np.nonzero(tmp_var)[0]]

        fig, axis = plt.subplots()
        for i in range(len(vals)):
            axis.scatter(
                np.ones(len(data[i])) * i,
                data[i],
                c='blue',
                marker='o'
            )
            axis.scatter(i, np.mean(data[i]), c='red', marker='+')
        axis.set_ylabel(f"Attractor Number")
        fig.savefig(f"data_Nassi_{param}_RI{params['ri'][0]}.png")
        plt.close(fig)
    sys.exit(0)

# %% Nassi meeting MULTIDEND/DIFF SIZE:
# %% Load NEURON data
if False:
    print("Running the morphological analysis")
    # This partial combines the simulations ran:
    random_input_dend_multidend_sims = \
        analysis.append_results_to_array(array=data)(
            partial(analysis.load_nwb_from_neuron,
                    glia_dir,
                    excitation_bias=1.75,
                    nmda_bias=6.0,
                    sim_duration=5,
                    template_postfix='_ri',
                    )
        )

    params = {
        'prefix': ['t3d2'],
        'dendlen': [ 'medium'],
        'dendno': [4],
        'connectivity_type': 'structured',
        'ri': [50],
        'ntrials': [100],
        'learning_condition': [1],
        'inhibition_bias':[2.5, 3.0],
    }

    analysis.run_for_all_parameters(
        random_input_dend_multidend_sims,
        **{'auto_param_array': params}
    )

    print("Creating dataframe")
    df = pd.DataFrame(data)
    # Use len for speed and float since we get a percentage:
    df['PA'] = [-1.0] * len(df.index)
    df['sparsness'] = [-1.0] * len(df.index)

    # Dont use the ntrials, since can vary between configurations:
    params.pop('ntrials')

    '''
    df['attractors_len'] = [-1] * len(df.index)
    # Add the number of attractors found on the dataframe
    analysis.run_for_all_parameters(
        analysis.query_and_add_attractors_len_column,
        df,
        **{'auto_param_array': params}
    )
    '''

    analysis.run_for_all_parameters(
        analysis.query_and_add_pa_column,
        df,
        **{'auto_param_array': params}
    )
    analysis.run_for_all_parameters(
        analysis.query_and_add_sparsness_column,
        df,
        **{'auto_param_array': params}
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    print("checkpoint")
    #sys.exit(0)

    NWB_array = []
    for index in range(df.shape[0]):
        NWB_array.append(df.loc[index, 'NWBfile'])

    for index, NWBfile in enumerate(NWB_array):
        #Manually plot different trials of the same stimuli:
        #This is a manual version of the compare_dend_params()
        trial_len, ntrials = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len', 'ntrials']
        )
        delay_range = (20, int(trial_len / 50))
        all_range = (0, int(trial_len / 50))

        # Assume 10 trials per stimulus:
        k_labels_arrays = [
            [i+1]
            for i in range(int(ntrials))
        ]
        K_labels = np.array(list(chain(*k_labels_arrays)))

        label_tags = [
            f'Stimulus {i+1}'
            for i in range(int(ntrials))
        ]

        try:
            fig = plt.figure()
            plot_axis = fig.add_subplot(111)

            analysis.pcaL2(
                NWBfile_array=[NWBfile],
                custom_range=delay_range,
                klabels=K_labels,
                smooth=True,
                plot_2d=True,
                plot_stim_color=True,
                plot_axes=plot_axis,
                legend_labels=label_tags
            )
            print('saving figure')
            IB = df.loc[index, 'inhibition_bias']
            LC = df.loc[index,'learning_condition']
            dendno = df.loc[index,'dendno']
            dendlen = df.loc[index,'dendlen']
            fig.savefig(
                f"MORPH_STIM_{params['prefix']}_IB{IB}"
                f"_LC{LC}"
                f"_{dendno}{dendlen}dend.png"
            )
        except Exception as e:
            print(f"Exception during PCA: {str(e)}")

    for index in range(df.shape[0]):
        print(f'On index {index}')
        NWBfile = df.loc[index, 'NWBfile']
        trial_len = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len']
        )
        delay_range = (20, int(trial_len / 50))
        K_star, K_labels, *_ = analysis.determine_number_of_clusters(
            NWBfile_array=[NWBfile],
            max_clusters=20,
            custom_range=delay_range
        )
        try:
            fig = plt.figure()
            #plot_axes = fig.add_subplot(111, projection='3d')
            plot_axes = fig.add_subplot(111)
            analysis.pcaL2(
                NWBfile_array=[NWBfile],
                custom_range=delay_range,
                klabels=K_labels,
                smooth=True,
                plot_2d=True,
                plot_3d=False,
                plot_stim_color=True,
                plot_axes=plot_axes,
            )
            print('saving figure')
            IB = df.loc[index, 'inhibition_bias']
            LC = df.loc[index,'learning_condition']
            dendno = df.loc[index,'dendno']
            dendlen = df.loc[index,'dendlen']
            fig.savefig(
                f"MORPH_ATTR_{params['prefix']}_IB{IB}"
                f"_LC{LC}"
                f"_{dendno}{dendlen}dend.png"
            )
        except Exception as e:
            print(f"Exception during PCA: {str(e)}")

    sys.exit(0)

    # Now run the analysis/plotting to check if you have a paper:
    # Print the correlation of free variables with attractor number:
    for param, vals in params.items():
        data = {}
        for i, val in enumerate(vals):
            query_d = { param: val }
            nwb_index = (df[list(query_d)] == pd.Series(query_d)).all(axis=1)
            df_index = nwb_index.values.astype(int).nonzero()[0]
            tmp_var = df.loc[df_index, 'attractors_len'].values
            # Since (yet) zero attractors means no data, make them nan:
            data[i] = tmp_var[np.nonzero(tmp_var)[0]]

        fig, axis = plt.subplots()
        for i in range(len(vals)):
            axis.scatter(
                np.ones(len(data[i])) * i,
                data[i],
                c='blue',
                marker='o'
            )
            axis.scatter(i, np.mean(data[i]), c='red', marker='+')
        axis.set_ylabel(f"Attractor Number")
        fig.savefig(f'data_Nassi_{param}_RI50.png')
        plt.close(fig)
    sys.exit(0)

# %% Nassi meeting CP
# %% Load NEURON data
if False:
    # This partial combines the simulations ran:
    anatomical_cluster_cp_sims = \
        analysis.append_results_to_array(array=data)(
            partial(analysis.load_nwb_from_neuron,
                    glia_dir,
                    excitation_bias=1.75,
                    inhibition_bias=3.0,
                    nmda_bias=6.0,
                    sim_duration=5,
                    prefix='',
                    template_postfix='',
                    experiment_config='structured_allt',
                    reload_raw=True
                    )
        )

    params = {
          'cp': [2,3,4,5],
        'ntrials': [50],
    }

    analysis.run_for_all_parameters(
        anatomical_cluster_cp_sims,
        **{'auto_param_array': params}
    )

    df = pd.DataFrame(data)
    # Use len for speed and float since we get a percentage:
    df['attractors_len'] = [-1] * len(df.index)
    df['PA'] = [-1.0] * len(df.index)
    df['sparsness'] = [-1.0] * len(df.index)

    # Dont use the ntrials, since can vary between configurations:
    params.pop('ntrials')

    analysis.run_for_all_parameters(
        analysis.query_and_add_pa_column,
        df,
        **{'auto_param_array': params}
    )
    # Add the number of attractors found on the dataframe
    analysis.run_for_all_parameters(
        analysis.query_and_add_attractors_len_column,
        df,
        **{'auto_param_array': params}
    )
    analysis.run_for_all_parameters(
        analysis.query_and_add_sparsness_column,
        df,
        **{'auto_param_array': params}
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    for index in range(df.shape[0]):
        NWBfile = df.loc[index, 'NWBfile']
        trial_len = analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=['trial_len']
        )
        delay_range = (20, int(trial_len / 50))
        K_star, K_labels, *_ = analysis.determine_number_of_clusters(
            NWBfile_array=[NWBfile],
            max_clusters=20,
            custom_range=delay_range
        )
        fig = plt.figure()
        #plot_axes = fig.add_subplot(111, projection='3d')
        plot_axes = fig.add_subplot(111)
        analysis.pcaL2(
            NWBfile_array=[NWBfile],
            custom_range=delay_range,
            klabels=K_labels,
            smooth=True,
            plot_2d=True,
            plot_3d=False,
            plot_stim_color=True,
            plot_axes=plot_axes,
        )
        fig.savefig(f"PCA_DATA_DETAILED_CP_index_{index}_2d_new.png")
    sys.exit(0)

# this is exploratory to see where in the intermediate connectivity I need to
# change the E/I in order to get PA:
# This partial combines the simulations ran:
anatomical_cluster_iid_sims_intermediate = \
    analysis.append_results_to_array(array=data)(
        partial(analysis.load_nwb_from_neuron,
                glia_dir,
                excitation_bias=1.75,
                inhibition_bias=3.0,
                nmda_bias=6.0,
                sim_duration=5,
                prefix='iid2_',
                template_postfix='_iid'
                )
    )

# Load each run simulation:
dcp_array = [25, 50, 75, 100]
dcs_array = [0, 1, 2]

params = {'dend_clust_perc': dcp_array, 'dend_clust_seg': dcs_array,
          'cp': cp_array, 'ntrials': sparse_cp_trials,
          'dendlen': ['small','medium','long'],
          'dendno': [1,2],
          'connectivity_type': [
             'intermediate_20',
              'intermediate_40',
              'intermediate_60',
          ]}

analysis.run_for_all_parameters(
    anatomical_cluster_iid_sims_intermediate,
    **{'auto_param_array': params}
)

sys.exit(0)


df = pd.DataFrame(data)
# Use len for speed and float since we get a percentage:
df['attractors_len'] = [-1] * len(df.index)

# Dont use the ntrials, since can vary between configurations:
params.pop('ntrials')

tic = time.perf_counter()
# Add the number of attractors found on the dataframe
analysis.run_for_all_parameters(
    analysis.query_and_add_attractors_len_column,
    df,
    **{'auto_param_array': params}
)
toc = time.perf_counter()
print(f'Finding attractors took {toc - tic} seconds.')

# Now run the analysis/plotting to check if you have a paper:
print('Must do analysis')

# Print the correlation of free variables with attractor number:
for param, vals in params.items():
    data = {}
    for i, val in enumerate(vals):
        query_d = { param: val }
        nwb_index = (df[list(query_d)] == pd.Series(query_d)).all(axis=1)
        df_index = nwb_index.values.astype(int).nonzero()[0]
        tmp_var = df.loc[df_index, 'attractors_len'].values
        # Since (yet) zero attractors means no data, make them nan:
        data[i] = tmp_var[np.nonzero(tmp_var)[0]]

    fig, axis = plt.subplots()
    for i in range(len(vals)):
        axis.scatter(
            np.ones(len(data[i])) * i,
            data[i],
            c='blue',
            marker='o'
        )
        axis.scatter(i, np.mean(data[i]), c='red', marker='+')
    axis.set_ylabel(f"Attractor Number")
    fig.savefig(f'data_{param}_CP7_th8.png')
    plt.close(fig)

sys.exit(0)

query_d = { 'dendno' :1 }
nwb_index = (df[list(query_d)] == pd.Series(query_d)).all(axis=1)
df_index = nwb_index.values.astype(int).nonzero()[0]
tmp_var = df.loc[df_index, 'attractors_len'].values
# Since (yet) zero attractors means no data, make them nan:
data = tmp_var[np.nonzero(tmp_var)[0]]
data_len = tmp_var[np.nonzero(tmp_var)[0]].size

fig, axis = plt.subplots()
axis.plot(np.arange(data_len), data)
fig.savefig('data_dendno_1.png')
plt.close(fig)


df = pd.DataFrame(data)
# Use len for speed and float since we get a percentage:
df['PA'] = [-1.0] * len(df.index)

analysis.run_for_all_parameters(
    analysis.query_and_add_pa_column,
    df,
    **{'auto_param_array': params}
)

# make a function that will traverse the dataframe and add a PA column with
# the average PA on each NWB file.

dendno_array = [1,2]
dendlen_array = ['small', 'medium', 'long']

# Do this list a picture with named axis:
std_params = {'cp': 2, 'connectivity_type': 'intermediate_60'}
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
for dn, dendno in enumerate(dendno_array):
    for dl, dendlen in enumerate(dendlen_array):
        pa_array = np.ones((len(dcp_array), len(dcs_array)), dtype=float)
        pa_array = np.multiply(pa_array, -1)
        for i, dcp in enumerate(dcp_array):
            for j, dcs in enumerate(dcs_array):
                # this should be a single entry on the dataframe!
                df_query = {
                    'dend_clust_perc': dcp,
                    'dend_clust_seg': dcs,
                    'dendno': dendno,
                    'dendlen': dendlen,
                    **std_params
                }
                df_indices = (df[list(df_query)] == pd.Series(
                    df_query)).all(axis=1)
                df_indices_int = df_indices.values.astype(int).nonzero()[0]
                if df_indices_int.size < 1:
                    print(f'ERROR: Nonexistent on dataframe! Entry:'
                          f' {df_query}!')
                else:
                    pa_array[i,j] = df.at[df_indices_int[0], 'PA']
        axes[dn][dl].imshow(pa_array, cmap='Blues', vmin=0.0, vmax=1.0)
        print(f'on dend no {dn}, dend len {dl}:')
        print(pa_array)

for i, (row, dendno) in enumerate(zip(axes, dendno_array)):
    for j, (cell, dendlen) in enumerate(zip(row, dendlen_array)):
        if i == len(axes) - 1:
            cell.set_xlabel(f"Segment ({dendlen})")
        if j == 0:
            cell.set_ylabel(f"percent (#dend:{dendno})")

fig.savefig(f"{std_params['connectivity_type']}_CP{std_params['cp']}_IB3.0.png")
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
