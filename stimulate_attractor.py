# <codecell>
import notebook_module as nb
import analysis_tools as analysis
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from pathlib import Path
from pynwb import NWBHDF5IO
from itertools import chain
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
import seaborn as sb
import math
import pandas as pd
from scipy import stats
from itertools import chain
import sys

plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
axis_label_font_size = 12
tick_label_font_size = 12
labelpad_x = 10
labelpad_y = 10
plt.rcParams['xtick.labelsize']=tick_label_font_size
plt.rcParams['ytick.labelsize']=tick_label_font_size

simulations_dir = Path.cwd().joinpath('simulations')
glia_dir = Path("\\\\139.91.162.90\\cluster\\stefanos\\Documents\\Glia\\")
glia_home_dir = Path("\\\\139.91.162.90\\cluster\\stefanos\\Documents\\Glia_home_sims\\")

cp_array = [2,3,4,5,6,7]
sparse_cp_trials = lambda cp: (cp - 1) * 10 + 1

# REad random input/stimulation:
NWBfiles_1longdend_ri = []
NWBfile = analysis.load_nwb_from_neuron(
    glia_dir,
    reload_raw=True,
    new_params={
        'excitation_bias': 1.0,
        'inhibition_bias': 2.0,
        'nmda_bias': 6.0,
        'ntrials': 20,
        'sim_duration': 5,
        'cp': 2,
        'experiment_config': 'structured_1longdend_ri'
    }
)
if NWBfile:
    NWBfiles_1longdend_ri.append(NWBfile)

NWBfiles_1longdend = []
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

analysis.compare_dend_params([
    NWBfiles_1longdend,
    NWBfiles_1longdend_ri
],[
    '1 long dend clustered input (100um)',
    '1 long dend random input (100um)'
],
plot_3d=True)

# Check RI/CP of 1 medium dendrite:
NWBfiles_1mediumdend = []
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
            'experiment_config': 'structured_1mediumdend'
        }
    )
    if NWBfile:
        NWBfiles_1mediumdend.append(NWBfile)

NWBfiles_1mediumdend_ri = []
NWBfile = analysis.load_nwb_from_neuron(
    glia_dir,
    reload_raw=True,
    new_params={
        'excitation_bias': 1.0,
        'inhibition_bias': 2.0,
        'nmda_bias': 6.0,
        'ntrials': 20,
        'sim_duration': 5,
        'cp': 2,
        'experiment_config': 'structured_1mediumdend_ri'
    }
)
if NWBfile:
    NWBfiles_1mediumdend_ri.append(NWBfile)

# This is the comparison of the random and clustered inputs.
analysis.compare_dend_params([
    NWBfiles_1mediumdend,
    NWBfiles_1mediumdend_ri
],[
    '1 medium dend clustered input (100um)',
    '1 medium dend random input (100um)'
],
    plot_3d=False)

NWBfiles_1smalldend_ri = []
NWBfile = analysis.load_nwb_from_neuron(
    glia_dir,
    reload_raw=False,
    new_params={
        'excitation_bias': 1.0,
        'inhibition_bias': 2.0,
        'nmda_bias': 6.0,
        'ntrials': 20,
        'sim_duration': 5,
        'cp': 2,
        'experiment_config': 'structured_1smalldend_ri'
    }
)
if NWBfile:
    NWBfiles_1smalldend_ri.append(NWBfile)

analysis.compare_dend_params([
    NWBfiles_1smalldend_ri
],[
    '1small dend random input (100um)'
],
    plot_3d=True)


NWBfiles_ri = []
NWBfile = analysis.load_nwb_from_neuron(
    glia_home_dir,
    reload_raw=False,
    new_params={
        'excitation_bias': 1.75,
        'inhibition_bias': 3.0,
        'nmda_bias': 6.0,
        'ntrials': 10,
        'sim_duration': 5,
        'experiment_config': 'structured'
    },
    template_postfix='_old'
)
if NWBfile:
    NWBfiles_ri.append(NWBfile)




# read stimulation of atractor basins.
NWBfile = analysis.load_nwb_from_neuron(
    glia_dir,
    reload_raw=False,
    new_params={
        'excitation_bias': 1.0,
        'inhibition_bias': 2.2,
        'nmda_bias': 6.0,
        'ntrials': 270 + 1,
        'sim_duration': 5.1,
        'cp': 27,
        'experiment_config': 'structured_1longdend'
    }
)
# compare result of stimulating the attractors with the original simulations:
NWBfiles_1longdend = []
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

#TODO: How acn it fucking be that the random input has no attractors whereas the
# clustered has, with same EB/IB ratio and same number of neurons stimulated?
analysis.compare_dend_params([
    NWBfiles_1longdend,
    NWBfiles_1longdend_ri
],[
    '1long dend clustered input (100um)',
    '1long dend random input (100um)'
],
    plot_3d=True)

analysis.compare_dend_params([
    NWBfiles_1longdend,
    NWBfiles_ri,
    [NWBfile]
],[
    '2 dends cluster input (100um)',
    '2 dends random input (100um)',
    'Stimulated attractors from cluster'
],
plot_3d=True)

analysis.compare_dend_params([
    NWBfiles_ri
],[
    '2 dends random input (100um)'
],
    plot_3d=True)

analysis.compare_dend_params([
    NWBfiles_1longdend,
    [NWBfile]
],[
    'NWBfiles_2longdend',
    'Stimulated attractors'
],
plot_3d=True)


plt.savefig('Stimulated_attractors_2.pdf')
plt.savefig('Stimulated_attractors_2.png')
print("tutto pronto!")