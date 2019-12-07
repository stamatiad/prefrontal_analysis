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

simulations_dir = Path.cwd().joinpath('simulations')
glia_dir = Path("\\\\139.91.162.90\\cluster\\stefanos\\Documents\\Glia\\")

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
cp_array = [2,3,4,5,6,7]
NWBfiles_1longdend = []
sparse_cp_trials = lambda cp: (cp - 1) * 10 + 1
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
    [NWBfile]
],[
    'NWBfiles_2longdend',
    'Stimulated attractors'
])


plt.savefig('Stimulated_attractors.pdf')
plt.savefig('Stimulated_attractors.png')
print("tutto pronto!")