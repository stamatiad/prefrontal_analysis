# This is the module containing utility methods for running the jupyter notebook
# so as not to pollute it with definitions all over the place.
#TODO: If this file gets to large (counteracting the purpose of readability),
# merge it with the analysis.py

import analysis_tools as analysis
from analysis_tools import from_zero_to, from_one_to
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pandas as pd
import h5py
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
from scipy import spatial
from functools import partial
from collections import defaultdict
from scipy import stats
import sys
from pynwb import NWBHDF5IO

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def datasetName(id):
    return f'Animal {id}'

def statisticalAnnotation(columns=None, datamax=None, axobj=None):
    # statistical annotation
    x1, x2 = columns  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = datamax + datamax/10, datamax/10, 'k'
    axobj.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    axobj.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color=col)

def is_significant(data=None, pthreshold=[0.05, 0.01]):
    # Compare data in columns for significance and return significance
    # level.
    #TODO: make it to return significance level, so the annotation function
    # to know how many stars to plot.
    statistic, p = stats.ttest_ind(
        data[:, 0], data[:, 1],
        equal_var=False
    )
    print(f'p is {p}')
    if p[0] < pthreshold:
        return True
    else:
        return False

def hide_axis_border(axis=None):
    # Hide the right and top spines
    for axis_loc in ['top', 'bottom', 'left', 'right']:
        axis.spines[axis_loc].set_visible(False)
        axis.spines[axis_loc].set_color(None)
        axis.tick_params(axis=False)
    axis.xaxis.set_ticks_position('none')
    axis.yaxis.set_ticks_position('none')
    axis.xaxis.set_ticklabels([])
    axis.yaxis.set_ticklabels([])

def axis_normal_plot(axis=None):
    # Hide the right and top spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    axis.spines['left'].set_position('zero')
    axis.spines['bottom'].set_position('zero')

    axis.tick_params(axis='both', which='major', labelsize=8)
    axis.tick_params(axis='both', which='minor', labelsize=6)

    for axis_loc in ['top', 'bottom', 'left', 'right']:
        axis.spines[axis_loc].set_linewidth(2)

def mark_figure_letter(axis=None, letter=None):
    axis.text(0.01, 0.99, f'{letter}       ',
                  fontsize=14,
                  horizontalalignment='right',
                  verticalalignment='top',
                  transform=axis.transAxes)


def plot_trial_spiketrains(NWBfile=None, trialid=None, plot_axis=None):

    animal_model_id, learning_condition_id, ncells, pn_no, ntrials, \
    trial_len, q_size, trial_q_no, correct_trials_idx, correct_trials_no = \
        analysis.get_acquisition_parameters(
            input_NWBfile=NWBfile,
            requested_parameters=[
                'animal_model_id', 'learning_condition_id', 'ncells',
                'pn_no', 'ntrials', 'trial_len', 'q_size', 'trial_q_no',
                'correct_trials_idx', 'correct_trials_no'
            ]
        )

    network_spiketrains = analysis.get_acquisition_spikes(
        NWBfile=NWBfile,
        acquisition_name='membrane_potential',
    )

    # Unpack cell ids and their respective spike trains:
    cell_ids, cell_spiketrains = zip(*[
        cell_spiketrain
        for cell_spiketrain in network_spiketrains[trialid]
    ])

    if not plot_axis:
        fig, plot_axis = plt.subplots()
        plt.ion()
    plot_axis.eventplot(
        cell_spiketrains,
        lineoffsets=cell_ids,
        colors='k'
    )
    plot_axis.set_xlim([-200.0, trial_len])
    plot_axis.set_ylim([0.0, ncells])
    plot_axis.margins(0.0)

    plot_axis.yaxis.set_ticks_position('none')
    plot_axis.yaxis.set_ticklabels([])

    #TODO: compute stimulus interval!
    plot_axis.spines['left'].set_position('zero')
    plot_axis.spines['bottom'].set_position('zero')
    #plot_axis.spines['left'].set_position(('axes', 0.6))

    plot_axis.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
    # Annotate excitatory/inhibitory population:
    plot_axis.axvspan(-200.0, 0.0, ymin=0, ymax=250/333, color='b', alpha=0.2)
    plot_axis.axvspan(-200.0, 0.0, ymin=250/333, ymax=1, color='r', alpha=0.2)

    plot_axis.set(xlabel='Time (ms)', ylabel='Cell type')
    plot_axis.set(title='Spike events')

    if not plot_axis:
        plt.show()

