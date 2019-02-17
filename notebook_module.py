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
