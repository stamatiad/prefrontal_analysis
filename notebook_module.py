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

