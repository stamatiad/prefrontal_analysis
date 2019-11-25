# This is the module containing utility methods for running the jupyter notebook
# so as not to pollute it with definitions all over the place.
#TODO: If this file gets to large (counteracting the purpose of readability),
# merge it with the analysis.py

import analysis_tools as analysis
from analysis_tools import from_zero_to, from_one_to
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, axes
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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.axes._subplots as mpsp
import matplotlib.projections as mppr

class traversable_1d_array(object):
    '''
    Make a callback class to traverse an array, column wise
    This is intended to be used in a notebook, so no fancy plot stuff.
    '''
    def __init__(self, plt_handle, np_array):
        self.ind = 0
        self.max_idx = np_array.shape[1]
        self.h = plt_handle
        self.data = np_array

    def next(self, event):
        # get next column:
        self.ind += 1
        i = self.ind % self.max_idx
        # Can I set multiple lines' data?
        self.h.set_data(self.data[:, self.idx])
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % self.max_idx
        # Can I set multiple lines' data?
        self.h.set_data(self.data[:, self.idx])
        plt.draw()

def p_figure_space(h, c, r):
    return (c*r)/(h-c*r+c)

def extract_subgridspec_bbox(fig, sgs):
    projection_class, kwargs, key = mppr.process_projection_requirements(
        fig, sgs)
    a = mpsp.subplot_class_factory(projection_class)(fig, sgs, **kwargs)
    return a.bbox

def split_gridspec(nrows, ncols, ch, cw, gs=None, left=0.0, bottom=0.0,
                   right=1.0, top=1.0):
    # I don't know what I'm doing...
    fig = plt.gcf()
    if gs:
        # If a gridspec is given, split that.
        # Extract gs bounding box:
        bbox = extract_subgridspec_bbox(fig, gs)
        h = bbox._bbox.y1 - bbox._bbox.y0
        w = bbox._bbox.x1 - bbox._bbox.x0
        hpercent = p_figure_space(h, ch, nrows)
        wpercent = p_figure_space(w, cw, ncols)
        gs_split = gs.subgridspec(
            nrows, ncols, wspace=wpercent, hspace=hpercent
        )
    else:
        h = top - bottom
        w = right - left
        hpercent = p_figure_space(h, ch, nrows)
        wpercent = p_figure_space(w, cw, ncols)
        gs_split = gridspec.GridSpec(
            nrows, ncols, left=left, right=right,
            top=top, bottom=bottom,
            wspace=wpercent, hspace=hpercent
        )

    return gs_split

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

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

def adjust_spines(ax, spines, blowout=2):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', blowout))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        pass
        # no yaxis ticks
        #ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        pass
        # no xaxis ticks
        #ax.xaxis.set_ticks([])

def axis_normal_plot(axis=None, xlim=None):
    # Hide the right and top spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    #axis.spines['left'].set_position('zero')
    #axis.spines['bottom'].set_position('zero')

    axis.tick_params(axis='both', which='major', labelsize=8)
    axis.tick_params(axis='both', which='minor', labelsize=6)
    axis.xaxis.set_tick_params(width=2)
    axis.yaxis.set_tick_params(width=2)

    for axis_loc in ['bottom', 'left']:
        axis.spines[axis_loc].set_linewidth(2)
    if xlim:
        axis.set_xlim(xlim[0], xlim[1])
        axis.spines['left'].set_position(xlim[0])

def axis_box_plot(axis=None, ylim=None):
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.xaxis.set_ticks_position('none')
    axis.spines['left'].set_linewidth(2)
    axis.yaxis.set_tick_params(width=2)

def plot_clear_abscissa(axis=None):
    axis.spines['bottom'].set_visible(False)
    axis.spines['bottom'].set_color(None)
    axis.tick_params(axis=False)
    axis.xaxis.set_ticks_position('none')
    axis.xaxis.set_ticklabels([])

def mark_figure_letter(axis=None, letter=None):
    #TODO: why is this a problem? Since its of that class, how comes the
    # class is nonexistent?
    #if isinstance(ax, axes._subplots.Axes3DSubplot)

    if isinstance(axis, Axes3D):
        axis.text2D(0.01, 0.99, f'{letter}       ',
                      fontsize=14,
                      horizontalalignment='right',
                      verticalalignment='top',
                      transform=axis.transAxes)
    else:
        axis.text(0.01, 0.99, f'{letter}       ',
                  fontsize=14,
                  horizontalalignment='right',
                  verticalalignment='top',
                  transform=axis.transAxes)


def plot_trial_spiketrains(NWBfile=None, trialid=None, plot_axis=None,
                           axis_label_font_size=12, tick_label_font_size=12,
                           labelpad_x=10, labelpad_y=10):

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
        colors='k',
        orientation='horizontal'
    )
    plot_axis.set_xlim([0.0, trial_len])
    plot_axis.set_ylim([0.0, ncells])
    plot_axis.margins(0.0)

    plot_axis.xaxis.set_ticks(np.arange(0, trial_len + 1000, 1000))
    plot_axis.xaxis.set_ticklabels(np.arange(0, 5, 1), fontsize=tick_label_font_size)
    plot_axis.yaxis.set_ticks_position('none')
    plot_axis.yaxis.set_ticklabels([])

    #TODO: compute stimulus interval!
    plot_axis.spines['left'].set_position('zero')
    plot_axis.spines['bottom'].set_position('zero')
    #plot_axis.spines['left'].set_position(('axes', 0.6))

    # pyramidals:
    x = 0.0
    y = 0.0
    w = -0.05
    h = 250 / 333
    rect = Rectangle(
        (x, y), w, h, 0.0, fill=True, facecolor='b', alpha=0.2,
        clip_on=False,
        transform=plot_axis.transAxes
    )
    plot_axis.add_patch(rect)
    # interneurons
    x = 0.0
    y = 1.0
    w = -0.05
    h = -83 / 333
    rect = Rectangle(
        (x, y), w, h, 0.0, fill=True, facecolor='r', alpha=0.2,
        clip_on=False,
        transform=plot_axis.transAxes
    )
    plot_axis.add_patch(rect)

    plot_axis.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
    # Annotate excitatory/inhibitory population:
    #plot_axis.axvspan(-200.0, 0.0, ymin=0, ymax=250/333, color='b', alpha=0.2)
    #plot_axis.axvspan(-200.0, 0.0, ymin=250/333, ymax=1, color='r', alpha=0.2)

    plot_axis.set_xlabel(
        'Time (ms)', fontsize=axis_label_font_size, labelpad=labelpad_x
    )
    plot_axis.set_ylabel(
        'Cell type', fontsize=axis_label_font_size, labelpad=labelpad_x
    )

    if not plot_axis:
        plot_axis.set(title='Spike events')
        plt.show()


def setBoxAttribtes(boxplot_handles, colors):
    for bplot in boxplot_handles:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_color(color)
            patch.set_facecolor(color)
            patch.set_linewidth(2)
        for line in bplot['medians']:
            line.set_linewidth(2)
        for line in bplot['whiskers']:
            line.set_linewidth(2)
        for line in bplot['fliers']:
            line.set_linewidth(2)
        for line in bplot['caps']:
            line.set_linewidth(2)

def set_horizontal_scalebar(axis=None, label=None, relativesize=None,
                            distfromx=0.1, distfromy=0.1):

    scalebar = AnchoredSizeBar(
        transform=axis.transData,
        size=relativesize,
        label=label,
        loc='lower left',
        pad=0.1,
        borderpad=0.1,
        color='black',
        frameon=False,
        label_top=False,
        size_vertical=relativesize/100,
        fontproperties=fm.FontProperties(size=8),
        bbox_to_anchor=(distfromy, -distfromx),
        bbox_transform=axis.transAxes
    )
    #scalebar.size_bar.get_children()[0].fill = True
    axis.add_artist(scalebar)

    pass

def report_value(name, value):
    print(f'Reporting value: \n\t{name} -> {value}\n')
    pass