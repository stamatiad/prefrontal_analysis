# <markdowncell>
# # # Generate Figure 1
# Single neuron and network responses following stimulation. A. Top: Schematic representation of the network model. Excitatory (E) connectivity profile was based on experimental data. The excitatory population was reciprocally connected with the Inhibitory (I) in a feedback way. Bottom: Random network connectivity changes only excitatory connectivity. B. Top: Cartoon morphology of the pyramidal model. Bottom: Same for fast-spiking interneuron. C. Top: three exemplar responses of pyramidals in a single trial. Bottom: Same for interneurons. D. Top: Network response activity raster plot of pyramidal (blue) and interneurons (red) to a 1 sec stimulus. Bottom: Same trial’s instantaneous firing frequencies of each pyramidal (> 20Hz), showing its highly dynamic response during delay period. E. Histograms of inter spike interval length (top) and Coefficient of Variation (bottom) of all the structured trials for the stimulus period (blue) and delay period (red). F. Top: Non-linear NMDA responses are generated in the basal dendrites of the pyramidal neurons (top) as in (Nevian et al. 2007b) (bottom). Somatic (blue) and dendritic (red) depolarization from resting potential in response to increasing stimulus intensity. G. Overall network response energy (mean firing rate; top) and multidimensional velocity (bottom) aligned on stimulus period onset. H. Top: Cross correlation of network states between the stimulus period and the delay period over time (aligned on stimulus onset, 1 s stimulus). Bottom: Experimentally reported correlation from (Murray et al. 2017). I.  Network responses for 10 trials, under one learning condition, reduced to their first three principal components. Colormap denotes time.
# <markdowncell>
# Import necessary modules:

# <codecell>
import notebook_module as nb
import analysis_tools as analysis
import network_tools as nt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from pathlib import Path
from pynwb import NWBHDF5IO
from itertools import chain
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

# <markdowncell>
# # Create figure 1.

# <codecell>
simulations_dir = Path.cwd().joinpath('simulations')
attributes_dir = Path.cwd().joinpath('new_files')
glia_dir = Path(r'G:\Glia')
plt.rcParams.update({'font.family': 'Helvetica'})
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.ion()
axis_label_font_size = 10
no_of_conditions = 10
no_of_animals = 4
#===============================================================================
#===============================================================================
subplot_width = 2
subplot_height = 1
figures1 = plt.figure(figsize=plt.figaspect(subplot_height / subplot_width))
#TODO: I tend to believe that the w/hspace is RELATIVE to the size of this grid.
# This asks for a absolute number, in order to have a visually pleasing grid.
gs1 = gridspec.GridSpec(2, 4, left=0.05, right=0.95, top=0.95, bottom=0.10, wspace=0.35, hspace=0.35)
#gs1.update(left=0.05, right=0.30, wspace=0.05)
A_axis = plt.subplot(gs1[0, 0], projection='3d')
B_axis = plt.subplot(gs1[0, 1])
C_axis = plt.subplot(gs1[0, 2])
D_axis = plt.subplot(gs1[0, 3])
E_axis = plt.subplot(gs1[1, 0])
F_axis = plt.subplot(gs1[1, 1])
G_axis = plt.subplot(gs1[1, 2])
H_axis = plt.subplot(gs1[1, 3])
nb.mark_figure_letter(A_axis, 'A')
nb.mark_figure_letter(B_axis, 'B')
nb.mark_figure_letter(C_axis, 'C')
nb.mark_figure_letter(D_axis, 'D')
nb.mark_figure_letter(E_axis, 'E')
nb.mark_figure_letter(F_axis, 'F')
nb.mark_figure_letter(G_axis, 'G')
nb.mark_figure_letter(H_axis, 'H')

# Import network attributes
filename = attributes_dir.joinpath(f'structured_network_SN1.hdf5')
network_attributes = pd.read_hdf(filename, key='attributes').to_dict()
serial_no = network_attributes['serial_no'][0]
configuration_alias = network_attributes['configuration_alias'][0]
pc_no = network_attributes['pc_no'][0]
pv_no = network_attributes['pv_no'][0]
connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
dist_mat = pd.read_hdf(filename, key='dist_mat').values
pc_somata = pd.read_hdf(filename, key='pc_somata').values
pv_somata = pd.read_hdf(filename, key='pv_somata').values
# Just gain access to the connection functions of the network class:
net_tmp = nt.Network(serial_no=serial_no, pc_no=pc_no, pv_no=pv_no)
# Plot network attributes.

A_axis.scatter(
    pc_somata[:, 0],
    pc_somata[:, 1],
    pc_somata[:, 2],
    c='b', marker='^'
)
A_axis.scatter(
    pv_somata[:, 0],
    pv_somata[:, 1],
    pv_somata[:, 2],
    c='r', marker='o'
)
A_axis.set_xlabel('X (um)')
A_axis.set_ylabel('Y (um)')
A_axis.set_zlabel('Z (um)')
A_axis.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
A_axis.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
A_axis.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
pca_axis_limits = (0, 180)
A_axis.set_xlim(pca_axis_limits)
A_axis.set_ylim(pca_axis_limits)
A_axis.set_zlim(pca_axis_limits)
A_axis.set_xticks(pca_axis_limits)
A_axis.set_yticks(pca_axis_limits)
A_axis.set_zticks(pca_axis_limits)


# Fig S1B
# Plot experimental PC to PC connection probabilities:
plt_range = np.arange(0, 200, 5)
mpl.style.use('seaborn')
B_axis.plot(plt_range, net_tmp.connection_functions_d['PN_PN']['total'](plt_range), 'C0', label='Overall')
B_axis.plot(plt_range, net_tmp.connection_functions_d['PN_PN']['unidirectional'](plt_range), 'C1', label='Unidirectional')
B_axis.plot(plt_range, net_tmp.connection_functions_d['PN_PN']['reciprocal'](plt_range), 'C2', label='Reciprocal')
B_axis.set_xlabel('Intersomatic distance (um)')
B_axis.set_ylabel('Connection Probability')
B_axis.legend()
mpl.style.use('default')

# Fig S1C
configuration_alias = 'structured'
for sn in range(1, 5):
    filename = Path(rf'C:\Users\steve\Documents\analysis\Python\new_files'). \
        joinpath(f'{configuration_alias}_network_SN{sn}_clust_coeff.hdf5')
    cc = pd.read_hdf(filename, key='clust_coeff').values
    C_axis.plot(np.arange(999), cc[0,1:], linewidth=0.1, color='black')

C_axis.set_xlabel('Iterations')
C_axis.set_ylabel('Average Clustering Coefficient')


#TODO: replace these network spesific functions with paper-wise ones:
# Figure S1B
# Paper-wise connectivity across distance:
#TODO: hardcoded histo len.
histo_acc = np.full([4, 15], 0)
histo_dist_acc = np.full([4, 15], 0)
ground_truth_acc = []
for sn in range(1, 5):
    # Load network attributes/data for each serial no:
    filename = attributes_dir.joinpath(f'structured_network_SN{sn}.hdf5')
    connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
    dist_mat = pd.read_hdf(filename, key='dist_mat').values


    histo, histo_bins, histo_dist, ground_truth, bin_width = \
        nt.compute_unidirectional_across_distance(
        connectivity_mat[:pc_no, :pc_no],
        dist_mat[:pc_no, :pc_no],
        net_tmp.connection_functions_d['PN_PN']['unidirectional'],
        plot=False
    )

    # accumulate all network instances:
    histo_acc[sn - 1, :] = histo
    histo_dist_acc[sn - 1, :] = histo_dist
    ground_truth_acc.append(ground_truth)
    print('blah!')

nt.plot_average_connectivity(
    histo_acc, histo_bins, histo_dist_acc, ground_truth_acc, bin_width,
ax=D_axis)
#D_axis.set_xticks(np.arange(0, 160, 20))
#D_axis.set_xticklabels(np.arange(0, 160, 20))
D_axis.set_xlabel('Distance (um)')
D_axis.set_ylabel('PN-PN Unidirectional Probability')

# Fig S1C
histo_acc = np.full([4, 15], 0)
histo_dist_acc = np.full([4, 15], 0)
ground_truth_acc = []
for sn in range(1, 5):
    # Load network attributes/data for each serial no:
    filename = attributes_dir.joinpath(f'structured_network_SN{sn}.hdf5')
    connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
    dist_mat = pd.read_hdf(filename, key='dist_mat').values


    histo, histo_bins, histo_dist, ground_truth, bin_width = \
        nt.compute_reciprocal_across_distance(
            connectivity_mat[:pc_no, :pc_no],
            dist_mat[:pc_no, :pc_no],
            net_tmp.connection_functions_d['PN_PN']['reciprocal'],
            plot=False
        )

    # accumulate all network instances:
    histo_acc[sn - 1, :] = histo
    histo_dist_acc[sn - 1, :] = histo_dist
    ground_truth_acc.append(ground_truth)
    print('blah!')

nt.plot_average_connectivity(
    histo_acc, histo_bins, histo_dist_acc, ground_truth_acc, bin_width,
    ax=E_axis)
#E_axis.set_xticks(np.arange(0, 200, 20))
#E_axis.set_xticklabels(np.arange(0, 200, 20))
E_axis.set_xlabel('Distance (um)')
E_axis.set_ylabel('PN-PN Reciprocal Probability')

# Fig S1D:
histo_acc = np.full([4, 15], 0)
histo_dist_acc = np.full([4, 15], 0)
ground_truth_acc = []
for sn in range(1, 5):
    # Load network attributes/data for each serial no:
    filename = attributes_dir.joinpath(f'structured_network_SN{sn}.hdf5')
    connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
    dist_mat = pd.read_hdf(filename, key='dist_mat').values


    histo, histo_bins, histo_dist, ground_truth, bin_width = \
        nt.compute_pn2pv_unidirectional_across_distance(
            mat_pn_pv=connectivity_mat[:pc_no, pc_no:],
            mat_pv_pn=connectivity_mat[pc_no:, :pc_no],
            dist_mat=dist_mat[:pc_no, pc_no:],
            ground_truth=net_tmp.connection_functions_d['PN_PV']['A2B'],
            plot=False
        )

    # accumulate all network instances:
    histo_acc[sn - 1, :] = histo
    histo_dist_acc[sn - 1, :] = histo_dist
    ground_truth_acc.append(ground_truth)
    print('blah!')

nt.plot_average_connectivity(
    histo_acc, histo_bins, histo_dist_acc, ground_truth_acc, bin_width,
    ax=F_axis)
F_axis.set_xlabel('Distance (um)')
F_axis.set_ylabel('PN2PV Unidirectional Probability')

# Fig S1E
histo_acc = np.full([4, 15], 0)
histo_dist_acc = np.full([4, 15], 0)
ground_truth_acc = []
for sn in range(1, 5):
    # Load network attributes/data for each serial no:
    filename = attributes_dir.joinpath(f'structured_network_SN{sn}.hdf5')
    connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
    dist_mat = pd.read_hdf(filename, key='dist_mat').values


    histo, histo_bins, histo_dist, ground_truth, bin_width = \
        nt.compute_pv2pn_unidirectional_across_distance(
            mat_pn_pv=connectivity_mat[:pc_no, pc_no:],
            mat_pv_pn=connectivity_mat[pc_no:, :pc_no],
            dist_mat=dist_mat[:pc_no, pc_no:],
            ground_truth=net_tmp.connection_functions_d['PN_PV']['B2A'],
            plot=False
        )

    # accumulate all network instances:
    histo_acc[sn - 1, :] = histo
    histo_dist_acc[sn - 1, :] = histo_dist
    ground_truth_acc.append(ground_truth)
    print('blah!')

nt.plot_average_connectivity(
    histo_acc, histo_bins, histo_dist_acc, ground_truth_acc, bin_width,
    ax=G_axis)
G_axis.set_xlabel('Distance (um)')
G_axis.set_ylabel('PV2PN Unidirectional Probability')

# Fig S1F
histo_acc = np.full([4, 15], 0)
histo_dist_acc = np.full([4, 15], 0)
ground_truth_acc = []
for sn in range(1, 5):
    # Load network attributes/data for each serial no:
    filename = attributes_dir.joinpath(f'structured_network_SN{sn}.hdf5')
    connectivity_mat = pd.read_hdf(filename, key='connectivity_mat').values
    dist_mat = pd.read_hdf(filename, key='dist_mat').values


    histo, histo_bins, histo_dist, ground_truth, bin_width = \
        nt.compute_pn_pv_reciprocal_across_distance(
            connectivity_mat[:pc_no, pc_no:],
            connectivity_mat[pc_no:, :pc_no],
            dist_mat[:pc_no, pc_no:],
            net_tmp.connection_functions_d['PN_PV']['reciprocal'],
            plot=False
        )

    # accumulate all network instances:
    histo_acc[sn - 1, :] = histo
    histo_dist_acc[sn - 1, :] = histo_dist
    ground_truth_acc.append(ground_truth)
    print('blah!')

nt.plot_average_connectivity(
    histo_acc, histo_bins, histo_dist_acc, ground_truth_acc, bin_width,
    ax=H_axis)
H_axis.set_xlabel('Distance (um)')
H_axis.set_ylabel('PN2PV Reciprocal Probability')
plt.show()


# <codecell>
figures1.savefig('Figure_S1_new.pdf')
figures1.savefig('Figure_S1_new.png')
print('Tutto pronto!')


#%%



