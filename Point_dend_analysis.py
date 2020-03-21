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
from sklearn.linear_model import LinearRegression
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
glia_dir_pd = Path("\\\\139.91.162.90\\cluster\\stefanos\\Documents\\Glia\\point_dendrite\\")

# <codecell>
#===============================================================================
# Compare attractor landscape while dend length increases
#===============================================================================
cp_array = [2,3,4,5,6,7]
NWBfile_array = []
NWBfiles_1smalldend = []
NWBfiles_2smalldend = []
NWBfiles_1mediumdend = []
NWBfiles_2mediumdend = []
NWBfiles_1longdend = []
NWBfiles_2longdend = []
#K_labels_array = []
sparse_cp_trials = lambda cp: (cp - 1) * 10 + 1
cp_trials_len = 0
for cp in cp_array:
    cp_trials_len += cp

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

for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 1.5,
            'nmda_bias': 6.0,
            'ntrials': sparse_cp_trials(cp),
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_2smalldend'
        }
    )
    if NWBfile:
        NWBfiles_2smalldend.append(NWBfile)
        #K_labels_array.append(1)
        
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
            'experiment_config': 'structured_1smalldend'
        }
    )
    if NWBfile:
        NWBfiles_1smalldend.append(NWBfile)
        #K_labels_array.append(2)

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
        #K_labels_array.append(3)

for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 2.5,
            'nmda_bias': 6.0,
            'ntrials': sparse_cp_trials(cp),
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_2mediumdend'
        }
    )
    if NWBfile:
        NWBfiles_2mediumdend.append(NWBfile)
        #K_labels_array.append(4)
for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias':  2.4,
            'nmda_bias': 6.0,
            'ntrials': sparse_cp_trials(cp),
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_2longdend'
        }
    )
    if NWBfile:
        NWBfiles_2longdend.append(NWBfile)
        #K_labels_array.append(4)

'''
# <codecell>
# Create anatomical cluster (affinity propagation) VS No of trajectories
trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len']
)
delay_range = (20, int(trial_len / 50))
all_range = (0, int(trial_len / 50))

K_star_array = []
k_labels_array = []
for i, NWBfile in enumerate(NWBfiles_1mediumdend):
    K_star, k_labels, *_ = analysis.determine_number_of_clusters(
        NWBfile_array=[NWBfile],
        max_clusters=20,
        custom_range=delay_range
    )
    K_star_array.append(K_star)
    k_labels_array.append(k_labels)

fig = plt.figure()
plt.ion()
plot_axes_3d = fig.add_subplot(111, projection='3d')
analysis.pcaL2(
    NWBfile_array=[NWBfiles_1mediumdend[5]],
    custom_range=delay_range,
    klabels=k_labels_array[-1],
    smooth=True, plot_3d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_3d
)
fig, ax = plt.subplots()
ax.plot(range(2, len(K_star_array)+2), K_star_array)
ax.set_xlabel('Anatomical clusters')
ax.set_ylabel('Number of trajectories')
fig.savefig('Clusters_VS_trajectories.pdf')
fig.savefig('Clusters_VS_trajectories.png')
print("Tutto pronto!")
'''



'''
activity, *_ = analysis.get_binned_activity(
    NWBfiles_1mediumdend[5],delay_range)
mymat = activity.mean(axis=2)
x = np.array(range(1, mymat.shape[1] + 1)).reshape(-1,1)
y = mymat[6, :].reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
beta = model.coef_
x_pred = model.predict(x)
# Plot outputs
plt.scatter(x, y,  color='black')
plt.plot(x, x_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
print("BLAH")
'''

'''
# <codecell>
# Create Murray figures:
trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len']
)
delay_range = (20, int(trial_len / 50))
all_range = (0, int(trial_len / 50))

# PCA with time variance axis:
fig = plt.figure()
plt.ion()
plot_axes_3d = fig.add_subplot(111, projection='3d')
t_L, L, S, T, T_all = analysis.pcaL2_with_time_variance(
    input_NWBfile=NWBfiles_1mediumdend[5],
    smooth=True, plot_3d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_3d,
    stim_and_delay_range=all_range,
    delay_range=delay_range
)
plt.show()
fig.savefig('Murray_2.pdf')
fig.savefig('Murray_2.png')

analysis.stim_variance_captured(
    input_NWBfile=NWBfiles_1mediumdend[5],
    S=S,
    T=T,
    T_all=T_all,
    stim_and_delay_range=all_range,
    delay_range=delay_range
)
plt.gcf().savefig('Murray_3c.pdf')
plt.gcf().savefig('Murray_3c.png')
'''


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
plt.gcf().savefig('Dend_compare_1dend_all_2d_new.pdf')
plt.gcf().savefig('Dend_compare_1dend_all_2d_new.png')

#Kanonika prepei na pairnw se olous PA, alla pairnw? einai ypopta ligotera ta clusters,
# otan exw perissoterous dendrites..
analysis.compare_dend_params([
    NWBfiles_2smalldend,
    NWBfiles_2mediumdend,
    NWBfiles_2longdend
],[
    'NWBfiles_2smalldend',
    'NWBfiles_2mediumdend',
    'NWBfiles_2longdend'
])
plt.gcf().savefig('Dend_compare_2dend_all_2d_new.pdf')
plt.gcf().savefig('Dend_compare_2dend_all_2d_new.png')

fig,ax = plt.subplots()
ax.plot([0,1,2,3,4],[2,12,20,10,5])
ax.set_ylabel('Number of states')
ax.set_xlabel('Total dendritic length')
plt.gcf().savefig('Total_dend_length.pdf')
ax.xaxis.set_ticks = [0,1,2,3,4]
ax.set_tickslabels = [10,20,50,100,200]
plt.show()
plt.gcf().savefig('Total_dend_length.pdf')
plt.gcf().savefig('Total_dend_length.png')


K_labels = np.array(list(chain(
    *[ [1] * cp_trials_len, \
       [2] * cp_trials_len] \
    )))
NWBfile_array = list(chain(*[
    NWBfiles_1longdend,
    NWBfiles_2longdend
]))
newfa = [NWBfile_array[0], NWBfile_array[1], NWBfile_array[6], NWBfile_array[7]]
newkl = np.array([1,1,1,1,1,2,2,2,2,2])
fig = plt.figure()
plt.ion()
plot_axes_3d = fig.add_subplot(111, projection='3d')
analysis.pcaL2(
    NWBfile_array=newfa,
    custom_range=delay_range,
    klabels=newkl,
    smooth=True, plot_3d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_3d
)
plt.title("Long")

K_labels = np.array(list(chain(
    *[ [1] * cp_trials_len] \
    )))
NWBfile_array = list(chain(*[
    NWBfiles_1smalldend,
]))

K_labels = np.array(list(chain(
    *[ [1] * cp_trials_len, \
       [2] * cp_trials_len] \
    )))
NWBfile_array = list(chain(*[
    NWBfiles_1smalldend,
    NWBfiles_2smalldend
]))

K_labels = np.array(list(chain(
    *[ [1] * analysis.get_nwb_list_ntrials(NWBfiles_1mediumdend), \
       [2] * analysis.get_nwb_list_ntrials(NWBfiles_2mediumdend)] \
    )))
NWBfile_array = list(chain(*[
    NWBfiles_1mediumdend,
    NWBfiles_2mediumdend
]))

#prosoxh, 8elw to length twn trials!
K_labels = np.array(list(chain(
    *[ [1] * cp_trials_len] \
    )))
NWBfile_array = list(chain(*[
    NWBfiles_1longdend
]))

K_labels = np.array(list(chain(
    *[ [1] * cp_trials_len, \
       [2] * cp_trials_len] \
    )))
NWBfile_array = list(chain(*[
    NWBfiles_1longdend,
    NWBfiles_2longdend
]))

K_labels = np.array(list(chain(
    *[ [1] * cp_trials_len, \
       [2] * cp_trials_len, \
      [3] * cp_trials_len, \
      [4] * cp_trials_len] \
    )))
NWBfile_array = list(chain(*[
    NWBfiles_1mediumdend,
    NWBfiles_2mediumdend,
    NWBfiles_1longdend,
    NWBfiles_2longdend
]))

blah, *_ = analysis.get_binned_activity(NWBfiles_1longdend[0])
plt.figure()
plt.imshow(blah[:,0,:])

K_star_1, * _ = analysis.determine_number_of_clusters(
    NWBfile_array=NWBfiles_1longdend,
    max_clusters=20,
    custom_range=delay_range
)
K_star_2, * _ = analysis.determine_number_of_clusters(
    NWBfile_array=NWBfiles_2longdend,
    max_clusters=20,
    custom_range=delay_range
)

fig = plt.figure()
plt.ion()
plot_axes_3d = fig.add_subplot(111, projection='3d')
analysis.pcaL2(
    NWBfile_array=NWBfile_array,
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_3d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_3d
)


fig = plt.figure()
plt.ion()
plot_axes_2d = fig.add_subplot(111)
# Plot different stimulated clusters on their on color
analysis.pcaL2(
    NWBfile_array=NWBfile_array,
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_2d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_2d
)
plt.title('Medium VS Lond dend attractor landscape')


# <codecell>
#===============================================================================
# Calculate parameter range having PA (CLUSTERS, 2 medium dendrite):
#===============================================================================
# Preload the per cluster E/Isparse threshold.
# This is a E/I parameter estimation for the 1dend small model for different
# clusters:
i_range = np.arange(1.0, 3.1, 0.1)

cp_array = [2,3,4,5,6,7]
persistent_percent = np.full((len(i_range), len(cp_array)), 0.0, dtype=float)
for p, cp in enumerate(cp_array):
    for i, inhibias in enumerate(i_range):
        tmp_array = []
        trials = (cp-1)*10
        NWBfile = analysis.load_nwb_from_neuron(
            glia_dir,
            reload_raw=False,
            new_params={
                'excitation_bias': 1.0,
                'inhibition_bias': inhibias,
                'nmda_bias': 6.0,
                'ntrials': trials,
                'sim_duration': 2,
                'cp': cp,
                'experiment_config': 'structured_2mediumdend'
            }
        )
        if NWBfile:
            for t in range(cp):
                tmp_array.append(
                    int(analysis.nwb_iter(
                        NWBfile.trials['persistent_activity']
                    ).__next__()) )
            persistent_percent[i, p] = np.mean(tmp_array)


fig, ax = plt.subplots()
plt.imshow(persistent_percent[:,:])
plt.yticks(range(len(i_range)), np.around(i_range, decimals=1))
plt.xticks(range(len(cp_array)), cp_array)
plt.xlabel('Cluster ID Bias')
plt.ylabel('Inhibition Bias')
plt.gca().invert_yaxis()
plt.show()
print("Tutto pronto!")



# <codecell>
#===============================================================================
# Calculate parameter range having PA (CLUSTERS, small dendrite):
#===============================================================================
# Preload the per cluster E/Isparse threshold.
# This is a E/I parameter estimation for the 1dend small model for different
# clusters:
i_range = np.arange(1.0, 3.1, 0.1)

cp_array = [2,3,4,5,6,7]
persistent_percent = np.full((len(i_range), len(cp_array)), 0.0, dtype=float)
for p, cp in enumerate(cp_array):
    for i, inhibias in enumerate(i_range):
        tmp_array = []
        trials = (cp-1)*10
        NWBfile = analysis.load_nwb_from_neuron(
            glia_dir,
            reload_raw=False,
            new_params={
                'excitation_bias': 1.0,
                'inhibition_bias': inhibias,
                'nmda_bias': 6.0,
                'ntrials': trials,
                'sim_duration': 2,
                'cp': cp,
                'experiment_config': 'structured_1smalldend'
            }
        )
        if NWBfile:
            for t in range(cp):
                tmp_array.append(
                    int(analysis.nwb_iter(
                        NWBfile.trials['persistent_activity']
                    ).__next__()) )
            persistent_percent[i, p] = np.mean(tmp_array)


fig, ax = plt.subplots()
plt.imshow(persistent_percent[:,:])
plt.yticks(range(len(i_range)), np.around(i_range, decimals=1))
plt.xticks(range(len(cp_array)), cp_array)
plt.xlabel('Cluster ID Bias')
plt.ylabel('Inhibition Bias')
plt.gca().invert_yaxis()
plt.show()
print("Tutto pronto!")

# <codecell>
#===============================================================================
# Compare attractor landscape while dend length increases
#===============================================================================

cp_array = [2,3,4,5,6,7]
NWBfile_array = []
K_labels_array = []
for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 2.0,
            'nmda_bias': 6.0,
            'ntrials': (cp-1)*10,
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_1smalldend'
        }
    )
    if NWBfile:
        NWBfile_array.append(NWBfile)
        K_labels_array.append(1)
for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 2.0,
            'nmda_bias': 6.0,
            'ntrials': (cp-1)*10,
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_1mediumdend'
        }
    )
    if NWBfile:
        NWBfile_array.append(NWBfile)
        K_labels_array.append(2)
for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 2.0,
            'nmda_bias': 6.0,
            'ntrials': (cp-1)*10,
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_1longdend'
        }
    )
    if NWBfile:
        NWBfile_array.append(NWBfile)
        K_labels_array.append(3)

trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len']
)
delay_range = (20, int(trial_len / 50))
all_range = (0, int(trial_len / 50))
K_labels = np.array(K_labels_array)

fig = plt.figure()
plt.ion()
plot_axes_3d = fig.add_subplot(111, projection='3d')
fig = plt.figure()
plt.ion()
plot_axes_2d = fig.add_subplot(111)

# Plot different stimulated clusters on their on color
analysis.pcaL2(
    NWBfile_array=NWBfile_array,
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_3d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_3d
)
analysis.pcaL2(
    NWBfile_array=NWBfile_array,
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_2d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_2d
)
plt.title('Medium VS Lond dend attractor landscape')

# <codecell>
#===============================================================================
# Calculate parameter range having PA (CLUSTERS, long dendrite):
#===============================================================================
# Preload the per cluster E/Isparse threshold.
# This is a E/I parameter estimation for the 1dend small model for different
# clusters:
i_range = np.arange(1.0, 3.1, 0.1)

cp_array = [2,3,4,5,6,7]
persistent_percent = np.full((len(i_range), len(cp_array)), 0.0, dtype=float)
for p, cp in enumerate(cp_array):
    for i, inhibias in enumerate(i_range):
        tmp_array = []
        trials = (cp-1)*10
        NWBfile = analysis.load_nwb_from_neuron(
            glia_dir,
            reload_raw=False,
            new_params={
                'excitation_bias': 1.0,
                'inhibition_bias': inhibias,
                'nmda_bias': 6.0,
                'ntrials': trials,
                'sim_duration': 2,
                'cp': cp,
                'experiment_config': 'structured_1longdend'
            }
        )
        if NWBfile:
            for t in range(cp):
                tmp_array.append(
                    int(analysis.nwb_iter(
                    NWBfile.trials['persistent_activity']
                    ).__next__()) )
            persistent_percent[i, p] = np.mean(tmp_array)


fig, ax = plt.subplots()
plt.imshow(persistent_percent[:,:])
plt.yticks(range(len(i_range)), np.around(i_range, decimals=1))
plt.xticks(range(len(cp_array)), cp_array)
plt.xlabel('Cluster ID Bias')
plt.ylabel('Inhibition Bias')
plt.gca().invert_yaxis()
plt.show()

# <codecell>
#===============================================================================
# Calculate parameter range having PA (CLUSTERS):
#===============================================================================
# This is a E/I parameter estimation for the 1dend small model for different
# clusters:

cp_array = [2,3,4,5,6,7]
NWBfile_array = []
K_labels_array = []
for p, cp in enumerate(cp_array):
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 2.0,
            'nmda_bias': 6.0,
            'ntrials': (cp-1)*10,
            'sim_duration': 5,
            'cp': cp,
            'experiment_config': 'structured_1longdend'
        }
    )
    if NWBfile:
        NWBfile_array.append(NWBfile)
        K_labels_array.append([cp-1]*cp)

trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len']
)
delay_range = (20, int(trial_len / 50))
all_range = (0, int(trial_len / 50))
K_labels = np.array(list(chain(*K_labels_array)))

fig = plt.figure()
plt.ion()
plot_axes_3d = fig.add_subplot(111, projection='3d')
fig = plt.figure()
plt.ion()
plot_axes_2d = fig.add_subplot(111)

# Plot different stimulated clusters on their on color
analysis.pcaL2(
    NWBfile_array=NWBfile_array,
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_3d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_3d
)
analysis.pcaL2(
    NWBfile_array=NWBfile_array,
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_2d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_2d
)
# Plot the same clusters, grouped together with k-means, to show the attractors.
K_star, K_labels_kmeans, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=NWBfile_array,
    max_clusters=20,
    custom_range=delay_range
)
fig = plt.figure()
plt.ion()
plot_axes_2 = fig.add_subplot(111, projection='3d')
analysis.pcaL2(
    NWBfile_array=NWBfile_array,
    custom_range=delay_range,
    klabels=K_labels_kmeans,
    smooth=True, plot_3d=True,
    plot_stim_color=False,
    plot_axes=plot_axes_2
)



# <codecell>
NWBfile = analysis.load_nwb_from_neuron(
glia_dir,
reload_raw=False,
include_membrane_potential=False,
new_params={
    'excitation_bias': 1.0,
    'inhibition_bias': 2.0,
    'nmda_bias': 6.0,
    'ntrials': 1,
    'sim_duration': 2,
    'cp': 2,
    'experiment_config': 'structured_1smalldend'
}
)

# <codecell>
vsoma = analysis.get_acquisition_potential(
    NWBfile=NWBfile, cellid=190, trialid=0
)
fig, ax = plt.subplots()
plt.ion()
ax.plot(vsoma, color='white')
ax.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
plt.show()
# <codecell>
trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len']
)
delay_range = (20, int(trial_len / 50))
all_range = (0, int(trial_len / 50))
data, *_ = analysis.get_binned_activity(NWBfile, custom_range=all_range)
plt.imshow(data[:,0,:])

# <codecell>
#===============================================================================
# Calculate parameter range having PA (CLUSTERS):
#===============================================================================
# This is a E/I parameter estimation for the 1dend small model for different
# clusters:
e_range = np.arange(1.0, 7.0, 1.0)
i_range = np.arange(1.0, 7.0, 1.0)

cp_array = [2,3,4,5,6,7]
has_persistent = np.full((len(i_range), len(e_range), len(cp_array)), False, dtype=bool)
for e, excitbias in enumerate(e_range):
    for i, inhibias in enumerate(i_range):
        for p, cp in enumerate(cp_array):
            NWBfile = analysis.load_nwb_from_neuron(
                glia_dir,
                reload_raw=False,
                new_params={
                    'excitation_bias': excitbias,
                    'inhibition_bias': inhibias,
                    'nmda_bias': 6.0,
                    'ntrials': 1,
                    'sim_duration': 2,
                    'cp': cp,
                    'experiment_config': 'structured_1longdend'
                }
            )
            if NWBfile:
                has_persistent[i, e, p] = analysis.nwb_iter(
                    NWBfile.trials['persistent_activity']
                ).__next__()

clst=3
fig, ax = plt.subplots()
plt.imshow(has_persistent[:,:,clst-1])
plt.xlabel('Excitation Bias')
plt.ylabel('Inhibition Bias')
plt.gca().invert_yaxis()
plt.title(f"Long dend, C={clst}")


# <codecell>
#===============================================================================
# Calculate parameter range having PA:
#===============================================================================
# This is a E/I parameter estimation for the 2dend small model:

e_range = np.arange(1.0, 5.0, 1.0)
i_range = np.arange(1.0, 5.0, 1.0)

sps_array = [20,40,60,80,100]
has_persistent = np.full((len(i_range), len(e_range), len(sps_array)), False, dtype=bool)
for e, excitbias in enumerate(e_range):
    for i, inhibias in enumerate(i_range):
        for p, sps in enumerate(sps_array):
            NWBfile = analysis.load_nwb_from_neuron(
                glia_dir,
                reload_raw=False,
                new_params={
                    'excitation_bias': excitbias,
                    'inhibition_bias': inhibias,
                    'nmda_bias': 6.0,
                    'ntrials': 1,
                    'sim_duration': 2,
                    'sps': sps,
                    'experiment_config': 'structured_2smalldend'
                }
            )
            if NWBfile:
                has_persistent[i, e, p] = analysis.nwb_iter(
                    NWBfile.trials['persistent_activity']
                ).__next__()

fig, ax = plt.subplots()
plt.imshow(has_persistent[:,:,0])
plt.gca().invert_yaxis()

# <codecell>
#===============================================================================
# Calculate parameter range having PA:
#===============================================================================
# This is a E/I parameter estimation for the 2dend normal model:

e_range = np.arange(1.0, 6.0, 1.0)
i_range = np.arange(1.0, 6.0, 1.0)

sps_array = [50]
has_persistent = np.full((len(i_range), len(e_range), len(sps_array)), False, dtype=bool)
for e, excitbias in enumerate(e_range):
    for i, inhibias in enumerate(i_range):
        for p, sps in enumerate(sps_array):
            NWBfile = analysis.load_nwb_from_neuron(
                glia_dir,
                reload_raw=False,
                new_params={
                    'excitation_bias': excitbias,
                    'inhibition_bias': inhibias,
                    'nmda_bias': 6.0,
                    'ntrials': 1,
                    'sim_duration': 2,
                    'sps': sps,
                    'experiment_config': 'structured_2normaldend'
                }
            )
            if NWBfile:
                has_persistent[i, e, p] = analysis.nwb_iter(
                    NWBfile.trials['persistent_activity']
                ).__next__()

fig, ax = plt.subplots()
plt.imshow(has_persistent[:,:,0])
plt.xlabel('Excitation Bias')
plt.ylabel('Inhibition Bias')
plt.gca().invert_yaxis()
# <codecell>
# This is the soma voltage for specific runs in the 2dend normal model, to 
# validate that the spiking activity is not atrocious.
NWBfile = analysis.load_nwb_from_neuron(
    glia_dir,
    reload_raw=True,
    include_membrane_potential=True,
    new_params={
        'excitation_bias': 1.0,
        'inhibition_bias': 2.0,
        'nmda_bias': 6.0,
        'ntrials': 1,
        'sim_duration': 2,
        'sps': 50,
        'experiment_config': 'structured_2normaldend'
    }
)
vsoma = analysis.get_acquisition_potential(
    NWBfile=NWBfile, cellid=[2,3], trialid=0
)
fig, ax = plt.subplots()
ax.plot(vsoma, color='white')
ax.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
# Ta apotelesmata fainontai normal gia Eb=1, Ib=2. kai poly kalo FF.
# Na tre3w runs, wste na parw apotelesmata gia thn posothta twn attractors.

# <codecell>
# This cell performs PCA on the previous 2dend normal model, within their
# good responding parameter space.
NWBfile = analysis.load_nwb_from_neuron(
    glia_dir,
    reload_raw=False,
    include_membrane_potential=False,
    new_params={
        'excitation_bias': 1.0,
        'inhibition_bias': 2.0,
        'nmda_bias': 6.0,
        'ntrials': 50,
        'sim_duration': 5,
        'sps': 50,
        'experiment_config': 'structured_2normaldend'
    }
)
trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len']
)
delay_range = (20, int(trial_len / 50))
# This should be reproducable with the call above:
K_star, K_labels, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=[NWBfile],
    max_clusters=20,
    custom_range=delay_range
)
analysis.pcaL2(
    NWBfile_array=[NWBfile],
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_3d=True,
    plot_stim_color=False
)

# <codecell>
trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=NWBfile,
    requested_parameters=['trial_len']
)
custom_range = (20, int(trial_len / 50))
data = analysis.get_binned_activity(NWBfile, custom_range=custom_range)

fig, ax = plt.subplots()
plt.ion()
plt.show()
plt.imshow(data[:,0,:])

# <codecell>
vsoma = analysis.get_acquisition_potential(
    NWBfile=NWBfile, cellid=11, trialid=0
)
fig, ax = plt.subplots()
plt.ion()
plt.show()
ax.plot(vsoma, color='k')
ax.axvspan(50.0, 1050.0, ymin=0, ymax=1, color='g', alpha=0.2)
#nb.hide_axis_border(axis=ax)

# <codecell>
#===============================================================================
# Load specific trials and check their PCA activity.
#===============================================================================
e_range = np.arange(1.0, 2.8, 0.2)
i_range = np.arange(1.0, 2.6, 0.2)
for sps in [20]:
    NWBfile = analysis.load_nwb_from_neuron(
        glia_dir,
        reload_raw=False,
        new_params={
            'excitation_bias': 1.0,
            'inhibition_bias': 1.0,
            'nmda_bias': 6.0,
            'ntrials': 20,
            'sps': sps,
            'sim_duration': 5,
            'experiment_config': 'structured_2smalldend'
        }
    )
    #data , *_= analysis.get_binned_activity(NWBfile)
    #fig, ax = plt.subplots()
    #plt.ion()
    #plt.show()
    #plt.imshow(data[:,0,:])

# <codecell>
filename = r'\\139.91.162.90\cluster\stefanos\Documents\Glia\SN1LC1TR19_EB1.000_IB1.000_GBF2.000_NMDAb6.000_AMPAb1.000_SPS20_structured_2smalldend_simdur5.nwb'
nwbfile = NWBHDF5IO(str(filename), 'r').read()
trial_len = analysis.get_acquisition_parameters(
    input_NWBfile=nwbfile,
    requested_parameters=['trial_len']
)
delay_range = (20, int(trial_len / 50))
# This should be reproducable with the call above:
K_star, K_labels, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=[nwbfile],
    max_clusters=20,
    custom_range=delay_range
)
analysis.pcaL2(
    NWBfile_array=[nwbfile],
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_3d=True,
    plot_stim_color=False
)

# <codecell>
t_L, L, S, T = analysis.pcaL2_with_time_variance(
    input_NWBfile=NWBfile,
    custom_range=delay_range,
    smooth=True, plot_3d=True,
    plot_axes=plot_axes,
    axis_label_font_size=axis_label_font_size,
    tick_label_font_size=tick_label_font_size,
    labelpad_x=labelpad_x,
    labelpad_y=labelpad_y,
    plot_stim_color=False
)
plt.ion()
plt.show()

# <codecell>
fig = plt.figure()
plot_axes = fig.add_subplot(111, projection='3d')
# This should be reproducable with the call above:
K_star, K_labels, BIC_val, _ = analysis.determine_number_of_clusters(
    NWBfile_array=[NWBfile],
    max_clusters=no_of_conditions,
    custom_range=delay_range
)
analysis.pcaL2(
    NWBfile_array=[NWBfile],
    custom_range=delay_range,
    klabels=K_labels,
    smooth=True, plot_3d=True,
    plot_axes=plot_axes,
    axis_label_font_size=axis_label_font_size,
    tick_label_font_size=tick_label_font_size,
    labelpad_x=labelpad_x,
    labelpad_y=labelpad_y,
    plot_stim_color=False
)
plt.ion()
plt.show()

analysis.stim_variance_captured(
    input_NWBfile=NWBfile,
    S=S,
    T=T,
    custom_range=delay_range,
)



# <codecell>
#===============================================================================
# Validate multidend neuron.
#===============================================================================
if False:
    valid_dir = Path(r'G:\Glia\publication_validation\excitatory_validation_multidend')
    read_potential = partial(
        analysis.read_validation_potential,
        inputdir=valid_dir,
        synapse_activation=list(range(1, 5)),
        condition='normal',
        currents='AMPA',
        nmda_bias=0.0,
        ampa_bias=1.0,
        trialid=3
    )
    vsoma = read_potential(
        location='vsomamd'
    )
    vsomamd = read_potential(
        location='vdendmd',
    )
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    plt.plot(vsoma, color='blue')
    plt.plot(vsomamd, color='red')
    print('Tutto pronto!')






# <codecell>
print('Tutto pronto!')


