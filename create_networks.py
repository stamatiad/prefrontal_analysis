"""
Created on Tue Nov  6 13:07:01 2018

@author: Stamatiadis Stefanos
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from net_structuredwork_tools import Network
from collections import namedtuple
import sys
import time
import copy

# Check that user is running a supported python version:
py_major = sys.version_info[0]
py_minor = sys.version_info[1]
if py_major < 3 or py_minor < 7:
    raise Exception('You are running an unsupported python version ({}.{}). Please use 3.7 or newer!' \
                    .format(py_major, py_minor))

# set parameters:
pn_no = 250
pv_no = round(250*25/75)
cube_side_length = 180  # um
trial_no = 100

# Create a net_structuredwork configuration with given serial number (used to seed the RNG for reproducability)
net_structured = Network(serial_no=1, pc_no=pn_no, pv_no=pv_no)

# A cube of side 180 um, has about 60000 cells.
net_structured.populate_net_structuredwork(cube_side_len=cube_side_length, plot=False)

# Create both structured and random configurations:
tic = time.perf_counter()
net_structured.connectivity_mat = net_structured.create_connections(configuration='structured', rearrange_iterations=10, plot=True)
toc = time.perf_counter()
print('Create Connections time {}'.format(toc-tic))

#net_structured.create_weights()

# Initialize 100 trials:
net_structured.initialize_trials(trial_no=trial_no)

# Export parameters to NEURON hoc files:
net_structured.export_net_structuredwork_parameters(configuration='structured')
net_structured.export_net_structuredwork_parameters(configuration='random')
net_structured.export_stimulation_parameters()

# Save Network parameters and data to a HDF5 file:
net_structured.save_data()

# Create a random network as a copy of the structured:
net_random = copy.deepcopy(net_structured)

# Change random net connectivity and weights:
tic = time.perf_counter()
net_random.configurations['random'] = net_random.create_connections(configuration='random', uniform_probability=0.1752, plot=True)
toc = time.perf_counter()
print('Create Connections time {}'.format(toc-tic))

#net_random.create_weights()

# Initialize 100 trials:
net_random.initialize_trials(trial_no=trial_no)

# Export parameters to NEURON hoc files:
net_random.export_net_randomwork_parameters(configuration='structured')
net_random.export_net_randomwork_parameters(configuration='random')
net_random.export_stimulation_parameters()

# Save Network parameters and data to a HDF5 file:
net_random.save_data()
pass

