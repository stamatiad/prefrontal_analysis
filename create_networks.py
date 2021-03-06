"""
Created on Tue Nov  6 13:07:01 2018

@author: Stamatiadis Stefanos
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from network_tools import Network
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
serial_no = 4

# Create a network configuration with given serial number (used to seed the RNG for reproducability)
net_structured = Network(serial_no=serial_no, pc_no=pn_no, pv_no=pv_no)

# A cube of side 180 um, has about 60000 cells.
net_structured.populate_network(cube_side_len=cube_side_length, plot=False)

# Create both structured and random configurations:
tic = time.perf_counter()
net_structured.create_connections(alias='structured', rearrange_iterations=1000, plot=False)
toc = time.perf_counter()
print('Create Connections time {}'.format(toc-tic))

net_structured.create_weights()
net_structured.create_network_stats()

# Initialize 100 trials:
net_structured.initialize_trials(trial_no=trial_no)

# Export parameters to NEURON hoc files:
net_structured.export_network_parameters()

# Save Network parameters and data to a HDF5 file:
net_structured.save_data()

## Create a random network as a copy of the structured:
#net_random = copy.deepcopy(net_structured)
# Create a network configuration with given serial number (used to seed the RNG for reproducability)
net_random = Network(serial_no=serial_no, pc_no=pn_no, pv_no=pv_no)

# A cube of side 180 um, has about 60000 cells.
net_random.populate_network(cube_side_len=cube_side_length, plot=False)

# Change random net connectivity and weights:
overall_conn_prob = net_structured.stats['averageConnectivity']
# Create a random/uniform connected network, with the same overall connection probability as the structured one.
tic = time.perf_counter()
net_random.create_connections(alias='random', average_conn_prob=overall_conn_prob, plot=False)
toc = time.perf_counter()
print('Create Connections time {}'.format(toc-tic))

net_random.create_weights()
net_random.create_network_stats()

# Initialize 100 trials:
net_random.initialize_trials(trial_no=trial_no)

# Export parameters to NEURON hoc files:
net_random.export_network_parameters()

# Save Network parameters and data to a HDF5 file:
net_random.save_data()
pass

