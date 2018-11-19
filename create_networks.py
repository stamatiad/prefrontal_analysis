"""
Created on Tue Nov  6 13:07:01 2018

@author: stefanos
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from network_tools import Network
from collections import namedtuple
import sys
import time

# Check that user is running a supported python version:
py_major = sys.version_info[0]
py_minor = sys.version_info[1]
if py_major < 3 or py_minor < 7:
    raise Exception('You are running an unsupported python version ({}.{}). Please use 3.7 or newer!' \
                    .format(py_major, py_minor))

# Create a network configuration with given seed (for reproducability)
net = Network(serial_no=1, pc_no=250, pv_no=round(250*25/75))
np.random.seed(net.serial_no)

#Number of neurons
#net.pc_no = 250
#net.pv_no = round(net.pc_no*25/75)

# A cube of side 180 um, has about 60000 cells.
net.populate_network(cube_side_len=180, plot=False)

# Make network connections utilizing the above functions:
tic = time.perf_counter()
net.create_connections()
toc = time.perf_counter()
print('Create Connections time {}'.format(toc-tic))

#net.create_weights()

# Save Network parameters and data to a HDF5 file:
net.save_data()
pass

