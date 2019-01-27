"""
Created on Tue Nov  6 13:07:01 2018

@author: Stamatiadis Stefanos
"""
import network_tools as nt

# Check python version and installed packages:
nt.check_requirements()

# set parameters:
pn_no = 250
pv_no = round(250*25/75)
cube_side_length = 180  # um
trial_no = 100
serial_no = 1

for serial_no in range(1, 5):
    # Create a network configuration with given serial number (used to seed the RNG for reproducability)
    net_structured = nt.Network(serial_no=serial_no, pc_no=pn_no, pv_no=pv_no)

    # A cube of side 180 um, has about 60000 cells.
    net_structured.populate_network(cube_side_len=cube_side_length, plot=False)

    # Create both structured and random configurations:
    net_structured.create_connections(alias='structured', rearrange_iterations=1000, plot=False)

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
    net_random = nt.Network(serial_no=serial_no, pc_no=pn_no, pv_no=pv_no)

    # A cube of side 180 um, has about 60000 cells.
    net_random.populate_network(cube_side_len=cube_side_length, plot=False)

    # Change random net connectivity and weights:
    overall_conn_prob = net_structured.stats['averageConnectivity']
    # Create a random/uniform connected network, with the same overall connection probability as the structured one.
    net_random.create_connections(alias='random', average_conn_prob=overall_conn_prob, plot=False)

    net_random.create_weights()
    net_random.create_network_stats()

    # Initialize 100 trials:
    net_random.initialize_trials(trial_no=trial_no)

    # Export parameters to NEURON hoc files:
    net_random.export_network_parameters()

    # Save Network parameters and data to a HDF5 file:
    net_random.save_data()
