#from analysis_tools import quick_spikes
#from analysis_tools import generate_slices_g
import analysis_tools as at
from analysis_tools import from_zero_to, from_one_to
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pandas as pd
import h5py
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
from scipy import spatial

# Quick class, containing the properties of the analysis:
Properties = namedtuple('Properties', [
    'sim_stop',
    'q_size',
    'total_qs',
    'ntrials'
])

class open_hdf_dataframe():

    def __init__(self, filename=None, hdf_key=None):
        self.filename = filename
        self.hdf_key = hdf_key

    def __enter__(self):
        try:
            return pd.read_hdf(self.filename, key=self.hdf_key)
        except Exception as e:
            print(f'File ({self.filename}) or key ({self.hdf_key}) not found!')
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    pass

experiment_config_str = 'SN{animal_model}LC{learning_condition}'.format

simulation_template = (
    'SN{animal_model}'
    'LC{learning_condition}'
    'TR{trial}'
    '_EB1.750'
    '_IB1.500'
    '_GBF2.000'
    '_NMDAb6.000'
    '_AMPAb1.000'
    '_randomdur3').format

def simulation_to_network_activity(tofile=None, animal_models=None, learning_conditions=None, properties=None):
    # Convert voltage traces to spike trains:
    cellno = 333
    upper_threshold = 0
    lower_threshold = -10

    # For multiple runs:
    # Use panda dataframes to keep track across learning conditions and animal models.
    # Group multiple trials in the same dataframe, since these are our working unit.
    # Touch the output file:
    open(tofile, 'w').close()


    for animal_model in animal_models:
        for learning_condition in learning_conditions:
            windowed_activity = np.zeros((properties.ntrials, properties.total_qs, cellno), dtype=int)
            #data_key = f'SN{animal_model}LC{learning_condition}'
            dataset = experiment_config_str(
                animal_model=animal_model,
                learning_condition=learning_condition
            )
            print(f'Handling: {dataset}')
            for trial in range(properties.ntrials):
                inputfile = Path('G:\Glia')\
                    .joinpath(
                    simulation_template(animal_model=animal_model,
                                        learning_condition=learning_condition,
                                        trial=trial))\
                    .joinpath('vsoma.hdf5')
                #with open_hdf_dataframe(filename=filename, hdf_key='vsoma') as df:
                if inputfile.exists():
                    # Convert dataframe to ndarray:
                    voltage_traces = pd.read_hdf(inputfile, key='vsoma').values
                    # Reduce each voltage trace to a list of spike times:
                    spike_trains = [
                        at.quick_spikes(voltage_trace=voltage_trace,
                                     upper_threshold=upper_threshold,
                                     lower_threshold=lower_threshold,
                                     plot=False)
                        for voltage_trace in voltage_traces
                    ]
                    # Sum spikes inside a window Q:
                    for cellid, spike_train in enumerate(spike_trains):
                        if len(spike_train) > 0:
                            for q, (q_start, q_end) in enumerate(at.generate_slices_g(size=properties.q_size, number=properties.total_qs)):
                                windowed_activity[trial][q][cellid] = sum([1 for spike in spike_train if q_start <= spike and spike < q_end])
            #fig, ax = plt.subplots()
            #ax.imshow(windowed_activity.reshape(total_qs * ntrials, cellno, order='C'))
            #plt.show()
            df = pd.DataFrame(windowed_activity.reshape(properties.total_qs * properties.ntrials, cellno, order='C'))
            df.to_hdf(tofile, key=dataset, mode='a')


#===================================================================================================================
def read_network_activity(fromfile=None, dataset=None, properties=None):
    data = None
    # Read spiketrains and plot them
    df = pd.read_hdf(fromfile, key=dataset)
    #with open_hdf_dataframe(filename=filename, hdf_key=data_key) as df:
    if df is not None:
        data = df.values.T
        # Also reshape the data into a 3d array:
        data = data.reshape(data.shape[0], properties.ntrials, properties.total_qs, order='C')

    return data

def pcaL2(data=None, plot=False, custom_range=None, properties=None):
    # how many components
    L = 2
    pca = decomposition.PCA(n_components=L)
    # Use custom_range to compute PCA only on a portion of the original data:
    new_len = len(custom_range)
    t_L = pca.fit_transform(
        data[:250, :, custom_range].reshape(250, properties.ntrials * new_len).T
    ).T
    t_L_reshaped = t_L.reshape(L, properties.ntrials, new_len, order='C')
    if plot:
        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(111, projection='3d')
        colors = cm.viridis(np.linspace(0, 1, new_len - 1))
        for trial in range(properties.ntrials):
            for t, c in zip(range(new_len - 1), colors):
                ax.plot(t_L_reshaped[0][trial][t:t+2], t_L_reshaped[1][trial][t:t+2], [t, t+1], color=c)
        plt.show()
    return t_L_reshaped

#===================================================================================================================
if __name__ == '__main__':
    # The file where network activity lies:
    network_activity_hdf5 = Path(fr'G:\Glia\all_data.hdf5')

    # Since namedtuple is immutable, pass arguments from a dictionary to make it easier on the eyes.
    p_d = {
        'sim_stop': 3000,
        'q_size': 50,
        'total_qs': None, # this must be calculated later on.
        'ntrials': 10
    }
    p_d['total_qs'] = int(np.floor(p_d['sim_stop'] / p_d['q_size']))
    p = Properties(**p_d)

    # Parse voltage traces into network activity matrices once and save to file for later use:
    '''
    simulation_to_network_activity(
        tofile=network_activity_hdf5,
        animal_models=from_one_to(1),
        learning_conditions=from_one_to(10),
        ntrials=10,
        properties=p
    )
    '''

    # Load and plot network activity:
    # cellno, ntrials, total_qs
    network_activity = read_network_activity(
        fromfile=network_activity_hdf5,
        dataset=experiment_config_str(
            animal_model=1,
            learning_condition=1
        ),
        properties=p
    )
    #fig, ax = plt.subplots()
    #ax.imshow(data)
    #plt.show()

    t_L = pcaL2(
        data=network_activity,
        custom_range=range(20, 60),
        plot=False,
        properties=p
    )

    # k-means cluster network activity:
    #k_means_cluster_network_activity()
    at.my_mean_shift(data=t_L, k=4, plot=True, properties=p, dims=2)
    print('here!')




