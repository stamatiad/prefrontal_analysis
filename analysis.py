from analysis_tools import quick_spikes
from analysis_tools import q_generator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pandas as pd
import h5py
from contextlib import contextmanager
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

'''
@contextmanager
def open_hdf_dataframe(filename=None, hdf_key=None):
    try:
        yield pd.read_hdf(filename, key=hdf_key)
    except Exception:
        print(f'File ({filename}) or key ({hdf_key}) not found!')
        yield
    finally:
        pass
    pass
'''
window_size = 50
total_qs = int(np.floor(3000 / window_size))
ntrials = 10

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

def voltage_to_network_activity():
    # Convert voltage traces to spike trains:
    cellno = 333
    upper_threshold = 0
    lower_threshold = -10

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

    # For multiple runs:
    # Use panda dataframes to keep track across learning conditions and animal models.
    # Group multiple trials in the same dataframe, since these are our working unit.
    filename_hdf5 = Path(fr'G:\Glia\all_data.hdf5')
    open(filename_hdf5, 'w').close()

    animal_models = range(1, 4 + 1)
    learning_conditions = range(1, 10 + 1)
    trials = range(10)

    for animal_model in animal_models:
        for learning_condition in learning_conditions:
            windowed_activity = np.zeros((ntrials, total_qs, cellno), dtype=int)
            data_key = f'SN{animal_model}LC{learning_condition}'
            print(f'Handling: {data_key}')
            for trial in trials:
                filename = Path('G:\Glia').joinpath(
                    simulation_template(animal_model=animal_model,
                                        learning_condition=learning_condition,
                                        trial=trial)
                ).joinpath('vsoma.hdf5')

                #df = pd.read_hdf(Path('G:\Glia').joinpath(filename).joinpath('vsoma.hdf5'), key='vsoma')
                with open_hdf_dataframe(filename=filename, hdf_key='vsoma') as df:
                    if df is not None:
                        # Convert dataframe to ndarray:
                        voltage_traces = df.values
                        # Reduce each voltage trace to a list of spike times:
                        spike_trains = [
                            quick_spikes(voltage_trace=voltage_trace,
                                         upper_threshold=upper_threshold,
                                         lower_threshold=lower_threshold,
                                         plot=False)
                            for voltage_trace in voltage_traces
                        ]
                        # Sum spikes inside a window Q:
                        for cellid, spike_train in enumerate(spike_trains):
                            if len(spike_train) > 0:
                                for q, (q_start, q_end) in enumerate(q_generator(q_size=window_size, q_total=total_qs)):
                                    windowed_activity[trial][q][cellid] = sum([1 for spike in spike_train if q_start <= spike and spike < q_end])
            #fig, ax = plt.subplots()
            #ax.imshow(windowed_activity[trial])
            #plt.show()
            #fig, ax = plt.subplots()
            #ax.imshow(windowed_activity.reshape(total_qs * ntrials, cellno, order='C'))
            #plt.show()
            df = pd.DataFrame(windowed_activity.reshape(total_qs * ntrials, cellno, order='C'))
            df.to_hdf(filename_hdf5, key=data_key, mode='a')


#===================================================================================================================
def read_network_activity(filename=None, data_key=None, plot=False):
    data = None
    # Read spiketrains and plot them
    with open_hdf_dataframe(filename=filename, hdf_key=data_key) as df:
        if df is not None:
            data = df.values
    return data

def pcaL2(data=None, plot=False):
    # how many components
    L = 2
    pca = decomposition.PCA(n_components=L)
    t_L = pca.fit_transform(data).T
    t_L_reshaped = t_L.reshape(L, ntrials, total_qs, order='C')
    if plot:
        #fig, ax = plt.subplots()
        #ax.imshow(data)
        #plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for trial in range(ntrials):
            ax.plot(t_L_reshaped[0][trial], t_L_reshaped[1][trial], range(total_qs))
        plt.show()

#===================================================================================================================
# Do k-means in spiketrains:
if __name__ == '__main__':
    # Parse voltage traces into network activity matrices:
    #voltage_to_network_activity()

    # Load and plot network activity:
    filename_hdf5 = Path(fr'G:\Glia\all_data.hdf5')
    animal_model = 1
    learning_condition = 4
    data_key = f'SN{animal_model}LC{learning_condition}'
    data = read_network_activity(filename=filename_hdf5, data_key=data_key, plot=True)
    #fig, ax = plt.subplots()
    #ax.imshow(data)
    #plt.show()

    pcaL2(data=data, plot=True)


    # k-means cluster network activity:
    #k_means_cluster_network_activity()
