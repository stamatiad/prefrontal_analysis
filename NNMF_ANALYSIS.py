from pynwb import NWBFile, NWBHDF5IO
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
#===============================================================================
no_of_animals = 4
no_of_conditions = 10

K_star_mat = np.zeros((no_of_animals, no_of_conditions), dtype=int)

animal_model = 1
learning_condition = 5

print(f'NT:{animal_model}, LC:{learning_condition}')
data_dir = Path.cwd()
#inputfile = data_dir.joinpath(
#    f'cross_validation_errors_structured_AM{animal_model}_LC{learning_condition}_RI5.hdf'
#)
K_max = 10
K_cv = 20
rng_max_iters = 1
fn_str = (
    'cross_validation_errors_structured'
    '_AM{animal_model_id}'
    '_LC{learning_condition_id}'
    '_Kmax{K_max}'
    '_Kcv{K_cv}'
    '_RI{rng_max_iters}.hdf').format
inputfile = Path(fn_str(
    animal_model_id=animal_model,
    learning_condition_id=learning_condition,
    K_max=K_max,
    K_cv=K_cv,
    rng_max_iters=rng_max_iters
))

# Read CV results.
try:
    attribs = pd.read_hdf(inputfile, key='attributes').to_dict()
    K_cv = attribs['K_cv'][0]
    K_max = attribs['K_max'][0]
    rng_max_iters = attribs['rng_max_iters'][0]
    error_train = pd.read_hdf(inputfile, key='error_train') \
        .values.reshape(K_max, K_cv, rng_max_iters) \
        .mean(axis=2)
    error_test = pd.read_hdf(inputfile, key='error_test') \
        .values.reshape(K_max, K_cv, rng_max_iters) \
        .mean(axis=2)
    if False:
        error_train_rm = pd.read_hdf(inputfile, key='error_train_rm') \
            .values.reshape(K_max, K_cv, rng_max_iters) \
            .mean(axis=2)
        error_test_rm = pd.read_hdf(inputfile, key='error_test_rm') \
            .values.reshape(K_max, K_cv, rng_max_iters) \
            .mean(axis=2)
    fig,ax = plt.subplots()
    ax.plot(error_train, 'black')
    ax.plot(error_test, 'red')
    plt.savefig(str(inputfile)+'.png')

    K_str_cv = np.argmin(error_test.mean(axis=1))
    K_star_mat[animal_model - 1, learning_condition - 1] = K_str_cv

except Exception as e:
    print(f'Exception! {str(e)}')
    pass


#===============================================================================

nwbfile = NWBFile(session_description='ADDME',
                  identifier='ADDME',
                  session_start_time=datetime.now().astimezone(),
                  keywords=[])

with NWBHDF5IO('test_keywords.nwb', 'w') as io:
    io.write(nwbfile)

print('tutto pronto')

import numpy as np
from dateutil import tz
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from datetime import datetime
from hdmf.data_utils import DataChunkIterator
import h5py

h5py.h5.get_config().mpi

start_time = datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz('US/Pacific'))
fname = 'test_parallel_pynwb.nwb'
nwbfile = NWBFile('aa', 'aa', start_time, keywords=[])

with NWBHDF5IO('test_keywords.nwb', 'w') as io:
    io.write(nwbfile)

#data = DataChunkIterator(data=[1,2], maxshape=(4,), dtype=np.dtype('int'))
data = [1,2]

nwbfile.add_acquisition(TimeSeries('ts_name', description='desc', data=data,
                                   rate=100., unit='m'))
with NWBHDF5IO(fname, 'w') as io:
    io.write(nwbfile)

