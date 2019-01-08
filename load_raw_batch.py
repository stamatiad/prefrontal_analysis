import multiprocessing as mp
from pathlib import Path
from glob import glob
import numpy as np
import os
import pandas as pd
import re

print(__name__)
ncores = 2 #mp.cpu_count()
#data_dir = Path(r'C:\Users\steve\Desktop\data')
data_dir = Path(r'F:\backendSN2LC1TR0_EB1.750_IB1.500_GBF2.000_NMDAb6.000_AMPAb1.000_randomdur3')

def read_train(fn):
    '''
    Read train from file, given filename
    :param arg:
    :return:
    '''
    # TODO: more elegant?
    def str2float(s):
        result = None
        try:
            result = float(s)
        except:
            pass
        return result

    with open(fn) as f:
        # Why not f.readlines() ?
        values = list(map(str2float, f.readlines()))
        # Remove empty rows/values:
        values = list(filter(None.__ne__, values))
    return values

def read_somatic_voltage(ncells = 0):
    '''
    Reads train from multiple txt files, onto a single HDF5 one.
    :return:
    '''
    # Glob all filenames of the somatic voltage trace:
    files_v_soma = list(data_dir.glob('v_soma*'))
    #ncells = len(files_v_soma)
    if ncells < 1:
        raise FileExistsError('No vsoma files found!')

    # Length of simulation samples:
    nsamples = len(read_train(files_v_soma[0]))

    vsoma = np.empty([ncells, nsamples], dtype=float)

    # Load all the voltage trains:
    for cell, fn in enumerate(files_v_soma):
        vsoma[cell][:] = read_train(fn)

    filename = Path(r'F:\vsoma.hdf5')
    df = pd.DataFrame(vsoma)
    df.to_hdf(filename, key='vsoma', mode='w')

    return nsamples

def read_dendritic_voltage(ncells = 0, nsamples = 0, nseg = 5):
    '''
    Reads train from multiple txt files, onto a single HDF5 one.
    :return:
    '''
    # Glob all filenames of the dendritic voltage trace:
    files_v_dend = list(data_dir.glob('v_dend*'))
    if len(files_v_dend) < 1:
        print('No dendritic voltage trace files!')
        return

    filename = Path(r'F:\vdend.hdf5')
    # Touch the file. Is there a better way?
    open(filename, 'w').close()

    for seg in range(nseg):
        # All files of segment i:
        files_dend_seg = list(data_dir.glob(f'v_dend_{seg:02d}*'))
        # TODO: properly, vsoma have also PV cells...
        #if len(files_dend_seg) is not ncells:
            #raise Exception('Error in file number: vcend.')

        vdend = np.empty([ncells, nsamples], dtype=float)

        # Load all the voltage trains:
        for cell, fn in enumerate(files_dend_seg):
            vdend[cell][:] = read_train(fn)

        df = pd.DataFrame(vdend)
        # Append to the same file:
        df.to_hdf(filename, key=f'vdend{seg}', mode='a')

def read_synapse_info(type=None, alias=None):
    '''
    Read information of synaptic locations, per given connection alias.
    :param alias:
    :return:
    '''
    # Glob all filenames of the PN2PN synaptic locations:
    files = list(data_dir.glob(f'{type}_{alias}*'))
    pid_d = {}
    for fn in files:
        with open(fn, 'r') as f:
            for line in f:
                # if line.startswith('src='):
                m = re.search(r'src=(\d+) trg=(\d+)', line)
                # else, read just the pid:
                n = re.search(r'\d+.\d+', line)
                if m is not None:
                    src = m.group(1)
                    trg = m.group(2)
                    pid_d[(src, trg)] = []
                elif n is not None:
                    pid = n.group()
                    pid_d[(src, trg)].append(pid)
                else:
                    continue

    df = pd.DataFrame(pid_d)
    df.to_hdf(Path(fr'F:\{type}_{alias}.hdf5'), key=f'{type}_{alias}', mode='w')


def main():
    #  Read somatic voltage:
    print('Reading vsoma.')
    #nsamples = read_somatic_voltage(ncells = 333)

    # Load dendritic voltage traces, if any:
    print('Reading vdend.')
    #read_dendritic_voltage(ncells=250, nsamples=nsamples, nseg=5)

    # Read synaptic locations in PN2PN connections:
    print('Reading pid pyramidal.')
    read_synapse_info(type='pid',alias='pyramidal')

    # Read synaptic locations in PV2PN connections:
    print('Reading pid interneurons.')
    read_synapse_info(type='pid', alias='interneurons')

    # Read synaptic locations in PN2PN connections:
    print('Reading delay pyramidal.')
    read_synapse_info(type='delay', alias='pyramidal')

    # Read synaptic locations in PV2PN connections:
    print('Reading delay interneurons.')
    read_synapse_info(type='delay', alias='interneurons')

    # TODO: Check that you can load these.

    # TODO: if ok, then load in parallel
    '''
    # split cells across cores:
    slice_len = np.ceil(333 / ncores)
    slices = np.arange(slice_len * ncores).reshape((ncores, -1))

    output_files = [x for x in data_dir.glob('*/*')]
    [file for file in output_files if file.startswith('vsoma')]

    with mp.Pool(ncores) as p:
        blah = p.map(read_train, slices)

    pass
    '''


if __name__ == '__main__':
    main()