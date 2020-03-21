import multiprocessing as mp
from pathlib import Path
from glob import glob
import numpy as np
import os
import pandas as pd
import re
import sys
from collections import defaultdict


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

def read_somatic_voltage(data_dir, ncells = 0):
    '''
    Reads train from multiple txt files, onto a single HDF5 one.
    :return:
    '''
    # Glob all filenames of the somatic voltage trace:
    #TODO: this line is wrong, since it is not sorted!
    # FIX: you must know a priori the number of cells./ or sort them
    files_v_soma = [
        data_dir.joinpath(f'v_soma_{id:03}.txt')
        for id in range(ncells)
    ]
    #files_v_soma = list(data_dir.glob('v_soma*')).sort()
    # Make sure that vsoma files exist:
    for id, file in enumerate(files_v_soma):
        if not file.is_file():
            raise FileNotFoundError(f'File {file} was not found!')

    #if len(files_v_soma) < 1:
    #    raise FileExistsError(f'No vsoma files found on {data_dir} !')

    # Use ncells as a validation variable: if the value given is the same as
    # the one found, raise error:
    #if len(files_v_soma) != ncells:
    #    raise ValueError('Ncells provided is not the same as the one found!')

    # Length of simulation samples:
    nsamples = len(read_train(files_v_soma[0]))

    vsoma = np.empty([ncells, nsamples], dtype=float)

    # Load all the voltage trains:
    for cell, fn in enumerate(files_v_soma):
        vsoma[cell][:] = read_train(fn)

    filename = data_dir.joinpath(r'vsoma.hdf5')
    df = pd.DataFrame(vsoma)
    df.to_hdf(filename, key='vsoma', mode='w')

    return nsamples

def read_dendritic_voltage(data_dir, ncells = 0, nsamples = 0, nseg = 5):
    '''
    Reads train from multiple txt files, onto a single HDF5 one.
    :return:
    '''
    # Glob all filenames of the dendritic voltage trace:
    #TODO: need to rewrite this!
    files_v_dend = list(data_dir.glob('v_dend*'))
    if len(files_v_dend) < 1:
        print('\tNo dendritic voltage trace files!')
        return

    # Use ncells as a validation variable: if the value given is the same as
    # the one found, raise error:
    if len(files_v_dend) != ncells:
        raise ValueError('Ncells provided is not the same as the one found!')

    filename = data_dir.joinpath(r'vdend.hdf5')
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

def read_synapse_info(data_dir, type=None, alias=None):
    '''
    Read information of synaptic locations, per given connection alias.
    :param alias:
    :return:
    '''
    #TODO: this does not work! fix it first.
    # Glob all filenames of the PN2PN synaptic locations:
    files = list(data_dir.glob(f'{type}_{alias}*'))
    pid_d = {}
    for fn in files:
        with open(fn, 'r') as f:
            for line in f:
                print(f'Line is: {line}')
                # if line.startswith('src='):
                m = re.search(r'src=(\d+) trg=(\d+)', line)
                # else, read just the pid:
                # This regexp is cursed...
                n = re.search(r'([0-9]{1}[.][0-9]{4})', line)
                if m is not None:
                    src = int(m.group(1))
                    trg = int(m.group(2))
                    pid_d[(src, trg)] = []
                    if src == 223 and trg == 243:
                        print(f'found key {(src, trg)}')
                elif n is not None:
                    pid = float(n.group(1))
                    pid_d[(src, trg)].append(pid)
                else:
                    continue

    # BLAH DE BLAH:
    print(f'PID dict is of length {len(pid_d)}')
    len_set = {
        len(v)
        for v in pid_d.values()
    }
    print(f'length of set lengths is: {len(len_set)}')
    print(len_set)

    for k, v in pid_d.items():
        if len(v) > 5:
            print(f'key {k} with value: {v} is wrong!')

    df = pd.DataFrame(pid_d)
    df.to_hdf(data_dir.joinpath(fr'{type}_{alias}.hdf5'), key=f'{type}_{alias}', mode='w')


def main(data_dir):
    #TODO: Must rewrite this fine!
    #ncores = 2 #mp.cpu_count()
    if __name__ == "__main__":
        print(f'Neuron output data_dir:\n\t{data_dir}')
    #  Read somatic voltage:
    print('\tReading vsoma.', end='')
    nsamples = read_somatic_voltage(data_dir, ncells = 333)

    # Load dendritic voltage traces, if any:
    #print('\tReading vdend.')
    #read_dendritic_voltage(data_dir, ncells=250, nsamples=nsamples, nseg=5)

    ## Read synaptic locations in PN2PN connections:
    #print('Reading pid pyramidal.')
    #read_synapse_info(type='pid',alias='pyramidal')

    ## Read synaptic locations in PV2PN connections:
    #print('Reading pid interneurons.')
    #read_synapse_info(type='pid', alias='interneurons')

    ## Read synaptic locations in PN2PN connections:
    #print('Reading delay pyramidal.')
    #read_synapse_info(type='delay', alias='pyramidal')

    ## Read synaptic locations in PV2PN connections:
    #print('Reading delay interneurons.')
    #read_synapse_info(type='delay', alias='interneurons')

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
    main(Path(sys.argv[1]))