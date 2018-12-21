import multiprocessing as mp
from pathlib import Path
import numpy as np
import os

print(__name__)
ncores = 2 #mp.cpu_count()
data_dir = Path(r'C:\Users\steve\Desktop\data')

def read_train(fn):
    '''
    Read train from file, given filename
    :param arg:
    :return:
    '''
    with open(fn) as f:
        values = map(float, f)
    return values

def main():
    # split cells across cores:
    slice_len = np.ceil(333 / ncores)
    slices = np.arange(slice_len * ncores).reshape((ncores, -1))

    output_files = [x for x in data_dir.glob('*/*')]
    [file for file in output_files if file.startswith('vsoma')]

    with mp.Pool(ncores) as p:
        blah = p.map(read_train, slices)

    pass

if __name__ == '__main__':
    main()