from nbformat import v3, v4
from pathlib import Path
import sys

'''
This script converts python files to ipython notebooks. Since in the v4
of nbformat module support for automatic conversion is dropped, we use
the v3 module and them upgrade the notebook.
For more info, check the source code repo:
https://github.com/jupyter/nbformat/blob/master/nbformat/v3/nbpy.py

stamatiad.st@gmail.com 
'''

def convert_file(filename: Path):
    if not filename.is_file():
        print(f'File: {filename} was not found!')
        raise FileNotFoundError

    outputfile = filename.with_suffix('.ipynb')
    print('File exists. Attempting to convert it...')
    # Get a python file
    with open(filename) as python_file:
        python_code = python_file.read()

    notebook_v3 = v3.reads_py(python_code)
    notebook_v4 = v4.upgrade(notebook_v3)  # Upgrade v3 to v4

    jsonform = v4.writes(notebook_v4) + "\n"
    with open(outputfile, 'w') as fpout:
        fpout.write(jsonform)

    print(f'Done converting python file: {filename}\n to ipython notebook: {outputfile}\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('py2ipynb takes one argument: the python filename!\nExiting...')
        sys.exit(-1)

    convert_file(Path(sys.argv[1]))