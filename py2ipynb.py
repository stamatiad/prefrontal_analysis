from nbformat import v3, v4
from pathlib import Path

'''
This script converts python files to ipython notebooks. Since in the v4
of nbformat module support for automatic conversion is dropped, we use
the v3 module and them upgrade the notebook.
For more info, check the source code repo:
https://github.com/jupyter/nbformat/blob/master/nbformat/v3/nbpy.py

stamatiad.st@gmail.com 
'''

figure_files = [
    Path('Generate_Publication_Figure_1.py'),
    Path('Generate_Publication_Figure_2.py'),
    Path('Generate_Publication_Figure_3.py'),
    Path('Generate_Publication_Figure_4.py'),
    Path('Generate_Publication_Figure_S2.py'),
    Path('Generate_Publication_Figure_S3.py'),
]
for fig_file in figure_files:
    with open(fig_file) as python_file:
        python_code = python_file.read()

    notebook_v3 = v3.reads_py(python_code)
    notebook_v4 = v4.upgrade(notebook_v3)  # Upgrade v3 to v4

    jsonform = v4.writes(notebook_v4) + "\n"
    with open(Path(fig_file.stem).with_suffix('.ipynb'), "w") as fpout:
        fpout.write(jsonform)

print('Done converting python files to ipython notebooks!')