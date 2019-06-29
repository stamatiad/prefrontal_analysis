import nbformat
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
    Path('Generate_Publication_Figure_S1.py'),
    Path('Generate_Publication_Figure_S2.py'),
    Path('Generate_Publication_Figure_S3.py'),
]
for fig_file in figure_files:
    with open(fig_file) as python_file:
        python_code = python_file.read()

    notebook_v3 = v3.reads_py(python_code)
    notebook_v4 = v4.upgrade(notebook_v3)  # Upgrade v3 to v4

    # Insert a cell that creates/initializes the environment:
    # Since this is just a placeholder: nbformat.v3.nbbase.NotebookNode()
    # \nbformat\v4\nbjson.py
    # Define it as you like:
    init_cell = nbformat.v3.nbbase.new_code_cell()
    # init_cell = nbformat.v3.nbbase.NotebookNode()
    # init_cell.cell_type = u'code'
    # init_cell.outputs = []
    init_cell.source = """\
    # Need to setup tools on our machine first:
    !sudo apt-get install git-lfs
    
    # Due to github limitation in git lfs, I clone from an identical
    # repo over bitbucket. To make sure the code in Github is identical
    # you can clone the Github repo and run:
    # >git diff review remotes/bitbucket/review
    #!git clone https://github.com/stamatiad/prefrontal_analysis.git
    !git clone https://bitbucket.org/stevest/prefrontal_analysis.git

    import os
    os.chdir('prefrontal_analysis')
    !git checkout review

    !git lfs install
    !git lfs fetch
    !git lfs checkout

    !pip install -r requirements.txt

    # numpy has issue: use version numpy==1.16.4
    """

    # add the environment cell after ipynb description:
    notebook_v4.cells.insert(1, init_cell)

    # Write notebook as a json file:
    jsonform = v4.writes(notebook_v4) + "\n"

    # Write notebook to ipynb file:
    with open(Path(fig_file.stem).with_suffix('.ipynb'), "w") as fpout:
        fpout.write(jsonform)

print('Done converting python files to ipython notebooks!')