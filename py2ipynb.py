from nbformat import v3, v4

'''
This script converts python files to ipython notebooks. Since in the v4
of nbformat module support for automatic conversion is dropped, we use
the v3 module and them upgrade the notebook.
For more info, check the source code repo:
https://github.com/jupyter/nbformat/blob/master/nbformat/v3/nbpy.py

stamatiad.st@gmail.com 
'''

for id in range(1, 5):
    with open(f'Generate_Publication_Figure_{id}.py') as python_file:
        python_code = python_file.read()

    notebook_v3 = v3.reads_py(python_code)
    notebook_v4 = v4.upgrade(notebook_v3)  # Upgrade v3 to v4

    jsonform = v4.writes(notebook_v4) + "\n"
    with open(f'Generate_Publication_Figure_{id}.ipynb', "w") as fpout:
        fpout.write(jsonform)

print('Done converting python files to ipython notebooks!')