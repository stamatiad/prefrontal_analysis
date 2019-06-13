# prefrontal_analysis
This is the code used to analyse the NEURON simulations produced from [here].

### Run on Google Colab
> Now you can run the code, replicating the results on Google Colab platform, without the need to setup a python environment.

> :grey_exclamation: These links need to change when I will merge to main branch!

Replicate figures using jupyter notebooks:


Figure 1 Notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stamatiad/prefrontal_analysis/blob/review/Generate_Publication_Figure_1.ipynb)

Figure 2 Notebook [![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stamatiad/prefrontal_analysis/blob/review/Generate_Publication_Figure_2.ipynb)

Figure 3 Notebook [![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stamatiad/prefrontal_analysis/blob/review/Generate_Publication_Figure_3.ipynb)

Figure 4 Notebook [![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stamatiad/prefrontal_analysis/blob/review/Generate_Publication_Figure_4.ipynb)

Figure S2 Notebook [![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stamatiad/prefrontal_analysis/blob/review/Generate_Publication_Figure_S2.ipynb)

Figure S3 Notebook [![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stamatiad/prefrontal_analysis/blob/review/Generate_Publication_Figure_S3.ipynb)




### Recommendations
This code is pure python 3, although there are some recommendations to run it trouble-free:
- pip, for easy python package management (included with most python installations).
- git VCS.
- About 200 MB of storage, because the repo includes parts of the simulations for easy figure reproducibility. 
- Jupyter notebook, for an annotated and concise way of presenting the results.
You can install a python 3.7 [anaconda distribution](https://www.anaconda.com/distribution/), that will contain most of the required packages. 


### Setup python environment
Using conda is recommended:
```
conda update -n base -c defaults conda
conda create -n publication python=3.7.0
activate publication
```

To run the code please:
1. [Install git](https://git-scm.com/downloads) for your machine. 
2. Clone this repository to your local working directory:
```
mkdir prefrontal_analysis && cd prefrontal_analysis
git clone https://github.com/stamatiad/prefrontal_analysis.git
```
This *might take a while*, since repo contains parts of the simulations (~200MB).
> !!! Since this is work in progress, please change from master to the review branch!

```
git checkout review
```

3. Install required python packages in your environment.
This can be done either traditionally with pip (inside the repo folder):
```
pip install -r requirements.txt
```
or utilizing the newer pipenv (again inside the repo folder):
```
pipenv install
```


### Run and replicate result figures
In order to ease the reproducibility, a Jupyter notebook is included. 
To run a Jupyter server and display/run it you must:
1. Run Jupyter server (from inside the repo folder):
```
jupyter notebook
```
or if you have pipenv, let it handle the requirements:
```
pipenv run jupyter notebook
```
2. Open your browser in the displayed address (it will be something like http://localhost:8888/tree#notebooks)
3. [Run the notebook](https://jupyter.readthedocs.io/en/latest/running.html).

### Please report any issues encountered!