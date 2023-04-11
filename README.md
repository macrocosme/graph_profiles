# Sequencing pulsar average profiles

This package includes mechanisms to investigates the pulse profile distribution of pulsars using graph topology analysis techniques, inspired by [Baron & Ménard (2019)](https://academic.oup.com/mnras/article-abstract/487/3/3404/5511907). The default input data are that of the [European Pulsar Network (EPN) Data Archive](http://www.epta.eu.org/epndb/).

## Quick start guide

In a shell:

```shell
git clone https://github.com/macrocosme/epn_mining.git
cd epn_mining
lftp -c ’mirror -c –parallel=100 http://www.epta.eu.org/epndb/json ; exit’
```

This next step can be done in a virtual environment, for a user only, or system-wide. The example below is for a user session only. It will install the following packages: dtaidistance==1.2.3, pandas==1.0.3, tqdm==4.24.0, pdat==0.2.1, psrqpy==1.0.0.
It assumes matplotlib, numpy and scipy are installed.

Virtual environment (from within the root repository (reached with `cd epn_mining`):
```shell
# (Install virtualenv if necessary)
pip3 install virtualenv

# Create a virtual environment coined `topology-env` for the project
virtualenv topology-env

# Activate the environment
source topology-env/bin/activate

# Install required packages
pip3 install -r requirements.txt

# Deactivate when done working on this project to return to global settings
deactivate
```

A virtual environment is useful to avoid package conflicts, or avoid updating a package for which a previous version is required by another program/package). If this is not a concern, general use of `pip` is just as fine.

User only:
```shell
pip3 install -r requirements.txt --user
```

System-wide:
```shell
pip3 install -r requirements.txt
```

### Running the code

#### How to run the code

The code shown in the next section can be exectuted from a python environment such as `ipython3`, or a `jupyter notebook` session:

```shell
ipython3 --pylab
```

```shell
jupyter notebook
```

#### Where to look 

The simplest way to run the whole pipeline is via `epn_mining.main.get_epn_sequence()`. It proceeds in the following order:

1. **load epn metadata** 
    - outputs: epn_metadata (of type pandas.DataFrame)
2. **load epn data**
    - output: population (of type list of Pulsar objects -- see file `epn_mining/preparation/pulsar.py` for details)
3. **compute complete (undirected) weighted graph**
    - output: distances (of type list of Tuples of the form (u, v, w) where u and v are pulsar indices, and w is the distance)
4. **compute minimum spanning tree**
    - output: mst (of type Graph -- see file `epn_mining/topology/graph.py` for details)
5. **find longest path (sequence)**
    - output: sequence_population (list of Pulsars)
    - output: sequence_indices (list of integers representing pulsar indices)

```python
from epn_mining.main import get_epn_sequence, load
from epn_mining.analysis.plotting import (
    sequence_to_pdf, 
    solution_space_pruning
)

# Set some global variables
verbose = True
reference = 'gl98'
if reference is not None:
    state_prefix = reference
else:
    state_prefix = ''
    
sequence_population, sequence_indices = get_epn_sequence(reference=reference,
                                                state_store=True, 
                                                state_prefix=state_prefix,
                                                state_reload=False,
                                                skip_steps=[
#                                                     'load_epn_metadata', 
#                                                     'load_epn_data',
                                                ],
                                                shift=True,
                                                normalize=True,
                                                remove_baseline=True,                                                
                                                verbose=verbose)
```

### Reload state variables
The output of previously completed processing steps can be stored (option `state_store=True`) to avoid re-running all steps. The output of one step is the input of the other. States can be loaded with the `load` function. State files can be marked by a prefix, else a default name will be used. We can skip steps in `get_epn_sequence` by adding one or more step name (as in the variables below). Skipped steps will reload previous states, and the other ones will be computed. 

```python
epn_metadata = load('epn_metadata', state_prefix)
population = load('population', state_prefix)
distances = load('distances', state_prefix)
mst = load('mst', state_prefix).as_dataframe()
sequence_indices = load('sequence_indices', state_prefix)
sequence_population = load('sequence_population', state_prefix)
```

### Output sequence to PDF
```python
sequence_to_pdf(epn_metadata, sequence_population, mst, state_prefix)
```

### Visualize solution space pruning steps
```python
solution_space_pruning(distances, state_prefix)
```

---

Copyright (c) Dany Vohl / macrocosme, 2019-2021.    The code is currently under development.
