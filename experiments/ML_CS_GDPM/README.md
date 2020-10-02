# Closure structure and GDPM (Gradual Discovery in Pattern Mining)



## MLJ experiments
---
## Data source
The datasets can be download from [LUCS-KDD](https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/DataSets/dataSets.html)

The class labels were removed from datasets. The datasets should be stored in [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html). 


## Pipeline

Computing closure structure
    The closure sturcture is comuted by the GDPM algorithm available at `GDPM` repository. 
    To compute the closure structure for `example.dat`, run 
    `gdmp example.dat` .
    
Summary on analyzed datasets is computed by running `scripts/dataset_characteristics.py`.
    
To analyse itemsets in supervised settings run `scripts/f1.py`.
