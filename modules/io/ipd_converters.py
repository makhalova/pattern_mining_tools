""" Converters for IPD-discretizer:
Nguyen, Hoang-Vu, et al. "Unsupervised interaction-preserving discretization
of multivariate data." Data Mining and Knowledge Discovery 28.5-6 (2014):
1366-1397.
"""

import pandas as pd
import numpy as np

def get_points(file_name):
    dims = {}
    dim_id = 0
    with open(file_name) as f:
        s = f.readline()
        while s:
            n_bines = int(s.split('(')[1].split(' ')[0])
            dims[dim_id] = []
            for i in range(n_bines):
                s = f.readline()
                dims[dim_id].append(float(s))
            f.readline()
            dim_id += 1
            s = f.readline()
    return dims
    
    
def discretise_with_borders(original_data_file, border_file, output_file, output_bin_file):

    dim = get_points(border_file)
    X = pd.read_csv(original_data_file, sep=';', header=None, index_col=False).values
                
    X_disc = np.zeros((X.shape[0], X.shape[1] - 1), dtype = int)

    attribute_id = -1
    for i, borders in dim.items():
        attribute_id += 1
        X_disc[:, i] = attribute_id
        for val, bord in enumerate(borders[:-1]):
            attribute_id += 1
            selector = X[:,i]>=bord
            X_disc[selector, i] = attribute_id

    total_numb_attributes = np.sum([len(v) for v in dim.values()])
    assert(total_numb_attributes == attribute_id + 1)

    with open(output_file, 'w') as f:
        for row_id in range(X_disc.shape[0]):
            row = X_disc[row_id, :]
            row_to_print = ' '.join([str(i) for i in row])
            f.writelines(row_to_print + '\n' )
            
    with open(output_bin_file, 'w') as f:
        f.writelines(str(np.sum([len(v) for v in dim.values()])) + '\n' )
            
