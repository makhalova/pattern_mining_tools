import json
import pickle
import numpy as np
import os

def read_json(input_file, n_obj, output_file = None):
    """
    Sofia's output json reader.
    """
    itemsets = {}
    list_stability = []
    with open(input_file) as json_file:
        data = json.load(json_file)
        for concept in data[1]['Nodes']:
            
            extent = concept['Ext']
            intent = concept['Int']
            
            binary_extent = np.zeros(n_obj, dtype = int)
            binary_extent[extent['Inds']] = 1
            itemsets[frozenset([int(v) for v in intent['Inds']])] = (binary_extent, concept['Interest'])

    if output_file is None:
        return itemsets
    else:
        dirname = os.path.dirname(output_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(output_file, 'wb') as fpkl:
            pickle.dump(itemsets, fpkl)
