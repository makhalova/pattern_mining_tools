"""
This module includes function for computing n-closed itemsets.
"""

import numpy as np
import pickle
import os
import sys


def _get_closure(candidate, data):
    p_attributes = candidate[0]
    p_obj_mmb = candidate[1]
    p_obj_mmb_size = np.sum(p_obj_mmb)
    r_cand = np.tile(p_obj_mmb,(data.shape[1], 1)).T
    closed_set = set([v[0] for v in np.argwhere(np.sum(data[p_obj_mmb == 1], axis = 0) == p_obj_mmb_size)])
    # we do not need union!!!
    return (frozenset(closed_set.union(p_attributes)), (data[:, [v for v in p_attributes]].sum(axis = 1) == len(p_attributes)).astype(int))


def _compute_merged_next(patterns, attributes, attribute_set, data, entire_pattern_dict):
    n_patterns = len(patterns)
    candidates = {}

    keys = [v for v in  patterns.keys()]

    for i1, k1 in enumerate(keys):

        attributes_to_merge = attribute_set.difference(k1)
        for k2 in attributes_to_merge:
            ids1 = k1
            ids2 = frozenset([k2])
            vec1 = patterns[ids1]
            vec2 = attributes[ids2]
            
            vec = vec1 & vec2
            n12 = np.sum(vec)
            
            if (n12 > 0) : #& (n1 > 0) & (n2 > 0):
                closed_itemset = _get_closure((ids1.union(ids2), vec), data)
                s = closed_itemset[0]
                if (candidates.get(s) is None) and (entire_pattern_dict.get(s) is None):
                    candidates[s] = closed_itemset[1]
                    entire_pattern_dict[s] = closed_itemset[1]
    return candidates, entire_pattern_dict


def _save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def _try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        print(filepath)
        input_size = os.path.getsize(filepath)
        print(input_size)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj



def compute_closed_by_level(data, max_level = None, make_log = False, dataset_output_folder = './results/'):
    """Compute closed itemsets in data by closure levels.

    Parameters
    ----------
    
    data : ndarray
        The binary dataset. Contains objects in rows. The columns are attrubtes.
        
    max_level : int, default = None
        The maximum number of closure levels that are computed for data.
        If the value is not given, all closure levels are computed.
        
    make_log : bool; default = None
        The flag for the execution tracking.
        
        
    dataset_output_folder : string, default = None
        The path of the output directoty. By default the results are stored in
        the child folder "./results"
    
    
    Returns
    -------
    closed_dict : dictionary
        The dictionary "#level : 'itemset dictionary'", where 'itemset dictionary'
        is the dictionary " 'frosen set of int' : binary vect of the object including it'
        
        
    Example
    -------
    >>> dataset = np.array([[1,0,1,0,0,1],
                            [1,0,1,0,1,0],
                            [1,0,0,1,1,0],
                            [0,0,1,0,1,0],
                            [0,1,0,1,1,1],
                            [0,1,0,1,0,1]])
                     
    {1: {frozenset({0}): array([1, 1, 1, 0, 0, 0]),
      frozenset({1, 3, 5}): array([0, 0, 0, 0, 1, 1]),
      frozenset({2}): array([1, 1, 0, 1, 0, 0]),
      frozenset({3}): array([0, 0, 1, 0, 1, 1]),
      frozenset({4}): array([0, 1, 1, 1, 1, 0]),
      frozenset({5}): array([1, 0, 0, 0, 1, 1])},
     2: {frozenset({0, 2}): array([1, 1, 0, 0, 0, 0]),
      frozenset({0, 4}): array([0, 1, 1, 0, 0, 0]),
      frozenset({0, 3, 4}): array([0, 0, 1, 0, 0, 0]),
      frozenset({0, 2, 5}): array([1, 0, 0, 0, 0, 0]),
      frozenset({1, 3, 4, 5}): array([0, 0, 0, 0, 1, 0]),
      frozenset({2, 4}): array([0, 1, 0, 1, 0, 0]),
      frozenset({3, 4}): array([0, 0, 1, 0, 1, 0])},
     3: {frozenset({0, 2, 4}): array([0, 1, 0, 0, 0, 0])}}

    """
    
    closed_dict = {}
    attribute_set = set(range(data.shape[1]))
    try_to_upload = True

    # output folder
    if dataset_output_folder ==  './results/':
        try_to_upload = False
    else:
        dataset_output_folder += '/'
    if not os.path.exists(dataset_output_folder):
        os.makedirs(dataset_output_folder,exist_ok=True)

    # initialization
    if max_level is None:
        max_level = data.shape[1]
    patterns = {frozenset([i]): data[:,i] for i in range(data.shape[1])}

    # 1-closed patterns
    if try_to_upload and os.path.exists(dataset_output_folder + str(1)):
        closed = _try_to_load_as_pickled_object_or_None(dataset_output_folder + str(1))
    else:
        closed = {}
        for ids, vec in patterns.items():
            k, v = _get_closure((ids, vec), data)
            closed[k] = v
        _save_as_pickled_object(closed, dataset_output_folder + str(1))

    closed_dict[1] = closed
    entire_pattern_dict = closed.copy() # contains all generated early closed itemsets

    # n-closed patterns
    run = max_level > 1
    i = 2
    while run and (max_level >= i):
        if make_log:
            print('ready to compute', i)
        dataset_output_file = dataset_output_folder + str(i)
        if try_to_upload and os.path.exists(dataset_output_file):
            closed = _try_to_load_as_pickled_object_or_None(dataset_output_file)
            entire_pattern_dict.update(closed.copy())
        else:
            closed, entire_pattern_dict = _compute_merged_next(closed_dict[i-1], patterns, attribute_set, data, entire_pattern_dict)
            
        if len(closed) > 0:
            _save_as_pickled_object(closed, dataset_output_file)
            closed_dict[i] = closed
     
            if make_log:
                print(i,len(closed))
            i += 1
        else:
            run = False
    return closed_dict
