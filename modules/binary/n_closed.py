"""
This module includes function for computing n-closed itemsets.
"""


import os
import sys
import pickle
from time import time
from numpy import argwhere, apply_along_axis
from pattern_mining_tools.modules.io.converter_writer import write_itemsets, read_itemsets

def _get_closure(candidate, data):
    p_attributes = candidate[0]
    p_obj_mmb = candidate[1]
    p_obj_mmb_size = sum(p_obj_mmb)
    closed_set = set([
        v[0] for v in argwhere(sum(data[p_obj_mmb==1], axis=0) == p_obj_mmb_size)
        ])
    return (frozenset(closed_set.union(p_attributes)),
            (data[:, list(p_attributes)].sum(axis = 1) ==
                len(p_attributes)).astype(int))

def _compute_merged_next(patterns, attributes, attribute_set, data,
                         entire_pattern_dict):
    candidates = {}
    for k1 in [v for v in  patterns.keys()]:
        attributes_to_merge = attribute_set.difference(k1)
        for k2 in attributes_to_merge:
            ids1 = k1
            ids2 = frozenset([k2])
            vec = patterns[ids1] & attributes[ids2]
            if sum(vec) > 0: # n12 = sum(vec)
                closed_itemset = _get_closure((ids1.union(ids2), vec), data)
                s = closed_itemset[0]
                if (candidates.get(s) is None) and (entire_pattern_dict.get(s) is None):
                    candidates[s] = closed_itemset[1]
                    entire_pattern_dict[s] = closed_itemset[1]
    return candidates, entire_pattern_dict


def _save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files
    on all platforms.
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def _try_to_load_as_pickled(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files
    on all platforms.
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


def _is_all_attributes(row, val):
    return 1 if row.sum() == val else 0

def _get_extent(itemsets_with_support, data):
    itemsets_with_extents = {}
    for itemset, support in itemsets_with_support.items():
        itemsets_with_extents[itemset] = apply_along_axis(_is_all_attributes, 1,
            data[:, list(itemset)], len(itemset))
        assert (itemsets_with_extents[itemset].sum() == support), "Wrong extent"
    return itemsets_with_extents

def get_extent(itemsets_with_support, data):
    """
    Return the dictionary of itemsets with their extention.
    """
    # the same as _get_extent, but without checking support
    itemsets_with_extents = {}
    for itemset, _ in itemsets_with_support.items(): # _ is "support"
        itemsets_with_extents[itemset] = apply_along_axis(_is_all_attributes, 1,
            data[:, list(itemset)], len(itemset))
        # assert(itemsets_with_extents[itemset].sum() == support)
    return itemsets_with_extents

def compute_closed_by_level(data, max_level=None, make_log=False,
        dataset_output_folder=None, inverse_index_dict=None, use_pickled=False):
    """Compute closed itemsets in data by closure levels.

    !WARNING!
    Dumping of pickled objects is very memory-consuming. Try to avoid it.


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

    inverse_index_dict : dictionary, default = None
        The dictionary "new index : original index"

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

    if inverse_index_dict is None:
        index_dict = None
    else:
        index_dict = {v: k for k, v in inverse_index_dict.items()}
    closed_dict = {}
    attribute_set = set(range(data.shape[1]))
    # output folder
    if dataset_output_folder is None:
        do_io = False
    else:
        if not os.path.exists(dataset_output_folder):
            os.makedirs(dataset_output_folder, exist_ok=True)
        do_io = True # False if use 'dataset_output_folder' to write the results
    # initialization
    if max_level is None:
        max_level = data.shape[1]
    patterns = {frozenset([i]): data[:, i] for i in range(data.shape[1])}
    # 1-closed patterns
    if do_io and os.path.exists(dataset_output_folder + str(1)):
        if use_pickled:
            closed = _try_to_load_as_pickled(dataset_output_folder + str(1))
        else:
            closed_itemsets = read_itemsets(dataset_output_folder + str(1),
                                            index_dict)
            closed = _get_extent(closed_itemsets, data)
    else:
        closed = {}
        for ids, vec in patterns.items():
            k, v = _get_closure((ids, vec), data)
            closed[k] = v
        if do_io:
            if use_pickled:
                _save_as_pickled_object(closed, dataset_output_folder + str(1))
            else:
                write_itemsets(dataset_output_folder + str(1), closed,
                    inverse_index_dict)
    closed_dict[1] = closed
    entire_pattern_dict = closed.copy() # contains all generated early closed itemsets
    # n-closed patterns
    run = max_level > 1
    compute_it = False
    i = 2
    while run and (max_level >= i):
        if make_log:
            print('ready to compute', i)
        if do_io:
            dataset_output_file = dataset_output_folder + str(i)
            if os.path.exists(dataset_output_file):
                if use_pickled:
                    closed = _try_to_load_as_pickled(dataset_output_file)
                else:
                    closed_itemsets = read_itemsets(dataset_output_file,
                                                    index_dict)
                    closed = _get_extent(closed_itemsets, data)
                entire_pattern_dict.update(closed.copy())
                compute_it = False
            else:
                compute_it = True
        if not(do_io) or (compute_it):
            start_time = time()
            closed, entire_pattern_dict = _compute_merged_next(closed_dict[i-1],
                patterns, attribute_set, data, entire_pattern_dict)
            if make_log:
                print(i - 1, time() - start_time, len(closed))
        if len(closed) > 0:
            if do_io:
                if not os.path.exists(dataset_output_file):
                    if use_pickled:
                        _save_as_pickled_object(closed, dataset_output_file)
                    else:
                        write_itemsets(dataset_output_file, closed,
                                       inverse_index_dict)
            closed_dict[i] = closed
            if make_log:
                print(i, len(closed))
            i += 1
        else:
            run = False
    return closed_dict
