"""
Computing indices for itemsets w.r.t. a dataset
-----------------------------------------------
"""

from itertools import zip_longest
from numpy import argsort, where, zeros_like
from pattern_mining_tools.modules.indices import simple_indices_array as simple_indices

def sort_itemsets(indices, data, itemsets, custom_index_values=None,
        return_values=False):
    """Sort itemsets by list of ordered indices. Returns ranked itemsets.

    Parameters
    ----------
    indices : list of tupples ('name of index', order). 'name of index'
        is a string, order is a boolean value, True is ascending order, false otherwise.

    data : array (dataset) to assess itemsets.

    itemsets : dictionary of itemsets itemset_id:binary array of items.

    custom_index_values : list of list of values of indices

    Returns
    -------
    ordered_itemsets : dictionary of sorted itemsets rank:itemset_id.
    """
    values = []
    custom_index_id = 0
    for (name, order) in indices:
        if name == 'custom':
            try:
                values.append([order * v for v in custom_index_values[custom_index_id]])
                custom_index_id += 1
            except IndexError:
                pass
        else:
            values.append([order * v for v in simple_indices.func_dict(name, data, itemsets)])

    lexicographical = argsort([''.join([str(i) for i in where(v)[0]]) for v in itemsets])
    lexicographical_cpy = zeros_like(lexicographical)
    for i, v in enumerate(lexicographical):
        lexicographical_cpy[v] = i
    values.append(list(lexicographical_cpy))
    values.append(list(range(len(itemsets))))
    sorted_values = sorted(zip_longest(*values))
    index_col_id = len(values) - 1
    if return_values:
        return [i[index_col_id] for i in sorted_values], sorted_values[:index_col_id]
    return [i[index_col_id] for i in sorted_values]

def compute_indices(indices, data, itemsets, custom_index_values=None):
    values = []
    custom_index_id = 0
    for (name, order) in indices:
        if name == 'custom':
            try:
                values.append([order * v for v in custom_index_values[custom_index_id]])
                custom_index_id += 1
            except IndexError:
                pass
        else:
            values.append([order * v for v in simple_indices.func_dict(name, data, itemsets)])
    return list(zip_longest(*values))
