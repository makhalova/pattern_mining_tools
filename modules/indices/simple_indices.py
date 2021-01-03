"""
Functions for computing simple indices
--------------------------------------
"""

from numpy import where
from numpy import sum as npsum

simple_index_dict = {'support': support, 'frequency': frequency, 'lift': lift, \
'naive_bayes_probability': naive_bayes_probability, 'separation': separation, 'length': length}

def length(data, itemsets):
    if isinstance(itemsets, list):
        unordered_itemsets = dict(enumerate(itemsets))
    elif isinstance(itemsets, dict):
        unordered_itemsets = itemsets.copy()
    else:
        return None
    return {i: sum(item) for i, item in unordered_itemsets.items()}

def support(data, itemsets):
    if isinstance(itemsets, list):
        unordered_itemsets = dict(enumerate(itemsets))
    elif isinstance(itemsets, dict):
        unordered_itemsets = itemsets.copy()
    else:
        return None
    unordered_itemset_values = dict.fromkeys(unordered_itemsets.keys(), 0)
    unordered_itemset_lengths = {i: sum(item) for i, item in unordered_itemsets.items()}
    for item_id, item in unordered_itemsets.items():
        val = 0
        for i in range(data.shape[0]):
            if sum(data[i,:]*item) == unordered_itemset_lengths[item_id]:
                val += 1
        unordered_itemset_values[item_id] = val
    return unordered_itemset_values

def frequency(data, itemsets):
    if isinstance(itemsets, list):
        unordered_itemsets = dict(enumerate(itemsets))
    elif isinstance(itemsets, dict):
        unordered_itemsets = itemsets.copy()
    else:
        # send an error message
        return None
    unordered_itemset_values = dict.fromkeys(unordered_itemsets.keys(), 0)
    unordered_itemset_lengths = {i: sum(item) for i, item in unordered_itemsets.items()}
    n_objects = float(data.shape[0])
    for item_id, item in unordered_itemsets.items():
        val = 0
        for i in range(data.shape[0]):
            if sum(data[i,:]*item) == unordered_itemset_lengths[item_id]:
                val += 1
        unordered_itemset_values[item_id] = val / n_objects
    return unordered_itemset_values

def lift(data, itemsets, ascending = False):
    if isinstance(itemsets, list):
        unordered_itemsets = dict(enumerate(itemsets))
    elif isinstance(itemsets, dict):
        unordered_itemsets = itemsets.copy()
    else:
        # send an error message
        return None
    unordered_itemset_values = dict.fromkeys(unordered_itemsets.keys(), 0)
    unordered_itemset_lengths = {i: sum(item) for i, item in unordered_itemsets.items()}
    n_objects = data.shape[0]
    single_freq = [val / n_objects for val in npsum(data, axis=0)]
    for item_id, item in unordered_itemsets.items():
        val = 0
        for i in range(data.shape[0]):
            if sum(data[i,:]*item) == unordered_itemset_lengths[item_id]:
                val += 1
        unordered_itemset_values[item_id] = val / n_objects
    ind_freq = {}
    for item_id, item in unordered_itemsets.items():
        prob = 1.
        for val in where(item == 1)[0]:
            prob *= single_freq[val]
        ind_freq[item_id] = prob
    return {item_id : unordered_itemset_values[item_id]/ind_freq[item_id] \
        for item_id in unordered_itemsets.keys()}

def naive_bayes_probability(data, itemsets, ascending = False):
    if isinstance(itemsets, list):
        unordered_itemsets = dict(enumerate(itemsets))
    elif isinstance(itemsets, dict):
        unordered_itemsets = itemsets.copy()
    else:
        return None
    n_objects = data.shape[0]
    single_freq = [val / n_objects for val in npsum(data, axis=0)]
    ind_freq = {}
    for item_id, item in unordered_itemsets.items():
        prob = 1.
        for val in where(item == 1)[0]:
            prob *= single_freq[val]
        ind_freq[item_id] = prob
    return ind_freq

def separation(data, itemsets):
    if isinstance(itemsets, list):
        unordered_itemsets = dict(enumerate(itemsets))
    elif isinstance(itemsets, dict):
        unordered_itemsets = itemsets.copy()
    else:
        return None
    #to compute the big area
    single_attribute_support = list(npsum(data, axis=0))
    single_object_support = list(npsum(data, axis=1))
    # to compute frequency
    unordered_itemset_lengths = {i: sum(item) for i, item in unordered_itemsets.items()}
    unordered_itemset_values = dict.fromkeys(unordered_itemsets.keys(), 0)
    for item_id, item in unordered_itemsets.items():
        sup = 0
        sq = unordered_itemset_lengths[item_id]
        total_amount = 0
        for i in range(data.shape[0]):
            if sum(data[i, :]*item) == unordered_itemset_lengths[item_id]:
                sup += 1
                total_amount += single_object_support[i]
        for i in where(item == 1)[0]:
            total_amount += single_attribute_support[i]
        sq = unordered_itemset_lengths[item_id] * sup
        unordered_itemset_values[item_id] = sq/(total_amount-sq)
    return unordered_itemset_values

def compose_by_product(index_tupple, itemsets, data):
    if isinstance(itemsets, list):
        unordered_itemset_values = dict.fromkeys(range(len(itemsets)), 1)
    elif isinstance(itemsets, dict):
        unordered_itemset_values = dict.fromkeys(itemsets.keys(), 1)
    else:
        return None
    for func_name in index_tupple:
        res = simple_index_dict[func_name](data, itemsets)
        for i, val in res.items():
            unordered_itemset_values[i] *= val
    return unordered_itemset_values

def func_dict(index_name, data, itemsets):
    if simple_index_dict.get(index_name) is None:
        correct_names = True
        indices = []
        for iname in index_name.split(';'):
            indices.append(iname)
            correct_names = correct_names and \
            not (simple_index_dict.get(iname) is None)
        return compose_by_product(indices, itemsets, data)
    return simple_index_dict.get(index_name)(data, itemsets)
