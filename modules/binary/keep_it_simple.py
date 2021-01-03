"""
KeepItSimple: an algorithm for multi-model pattern mining
---------------------------------------------------------
"""

#import numpy as np
from numpy import argwhere, array, Infinity, where
from numpy import sum as npsum
from pattern_mining_tools.modules.models.codetable import CodeTable

class Stat:
    """ Statistic collector. """

    def __init__(self, n_attributes, n_items):
        self.n_attribute_list = [n_attributes]
        self.n_covered_cells_list = [0]
        self.n_generated_pattern_list = []
        self.n_selected_pattern_list = []
        self.n_uncovered_cells_list = [n_items]
        self.candidate_set_max_size = 0
        self.candidate_set_min_size = Infinity
        self.candidate_set_average_size = 0
        self.selected_candidate_set_max_size = 0
        self.selected_candidate_set_min_size = Infinity
        self.selected_candidate_set_avg_size = 0

    def update(self, candidate_set_size, selected_candidate_set_size, data_new,
               data_uncovered, new_dim):
        """ Updates the characteristics of the current epoch. """
        self.n_attribute_list.append(new_dim.sum())
        self.n_covered_cells_list.append((data_new - data_uncovered).sum())
        self.n_generated_pattern_list.append(candidate_set_size)
        self.n_selected_pattern_list.append(selected_candidate_set_size)
        self.n_uncovered_cells_list.append(data_uncovered.sum())
        if self.candidate_set_max_size < candidate_set_size:
            self.candidate_set_max_size = candidate_set_size
        if self.candidate_set_min_size > candidate_set_size:
            self.candidate_set_min_size = candidate_set_size
        self.candidate_set_average_size += candidate_set_size
        if self.selected_candidate_set_max_size < selected_candidate_set_size:
            self.selected_candidate_set_max_size = selected_candidate_set_size
        if self.selected_candidate_set_min_size > selected_candidate_set_size:
            self.selected_candidate_set_min_size = selected_candidate_set_size
        self.selected_candidate_set_avg_size += selected_candidate_set_size

    def get_stat(self, n_epochs):
        """ Returns the collected staticstics (of all epochs). """
        return {'n_epochs' : n_epochs, 'n_attribute_list' : self.n_attribute_list,
                'n_covered_cells_list' : self.n_covered_cells_list,
                'n_generated_pattern_list' : self.n_generated_pattern_list,
                'n_selected_pattern_list' : self.n_selected_pattern_list,
                'n_uncovered_cells_list' : self.n_uncovered_cells_list,
                'candidate_set_max_size': self.candidate_set_max_size,
                'candidate_set_min_size' : self.candidate_set_min_size,
                'candidate_set_average_size' : self.candidate_set_average_size / n_epochs,
                'selected_candidate_set_max_size' : self.selected_candidate_set_max_size,
                'selected_candidate_set_min_size' : self.selected_candidate_set_min_size,
                'selected_candidate_set_avg_size' : self.selected_candidate_set_avg_size / n_epochs
                }

def avg(lst):
    """Compute the average value of the numbers in the list 'lst'."""
    if len(lst) == 0:
        return 0
    return sum(lst)/len(lst)

def get_closure(candidate, data):
    """ Returns the closure of 'candidate'. """
    p_attributes, p_obj_mmb = candidate
    p_obj_mmb_size = sum(p_obj_mmb)
    closed_set = set([v[0] for v in argwhere(npsum(data[p_obj_mmb==1], axis=0)
        == p_obj_mmb_size)])
    return (frozenset(closed_set.union(p_attributes)), (data[:,
        list(p_attributes)].sum(axis=1) == len(p_attributes)).astype(int))

def get_closure_of_list(candidates, data, score=None):
    """ Returns the closure of the candidates from the list 'candidates'. """
    closed_candidates = {}
    if score is None:
        for p_attributes, extent in candidates.items():
            p_obj_mmb = extent
            p_obj_mmb_size = sum(p_obj_mmb)
            closed_set = frozenset([v[0] for v in argwhere(npsum(data[p_obj_mmb==1],
                axis=0) == p_obj_mmb_size)])
            closed_candidates[closed_set] = (data[:, list(p_attributes)].sum(axis=1)
                == len(p_attributes)).astype(int)
        return closed_candidates
    score_closed = {}
    for p_attributes, extent in candidates.items():
        p_obj_mmb = extent
        p_obj_mmb_size = sum(p_obj_mmb)
        closed_set = frozenset([v[0] for v in argwhere(npsum(data[p_obj_mmb==1], axis=0)
            == p_obj_mmb_size)])
        closed_candidates[closed_set] = (data[:,list(p_attributes)].sum(axis=1)
            == len(p_attributes)).astype(int)
        if score_closed.get(closed_set):
            score_closed[closed_set] = min(score_closed[closed_set], score[p_attributes])
        else:
            score_closed[closed_set] = score[p_attributes]
    return closed_candidates, score_closed

def compute_merged(patterns, data):
    """ Computing biclosed itemsets. """
    candidates = {}
    keys = list(patterns.keys())
    for i1, k1 in enumerate(keys):
        for k2 in keys[i1+1:]:
            ids1 = k1
            ids2 = k2
            vec = patterns[k1] & patterns[k2]
            if sum(vec) > 0:
                closed_itemset = get_closure((ids1.union(ids2), vec), data)
                c_itemset = closed_itemset[0]
                if candidates.get(c_itemset) is None:
                    candidates[c_itemset] = closed_itemset[1]
    return candidates

def keep_it_simple_s(data, covering='overlapping', cover_indices=[('length', -1),
                                                                  ('frequency', -1)],\
                     candidate_indices=[('frequency', -1), ('length', -1)],
                     return_statistic=False, frequency_threshold=0.0, **kwargs):
    n_attr = data.shape[1]
    CT = CodeTable(data, cover_indices=cover_indices)
    best_patterns = {}
    true_indices = array(list(range(n_attr)))
    data_new = data.copy()
    cr_len = CT.stand_total_len * 2
    cr_len_new = CT.stand_total_len
    st_total_length = CT.stand_total_len
    n_epochs = 0
    run = True
    if return_statistic:
        stat = Stat(n_attr, data.sum())
    while run:
        cr_len = cr_len_new
        pruned_closed_candidates = compute_merged({frozenset([i]): data_new[:,i]
            for i in range(data_new.shape[1])}, data_new)
        delete_them = [v for v in pruned_closed_candidates.keys()
            if avg(pruned_closed_candidates[v]) <= frequency_threshold]
        for k in delete_them:
            del pruned_closed_candidates[k]
        if len(pruned_closed_candidates) > 0:
            pruned_sets = list(pruned_closed_candidates.keys())
            CT = CodeTable(data_new, cover_indices=cover_indices)
            itemsets = CT.list_set_to_list_binary_vector(pruned_sets)
            selected_itemsets = CT.add_itemsets(itemsets, data_new,
                indices=candidate_indices, covering=covering, **kwargs)
            selected_keys = [pruned_sets[i] for i in selected_itemsets]
            selected_keys_true = {}
            for s in selected_keys:
                true_s = frozenset([true_indices[i] for i in s])
                selected_keys_true[true_s] = (pruned_closed_candidates[s], n_epochs)
            data_uncovered = data_new.copy()
            for itemset in CT.itemsets:
                if sum(itemset) > 1:
                    for i in range(data_new.shape[0]):
                        vec = data_new[i,:] & itemset
                        if sum(vec) == sum(itemset):
                            data_uncovered[i, where(itemset)[0]] = 0
            new_dim = data_uncovered.sum(axis=0) > 0
            true_indices = true_indices[new_dim]
            cr_len_new = CT.total_len
            n_epochs += 1
            if return_statistic:
                stat.update(len(pruned_sets), len(selected_keys), data_new,
                    data_uncovered, new_dim)
            if cr_len > cr_len_new:
                best_patterns.update(selected_keys_true)
                if sum(new_dim) > 0:
                    data_new = data_new[:,new_dim]
                else:
                    run = False
            run = run and (sum(new_dim)<data_uncovered.shape[1]) and (cr_len>cr_len_new)
        else:
            run = False
    if return_statistic:
        stat_dict = stat.get_stat(n_epochs)
        stat_dict.update({'L_ST' : st_total_length})
        return best_patterns, stat_dict
    return best_patterns

def keep_it_simple_d(data, covering='overlapping', cover_indices=[('length', -1),
                                                                  ('frequency', -1)],
                     candidate_indices=[('frequency', -1), ('length', -1)],
                     frequency_threshold=0.0, return_statistic=False, **kwargs):
    n_attr = data.shape[1]
    CT = CodeTable(data, cover_indices=cover_indices)
    best_patterns = {}
    true_indices = array(list(range(n_attr)))
    data_new = data.copy()
    cr_len = CT.stand_total_len * 2
    cr_len_new = CT.stand_total_len
    st_total_length = CT.stand_total_len
    n_epochs = 0
    run = True
    if return_statistic:
        stat = Stat(n_attr, data.sum())
    while run:
        cr_len = cr_len_new
        pruned_closed_candidates = compute_merged({frozenset([i]): data_new[:,i]
            for i in range(data_new.shape[1])}, data_new)
        delete_them = [v for v in pruned_closed_candidates.keys()
            if avg(pruned_closed_candidates[v]) <= frequency_threshold]
        for k in delete_them:
            del pruned_closed_candidates[k]
        if len(pruned_closed_candidates) > 0:
            pruned_sets = list(pruned_closed_candidates.keys())
            CT = CodeTable(data_new, cover_indices=cover_indices)
            itemsets = CT.list_set_to_list_binary_vector(pruned_sets)
            selected_itemsets = CT.add_itemsets(itemsets, data_new,
                indices=candidate_indices, covering=covering, **kwargs)
            selected_keys = [pruned_sets[i] for i in selected_itemsets]
            selected_keys_true = {}
            for s in selected_keys:
                true_s = frozenset([true_indices[i] for i in s])
                selected_keys_true[true_s] = (pruned_closed_candidates[s], n_epochs)
            data_uncovered = data_new.copy()
            for itemset in CT.itemsets:
                if sum(itemset) > 1:
                    for i in range(data_new.shape[0]):
                        vec = data_new[i, :] & itemset
                        if sum(vec) == sum(itemset):
                            data_uncovered[i, where(itemset)[0]] = 0
            new_dim = data_uncovered.sum(axis = 0) > 0
            true_indices = true_indices[new_dim]
            cr_len_new = CT.total_len
            n_epochs += 1
            if return_statistic:
                stat.update(len(pruned_sets), len(selected_keys), data_new,
                    data_uncovered, new_dim)
            if cr_len > cr_len_new:
                best_patterns.update(selected_keys_true)
                if sum(new_dim) > 0:
                    data_new = data_new[:,new_dim]
                else:
                    run = False
            run = run and (sum(new_dim)<data_uncovered.shape[1]) and (cr_len>cr_len_new)
        else:
            run = False
    if return_statistic:
        stat_dict = stat.get_stat(n_epochs)
        stat_dict.update({'L_ST' : st_total_length})
        return best_patterns, stat_dict
    return best_patterns
