"""
Code table computing
--------------------
"""
from math import log2
from bisect import bisect, insort
from collections import Counter
from numpy import vectorize, zeros_like, where, array, dot, zeros
from pattern_mining_tools.modules.indices.utiltools import sort_itemsets as sort_itemsets
from pattern_mining_tools.modules.indices.utiltools import compute_indices as compute_indices

def avg(lst):
    """Compute the average value of the numbers in the list 'lst'."""
    if len(lst) == 0:
        return 0
    return sum(lst)/len(lst)

class CodeTable:
    """
    A code table is a list of ordered itemsets and their lengths.

    Parameters
    ----------

    data : ndarray
        The binary arrray to build the stadard code table

    cover_indices : list, default=[('length', -1),('frequency', -1)])
        Defines the cover order; default value is the standard cover order.
        The supported indices 'support', 'frequency', 'lift',
        'naive_bayes_probability', 'separation', 'length'. See details in
        ``pattern_mining_tools.modules.indices.utiltools.simple_indices''.


    Attributes
    ----------

    n_candidates : int, default=0
        The number of candidates is set in 'add_itemsets' function.

    covering : string, default='disjoint'
        The strategy of covering: 'disjoint', 'overlapping', or 'parametrized'.
        For the parametrized strategy, an itemset is accepted if the rate of
        items covered uniquely by the current item is 'singleton_rate'.
        For the disjoint covering 'ingleton_rate=1', for the overlapping covering
        'singleton_rate=0'.

        The corresponding functions:
        'disjoint' : _compute_ct_disjoint_cover
        'overlapping' : _compute_ct_overlapped_cover
        'parametrized' : _compute_ct_cover_parametrised


    Private attributes
    ------------------

    _n_attributes : int
        The number of attributes.

    _n_objects : int
        The number of objects.

    _cover_indices : list of tuples ('index name', 'order'),
        default=[('length', -1),('frequency', -1)])
        Defines the order of covering, 'order' is 1 (descending)
        or -1 (asceding). It makes sense only for disjoint covering

    _alphabet_lens : list of float
        The length of the code words of corresponding to the singletons.

    _data_len : float
        L(D | CT).

    _code_table_len : float
        L(CT)

    _total_len : float
        L(CT) + L(D | CT)

    _stand_total_len : float
        L(ST) + L(D | ST)

    _stat : dictionary
        Characteristics of the code table.

    _singleton_frequency : list of int
        List of the frequencies of singletons.

    _nonsingleton_frequency : list of int
        List of the frequencies of nonsingletons.

    _nonsingletons : list of binary array.
        List of selected itemsets.


    Temporary attributes
    ------------------

    _candidates_cover_indices
    _candidate_extents
    _candidates_cover_indices
    """

    def __init__(self, data, cover_indices=[('length', -1),('frequency', -1)]):
        def func_data_len(usg, full_usg):
            return usg * log2(full_usg/usg) if usg > 0 else 0
        get_data_length = vectorize(func_data_len, cache = True)
        zero_usg_const = 0.05
        self._nonsingletons = []
        self._cover_indices = cover_indices
        self.n_candidates = 0
        self._n_objects, self._n_attributes = data.shape
        itemset_usg = data.sum(axis = 0)
        full_usg = sum(itemset_usg) + Counter(itemset_usg)[0] * zero_usg_const
        itemset_len = get_data_length(itemset_usg, full_usg)
        self._alphabet_lens = [
            log2(full_usg/x) if  x > 0 else log2(full_usg/zero_usg_const)
            for x in itemset_usg
            ]
        self._data_len = sum(itemset_len)
        self._code_table_len = 2 * sum(self._alphabet_lens)
        self._total_len = self._data_len + self._code_table_len
        self._stand_total_len = self._total_len
        self.candidate_indices = None
        self.covering = None

    def list_set_to_list_binary_vector(self, list_of_sets):
        list_of_binary_vectors = []
        append = list_of_binary_vectors.append
        for s in list_of_sets:
            bin_itemset = zeros(self._n_attributes, dtype=int)
            bin_itemset[list(s)] = 1
            append(bin_itemset)
        return list_of_binary_vectors

    def __repr__(self):
        return 'CT over {0} attributes, {1} itemsets'.format(self._n_attributes,
            len(self._nonsingletons))

    @property
    def total_len(self):
        # get the total two-part total description length
        return self._total_len

    @property
    def stand_total_len(self):
        # get the standard two-part total description length
        return self._stand_total_len

    @property
    def itemsets(self):
        return self._nonsingletons

    @property
    def stat(self):
        return self._stat

    def _set_stat(self, stats):
        singleton_usg, nonsingleton_usg, self._singleton_frequency, \
            self._nonsingleton_frequency = stats
        total_usg = sum(nonsingleton_usg) + sum(singleton_usg)
        self._data_len = sum([v * log2(total_usg/v) if v > 0 else 0 for v in nonsingleton_usg]) \
            + sum([v * log2(total_usg/v) if v > 0 else 0 for v in singleton_usg])
        self._code_table_len = self._total_len - self._data_len
        avg_sigleton_usg = avg(singleton_usg)
        avg_sigleton_fr = avg(self._singleton_frequency)
        ov = sum([u * sum(itemset) for u, itemset in zip(self._nonsingleton_frequency, \
            self._nonsingletons)]) / (sum(self._singleton_frequency)
            - sum(singleton_usg)) if len(self._nonsingleton_frequency) > 0 else 0
        self._stat = {'avg_sigleton_usg': avg_sigleton_usg,
                'avg_nonsigleton_usg': avg(nonsingleton_usg),
                'avg_sigleton_fr': avg_sigleton_fr,
                'avg_nonsigleton_fr': avg(self._singleton_frequency),
                'n_singletons': len([v for v in singleton_usg if v > 0]),
                'uncovered_cell_rate': avg_sigleton_usg / avg_sigleton_fr,
                'overlapping_ratio': ov,
                'code_table_length': self._data_len,
                'data_length': self._code_table_len,
                'total_length': self._total_len,
                'total_standard_length' : self._stand_total_len}

    def _compute_ct_overlapped_cover(self, data, lazy_computing=False):
        """
        Computes a code table using the 'overlapping' covering strategy.

        Parameters
        ----------
        data : binary ndarray

        lazy_computing : boolean
            If True, use a greedy covering.

        Returns
        -------
            temp_accepted_itemset_ids : list of int
                The kist of selected itemsets.
        """
        def set_val(val):
            covered_data[extent, val] += 1
        def rem_val(val):
            covered_data[extent, val] -= 1
        set_ones = vectorize(set_val, cache=True)
        remove_ones = vectorize(rem_val, cache=True)
        # compute cover
        cur_len = self._total_len
        cand_cover_order_indices = []
        temp_accepted_itemset_ids = []
        nonsingleton_usg = []
        nonsingleton_st_len = []
        usg = data.sum(axis = 0)
        singleton_usg = list(usg[usg > 0])
        if lazy_computing:
            covered_data = zeros_like(data)
            intent_len = 0
            for i in range(self.n_candidates):
                extent = self._candidate_extents[i]
                intent = self._candidate_intents[i]
                itemset_index_value = self._candidates_cover_indices[i]
                intent_st_len = sum([self._alphabet_lens[i] for i in intent])
                position = bisect(cand_cover_order_indices,
                    itemset_index_value)
                insort(cand_cover_order_indices, itemset_index_value)
                nonsingleton_usg.insert(position, len(extent))
                nonsingleton_st_len.insert(position, intent_st_len)
                set_ones(intent)
                uncovered_data = data - covered_data.clip(max=1)
                singleton_usg = uncovered_data.sum(axis=0)
                singleton_len = sum([
                    self._alphabet_lens[i] for i, v
                    in enumerate(singleton_usg) if v > 0
                    ])
                singleton_usg = list(singleton_usg[singleton_usg>0])
                intent_len += intent_st_len
                usg_list= singleton_usg + nonsingleton_usg
                new_len = sum(list(map(lambda x: (x+1) * log2(sum(usg_list)/x),
                                usg_list))) + intent_len + singleton_len
                if cur_len > new_len:
                    cur_len = new_len
                    temp_accepted_itemset_ids.append(i)
                else:
                    del nonsingleton_usg[position]
                    del nonsingleton_st_len[position]
                    del cand_cover_order_indices[position]
                    remove_ones(intent)
        else:
            candidate_extent_lens = []
            for i in range(self.n_candidates):
                itemset_index_value = self._candidates_cover_indices[i]
                candidate_extent_lens.append(len(self._candidate_extents[i]))
                covered_data = zeros_like(data)
                position = bisect(cand_cover_order_indices,
                    itemset_index_value)
                insort(cand_cover_order_indices, itemset_index_value)
                temp_accepted_itemset_ids.insert(position, i)
                temp_nonsingleton_usg = []
                temp_nonsingleton_st_code_len = []
                itemsets_ids_to_remove = []
                intent_len = 0
                for pos_cov, i_cov in enumerate(temp_accepted_itemset_ids):
                    extent = self._candidate_extents[i_cov]
                    intent = self._candidate_intents[i_cov]
                    extent_len = candidate_extent_lens[i_cov]
                    if (covered_data[extent,:][:,intent] == 0).sum() > 0:
                        set_ones(intent)
                        intent_st_len = sum([self._alphabet_lens[i] for i in intent])
                        intent_len += intent_st_len
                        temp_nonsingleton_usg.append(extent_len)
                        temp_nonsingleton_st_code_len.append(intent_st_len)
                    else:
                        itemsets_ids_to_remove.append(pos_cov)
                uncovered_data = data - covered_data.clip(max=1)
                temp_singleton_usg_with_zeros = uncovered_data.sum(axis=0)
                singleton_len = sum([
                    self._alphabet_lens[i] for i, v
                    in enumerate(temp_singleton_usg_with_zeros) if v > 0
                    ])
                usg_list = [v for v in temp_singleton_usg_with_zeros if v > 0] \
                    + [v for  v in temp_nonsingleton_usg if v > 0]
                new_len = sum(list(map(lambda x: (x+1) * log2(sum(usg_list)/x),
                    usg_list))) + intent_len + singleton_len
                if cur_len > new_len:
                    cur_len = new_len
                    for id_del in itemsets_ids_to_remove[::-1]:
                        del temp_accepted_itemset_ids[id_del]
                        del cand_cover_order_indices[id_del]
                    nonsingleton_usg = temp_nonsingleton_usg
                    nonsingleton_st_len = temp_nonsingleton_st_code_len
                    singleton_usg = list(temp_singleton_usg_with_zeros)
                else:
                    del cand_cover_order_indices[position]
                    del temp_accepted_itemset_ids[position]
        # finalize
        self._total_len = cur_len
        self._set_stat((singleton_usg, nonsingleton_usg, list(data.sum(axis = 0)), \
            [len(self._candidate_extents[i]) for i in temp_accepted_itemset_ids]))
        return temp_accepted_itemset_ids


    def _compute_ct_disjoint_cover(self, data):
        """
        Computes a code table using the 'disjoint' covering strategy.

        Parameters
        ----------
        data : binary ndarray

        Returns
        -------
            temp_accepted_itemset_ids : list of int
                The kist of selected itemsets.
        """
        def set_val(val):
            uncovered_data[true_extent, val] -= 1
        set_ones = vectorize(set_val, cache=True)
        # compute cover
        cur_len = self._total_len
        cand_cover_order_indices = []
        temp_accepted_itemset_ids = []
        nonsingleton_usg = []
        temp_intent_len = []
        usg = data.sum(axis = 0)
        singleton_usg = list(usg[usg > 0])
        for i in range(self.n_candidates):
            itemset_index_value = self._candidates_cover_indices[i]
            uncovered_data = data.copy()
            temp_intent_len.append(len(self._candidate_intents[i]))
            position = bisect(cand_cover_order_indices, itemset_index_value)
            insort(cand_cover_order_indices, itemset_index_value)
            temp_accepted_itemset_ids.insert(position, i)
            temp_nonsingleton_usg = []
            temp_nonsingleton_st_code_len = []
            itemsets_ids_to_remove = []
            intent_len = 0
            for pos_cov, i_cov in enumerate(temp_accepted_itemset_ids):
                intent = self._candidate_intents[i_cov]
                true_extent = where((uncovered_data[:,intent]).sum(axis=1)
                    == temp_intent_len[i_cov])[0]
                if len(true_extent) > 0:
                    set_ones(intent)
                    intent_st_len = sum([self._alphabet_lens[i] for i in intent])
                    intent_len += intent_st_len
                    temp_nonsingleton_usg.append(len(true_extent))
                    temp_nonsingleton_st_code_len.append(intent_st_len)
                else:
                    itemsets_ids_to_remove.append(pos_cov)
            temp_singleton_usg = uncovered_data.sum(axis = 0)
            singleton_len = sum([self._alphabet_lens[i] for i, v
                in enumerate(temp_singleton_usg) if v > 0])
            temp_singleton_usg = list(temp_singleton_usg)
            usg_list= [v for v in temp_singleton_usg if v > 0] + temp_nonsingleton_usg
            new_len = sum(list(map(lambda x : (x+1) * log2(sum(usg_list)/x),
                usg_list))) + intent_len + singleton_len
            if cur_len > new_len:
                cur_len = new_len
                for id_del in itemsets_ids_to_remove[::-1]:
                    del temp_accepted_itemset_ids[id_del]
                    del cand_cover_order_indices[id_del]
                nonsingleton_usg = temp_nonsingleton_usg
                singleton_usg = temp_singleton_usg
            else:
                del cand_cover_order_indices[position]
                del temp_accepted_itemset_ids[position]
        # finalize
        self._total_len = cur_len
        self._set_stat((singleton_usg, nonsingleton_usg, list(data.sum(axis = 0)), \
            [len(self._candidate_extents[i]) for i in temp_accepted_itemset_ids]))
        return temp_accepted_itemset_ids

    def _compute_ct_cover_parametrised(self, data, singleton_rate=1):
        """
        Computes a code table using the 'disjoint' covering strategy.

        Parameters
        ----------
        data : binary ndarray

        singleton_rate : float
            The minimal rate of the uniquely covered items needed
            to accept an itemset.

        Returns
        -------
        temp_accepted_itemset_ids : list of int
            The kist of selected itemsets.
        """
        def set_val(val):
            uncovered_data[true_extent, val] -= 1
        set_ones = vectorize(set_val, cache=True)
        # compute cover
        cur_len = self._total_len
        cand_cover_order_indices = []
        temp_accepted_itemset_ids = []
        nonsingleton_usg = []
        temp_intent_len = []
        usg = data.sum(axis = 0)
        singleton_usg = list(usg[usg > 0])
        for i in range(self.n_candidates):
            itemset_index_value = self._candidates_cover_indices[i]
            uncovered_data = data.copy()
            temp_intent_len.append(len(self._candidate_intents[i]))
            position = bisect(cand_cover_order_indices, itemset_index_value)
            insort(cand_cover_order_indices, itemset_index_value)
            temp_accepted_itemset_ids.insert(position, i)
            temp_nonsingleton_usg = []
            temp_nonsingleton_st_code_len = []
            itemsets_ids_to_remove = []
            intent_len = 0
            for pos_cov, i_cov in enumerate(temp_accepted_itemset_ids):
                intent = self._candidate_intents[i_cov]
                complete_extent = where((data[:,intent]).sum(axis=1) == temp_intent_len[i_cov])[0]
                true_extent = complete_extent[
                    where((uncovered_data[:,intent][complete_extent,:]).sum(axis=1)
                    >= temp_intent_len[i_cov]*singleton_rate)[0]]
                if len(true_extent) > 0:
                    set_ones(intent)
                    intent_st_len = sum([self._alphabet_lens[i] for i in intent])
                    intent_len += intent_st_len
                    temp_nonsingleton_usg.append(len(true_extent))
                    temp_nonsingleton_st_code_len.append(intent_st_len)
                else:
                    itemsets_ids_to_remove.append(pos_cov)
            temp_singleton_usg = uncovered_data.sum(axis = 0)
            singleton_len = sum([self._alphabet_lens[i] for i, v
                in enumerate(temp_singleton_usg) if v > 0])
            temp_singleton_usg = list(temp_singleton_usg)
            usg_list= [v for v in temp_singleton_usg if v > 0] \
                    + temp_nonsingleton_usg
            total_usg = sum(usg_list)
            new_len = sum(list(map(lambda x: (x+1) * log2(total_usg/x),
                usg_list))) + intent_len + singleton_len
            if cur_len > new_len:
                cur_len = new_len
                for id_del in itemsets_ids_to_remove[::-1]:
                    del temp_accepted_itemset_ids[id_del]
                    del cand_cover_order_indices[id_del]
                nonsingleton_usg = temp_nonsingleton_usg
                singleton_usg = temp_singleton_usg
            else:
                del cand_cover_order_indices[position]
                del temp_accepted_itemset_ids[position]
        # finalize
        self._total_len = cur_len
        self._set_stat((singleton_usg, nonsingleton_usg, list(data.sum(axis = 0)),\
            [len(self._candidate_extents[i]) for i in temp_accepted_itemset_ids]))
        return temp_accepted_itemset_ids

    def add_itemsets(self, itemsets, dataset,
                     indices=[('frequency', -1),('length', -1)],
                     covering='disjoint', custom_index_values=None,
                     lazy_computing=False, singleton_rate = 1):
        """
        Adds the itemsets to the code table. Only those that minimize
        the total length are accepted to the code table

        Attributes
        ---------

        itemsets : list of binary arrays

        indices: list of tupples (<index_name>, order)
        <index_name> is a string (the index name, might be complex, like '
        lift;frequency', that means lift*frequency), <order> is boolean
        (true is the default value, i.e., ascending order).

        covering : 'disjoint', 'overlapping', or 'parametrized'
        """
        self.n_candidates = len(itemsets)
        self.candidate_indices = indices
        self.covering = covering
        ranked_itemset_ids = sort_itemsets(indices, dataset, itemsets,
            custom_index_values=custom_index_values)
        candidates = [itemsets[i] for i in ranked_itemset_ids] # cache
        self._candidate_intents = [where(cand)[0] for cand in candidates]
        self._candidates_cover_indices = compute_indices(self._cover_indices,
            dataset, candidates)
        itemset_array = array(candidates).T
        sizes = itemset_array.sum(axis = 0)
        shared_attributes = dot(dataset, itemset_array)
        extents = shared_attributes == sizes# (_n_attributes x itemsets)
        self._candidate_extents = [where(extents[:,i])[0] for i in range(extents.shape[1])]
        added_itemset_id_list = []
        if covering == 'disjoint':
            added_itemset_id_list = self._compute_ct_disjoint_cover(dataset)
        elif covering == 'overlapping':
            added_itemset_id_list = self._compute_ct_overlapped_cover(dataset,
                lazy_computing=lazy_computing)
        else:
            added_itemset_id_list = self._compute_ct_cover_parametrised(dataset,
                    singleton_rate=singleton_rate)
        # finalize
        self._nonsingletons = [candidates[i] for i in added_itemset_id_list]
        del candidates
        del self._candidate_intents
        del self._candidates_cover_indices
        del self._candidate_extents
        return [ranked_itemset_ids[i] for i in added_itemset_id_list]

    def get_nonsingleton_itemset_number(self):
        """
        Returns the number of non-singleton itemsets in the CT.
        """
        return len(self._nonsingletons)

    def write_code_table(self, file_name):
        with open('_'.join([file_name,self.covering, str(self.n_candidates)])
                  + '.ct', 'w') as f:
            f.write('{0} {1} {2}\n'.format(self.covering, self.candidate_indices,
                self._cover_indices))
            f.write('{0} {1} {2}\n'.format(len(self._nonsingletons),
                self._n_attributes, max([len(where(itemset)[0]) for itemset
                in self._nonsingletons])))
            lst1 = [self._nonsingletons, self._nonsingleton_frequency]
            lst2 = [list(range(self._n_attributes)), self._singleton_frequency]
            s_format = '{0} (,{1})\n'
            if not self.stat is None:
                lst1.insert(1,self.stat['nonsingleton_usg'])
                lst2.insert(1,self.stat['singleton_usg'])
                s_format = '{0} ({1},{2})\n'
            for l in [lst1, lst2]:
                for itemset, i,j in zip(*l):
                    f.write(s_format.format(' '.join([str(v) for v
                    in where(itemset)[0]]), i, j))

    def cover_objects_disjoint(self, data):
        """
        Computes covering of the dataset 'data'. Returns a dictionary, where key
        of the form <object id> : <lists of itemsets>. If 'return_length' returns
        the length of the encoded data.

        Parameters
        ----------

        data : binary 2-dim array

        return_length: boolean
        """
        def set_val(x):
            uncovered_data[true_extent, x] -= 1
        set_ones = vectorize(set_val, cache=True)
        nonsingleton_usg = []
        nonsingleton_area = []
        uncovered_data = data.copy()
        for itemset in self._nonsingletons:
            intent = where(itemset)[0]
            intent_len = len(intent)
            true_extent = where((uncovered_data[:,intent]).sum(axis = 1) == intent_len)[0]
            extent_len = len(true_extent)
            nonsingleton_usg.append(extent_len)
            if extent_len > 0:
                set_ones(intent)
                nonsingleton_area.append(intent_len * extent_len)
        n_objects = data.shape[0]
        total_data_size = data.sum()
        singleton_usg = uncovered_data.sum(axis = 0)
        uncovered_cell_number = sum(singleton_usg)
        param_list = [
            'uncovered_cell_rate', 'average_nonsingleton_usg',
            'average_singleton_usg', 'used_nonsingleton', 'unused_nonsingleton',
            'used_singleton', 'unused_singleton',
            'average_area_used_nonsingleton', 'overlapping_rate', 'data_length'
            ]
        return dict(zip(param_list, [
                    uncovered_cell_number / total_data_size,
                    avg(nonsingleton_usg) / n_objects,
                    avg(singleton_usg) / n_objects,
                    len(nonsingleton_area),
                    len(nonsingleton_usg) - len(nonsingleton_area),
                    len(singleton_usg[singleton_usg > 0]),
                    len(singleton_usg) - len(singleton_usg[singleton_usg > 0]),
                    avg(nonsingleton_area),
                    sum(nonsingleton_area) / (total_data_size
                    - uncovered_cell_number),  self._data_len]
                ))

    def cover_objects_overlapping(self, data):
        """
        Computes covering of the dataset 'data'. Returns a dictionary, where key
        of the form <object id> : <lists of itemsets>. If 'return_length' returns
        the length of the encoded data.

        Parameters
        ----------

        data : binary 2-dim array

        return_length : boolean
        """
        def set_val(val):
            covered_data[extent, val] += 1
        set_ones = vectorize(set_val, cache=True)
        nonsingleton_usg = []
        nonsingleton_area = []
        covered_data = zeros_like(data)
        for itemset in self._nonsingletons:
            intent = where(itemset)[0]
            intent_len = len(intent)
            extent = where((data[:,intent]).sum(axis=1) == intent_len)[0]
            extent_len = len(extent)
            if (covered_data[extent,:][:,intent] == 0).sum() > 0:
                set_ones(intent)
                nonsingleton_usg.append(extent_len)
                nonsingleton_area.append(intent_len*extent_len)
            else:
                nonsingleton_usg.append(0)
        uncovered_data = data - covered_data.clip(max=1)
        n_objects = data.shape[0]
        total_data_size = data.sum()
        singleton_usg = uncovered_data.sum(axis=0)
        uncovered_cell_number = sum(singleton_usg)
        param_list = [
            'uncovered_cell_rate', 'average_nonsingleton_usg',
            'average_singleton_usg', 'used_nonsingleton', 'unused_nonsingleton',
            'used_singleton', 'unused_singleton',
            'average_area_used_nonsingleton', 'overlapping_rate', 'data_length'
            ]
        return dict(zip(param_list, [
                    uncovered_cell_number / total_data_size,
                    avg(nonsingleton_usg) / n_objects,
                    avg(singleton_usg) / n_objects,
                    len(nonsingleton_area),
                    len(nonsingleton_usg) - len(nonsingleton_area),
                    len(singleton_usg[singleton_usg > 0]),
                    len(singleton_usg)
                        - len(singleton_usg[singleton_usg > 0]),
                    avg(nonsingleton_area),
                    sum(nonsingleton_area) / (total_data_size
                        - uncovered_cell_number),  self._data_len
                ]))

    def get_overlapping_stat(self, data):
        """ Computes the statistics for the selected itemsets applied
            with the overlapping cover strategy. """
        def set_val(val):
            covered_data[extent, val] += 1
        set_ones = vectorize(set_val, cache=True)
        nonsingleton_usg = []
        nonsingleton_area = []
        covered_data = zeros_like(data)
        for itemset in self._nonsingletons:
            intent = where(itemset)[0]
            intent_len = len(intent)
            extent = where((data[:,intent]).sum(axis=1) == intent_len)[0]
            extent_len = len(extent)
            if (covered_data[extent,:][:,intent] == 0).sum() > 0:
                set_ones(intent)
                nonsingleton_usg.append(extent_len)
                nonsingleton_area.append(intent_len*extent_len)
            else:
                nonsingleton_usg.append(0)
        nonsingleton_st_code_len = 0
        for itemset in self._nonsingletons:
            nonsingleton_st_code_len += sum([self._alphabet_lens[i] for i
                in where(itemset)[0]])
        uncovered_data = data - covered_data.clip(max = 1)
        n_objects = data.shape[0]
        total_data_size = data.sum()
        singleton_usg = uncovered_data.sum(axis = 0)
        uncovered_cell_number = sum(singleton_usg)
        covered_cell_number = total_data_size - uncovered_cell_number
        overlapping_rate = sum(nonsingleton_area) / covered_cell_number
        average_nonsingleton_usg = avg(nonsingleton_usg) / n_objects
        average_singleton_usg = avg(singleton_usg) / n_objects
        total_usg = sum(nonsingleton_usg) + sum(singleton_usg)
        nonsingleton_lens = [log2(total_usg/v) if v > 0 else 0 for v in nonsingleton_usg]
        singleton_lens = [log2(total_usg/v) if v > 0 else 0 for v in singleton_usg]
        l_ct = sum(nonsingleton_lens) + sum(nonsingleton_st_code_len) \
            + sum([n + l for n, l in zip(self._alphabet_lens, singleton_lens)
            if l > 0])
        l_data = sum([n * l for n, l in zip(nonsingleton_usg,
            self._nonsingleton_lens)]) + sum([n * l for n, l in zip(singleton_usg,
            singleton_lens)])
        param_list = [
            'average_area_used_nonsingleton', 'average_nonsingleton_usg',
            'average_singleton_usg', 'overlapping_rate', 'overlapping_tuple',
            'uncovered_cell_rate', 'used_singleton', 'code_table_length',
            'data_length', 'total_length', 'total_standard_length'
            ]
        return dict(zip(param_list, [
                    avg(nonsingleton_area), average_nonsingleton_usg,
                    average_singleton_usg, overlapping_rate,
                    (sum(nonsingleton_area), covered_cell_number),
                    uncovered_cell_number / total_data_size,
                    len(singleton_usg[singleton_usg > 0]),
                    l_ct, l_data, l_ct + l_data, self.stand_total_len ]))

