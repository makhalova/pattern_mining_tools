"""
Likely-occuring itemset miner.

LO itemsets are stored in a trie (class 'Tree'), the other classes are parent.
"""

import operator
from numpy import zeros, dot, where, multiply, argsort, ones


def avg(lst):
    """Compute the average value of the numbers in the list 'lst'."""
    if len(lst) == 0:
        return 0
    return sum(lst)/len(lst)

class Node:
    """
    Class of nodes with arbitrary labels.
    """
    def __init__(self, extent, label):
        self.childs = []
        self.frequency = avg(extent)
        self.extent = extent
        self.labels = label

    def print_node(self):
        """ Prints the node label, frequency and the number of children. """
        for child in self.childs:
            print(child.labels, child.frequency, len(child.childs))
            child.print_node()

    def get_nodes_with_area(self, return_without_singletons=True):
        """ Returns the node labels with their areas. """
        length_threshold = 0
        if return_without_singletons:
            length_threshold = 1
        lst = []
        for child in self.childs:
            lst += child.get_nodes_with_area()
        return [(child.labels, child.frequency * len(child.labels), len(child.labels)) \
                for child in self.childs if len(child.labels) > length_threshold] + lst

    def get_nodes_with_frequencies(self, return_without_singletons=True):
        """ Returns the node labels with their frequencies. """
        length_threshold = 0
        if return_without_singletons:
            length_threshold = 1
        lst = []
        for child in self.childs:
            lst += child.get_nodes_with_frequencies()
        return [(child.labels, child.frequency, len(child.labels)) for child
                in self.childs if len(child.labels) > length_threshold] + lst

    def get_ranked_nodes(self, metric='area', return_value=False):
        """ Returns the nodes ranked by 'metric'. """
        if metric == 'area':
            lst = self.get_nodes_with_area()
            ordered_list = sorted(lst, key = operator.itemgetter(1, 2))[::-1]
        if  metric == 'frequency':
            lst = self.get_nodes_with_frequencies()
            ordered_list = sorted(lst, key = operator.itemgetter(1, 2))[::-1]
        if return_value:
            return ordered_list
        return [v for (v, _, _) in ordered_list]

    def get_binary_nodes(self, n_attributes, discard_frequent_threshold=1.):
        lst = self.get_nodes_with_frequencies()
        ordered_list = sorted(lst, key = operator.itemgetter(1, 2))[::-1]
        itemsets = []
        for (itemset, freq, _) in ordered_list:
            if freq < discard_frequent_threshold:
                bin_itemset = zeros(n_attributes, dtype=int)
                bin_itemset[itemset] = 1
                itemsets.append(bin_itemset)
        return itemsets

    def get_binary_closed_nodes(self, n_attributes, discard_frequent_threshold=1.):
        lst = self.get_nodes_with_frequencies()
        ordered_list = sorted(lst, key = operator.itemgetter(1, 2))[::-1]
        not_discarded = [True for v in ordered_list]
        for i, (itemset, freq, _) in enumerate(ordered_list):
            for j in range(i + 1, len(ordered_list)):
                if not_discarded[j]:
                    itemset1, freq1, _ = ordered_list[j]
                    if (freq - freq1) < 10e-5:
                        if len(set(itemset1).intersection(set(itemset))) \
                            == min(len(itemset), len(itemset1)):
                            not_discarded[j] = False
        itemsets = []
        for i, (itemset, freq, _) in enumerate(ordered_list):
            if (freq < 1.) and not_discarded[i]:
                bin_itemset = zeros(n_attributes, dtype=int)
                bin_itemset[list(itemset)] = 1
                itemsets.append(bin_itemset)
        return itemsets



class ClosedNode(Node):
    """
    Class of nodes with closed itemsets.
    """
    def __init__(self, dataset, extent, label):
        Node.__init__(self, extent, label)
        idx, = where(dot(dataset.T, extent) == extent.sum())
        self.labels = set(idx)



class Tree():
    """
    Tree that stores itemsets or their generators (in cased of closed itemsets)
    in a suffix tree.


    Parameters
    ----------
    Q : float, default = 1
        Lift threshold. All the itemsets Xm that fr(Xm) >= Q fr(X)fr(m)
        will be added to the tree.
    min_sup : float, default = 1.
        Minimal support threshold of itemsets
    node_type : string, 'closed' or 'arbitrary'
        Type of generating itemsets. Depending on the itemset type specific
        functions ('merge', 'create_node') will be chosed during building the tree.


    Attributes
    ----------
    itemsets : dict.
        Dictionary 'itemset': 'corresponding node' of all nodes in the tree
    data : ndarray of shape (n_objects, n_features) of 0-1
        Dataset for computing itemsets.
    """

    def __init__(self, data=None, min_sup=0.00, n_type='closed', Q = 1):
        self.root = None
        self.data = data
        self.min_sup = min_sup
        self.itemsets = {}
        self.node_type = n_type
        self.Q = Q
        self.n_attributes = 0
        if self.node_type == 'closed':
            self.create_node = self._create_closed_node
        else:
            self.create_node = self._create_arbitrary_node


    def _merge(self, current_node, candidate):
        """
        Add the closed itemset of the 'candidate' node to a subtree
        with the 'current_node' root.
        """
        if len(candidate.labels.difference(current_node.labels)) > 0:
            new_data = multiply(current_node.extent, candidate.extent)
            new_label = current_node.labels.union(candidate.labels)
            avg_val = avg(new_data)
            if current_node.frequency == avg_val:
                if self.itemsets.get(frozenset(new_label)) is None:
                    del self.itemsets[frozenset(current_node.labels)]
                    current_node.labels = current_node.labels.union(candidate.labels)
                    self.itemsets[frozenset(current_node.labels)] = current_node
                else: # just remove a redundant node
                    del self.itemsets[frozenset(current_node.labels)]
            elif (self.Q * current_node.frequency * candidate.frequency <= avg_val) \
                    and (avg_val >= self.min_sup):
                for child in current_node.childs:
                    self._merge(child, candidate)
                self._add_node(current_node, self.create_node(extent=new_data,
                    label=new_label))

    def _create_arbitrary_node(self, label=None, extent=None):
        return Node(extent=extent, label=label)

    def _create_closed_node(self, label=None, extent=None):
        return ClosedNode(dataset=self.data, extent=extent, label=label)

    def _add_node(self, current_node, child_node):
        flabels = frozenset(child_node.labels)
        if self.itemsets.get(flabels) is None:
            current_node.childs.append(child_node)
            self.itemsets[flabels] = child_node

    def build_tree(self, data):
        """ Builds a trie of LO itemsets. """
        self.data = data
        col_sums = data.sum(axis = 0)
        self.root = self.create_node(extent=ones(data.shape[0]), label=set([]))
        sorted_indices = argsort(col_sums)[::-1]
        for i in sorted_indices:
            labels = set([i])
            node = self.create_node(extent=data[:,i], label=labels)
            for child in self.root.childs:
                self._merge(child, node)
            self._add_node(self.root, node)
        self.n_attributes = data.shape[1]

    def print_tree(self):
        """ Prints tree starting from the root.
        See class 'Node' for details.
        """
        self.root.print_node()

    def get_nodes_with_area(self, return_without_singletons=True):
        """ Returns node labels and their areas.

        Parameters
        ----------

        return_without_singletons : bool
            If True, returns the singletons as well.

        See class 'Node' for details.
        """
        return self.root.get_nodes_with_area(return_without_singletons)

    def get_nodes_with_frequencies(self, return_without_singletons=True):
        """ Returns node labels and their frequencies.

        Parameters
        ----------

        return_without_singletons : bool
            If True, returns the singletons as well.

        See class 'Node' for details.
        """
        return self.root.get_nodes_with_frequencies(return_without_singletons)

    def get_ranked_nodes(self, metric='area', return_value=False):
        """ Returns node labels and ranked by 'metric'.

        Parameters
        ----------

        return_value : bool
            If True, returns the values of the metircs, otherwise only the labels.

        See class 'Node' for details.
        """
        return self.root.get_ranked_nodes(metric, return_value)

    def get_binary_nodes(self, discard_frequent_threshold=1.):
        return self.root.get_binary_nodes(self.n_attributes,
                                          discard_frequent_threshold)

    def get_binary_closed_nodes(self, discard_frequent_threshold=1.):
        return self.root.get_binary_closed_nodes(self.n_attributes,
                                                 discard_frequent_threshold)
