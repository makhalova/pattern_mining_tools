import time
from math import log
import operator
import gc
from collections import defaultdict

from numpy import where, minimum, maximum, unique, array
from numpy import round as npround
from scipy.special import gammaln #, binom

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KDTree

l2 = log(2)

def un_int(value): # the universal code for integers
    const =  2.865064
    logsum = log(const,2)
    cond = True # condition
    if value == 0:
        logsum = 0
    else:
        while cond:
            value = log(value,2)
            cond = value > 0.000001
            if value < 0.000001:
                break
            logsum += value
    return logsum

def mint(X, n_bins=10, n_neighbors=10, pattern_length='standard',
              eps=1, prune=True, max_nmb_candidates=2000,
              not_discretized=True):
    gamma_eps = gammaln(eps)/l2

    def not_equal(l1, l2):
        for v, w in zip(l1, l2):
            if v != w:
                return True
        return False

    def not_greater(l1, l2):
        for v, w in zip(l1, l2):
            if v > w:
                return False
        return True

    def get_gain_min(pid1, pid2):
        lo1, u1, usg1, g1 = patterns[pid1]
        lo2, u2, usg2, g2 = patterns[pid2]
        l = minimum(lo1, lo2)
        u = maximum(u1, u2)
        size = sum([log(v, 2) for v in (u-l) + 1])
        usg = usg1 + usg2
        c = (l,u,usg,usg*size - gammaln(usg+eps)/l2)
        cands[(pid1, pid2)] = c
        gains[(pid1, pid2)] = g1 + g2 - c[3]

    def prune_it(L_total):
        max_border = X_dun.max(axis = 0)
        min_border = X_dun.min(axis = 0)
        cands.clear()
        gains.clear()
        # compute candidates
        candidates = list(patterns.keys())
        cand_inds = [
            (k1, k2) for i, k1 in enumerate(candidates)
            for k2 in candidates[i+1 :]
            ]
        for f,l in cand_inds:
            get_gain_min(f, l)
        new_id = max(patterns.keys())
        new_patterns = []
        # start minimization: computing the candidates
        n_old_patterns = len(patterns) + 1
        while n_old_patterns > len(patterns):
            sorted_gains = sorted(gains.items(), key=operator.itemgetter(1),
                                  reverse=True)
            included = defaultdict(lambda: [])
            # creating the candidates for merging (more than a pair)
            for (pid1, pid2), _ in sorted_gains:
                if patterns.get(pid1) and patterns.get(pid2):
                    l, u, _, _ = cands[(pid1, pid2)]
                    for pid, (pl, pu, _, _) in patterns.items():
                        if (pid != pid1) and (pid != pid2):
                            if not_greater(l, pl) & not_greater(pu, u):
                                included[(pid1, pid2)].append(pid)
                    if len(included) == max_nmb_candidates:
                        break
                else:
                    del gains[(pid1, pid2)], cands[(pid1, pid2)]
            # getting the gains for candidates that includes for than a pair of patterns
            new_gains = {key: gains[key] for key in included.keys()}
            sorted_new_gains = sorted(new_gains.items(),
                                      key=operator.itemgetter(1),
                                      reverse=False)
            n_old_patterns = len(patterns)
            new_patterns.clear()
            #print('new iteration ', n_old_patterns, mid_time - time.time())
            while len(sorted_new_gains) > 0:
                # greedy strategy: chosing the best pair
                # (don't consider how many other patterns the candidate includes)
                (pid1, pid2), _ = sorted_new_gains.pop()
                if patterns.get(pid1) and patterns.get(pid2)\
                        and not_equal(cands[(pid1, pid2)][1], max_border)\
                        and not_equal(cands[(pid1, pid2)][0], min_border):
                    # avoid creating the pattern enveloping the whole space
                    n_patterns = len(patterns)
                    accepted_patterns = []  # list of patterns included in the candidate pattern
                    l, u, usg_total, _ = cands[(pid1, pid2)]
                    size = sum([log(v,2) / l2 for v in (u-l) + 1])  # the size of the candidate pattern
                    L_delta_stable = l_n[n_patterns] + gammaln(G+eps*n_patterns)/l2\
                        - gammaln(eps*n_patterns)/l2 + l_cell + patterns[pid1][3]\
                        + patterns[pid2][3] - usg_total*size + gamma_eps
                    d_pt = 1 # reduction in the number of patterns in the model
                    L_delta_old = L_delta_stable - l_n[n_patterns-d_pt]\
                        - gammaln(G+eps*(n_patterns-d_pt))/l2\
                        + gammaln(eps*(n_patterns-d_pt))/l2\
                        + gammaln(usg_total+eps)/l2
                    for pid in included[(pid1, pid2)]:
                        # start to add patterns that could improve the total length
                        if patterns.get(pid):
                            _,_, usg, g = patterns[pid]
                            d_pt_i = d_pt + 1
                            usg_total_i = usg_total + usg
                            L_delta_stable_i = l_cell + g - usg*size + gamma_eps
                            L_delta_variable = - l_n[n_patterns-d_pt_i]\
                                - gammaln(G+eps*(n_patterns-d_pt_i))/l2\
                                + gammaln(eps*(n_patterns-d_pt_i))/l2\
                                + gammaln(usg_total_i+eps)/l2
                            L_delta = L_delta_stable + L_delta_stable_i + L_delta_variable
                            if L_delta > L_delta_old:  # accept merging
                                d_pt = d_pt_i
                                usg_total = usg_total_i
                                accepted_patterns.append(pid)
                                L_delta_stable += L_delta_stable_i
                                L_delta_old = L_delta
                    # checki if a new candidate allows for a shorter length
                    if L_delta_old > 0:  # add pattern if it reduces the total length
                        new_id += 1
                        patterns[new_id] = (l, u, usg_total, usg_total*size\
                                            - gammaln(usg_total+eps)/l2)
                        del patterns[pid1], patterns[pid2]
                        for pid in accepted_patterns:
                            del patterns[pid]
                        accepted_patterns.clear()
                        new_patterns.append(new_id)
                        L_total -= L_delta_old
            # compute new candidates
            for pid1 in new_patterns:
                for pid2 in patterns:
                    if pid1 != pid2:
                        get_gain_min(pid1, pid2)
        return L_total
    n_neighbors += 1
    # reduce the number of columns if there are some const-value ones
    col_selected = [i for i in range(X.shape[1]) if len(unique(X[:,i])) > 1]
    if X.shape[1] > len(col_selected):
        X = X[:,col_selected]
    G, M = X.shape
    l_n = {key: un_int(key) for key in range(0, 10*max(G,M) + 1)} # length of int
    # discretization
    if not_discretized:
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                               strategy='uniform').fit(X)
        X_discr = est.transform(X).astype(int)
        L_CT_base = l_n[G] + l_n[M] + l_n[n_bins] #+ M*log(binom(G-1, n_bins - 1),2)
        if pattern_length == 'minimal':
            l_cell = M*log(n_bins,2)
        elif pattern_length == 'standard':
            l_cell = M*log(n_bins*(n_bins-1)/2 + n_bins,2)
           #l_cell = M*log(n_bins*(n_bins-1)/2,2)
           #l_cell = 2*M*log(n_bins,2)
        else:
            raise ValueError("Invalid pattern length type.")
    else:
        n_bins_list = [max(vals) - min(vals) + 1 for vals
            in [unique(X[:,i]) for i in range(M)]]
        X_discr = X
        L_CT_base = l_n[G] + l_n[M] + sum([l_n[n_bins] for n_bins in n_bins_list])
        if pattern_length == 'minimal':
            l_cell = sum([log(n_bins,2) for n_bins in n_bins_list])
        elif pattern_length == 'standard':
            l_cell = sum([log(n_bins*(n_bins-1)/2 + n_bins,2) for n_bins in n_bins_list])
        else:
            raise ValueError("Invalid pattern length type.")
    # remove repetitive rows
    X_dun, inverse_indices, counts = unique(X_discr, return_counts=True,
                                               return_inverse=True, axis=0)
    #print('elementary ', X_dun.shape)
    n_neighbors = min(n_neighbors, X_dun.shape[0])
    # to reconstruct original indices
    new2original = {i: set() for i in range(X_dun.shape[0])}
    for i, v in enumerate(inverse_indices):
        new2original[v].add(i)
    # computing the standard total length (the reconstruction error is 0)
    n_patterns = X_dun.shape[0]
    L_CT_var = l_n[n_patterns] + n_patterns*l_cell
    L_DCT_init = gammaln(G+eps*n_patterns)/l2 - gammaln(eps*n_patterns)/l2\
               - sum([gammaln(usg+eps)/l2 - gamma_eps for usg in counts])
    L_total = L_CT_base + L_CT_var + L_DCT_init
    # standard param on patterns
    n_patterns_stand = n_patterns
    L_total_stand = L_total
    # --- time check ---
    start_time = time.time()
    tree = KDTree(X_dun, leaf_size = int(n_bins*.5))
    # computing candidates using 'n_neighbors' closest points
    patterns = {i: (row, row, cnt, -gammaln(cnt+eps)/l2) for i,(row, cnt) \
                in enumerate(zip(X_dun, counts))}
    neighbours = {i : set(tree.query(array(x).reshape(1,-1), k=n_neighbors,
                  return_distance=False)[0][1:]) for i, x in enumerate(X_dun)}
    del tree
    cand_inds= set([(i, j) if i < j else (j, i) for i, lst in neighbours.items() for j in lst])
    # computing the length gains for candidates
    print('starts', len(cand_inds))
    # start length minimization
    cands = {}
    gains = {}
    cand_to_add = set([])  # to store individual patterns for candidate update
    while len(cand_inds) > 0:
        for f,l in cand_inds:
            get_gain_min(f, l)
        cand_inds.clear()
        sorted_gains = sorted(gains.items(), key=operator.itemgetter(1), reverse=False)
        L_total_old = L_total + 1
        while (L_total <= L_total_old) and (len(sorted_gains) > 0):
            n_patterns = len(patterns)
            new_id = max(patterns.keys()) + 1
            if len(sorted_gains)%50000 == 0:
                print('gain', len(sorted_gains))
                gc.collect()
            best_inds, best_gain = sorted_gains.pop()
            if patterns.get(best_inds[0]) and patterns.get(best_inds[1]):
                L_delta = l_n[n_patterns] - l_n[n_patterns-1] + l_cell\
                        + gammaln(G+eps*n_patterns)/l2\
                        - gammaln(G+eps*(n_patterns-1))/l2\
                        + gammaln(eps*(n_patterns-1))/l2\
                        - gammaln(eps*n_patterns)/l2 + gamma_eps + best_gain
                if L_delta > 0:
                    patterns[new_id] = cands[best_inds]
                    neighbours[new_id] = set([
                        v for v in neighbours[best_inds[0]].union(neighbours[best_inds[1]])
                        if patterns.get(v)
                        ])
                    for v in neighbours[new_id]:
                        neighbours[v].add(new_id)
                    del patterns[best_inds[0]], patterns[best_inds[1]], neighbours[best_inds[0]], \
                        neighbours[best_inds[1]]
                    cand_to_add.add(new_id)
                    del gains[best_inds], cands[best_inds]
                    L_total_old = L_total
                    L_total -= L_delta
                else:
                    break
            else:
                del gains[best_inds], cands[best_inds]
        if len(cand_to_add) > 0:
            clean_up = [v for v in cand_to_add if not patterns.get(v)]
            for i in clean_up:
                del cand_to_add[i]
            cand_inds = set([(i, j) if i < j else (j, i) for i in cand_to_add
                for j in neighbours[i] if (patterns.get(j)) and (i != j)])
            cand_to_add.clear()
    print('start pruning')
    mid_time = time.time()
    mid_L_total = L_total
    mid_n_patterns = len(patterns)
    if prune:
        L_total = prune_it(L_total)
    end_time = time.time()
    # compute pattern extensions
    pattern_inds = {k: where((X_dun >= l).all(axis=1) & (X_dun <= u).all(axis=1))[0]
        for k, (l, u, _, _) in patterns.items()}
    pattern_true_inds = {}
    for key, lst in pattern_inds.items():
        pattern_true_inds[key] = set([])
        for v in lst:
            pattern_true_inds[key] = pattern_true_inds[key].union(new2original[v])
    if prune:
        return pattern_true_inds, npround(L_total, 4), npround(L_total_stand, 4), \
            n_patterns_stand, npround(end_time-start_time, 4),\
            npround(mid_L_total-L_total, 3), mid_n_patterns - len(patterns),\
            npround(end_time-mid_time, 4)
    return pattern_true_inds, npround(L_total, 4), npround(L_total_stand, 4),\
        n_patterns_stand, npround(end_time-start_time, 4)
