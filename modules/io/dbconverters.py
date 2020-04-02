""" Krimp/Slim-converters """


import numpy as np

# read training / test data
def db2train(data_dir, data_name, write_with_targets = True,
             print_job_status = False,
             write_context_with_negations = True):
    
    input_filename = data_dir + data_name + '.db'
    output_filename = data_dir + data_name + '.csv'
    output_filename_with_negations = data_dir + data_name + '_neg.csv'
    output_target_filefname = data_dir + data_name + '.cl'
    
    with open(input_filename, 'r') as f:
        f.readline().replace('\n', '') # 'fic-1.5'
        info = {}
        
        key, s = f.readline().replace('\n', '').split(': ')
        while not key.isnumeric():
            info[key] = s
            key, s = f.readline().replace('\n', '').split(': ')
        
        # get parameters of a context
        parameters = info["mi"].split(' ')
        n_classes = 0
        for p in parameters:
            if p.startswith('nR'):
                _, n_rows = p.split('=')
                n_rows = int(n_rows)
            if p.startswith('aS'):
                _, n_cols = p.split('=')
                n_cols = int(n_cols)
        if info.get('cl') != None:
            classes_list = [int(num) for num in info["cl"].split(' ')]
            classes = {int(num): i for i, num in enumerate(classes_list)}
        else:
            classes = {}
        context = np.zeros((n_rows, n_cols))
        
        # fill the context
        for i in range(n_rows):
            s = s.replace('[', '')
            s = s.replace(']', '')
            items = s.split()
            key = int(key)
            for j in items:
                context[i][int(j)] = 1
            line = f.readline().replace('\n', '').split(': ')
            if len(line) > 1:
                s = line[1]
                key = int(line[0])
    f.close()

    # write to cxt file with the krimp-encoded items
    # attribute names
    attribute_names = info["ab"].split(' ')
    attribute_cols = [int(i) for i in attribute_names]
    next_negated = max(attribute_cols) + 1
    cxt_rows = n_rows
    
    target_list = []
    
    
    with open(output_target_filefname, 'w') as f:
        f.writelines('C\n\n')
        f.writelines(str(cxt_rows) + '\n' + str(len(classes)) + '\n\n' )
        
        for i in range(cxt_rows):
            f.writelines(str(i) + '\n')
        for i in classes: # print keys
            f.writelines(str(i) + '\n')
        f.writelines('\n')
        for i in range(n_rows):
            target_class = 0
            for j in classes:
                if context[i][j] > 0:
                    target_class = j
            f.writelines(str(classes[target_class]) + '\n')
            target_list.append(classes[target_class])
    f.close()


    if write_with_targets:
        for col in classes:
            attribute_cols.remove(col)
        cxt_cols = len(attribute_cols)


    with open(output_filename, 'w') as f:
        f.writelines('D\n\n')
        f.writelines(str(cxt_rows) + '\n' + str(cxt_cols) + '\n\n' )
        
        for i in range(cxt_rows):
            f.writelines(str(i) + '\n')
        for i in attribute_cols:
            f.writelines(str(i) + '\n')
        for i in range(cxt_rows):
            s = ''
            for j in attribute_cols:
                s+= '0 ' if context[i][j] == 0 else '1 '
            f.writelines(s + '\n')
    f.close()


    if write_context_with_negations:
        # dictionary: negeted original index (value) : new_index (key)
        negeted_attribute_dict = {}
        for i in sorted(attribute_cols):
            negeted_attribute_dict[i] = next_negated
            next_negated += 1
        with open(output_filename_with_negations, 'w') as f:
            f.writelines('D\n\n')
            f.writelines(str(cxt_rows) + '\n' + str(cxt_cols * 2) + '\n\n' )
            
            for i in range(cxt_rows):
                f.writelines(str(i) + '\n')
                for i in attribute_cols:
                    f.writelines(str(i) + '\n')
            for i in attribute_cols:
                f.writelines(str(negeted_attribute_dict[i]) + '\n')
                for i in range(cxt_rows):
                    s = ''
                    for j in attribute_cols:
                        s+= '0 ' if context[i][j] == 0 else '1 '
                    for j in attribute_cols:
                        s+= '1 ' if context[i][j] == 0 else '0 '
                    f.writelines(s + '\n')
            f.close()


    if print_job_status:
        print('Dataset {} ({} classes: {}) from {} is written to csv ({} x {})'\
          .format(data_name, len(classes), classes,data_dir, n_rows,\
                  len(attribute_cols)))
    attr_dict = {v: i for i, v in enumerate(attribute_cols)}
    if write_context_with_negations:
        return context[:, attribute_cols], np.hstack([context[:, attribute_cols], 1 - context[:, attribute_cols]] ), target_list, classes, attr_dict
    else:
        return context[:, attribute_cols], target_list, classes, attr_dict
    
    
def db2bindata(data_dir, data_name, return_attribute_names = False):
    
    input_filename = data_dir + data_name + '.db'
    output_filename = data_dir + data_name + '.csv'
    
    with open(input_filename, 'r') as f:
        f.readline().replace('\n', '') # 'fic-1.5'
        info = {}
        
        key, s = f.readline().replace('\n', '').split(': ')
        while not key.isnumeric():
            info[key] = s
            key, s = f.readline().replace('\n', '').split(': ')
        
        # get parameters of a context
        parameters = info["mi"].split(' ')
        for p in parameters:
            if p.startswith('nR'):
                _, n_rows = p.split('=')
                n_rows = int(n_rows)
            if p.startswith('aS'):
                _, n_cols = p.split('=')
                n_cols = int(n_cols)

        context = np.zeros((n_rows, n_cols))
        # fill the context
        for i in range(n_rows):
            items = s.split()
            key = int(key)
            for j in items:
                context[i][int(j)] = 1
            line = f.readline().replace('\n', '').split(': ')
            if len(line) > 1:
                s = line[1]
                key = int(line[0])
    f.close()

    # write to cxt file with the krimp-encoded items
    # attribute names
    attribute_names = info["ab"].split(' ')
    attribute_cols = [int(i) for i in attribute_names]
    next_negated = max(attribute_cols) + 1
    cxt_rows = n_rows
    if return_attribute_names:
        return context, attribute_cols
    else:
        return context

def readCT(file_name):
    """ Reads the code table from 'file_name', the output of Krimp/Slim.
    
    Parameters
    ----------
    file_name : string.
        The path of the code table file.


    Returns
    -------
        
    n_non_singleton_itemsets_usage : list of int
        The list of the usage of patterns in covering.
    
    n_non_singleton_itemsets_freq : list of int
        The list of the frequencies of patterns in data.
    
    n_non_singleton_itemsets_size  : list of int
        The length of non-singleton patterns.
    
    n_singleton_itemsets_usage: list of int
        The list of the usage of singletons in covering.

    n_singleton_itemsets_freq : list of int
           The list of the frequencies of singletons in data.

    itemsets : list of binary vectors (n_attributes,)
        The list of patterns from the code table.
    
    
    Example
    -------
    n_non_singleton_itemsets_usage, n_non_singleton_itemsets_freq, n_non_singleton_itemsets_size, \
    n_singleton_itemsets_usage, n_singleton_itemsets_freq, itemsets = readCT(filename.ct)
    
    """
    n_non_singleton_itemsets_usage = []
    n_non_singleton_itemsets_freq = []
    n_non_singleton_itemsets_size = []

    n_singleton_itemsets_usage = []
    n_singleton_itemsets_freq = []

    itemsets = []

    with open(file_name, 'r') as f:

        f.readline().replace('\n', '') # 'fic-1.5'
        paprams = f.readline().replace('\n', '').split(' ') # 'param'
        n_non_singleton_itemsets = int(paprams[0])
        n_singleton_itemsets = int(paprams[1])
        
        for i in range(n_non_singleton_itemsets):
            s = f.readline().replace(')', '')
            attributes, sep, usage_fr = s.rpartition(' (')
            usage, freq = usage_fr.split(',')
            n_non_singleton_itemsets_usage.append(int(usage))
            n_non_singleton_itemsets_freq.append(int(freq))
            n_non_singleton_itemsets_size.append(len(attributes.split(' ')))

            itemset = np.zeros(n_singleton_itemsets, dtype = np.int8)
            for i in attributes.split(' '):
                itemset[int(i)] = 1
            itemsets.append(itemset)

        for i in range(n_singleton_itemsets):
            s = f.readline().replace(')', '')
            attribute, sep, usage_fr = s.rpartition(' (')
            usage, freq = usage_fr.split(',')
            n_singleton_itemsets_usage.append(int(usage))
            n_singleton_itemsets_freq.append(int(freq))
    return n_non_singleton_itemsets_usage, n_non_singleton_itemsets_freq, n_non_singleton_itemsets_size, \
            n_singleton_itemsets_usage, n_singleton_itemsets_freq, itemsets


def read_candidate_list(file_name, n_attr, min_support, max_support, min_itemset_size = 1):
    """ Reading the itemsets from isc-file, the output of Krimp.
    
    
    Parameters
    ----------
    
    file_name : string
        The absolute path of the isc-file (the output of Krimp).
        
    n_attribute : int
        The size of the binary vector that contains items.
        
    min_support : int
        The minimum pattern support threshold (in integers).
        
    max_support : int
        The maximum pattern support threshold (in integer).
        
    min_itemset_size : int, default = 1
        The minimal size of attributes in a pattern. By default,
        singletons are not considered as patterns.


    Returns
    -------
    
    itemset_list : list of tupples (ndrarray, int)
        An element of the list is a tupple, where 'ndarray' is a
        binary verctor (n_attr,) and int is the pattern support in
        absolute values.
    
    n_total_itemsets : int
        The total number of patterns (with those that violate the
        specified constraints).
    
    """
    itemset_list = []
    
    with open(file_name, 'r') as f:
        
        f.readline().replace('\n', '') # 'fic'
        s1 = f.readline().replace('\n', '').split(' ') # 'param'
        x = s1[1]
        if x.startswith('numSets='):
            n_total_itemsets = int(x[x.rindex('=') + 1 : ])
        
        cur_support = min_support + 1
        s = f.readline().replace('\n', '')
        while (cur_support >= min_support) and len(s) > 0:
            itemset_size, itemsets_support = s.split(': ')
            itemsets, _, support = itemsets_support.replace(')', '').rpartition(' (')
            itemset_array = np.zeros(n_attr, dtype=int)
            itemset_array[[i for i in map(lambda x : int(x) ,itemsets.split(' '))]] = 1
            cur_support = int(support)
            if (cur_support < max_support) and (np.sum(itemset_array) > min_itemset_size):
                itemset_list.append((itemset_array, cur_support))
            s = f.readline().replace('\n', '')

    return itemset_list, n_total_itemsets
    
    
    
def write_isc(output_file_name, itemsets, dname):
    """Writing isc-files readable by Krimp.
        
    Parameters
    ----------
    
    output_file_name : string
        The path of the output file.
        
    itemsets : dictionary 'frozen_set' of int : binary vector (n_objects,)
        The dictionary of itemsets, where the keys is itemsets, and the values
        are binary vectors describing which objects containt the itemset.
        
    dname : the name of the dataset used to compute the itemsets.
    """

    numSets = len(itemsets)
    itemsets_sup_num = {k : np.sum(val) for k, val in itemsets.items()}
    sort_itemsets_sup_num = {k: v for k, v in sorted(itemsets_sup_num.items(), key=lambda item: item[1], reverse=True)}
    
    maxLen = np.max([len(itemset) for itemset in sort_itemsets_sup_num.keys()])
    minSupport = np.min([support for support in sort_itemsets_sup_num.values()])

    with open(output_file_name, 'w') as f:
        f.writelines('ficfis-1.3\n')
        f.writelines('mi: numSets={0:d} minSup={1:d} maxLen={2:d} sepRows=0 iscOrder=d patType=closed dbName={3:s}\n'.format(len(itemsets), minSupport, maxLen, dname))
        
        for itemset, support in sort_itemsets_sup_num.items():
            f.writelines('{0:d}: {1:s} ({2:d})\n'.format(len(itemset), ' '.join(map(str, itemset)), support))
    


def get_candidates_from_log(file_name):
    """Read candidates from Krimp/Slim log files.

    Parameters
    -------
    filename : string.


    Returns
    -------
    itemsets : list of tuples (intent, extent_size),
    where 'intent' is a list of integers,
    'extent_size' is an integer.

    Example
    -------
    INPUT

    Accepted: 0 1 2 4 6 9 51 (20,319) [814762.94, 814794.40, 1.60, 1404]
    Rejected: 0 1 2 3  (0,13) [814772.86, 814762.94, -59.86, 1405]
    Accepted: 8 10 12 18 19 28 (14,14) [814748.53, 814762.94, 0.56, 1406]
    Rejected: 19 21 28 36 (0,15) [814757.97, 814748.53, -23.47, 1407]
    Rejected: 21 28 36 (0,13) [814769.26, 814748.53, -34.76, 1408]

    OUTPUT
    itemsets = [([0, 1, 2, 4, 6, 9], 319), ([0, 1, 2, 3], 13),
    ([8, 10, 12, 18, 19, 28], 14), ([19, 21, 28, 36], 15), ([21, 28, 36], 13)]
    """
    itemsets = []
    with open(file_name, 'r') as f:
        s = f.readline()
        while s != '':
            #print(s)
            intent_raw, extent_raw = s.split('(')
            intent = [int(i) for i in intent_raw.split(':')[1].strip().split(' ')]
            extent_size = int(extent_raw.split(')')[0].split(',')[1])
            itemsets.append((intent,extent_size))
            s = f.readline()
    return itemsets
