import numpy as np

# CXT - converter

def write_dataframe_to_cxt(filename, df):

    with open(filename + ".cxt", "w") as fo:
        n_obj = df.shape[0]
        n_attr = df.shape[1]
        fo.writelines("B\n")
        fo.writelines('\n')
        fo.writelines(str(n_obj)+'\n')
        fo.writelines(str(n_attr)+'\n')
        fo.writelines('\n')
        for ind in df.index:
            fo.write(str(ind) + '\n')
        for ind in df.columns:
            fo.write(str(ind)+ '\n')
        
        new_objects = {i : ''.join( ['X' if c == 1  else '.' for c in v ]) for i, v in df.iterrows()}
        for ind in df.index:
            fo.write(new_objects[ind] + '\n')




# DAT - writer
def write_dat(file_name, X):
    """Write ndarray in dat-format in file"""
    
    with open(file_name, 'w') as f:
        for row in X:
            s = ' '.join([str(v) for v in row])
            f.writelines(s + '\n')

# dat to binary array
def dat_to_binary(file_name, return_index = False, return_inverse = False):
    """Covert dat-dataset into binary data table.
    
    
    Returns the binarized dataset. There are two optional
    outputs in addition to the binarized dataset:
    * the dictionary that stories "old indices to new ones"
    * the dictionary that stories "new indices to old ones"
    Parameters
    ----------
    file_name : string
        The file name of the dat-dataset.
    return_index : bool, optional
        If True, also return the dictionary "old index : new index"
    return_inverse : bool, optional
        If True, also return the dictionary "new index : new index"
    Returns
    -------
    unique : ndarray
        The binarized dataset.
    dict_index : dictionary, optional
        The keys are the indices in the original dataset, the values
        are the corresponding indices in the binary dataset.
        Only provided if `return_index` is True.
    unique_inverse : dictionary, optional
        The keys are the indices in the binarized dataset, the values
        are the corresponding indices in the original dataset.
        Only provided if `return_inverse` is True.
    """
    def to_binary(itemset):
        bin_itemset = np.zeros(n, dtype=int)
        for x in itemset:
            bin_itemset[itemset_dictionary[x]] += 1
        assert(sum(bin_itemset > 1) == 0)
        return bin_itemset

    with open(file_name, 'r') as f:
        itemsets = f.readlines()
    itemsets = [set(int(v) for v in itemset.split()) for itemset in itemsets]
    items = set().union(*[itemset for itemset in itemsets])
    n = len(items)
    itemset_dictionary = {item : i for i, item in enumerate(items)}

    res = (np.array(list(map(to_binary, itemsets))), )
    if return_index:
        res += (itemset_dictionary,)
    if return_inverse:
        inverse = {v : k for k, v in itemset_dictionary.items()}
        res += (inverse,)
    if len(res) == 1:
        res = res[0]
    return  res
    
def read_dat_to_binary(file_name):
    #read data
    lines = []
    max_id = -1
    sep = ' '
    with open(file_name, 'r') as f:
        for s in f:
            itemset = [int(v) for v in s.replace('\n', '').split(sep) if len(v) > 0]
            lines.append(itemset)
            if len(itemset) > 0:

                max_id = np.max([np.max(itemset), max_id])


    # chech empty lines
    for i, line in enumerate(lines):
        if len(line) == 0:
            print(i, len(line))
    #transform to binary
    new_data = np.zeros((len(lines),max_id + 1),dtype=int)
    for i, line in enumerate(lines):
        for j in line:
            new_data[i][j] = 1
    return new_data

# ITEMSET IO
def write_itemsets(file_name, itemsets, index_dict = None):
    if (index_dict is None):
        with open(file_name, 'w') as f:
            for item, e in itemsets.items():
                st = " ".join([str(i) for i in item])
                f.writelines(st + " ; " + str(e.sum()) + "\n")
    else:
        with open(file_name, 'w') as f:
            for item, e in itemsets.items():
                st = " ".join([str(index_dict[i]) for i in item])
                f.writelines(st + " ; " + str(e.sum()) + "\n")
    
            
def read_itemsets(file_name, index_dict = None):
    with open(file_name, 'r') as f:
            itemsets = f.readlines()

    itemsets_dict = {}
    if (index_dict is None):
        for itemset in itemsets:
            temp = itemset.split(' ;')
            itemsets_dict[frozenset([int(i) for i in temp[0].split(' ')])] = int(temp[1])
    else:
        for itemset in itemsets:
            temp = itemset.split(' ;')
            itemsets_dict[frozenset([index_dict[int(i)] for i in temp[0].split(' ')])] = int(temp[1])
    return itemsets_dict


def write_selector_writer(intput_file, indices, output_file):
    """
    Write lines with indices 'indices' from file 'intput_file' to file 'output_file'.
    """
    with open(intput_file, 'r') as f:
        itemsets = f.readlines()
    with open(output_file, 'w') as f:
        for i in indices:
            f.writelines(itemsets[i])


def read_line_csv(filename):
    with open(filename, "r") as f:
        s = f.readline()

    return np.array([int(v) for v in s.split(';')])
