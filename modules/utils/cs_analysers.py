"""
   Data topology
   _____________

   F1 evaluation
   _____________
   
   Coverage and overlaps

"""


import numpy as np
import pandas as pd
import glob

def get_data_topology(dir_name, n_bins, n_trans):
    """Computing data topology in the csv format.

    dir_name : string, directory name containing files with itemsets corresponding to
            closure levels
    n_trans : int, the number of transations
    """

    from sklearn.preprocessing import KBinsDiscretizer
    from ..io.converter_writer import read_itemsets

    frequency_param = sorted([ i/n_bins for i in range(1, n_bins)] + [0.,1.])
    array = np.array(frequency_param + [0.,1.]).reshape(-1, 1)
    kb_disc = KBinsDiscretizer(n_bins, encode='onehot', strategy='uniform')
    kb_disc.fit(array)

    file_list_glob = glob.glob(dir_name + '*')
    file_list = [s.split('/')[-1] for s in file_list_glob]
    file_level_list = sorted([int(v) for v in file_list if v.isdigit() and v !='0'])

    results = []
    for level_numb in file_level_list:
        file_name = dir_name + str(level_numb)
        itemsets_binary = read_itemsets(file_name)
        if len(itemsets_binary) > 0:
            array = np.array([np.sum(l)  for l in itemsets_binary.values()]).reshape(-1, 1) / n_trans # list of frequencies
            discretized = kb_disc.transform(array).toarray() # put frequencies into the bins
            by_frequency = discretized.sum(axis = 0) # count the number of itemsets
            total_number = by_frequency.sum()
            relative = by_frequency/total_number # the ratio of itemsets of particular frequency at a level
            dct = {j + 1: v for j, v in enumerate(relative)}
            dct.update({'k' : level_numb, 'total' : total_number})

        results.append(dct)
    df = pd.DataFrame(results)
    s = df.total.sum()
    df.rename({i + 1 : '(' + str(frequency_param[i])  + ',' + str(frequency_param[i + 1]) + ']'  for i in range(len(frequency_param[:-1]))}, axis = 1, inplace = True)#
    df['labels'] = df.apply(lambda r : '{1:>.2f}% ({0:.0f})'.format(int(r.k),  np.round(r.total/s * 100, decimals=1)), axis = 1)

    return df


def draw_topology(topology_data_file, data_name = "", output_file = None, font_size = 17):
    """ Drawing topology of a dataset.
        The data topology is computed using 'get_data_topology'.
        
    topology_data : string, name of the csv data topology file
    output_file : string, name of the picture with the data topology
    font_size : int, font size of the labels

    """
    import matplotlib.pyplot as plt
    df = pd.read_csv(topology_data_file, index_col=0)
    df_new = df.drop(['total', 'k', 'labels'], axis = 1)
    f, axes = plt.subplots(1, 1, figsize = (3.5, 2.8))

    ax = df_new.plot.barh(stacked=True, edgecolor='none', width=0.85, ax = axes, fontsize = font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size )
    ax.set_yticklabels(df.labels, fontsize = font_size)

    plt.title(data_name, fontsize = font_size)
    plt.tight_layout()
    plt.legend(prop = {'size' : font_size}, framealpha=0.6, ncol=5)#, bbox_to_anchor=(1., 1.05))
    ax.get_legend().remove()

    plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    if not(output_file is None):
        plt.savefig(output_file, dpi=300, transparent=True)
    return f

# F1 mesure

def get_f1_measure(X, y, dir_name, indices = None, output_file = None):
    """ Compute F1 measure for itemsets from the closure structure
        stored in the 'dir_name' directory.
        
        X : bool array, dataset
        y : int array, target labels
        dir_name : string, directory where the itemsets are stored
        indices : dictionary to transfrom the original indeces to
            indices of dataset X
            
        output_file : string
            
    """
    from ..io.converter_writer import read_itemsets
    from ..analysers.is_supervised_analyser import get_performance
    from ..binary.n_closed import _get_extent

    def get_interval(x):
        if x < 10:
            return '(0.{0},0.{1}]'.format(x-1, x)
        else:
            return '(0.9,1.0]'


    file_list_glob = glob.glob(dir_name + '*')
    file_list = [s.split('/')[-1] for s in file_list_glob]
    file_level_list = sorted([int(v) for v in file_list if v.isdigit()])
    
    results = []
    for level_numb in file_level_list:
        if level_numb == 0:
            continue
        file_name = dir_name + str(level_numb)
        itemsets_binary = read_itemsets(file_name, indices)
        itemsets_with_support = _get_extent(itemsets_binary, X)
        member_dict = {k: np.where(l)[0] for k, l in itemsets_with_support.items() if np.sum(l) < X.shape[0]}
        df = get_performance(member_dict, X, y)
        df["level"] = level_numb
        results.append(df)

    df_res = pd.concat(results)
    df_res['fr'] = (df_res.tp + df_res.fp) / (df_res.tp + df_res.fp + df_res.tn + df_res.fn)
    df_res['fr_discrete'] = df_res['fr'].apply(lambda x: int(np.floor(x * 10)) + 1)
    df_res['precision'] = df_res.tp/(df_res.tp + df_res.fp)
    df_res['recall'] = df_res.tp/(df_res.tp + df_res.fn)
    df_res['accuracy'] = (df_res.tp + df_res.tn)/(df_res.tp + df_res.fn + df_res.tn + df_res.fp)
    df_res['f1'] = (2 * df_res.tp)/(2 * df_res.tp + df_res.tn + df_res.fp)
    df_res['fr. range'] = df_res.fr_discrete.apply(get_interval)
    if not (output_file is None):
        df_res.to_csv(output_file)
    return df_res
    
    
def get_summary_f1(data_tuple_list, output_file_f1 = None, output_file_count = None):
    """ Compute F1 measure for a list of datasets. Returns a dataframe containing
        quality values of all itemsets and a dataframe with the sizes of all levels
        

        data_tuple_list : list of tuples ('dataset name', 'X path', 'y path',
            'closure structure path')
            
        output_file_f1 : string
        
        output_file_count : string
            
    """
    from ..io.converter_writer import dat_to_binary

    res = []
    res_count = []
    for (data_name, X_file, y_file, levels_dir_name) in data_tuple_list:
        X, indices, inverse_indices = dat_to_binary(X_file, return_index = True, return_inverse = True)
        y = pd.read_csv(y_file, header = None, sep=' ').values.T[0]
        if np.all(np.isnan(X[:, X.shape[1] - 1])):
            X = np.delete(X, X.shape[1] - 1, 1)

        df = get_f1_measure(X, y, levels_dir_name, indices = indices)
        df['name'] = data_name
        res.append(df)
        df_count = df[['fr_discrete','level', 'f1']].groupby(['fr_discrete','level'], as_index = False).count()
        df_count['name'] = data_name
        res_count.append(df_count)

    df_total = pd.concat(res)
    df_total_count = pd.concat(res_count)
    df_total_count.rename(columns = {'f1':'number'}, inplace=True)

    if not(output_file_f1 is None):
        df_total.to_csv(output_file_f1)
    if not(output_file_count is None):
        df_total_count.to_csv(output_file_count)
    return df_total, df_total_count
    
    
    
def get_coverage_overlap(X, y, dir_name, indices = None, output_file = None):
    """ Compute coverage and overlapping ratio?
        
        X : bool array, dataset
        y : int array, target labels
        dir_name : string, directory where the itemsets are stored
        indices : dictionary to transfrom the original indeces to
            indices of dataset X
            
        output_file : string
            
    """
    from ..io.converter_writer import read_itemsets
    from ..analysers.is_supervised_analyser import get_performance
    from ..binary.n_closed import _get_extent
    
    coverage_dict_mean = {}
    overlap_dict_mean = {}
    overlap_dict_std = {}
    
    file_list_glob = glob.glob(dir_name + '*')
    file_list = [s.split('/')[-1] for s in file_list_glob]
    file_level_list = sorted([int(v) for v in file_list if v.isdigit()])
    
    results = []
    for level_numb in file_level_list:
        if level_numb == 0:
            continue
        file_name = dir_name + str(level_numb)
        itemsets_binary = read_itemsets(file_name, indices)
        itemsets_with_support = _get_extent(itemsets_binary, X)
        
        covered_data = np.zeros_like(X)
        for itemset, extent in itemsets_with_support.items():
            for i in itemset:
                covered_data[:, i] += extent

        vals = covered_data[covered_data > 0]
        coverage_dict_mean[level_numb] = np.mean(covered_data[X == 1] > 0)
        overlap_dict_mean[level_numb] = np.mean(vals)
        overlap_dict_std[level_numb] = np.std(vals)
    return coverage_dict_mean, overlap_dict_mean, overlap_dict_std

