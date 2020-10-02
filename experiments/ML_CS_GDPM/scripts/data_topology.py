import glob
import numpy as np
import pandas as pd
import sys
sys.path.append('../'*4)
sys.path.append('../')

from pattern_mining_tools.modules.io.converter_writer import read_itemsets
from pattern_mining_tools.modules.io.converter_writer import dat_to_binary
from global_param import data_dir_name, data_name_list

from sklearn.preprocessing import KBinsDiscretizer

n_bins = 10
data_dir_name = '../'*5 + 'datasets/'

frequency_param = sorted([ i/n_bins for i in range(1, n_bins)] + [0.,1.])
array = np.array(frequency_param + [0.,1.]).reshape(-1, 1)
kb_disc = KBinsDiscretizer(n_bins, encode='onehot', strategy='uniform')
kb_disc.fit(array)


for data_name in data_name_list:
    dir_name = data_dir_name + data_name + '/transformed/closure_structure/dat/'
    file_list_glob = glob.glob(dir_name + '*')
    file_list = [s.split('/')[-1] for s in file_list_glob]
    file_level_list = sorted([int(v) for v in file_list if v.isdigit() and v !='0'])
    
    data_file_name = data_dir_name + data_name + '/transformed/' + data_name + '_liv_without_labels.dat'
    X_train, _ = dat_to_binary(data_file_name, return_index = True)
    n_trans = X_train.shape[0]

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
    df.to_csv(dir_name + 'frequency_distribution_by_levels.csv')
