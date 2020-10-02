# dataset characteristics
import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append('../'*4)
sys.path.append('../')

from global_param import data_name_list, true_numb_attributes
from pattern_mining_tools.modules.io.converter_writer import dat_to_binary

dat_suf = '_liv_without_labels.dat'
target_duf = '_liv_only_labels.dat'
data_dir_name = '../'*5 + 'datasets/'
output_dir = '../results/'


# compute data description
list_data_name_description = []
for data_name in data_name_list:
    data_file_name = data_dir_name + data_name + '/transformed/' + data_name + dat_suf
    target_file_name = data_dir_name + data_name + '/transformed/' + data_name + '_liv_only_labels.dat'

    # reading the whole dataset
    X, indices, inverse_indices = dat_to_binary(data_file_name, return_index = True, return_inverse = True)
    y = pd.read_csv(target_file_name, header = None, sep=' ').values.T[0]
    if np.all(np.isnan(X[:, X.shape[1] - 1])):
        X = np.delete(X, X.shape[1] - 1, 1)
    X = X.astype(int)

    dir_name = data_dir_name + data_name + '/transformed/closure_structure/dat/'
    file_list_glob = glob.glob(dir_name + '*')
    file_list = [s.split('/')[-1] for s in file_list_glob]
    file_level_list = [int(v) for v in file_list if v.isdigit()]
    file_summary_list = [v for v in file_list if not v.isdigit()]

    d = {'name': data_name, '|G|': X.shape[0], '|M|': X.shape[1], "density": np.round(X.mean(),2), \
         "#classes": len(np.unique(y)), "#attributes" : true_numb_attributes[data_name]}
    if os.path.exists(dir_name + 'summary_E.csv'):
        n_itemsets = np.sum(pd.read_csv(dir_name + 'summary_E.csv', sep = ';', header = None, index_col = 0).T.n_itemsets)
        d.update({'#levels': int(max(file_level_list)), '#concepts' : str(n_itemsets)})
    list_data_name_description.append(d)

df_data_description = pd.DataFrame(list_data_name_description)
df_data_description.to_csv(output_dir + 'data_description__.csv')



# summary on GDPM-INT and GDPM-EXT (summary_ME and summary_F, respectively)
#
list_ME_F = []
for data_name in data_name_list:
    dir_name = data_dir_name + data_name + '/transformed/closure_structure/dat/'
    file_list_glob = glob.glob(dir_name + '*')
    file_list = [s.split('/')[-1] for s in file_list_glob]
    file_level_list = [int(v) for v in file_list if v.isdigit()]
    file_summary_list = [v for v in file_list if not v.isdigit()]
    if len(file_summary_list) > 1:
        df_ME = pd.read_csv(dir_name + 'summary_ME.csv', sep = ';', header = None, index_col = 0).T
        df_F = pd.read_csv(dir_name + 'summary_F.csv', sep = ';', header = None, index_col = 0).T
        assert(df_ME.n_itemsets.sum() == df_F.n_itemsets.sum())
        list_ME_F.append([data_name, df_ME.n_itemsets.sum(), df_ME.time.sum(), df_F.time.sum(),df_ME.time.sum()/df_F.time.sum(), df_ME.n_nodes.max(), df_F.n_nodes.max(), df_ME.n_nodes.max()/ df_F.n_nodes.max()])
df_ME_F = pd.DataFrame(list_ME_F, columns=['name', 'n_itemsets', 'ME time', 'F time', 'ME/F', 'ME nodes', 'F nodes', 'ME/F nodes'])
df_ME_F.to_csv(output_dir + 'ME_F.csv', float_format="%.2f")
