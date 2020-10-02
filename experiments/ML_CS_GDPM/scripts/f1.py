import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append('../'*4)
sys.path.append('../')

from global_param import data_name_list, true_numb_attributes
from pattern_mining_tools.modules.io.converter_writer import dat_to_binary, read_itemsets
from pattern_mining_tools.modules.analysers.is_supervised_analyser import get_performance
from pattern_mining_tools.modules.binary.n_closed import _get_extent

def get_interval(x):
    if x < 10:
        return '(0.{0},0.{1}]'.format(x-1, x)
    else:
        return '(0.9,1.0]'

dat_suf = '_liv_without_labels.dat'
target_suf = '_liv_only_labels.dat'
data_dir_name = '../'*5 + 'datasets/'
output_dir = '../results/'

datasets = {}

for data_name in ['auto', 'ecoli', 'heart-disease', 'hepatitis', 'iris', 'led7', 'pima', 'tic_tac_toe']:

    data_file_name = data_dir_name + data_name + '/transformed/' + data_name + dat_suf
    target_file_name = data_dir_name + data_name + '/transformed/' + data_name + target_suf
    
    # reading the whole dataset
    X, indices, inverse_indices = dat_to_binary(data_file_name, return_index = True, return_inverse = True)
    y = pd.read_csv(target_file_name, header = None, sep=' ').values.T[0]
    if np.all(np.isnan(X[:, X.shape[1] - 1])):
        X = np.delete(X, X.shape[1] - 1, 1)
    X = X.astype(int)
    
    dir_name = data_dir_name + data_name + '/transformed/closure_structure/dat/'
    file_list_glob = glob.glob(dir_name + '*')
    file_list = [s.split('/')[-1] for s in file_list_glob]
    file_level_list = sorted([int(v) for v in file_list if v.isdigit()])
    
    print(file_level_list)

    if os.path.exists(dir_name + 'summary_F.csv'):
        results = []
        for level_numb in file_level_list:
            print(level_numb)
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
        df_res["fr"] = (df_res.tp + df_res.fp) / (df_res.tp + df_res.fp + df_res.tn + df_res.fn)
        df_res["fr_discrete"] = df_res["fr"].apply(lambda x: int(np.floor(x * 10)) + 1)
        datasets[data_name] = df_res

res = []
res_count = []
for name, data in datasets.items():
    data['name'] = name
    data['precision'] = data.tp/(data.tp + data.fp)
    data['recall'] = data.tp/(data.tp + data.fn)
    data['accuracy'] = (data.tp + data.tn)/(data.tp + data.fn + data.tn + data.fp)
    data['f1'] = (2 * data.tp)/(2 * data.tp + data.tn + data.fp)
    res.append(data)
    df_count = data[['fr_discrete','level', 'f1']].groupby(['fr_discrete','level'], as_index = False).count()
    df_count['name'] = name
    res_count.append(df_count)

    
df_total = pd.concat(res)
df_total_count = pd.concat(res_count)
df_total_count.rename(columns = {'f1':'number'}, inplace=True)

df_total['fr. range'] = df_total.fr_discrete.apply(get_interval)
df_total_count['fr. range'] = df_total_count.fr_discrete.apply(get_interval)

df_total.to_csv(output_dir + 'f1_by_levels_summary.csv')
df_total_count.to_csv(output_dir + 'f1_by_levels_count.csv')
