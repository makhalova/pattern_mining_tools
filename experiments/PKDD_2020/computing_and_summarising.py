from parameters import name_dict, max_cmp_dict, main_data_dir, data_list_without_labels

import numpy as np
import pickle
import pandas as pd

from glob import glob
import re
import logging
import os
import sys
sys.path.append('../../')

logging.basicConfig(level=logging.DEBUG)

from modules.io.dbconverters import db2bindata
from modules.binary.n_closed import compute_closed_by_level
from modules.analysers.is_supervised_analyser import get_average_performance_fast


# --------------------------------------------------
# initialisation
# --------------------------------------------------
# CHANGE ITEMSET DIR
itemsets_data_dir = '../CLA_2020/results/'
itemset_set_list = glob(itemsets_data_dir+ 'closed_itemsets/*')

start_data_name = len(itemsets_data_dir+ 'closed_itemsets/')
data_name_ads1 = '_without_labels'
data_name_ads2 = 'closed-1d.isc'

def get_level(file_name, f):
    return int(f[len(file_name) + 1 : ])
    
    
logging.debug('Compute closed itemsets by levels')
# --------------------------------------------------
# Computing closed itemsets level-by-level
# --------------------------------------------------
for data in data_list_without_labels:
    data_name = data['data_name']
    dname = data['dname']
    data_dir = main_data_dir + data_name + '/transformed/'
    dataset = db2bindata(data_dir, dname + '_without_labels').astype(int)
    res = compute_closed_by_level(dataset, max_level=10, dataset_output_folder = './results/closed_itemsets/' + data_name )
    


logging.debug('Convert itemsets from pickle-file to Krimp readable')
# --------------------------------------------------
# Converting itemsets to Krimp readable format ".isc"
# --------------------------------------------------
for file_name in itemset_set_list:
   if os.path.exists(file_name):
       itemsets = {}
       levels = np.sort([get_level(file_name, f) for f in glob(file_name + '/*') if re.match(r''+file_name + '/\d', f) ])
       #levels = [f for f in glob(file_name + '/*') if re.match(r''+file_name + '/\d', f) ]
       data_name = file_name[start_data_name:]
       output_dir = main_data_dir +  data_name + '/transformed/candidates/'
       start_level_num = len(file_name) + 1
       for num_level in levels:
           f_level = file_name + '/'+ str(num_level)
           with open(f_level, 'rb') as f:
               itemsets_new = pickle.load(f)
               itemsets.update(itemsets_new)
               output_file_name = output_dir + name_dict[data_name] + data_name_ads1 + '_' + f_level[start_level_num : ] + '-'+ data_name_ads2
               print(output_file_name, len(itemsets))
               #write_isc(output_file_name, itemsets, name_dict[data_name] + data_name_ads1)
               print(output_file_name)
               

logging.debug('Computing the dataset descriptions')
# --------------------------------------------------
# computing dataset descriptions
# --------------------------------------------------
result = []
for file_name in sorted(itemset_set_list):
    if os.path.exists(file_name):
        #print(file_name)
        data_name = file_name[start_data_name:]
        output_dir = main_data_dir +  data_name + '/transformed/'
        X_data = db2bindata(output_dir, name_dict[data_name] + data_name_ads1 )
        y = pd.read_csv(output_dir + name_dict[data_name] + '_only_labels.dat', header=None)
        result.append({'data_name' : data_name, 'n_objects' : X_data.shape[0],\
            'n_attributes' : X_data.shape[1], 'n_classes' : len(np.unique(y)),\
            'den' : X_data.mean(), 'max_level' : max_cmp_dict[data_name]})
df = pd.DataFrame(result)
df.to_csv('./results/data_summary.csv', float_format='%.2f')


logging.debug('Analysing the results of compression')
# --------------------------------------------------
# getting the results of compression
# --------------------------------------------------
# by levels
result = []
for file_name in itemset_set_list:
    if os.path.exists(file_name):
        itemsets = {}
        levels = np.sort([get_level(file_name, f) for f in glob(file_name + '/*') if re.match(r''+file_name + '/\d', f) ])
        fname = file_name[start_data_name:] # data_name
        dname = name_dict[fname]
        start_level_num = len(file_name) + 1
        
        for num_level in levels:
            f_level = file_name + '/'+ str(num_level)
            with open(f_level, 'rb') as f:   # n_patterns
                itemsets_new = pickle.load(f)
                itemsets.update(itemsets_new)
                n_patterns = len(itemsets)
            file_list = glob(main_data_dir + fname + '/transformed/compress/' + \
                            dname + '_without_labels_' + str(num_level) + '*/report*')
            if len(file_list) > 0:          # CR
                report_file_name = file_list[0]
                totalSize = pd.read_csv(report_file_name, sep = ';').totalSize
                cr = totalSize.min() / totalSize.max()
            else:
                cr = 1
            result.append({'data_name' : fname, 'CR' : cr, 'type' : 'level', \
                           'threshold' : num_level, 'n_patterns' : n_patterns})
            
# by frequency
vals = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
#result = []
for file_name in sorted(itemset_set_list):
    if os.path.exists(file_name):
        fname = file_name[start_data_name:]
        dname = name_dict[fname]
        output_dir = main_data_dir +  fname + '/transformed/'
        X_data = db2bindata(output_dir,dname  + data_name_ads1 )
        
        supports = np.ceil(vals * X_data.shape[0]).astype(int)
        support_dict = {ptg: int(np.ceil(i)) for i, ptg in zip(supports,[90, 80, 70, 60, 50, 40, 30, 20, 10])}
        support_dict[0] = 1
        for ptg, values in support_dict.items():
            if values > 1:
                st = main_data_dir + fname + '/transformed/candidates/' + dname +\
                                            '_without_labels-closed-' + str(values) + 'd*.isc'
            else:
                st = main_data_dir + fname + '_without_labels_krimp-closed-1d*.isc'
            
            file_list = glob(st)
            if len(file_list) > 0:
                itemsets_file_name = file_list[0]
                with open(itemsets_file_name, 'r') as f:
                    f.readline()
                    n_patterns = f.readline().split(' ')[1].split('=')[1]
            else:
                n_patterns = 0
            # CR
            if values > 1:
                st = main_data_dir + fname + '/transformed/compress/' + dname +\
                    '_without_labels-closed-' + str(values) +  'd*/report*'
            else:
                st = main_data_dir + fname + '/transformed/compress/' + dname +\
                    '_without_labels_krimp-closed-1d*/report*'
            file_list = glob(st)
            if len(file_list) > 0 :
                report_file_name = file_list[0]
                totalSize = pd.read_csv(report_file_name, sep = ';').totalSize
                cr = totalSize.min() / totalSize.max()
            else:
                cr = 1
            result.append({'data_name' : fname, 'CR' : cr, 'type' : 'frequency', \
                'threshold' : ptg, 'n_patterns' : n_patterns, 'threshold_val' : values})
df = pd.DataFrame(result)
df.to_csv('./results/gradual_characteristics.csv')


logging.debug('Supervised settings')
# --------------------------------------------------
# supervised settings
# --------------------------------------------------
result = []
for file_name in sorted(itemset_set_list):
    if os.path.exists(file_name):
        
        data_name = file_name[start_data_name:]
        output_dir = main_data_dir +  data_name + '/transformed/'
        X_data = db2bindata(output_dir, name_dict[data_name] + data_name_ads1 )
        y = pd.read_csv(output_dir + name_dict[data_name] + '_only_labels.dat', header=None).values
        
        itemsets_with_members_cum = {}
        pattern_labels_cum = {}
        levels = np.sort([get_level(file_name, f) for f in glob(file_name + '/*') if re.match(r''+file_name + '/\d', f) ])
        data_name = file_name[start_data_name:]
        output_dir = main_data_dir +  data_name + '/transformed/candidates/'
        start_level_num = len(file_name) + 1
        for num_level in levels:
            f_level = file_name + '/'+ str(num_level)
            with open(f_level, 'rb') as f:
                itemsets = pickle.load(f)
                
                # itemsets of one level
                itemsets_with_members = {k: np.where(support)[0] for k, support in itemsets.items()}
                dmeans,  pattern_labels = get_average_performance_fast(itemsets_with_members, X_data, y, True)
                classified = {v : np.zeros(X_data.shape[0]) for v in np.unique(y) }
                for itemset, members in itemsets_with_members.items():
                    classified[pattern_labels[itemset]][members] += 1
                maj_vote = pd.DataFrame(classified).idxmax(axis=1)
                result.append({'data_name' : data_name, 'num_level' : num_level,\
                    'accuracy' : (maj_vote == y.T[0]).mean(), 'type' : 'single'})
                
                # commulative itemsets
                itemsets_with_members_cum.update(itemsets_with_members)
                pattern_labels_cum.update(pattern_labels)
                classified = {v : np.zeros(X_data.shape[0]) for v in np.unique(y) }
                for itemset, members in itemsets_with_members_cum.items():
                    classified[pattern_labels_cum[itemset]][members] += 1
                maj_vote_cum = pd.DataFrame(classified).idxmax(axis=1)
                result.append({'data_name' : data_name, 'num_level' : num_level,\
                    'accuracy' : (maj_vote_cum == y.T[0]).mean(), 'type' : 'commulative'})

df = pd.DataFrame(result)
df.to_csv('./results/precision.csv')
