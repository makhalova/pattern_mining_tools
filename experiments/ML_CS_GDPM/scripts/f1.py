import pandas as pd
import numpy as np
import sys
sys.path.append('../'*4)
sys.path.append('../')


from global_param import data_name_list
from pattern_mining_tools.modules.io.converter_writer import dat_to_binary
from pattern_mining_tools.modules.utils.cs_analysers import get_f1_measure, get_summary_f1

data_dir_name = '../'*5 + 'datasets/'
data_list = ['auto', 'ecoli', 'heart-disease', 'hepatitis', 'iris', 'led7', 'pima', 'tic_tac_toe']

data_name_list = ['auto', 'breast', 'car_evaluation', 'dermatology', 'ecoli',
                  'glass', 'heart-disease', 'iris', 'led7', 'page_blocks',
                  'pima', 'tic_tac_toe', 'wine', 'zoo']

#for data_name in ['auto', 'ecoli', 'heart-disease', 'hepatitis', 'iris', 'led7', 'pima', 'tic_tac_toe']:
data_tuple_list = [(data_name, data_dir_name + data_name + '/transformed/' + data_name + '_liv_without_labels.dat', data_dir_name + data_name + '/transformed/' + data_name + '_liv_only_labels.dat', data_dir_name + data_name + '/transformed/closure_structure/dat/') for data_name in data_name_list]
df_total, df_total_count = get_summary_f1(data_tuple_list, output_file_f1 = '../results/f1_by_levels_summary.csv',  output_file_count = '../results/f1_by_levels_count.csv')
