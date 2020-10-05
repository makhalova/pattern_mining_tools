import glob
import numpy as np
import pandas as pd
import sys
sys.path.append('../'*4)
sys.path.append('../')

from pattern_mining_tools.modules.utils.cs_analysers import get_data_topology
from pattern_mining_tools.modules.io.converter_writer import dat_to_binary
from global_param import data_dir_name, data_name_list

n_bins = 10

for data_name in data_name_list:
    dir_name = data_dir_name + data_name + '/transformed/closure_structure/dat/'
    data_file_name = data_dir_name + data_name + '/transformed/' + data_name + '_liv_without_labels.dat'
    X_train = dat_to_binary(data_dir_name + data_name + '/transformed/' + data_name + '_liv_without_labels.dat')
    n_trans = X_train.shape[0]
    df = get_data_topology(dir_name, n_bins, n_trans)
    df.to_csv(dir_name + 'frequency_distribution_by_levels.csv')
