import numpy as np

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
