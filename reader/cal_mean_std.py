from pathlib import Path
from typing import Union
import numpy as np
import pdb


def calc_norm_param(X):
    """Assumes X to be a list of arrays (of differing sizes)"""
    total_len = 0
    mean_val = np.zeros(X[0].shape[1])
    std_val = np.zeros(X[0].shape[1])
    for obs in X:
        obs_len = obs.shape[0]
        mean_val += np.mean(obs,axis=0)*obs_len
        std_val += np.std(obs, axis=0)*obs_len
        total_len += obs_len
    
    mean_val /= total_len
    std_val /= total_len

    return mean_val, std_val, total_len


X_train = []
# max_cur = [0]*80
# std_list=[]
list_file = '/home/hk/voice_conversion/nonparaSeq2seqVC_text-dependent_SE/reader/vctk_train_100speakers.txt'
with open(list_file) as f:
    # pdb.set_trace()
    lines = f.readlines()
    for line in lines:
        # print("line ", line)
        # path, n_frame, n_phones = line.strip().split()
        # path = line.strip().split()
        path = line.strip()
        # print("path ", path)
        # if int(n_frame) >= 1000:
        #     continue
        mel = np.load(path)
        mel = np.transpose(mel)
        X_train.append(mel)

mean_val, std_val, _ = calc_norm_param(X_train)
np.save('std.npy', std_val)
np.save('mean.npy', mean_val)
