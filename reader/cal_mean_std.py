from pathlib import Path
from typing import Union
import numpy as np
import pdb

## why '\n' is not necessary?
#previous way
# file_path_list = []
# max_cur = [0]*80
# std_list=[]
# list_file = '/home/hk/voice_conversion/nonparaSeq2seqVC_code/pre-train/reader/vctk_train.txt'
# with open(list_file) as f:
#     # pdb.set_trace()
#     lines = f.readlines()
#     for line in lines:
#         # print("line ", line)
#         # path, n_frame, n_phones = line.strip().split()
#         # path = line.strip().split()
#         path = line.strip()
#         # print("path ", path)
#         # if int(n_frame) >= 1000:
#         #     continue
#         mel = np.load(path)
#         # print (mel.shape)
#         # print(mel.shape[1])
#         # if  int(mel.shape[1]) >= 1000:
#         #     continue

#         file_path_list.append(mel)

#         max_val = np.amax(mel, axis = 1)
#         max_val = max_val.tolist()
#         # max_cur = [max_val[i] if (max_val[i]>max_cur[i]) for i in range(len(max_val))]
#         for i in range (len(max_val)):
#             if max_val[i] > max_cur[i]:
#                 max_cur[i] = max_val[i]

#     # pdb.set_trace()
#     # print (len(max_cur))

#     # for line in lines:
#     for i in range(80):
#         # row = mel[i].tolist()
#         list = []
#         std = 0
#         n = 0
#         print(n)
#         for line in lines:
#             path = line.strip()
#             mel = np.load(path)
#             # mel = mel.tolist()
#             # list.append(mel[i])
#             # mel1 = np.concatenate((mel))
#             # std = np.sqrt(np.sum(np.square(mel[i]-max_cur[i]))/mel[i].shape[0])
#             std = std + np.sum(np.square(mel[i]-max_cur[i]))
#             n = n+ mel[i].shape[0]
#             print(n)
#         std = np.sqrt(std/n)
#         std_list.append(std)

#     std_list = np.asarray(std_list)
#     max_cur = np.asarray(max_cur)

#     np.save('std.npy', std_list)
#     np.save('mean.npy', max_cur)

#         # print(list)
#         # print(len(list))
#         # print(len)

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
list_file = '/home/hk/voice_conversion/nonparaSeq2seqVC_code/pre-train/reader/vctk_train.txt'
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
