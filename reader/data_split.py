from pathlib import Path
from typing import Union
import pdb

## why '\n' is not necessary?

def get_files(path: Union[str, Path], extension='.npy'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

wav = get_files('VCTK/wav48',extension='.npy')
last_letter = set()

with open('vctk_train_100speakers.txt',"w") as f:
    for i in range(len(wav)):
        # pdb.set_trace()
        speaker = str(wav[i]).split('/')[-2]
        idx = str(wav[i]).split('/')[-1].split('_')[-1].split('.')[0]

        if speaker == 'p225' or speaker == '315' or  speaker == 'p340' or  speaker == 'p363' or  speaker == 'p239' or  speaker == 'p240' or  speaker == 'p270' or  speaker == 'p271' or  speaker == 'p294':
            continue
        if int(idx) >= 390:
            continue
        # f.write(str(wav[i])+' ')
        f.write(str(wav[i]))
#         txt_add = str(wav[i]).replace('wav48','txt').replace('.npy','.phones')
#         txt_f = open(txt_add,"r")
#         txt = txt_f.readline()
#         # txt = txt + '*'
#         if txt[-1] == '\n':
# #            if txt[-2] != '.' and txt[-2] != '?' and txt[-2] != '!':
#             if txt[-2] == ')':# txt[-2] != '?' and txt[-2] != '!':
#                 last_letter.add(txt[-2])
# #                print(txt[-2])
#                 print(str(wav[i]).split('/')[-1])
# #        elif txt[-1] != '.':
# #            last_letter.add(txt[-1])
# #            print(str(wav[i]).split('/')[-1])

#         f.write(txt)
        f.write('\n')   ##
        # txt_f.close()
    
with open('vctk_val_100speakers.txt',"w") as f:
    for i in range(len(wav)):
        speaker = str(wav[i]).split('/')[-2]
        idx = str(wav[i]).split('/')[-1].split('_')[-1].split('.')[0]

#        if not speaker == 'p225' and not speaker == 'p226' and not speaker == 'p227':
#            continue
        if speaker == 'p225' or speaker == '315' or  speaker == 'p340' or  speaker == 'p363' or  speaker == 'p239' or  speaker == 'p240' or  speaker == 'p270' or  speaker == 'p271' or  speaker == 'p294':
            continue
        if int(idx) < 390:
            continue
        f.write(str(wav[i]))
#         txt_add = str(wav[i]).replace('wav48','txt').replace('.npy','.phones')
#         txt_f = open(txt_add,"r")
#         txt = txt_f.readline()
# #        if txt[-1] == '\n':
# #            if txt[-2] != '.':
# #                last_letter.add(txt[-2])
# #                print(str(wav[i]).split('/')[-1])
# #        elif txt[-1] ==
#         f.write(txt)
        f.write('\n')
        # txt_f.close()

print(last_letter)
