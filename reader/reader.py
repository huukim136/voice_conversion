import torch
import torch.utils.data
import random
import numpy as np
from reader.symbols import ph2id, sp2id, seen_speakers
from torch.utils.data import DataLoader

def read_text(fn):
    '''
    read phone alignments from file of the format:
    start end phone
    '''
    text = []
    with open(fn) as f:
        lines = f.readlines()
        for line in lines:
            # print("line ", line)
            # print("line_strip_split ", line.strip().split())
            phone = line.strip().split()
            # phone = phone + 'SOS/EOS'
            # text.append(phone)
    return phone

class TextMelIDLoader(torch.utils.data.Dataset):
    
    def __init__(self, list_file, mean_file, std_file, shuffle=True):
    # def __init__(self, list_file, shuffle=True):

        '''
        list_file: 3-column: (path, n_frames, n_phones)
        mean_std_file: tensor loadable into numpy, of shape (2, feat_dims), i.e. [mean_row, std_row]
        '''
        file_path_list = []
        with open(list_file) as f:
            lines = f.readlines()
            for line in lines:
                #import pdb
                #pdb.set_trace()
                # print("line ", line)
                # path, n_frame, n_phones = line.strip().split()
                # path = line.strip().split()
                path = line.strip()
                # print("path ", path)
                # if int(n_frame) >= 1000:
                #     continue
                mel = np.load(path)
                # print (mel.shape)
                # print(mel.shape[1])
                if  int(mel.shape[1]) >= 1000:
                    continue

                file_path_list.append(path)

        if shuffle:
            random.seed(1234)
            random.shuffle(file_path_list)
        
        self.file_path_list = file_path_list
        mel_mean = np.float32(np.load(mean_file))
        self.mel_mean = mel_mean[:, None]
        mel_std = np.float32(np.load(std_file))
        self.mel_std = mel_std[:,None]
        # self.mel_mean_std = np.float32(np.load(mean_std_file))
        # self.spc_mean_std = np.float32(np.load(mean_std_file.replace('mel', 'spec')))

    def get_text_mel_id_pair(self, path):
        '''
        You should Modify this function to read your own data.

        Returns:

        object: dimensionality
        -----------------------
        text_input: [len_text]
        mel: [mel_bin, len_mel]
        mel: [spc_bin, len_spc]
        speaker_id: [1]
        '''
        #import pdb
        #pdb.set_trace()
        # Deduce filenames
        text_path = path.replace('wav48', 'txt').replace('npy', 'phones')
        mel_path = path
        speaker_id = path.split('/')[-2]
        # print("dfdfd",path.split('/') )
        # print("speaker_id", speaker_id)
        speaker_id = speaker_id[0:4]

        # Load data from disk
        text_input = self.get_text(text_path)
        # print("text_input ", text_input)
        mel = np.load(mel_path)
        # print("type", type(mel))
        # print("shape", mel.shape)
        # mel = np.transpose(mel)
        # spc = np.load(path)
        # Normalize audio 
        mel = (mel - self.mel_mean)/ self.mel_std
        # spc = (spc - self.spc_mean_std[0]) / self.spc_mean_std[1]
        # Format for pytorch
        text_input = torch.LongTensor(text_input)
        mel = torch.from_numpy(mel)
        # print("mel shape", mel.shape)
        # spc = torch.from_numpy(np.transpose(spc))
        speaker_id = torch.LongTensor([sp2id[speaker_id]])
        # print("len(speakre_list", len(seen_speakers))

        # return (text_input, mel, spc, speaker_id)
        return (text_input, mel, speaker_id)
        
    def get_text(self,text_path):
        '''
        Returns:

        text_input: a list of phoneme IDs corresponding 
        to the transcript of one utterance
        '''
        text = read_text(text_path)
        text_input = []
        # print("text" , text)

        for ph in text:
            # print("ph ", ph)
            text_input.append(ph2id[ph])
        text_input.append(41)
        
        return text_input

    def __getitem__(self, index):
        return self.get_text_mel_id_pair(self.file_path_list[index])

    def __len__(self):
        return len(self.file_path_list)


class TextMelIDCollate():

    def __init__(self, n_frames_per_step=2):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        '''
        batch is list of (text_input, text_targets, mel, speaker_id)
        '''
        text_lengths = torch.IntTensor([len(x[0]) for x in batch])
        # print("text_lengths", text_lengths)
        mel_lengths = torch.IntTensor([x[1].size(1) for x in batch])
        # print("mel_lengths", mel_lengths)
        mel_bin = batch[0][1].size(0)
        # print("mel_bin", mel_bin )
        # spc_bin = batch[0][3].size(0)

        max_text_len = torch.max(text_lengths).item()
        max_mel_len = torch.max(mel_lengths).item()
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
            assert max_mel_len % self.n_frames_per_step == 0

        text_input_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), mel_bin, max_mel_len)
        # spc_padded = torch.FloatTensor(len(batch), spc_bin, max_mel_len)

        speaker_id = torch.LongTensor(len(batch))
        gate_padded = torch.FloatTensor(len(batch), max_mel_len)

        text_input_padded.zero_()
        mel_padded.zero_()
        # spc_padded.zero_()
        speaker_id.zero_()
        gate_padded.zero_()

        for i in range(len(batch)):
            text =  batch[i][0]
            mel = batch[i][1]
            spc = batch[i][2]

            text_input_padded[i,:text.size(0)] = text 
            mel_padded[i,  :, :mel.size(1)] = mel
            # spc_padded[i,  :, :spc.size(1)] = spc
            speaker_id[i] = batch[i][2][0]
            # make sure the downsampled gate_padded have the last eng flag 1. 
            gate_padded[i, mel.size(1)-self.n_frames_per_step:] = 1

        # return text_input_padded, mel_padded, spc_padded, speaker_id, \
        #             text_lengths, mel_lengths, gate_padded

        return text_input_padded, mel_padded,  speaker_id, \
                    text_lengths, mel_lengths, gate_padded
    
