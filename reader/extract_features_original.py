import librosa
import numpy as np
import glob
import os
from multiprocessing import Pool, cpu_count
import sys

min_level_db = -100
hop_length = 275
win_length = 1100
n_fft = 2048
# sample_rate = 22050
num_mels = 80
fmin = 40

# def normalize(S):
#     return np.clip((S - min_level_db) / -min_level_db, 0, 1)

# def amp_to_db(x):
#     return 20 * np.log10(np.maximum(1e-5, x))

# def stft(y):
#     return librosa.stft(
#         y=y,
#         n_fft=n_fft, hop_length=hop_length, win_length=win_length)

# def linear_to_mel(spectrogram):
#     return librosa.feature.melspectrogram(
#         S=spectrogram, sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin)

    
def extract_mel_spec(filename):
    '''
    extract and save both log-linear and log-Mel spectrograms.
    '''
    # print (filename)
    try:
        y, sample_rate = librosa.load(filename, sr=16000)

        spec = librosa.core.stft(y=y, 
                                 n_fft=2048, 
                                 hop_length=200, 
                                 win_length=800,
                                 window='hann',
                                 center=True,
                                 pad_mode='reflect')
        spec= librosa.magphase(spec)[0]
        log_spectrogram = np.log(spec).astype(np.float32)

     

        mel_spectrogram = librosa.feature.melspectrogram(S=spec, 
                                                         sr=sample_rate, 
                                                         n_mels=80,
                                                         power=1.0, #actually not used given "S=spec"
                                                         fmin=0.0,
                                                         fmax=None,
                                                         htk=False,
                                                         norm=1
                                                         )
        log_mel_spectrogram = np.log(mel_spectrogram).astype(np.float32)


        np.save(file=filename.replace(".wav", ".npy"), arr=log_mel_spectrogram)
    except Exception:
        print(filename)
    # file = filename.replace(".wav", ".npy")
    # os.remove(file)

def extract_phonemes(filename):
    from phonemizer.phonemize import phonemize
    from phonemizer.backend import FestivalBackend
    from phonemizer.separator import Separator
    
    with open(filename) as f:
        text=f.read()
        phones = phonemize(text,
                           language='en-us',
                           backend='festival',
                           separator=Separator(phone=' ',
                                               syllable='',
                                               word='')
        )

    with open(filename.replace(".txt", ".phones"), "w") as outfile:
        print(phones, file=outfile)

def extract_dir(root, kind):
    # import pdb
    # pdb.set_trace()
    if kind =="audio":
        extraction_function=extract_mel_spec
        ext=".wav"
    elif kind =="text":
        extraction_function=extract_phonemes
        ext=".txt"
    else:
        print("ERROR: invalid args")
        sys.exit(1)
    if not os.path.isdir(root):
        print("ERROR: invalid args")
        sys.exit(1)
        
    # traverse over all subdirs of the provided dir, and find
    # only files with the proper extension
    abs_paths=[]
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            abs_path = os.path.abspath(os.path.join(dirpath, f))
            if abs_path.endswith(ext):
                 abs_paths.append(abs_path)

    # with open('list_files.txt', 'w') as f:
    #     for item in abs_paths:
    #         f.write("%s\n" % item)
    # import pdb
    # pdb.set_trace()       
    pool = Pool(cpu_count())
    pool.map(extraction_function,abs_paths)
        
if __name__ == "__main__":
    try:
        path = sys.argv[1]
        kind = sys.argv[2]
    except:
        print(
        '''
        Usage:
        
        $ extract_features.py "path" "kind"
        
        path: (str) Root path to data directory, this dir will be traversed
                    and all files matching the appropriate file extension
                    (i.e. ".txt" or ".wav") will undergo feature extraction.

        kind: (kind) Either "audio" or "text". "audio" will trigger feature
                     extraction of Mel-spectrograms, and "text" will trigger
                     phoneme extraction with a Festival backend.
        '''
        )
        sys.exit(1)

    extract_dir(path,kind)
    

    # extract_mel_spec("/home/hk/voice_conversion/nonparaSeq2seqVC_text-dependent_SE/reader/VCTK/wav48/p245/p245_020.wav")