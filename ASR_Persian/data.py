import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import librosa
from   sklearn import preprocessing
import os
import random
from uyghur import uyghur_latin
import numpy as np


featurelen  = 128 #melspec, 60  #mfcc
sample_rate = 22050
fft_len     = 1024
window_len  = fft_len
window      = "hann"
hop_len     = 200

#white_noise,_=librosa.load('white.wav',sr=sample_rate, duration=15.0)
#perlin_noise,_=librosa.load('perlin.wav',sr=sample_rate, duration=15.0)
#cafe_noise, _ = librosa.load('cafe.wav',sr=sample_rate, duration=15.0)
#radio_noise, _ = librosa.load('radionoise.wav',sr=sample_rate, duration=15.0)

def addnoise(audio):
    rnd = random.random()
    if len(audio) > len(white_noise):
        pass
    elif rnd <0.25:
        audio = audio + white_noise[:len(audio)] 
    elif rnd <0.50:
        audio = audio + perlin_noise[:audio.shape[0]]
    elif rnd <0.75:
        audio = audio + radio_noise[:audio.shape[0]]
    else:
        audio = audio + cafe_noise[:audio.shape[0]]
    return audio

def randomstretch(audio):
    factor = random.uniform(0.8, 1.2)
    audio = librosa.core.resample(audio,orig_sr=sample_rate,target_sr=sample_rate*factor)
    return audio

#def spec_augment(feat, T=70, F=15, time_mask_num=1, freq_mask_num=1):
def spec_augment(feat, T=50, F=13, time_mask_num=1, freq_mask_num=1):
    rnd = random.random()

    feat_size = feat.size(0)
    seq_len = feat.size(1)

    if  rnd< 0.33:
        # time mask
        for _ in range(time_mask_num):
            t = random.randint(0, T)
            t0 = random.randint(0, seq_len - t)
            feat[:, t0 : t0 + t] = 0

    elif rnd <0.66:
        # freq mask
        for _ in range(freq_mask_num):
            f = random.randint(0, F)
            f0 = random.randint(0, feat_size - f)
            feat[f0 : f0 + f, :] = 0
    else:
        # time mask
        for _ in range(time_mask_num):
            t = random.randint(0, T)
            t0 = random.randint(0, seq_len - t)
            feat[:, t0 : t0 + t] = 0

        # freq mask
        for _ in range(freq_mask_num):
            f = random.randint(0, F)
            f0 = random.randint(0, feat_size - f)
            feat[f0 : f0 + f, :] = 0

    return feat


def melfuture(wav_path, augument = False, my_eps=0.00000000001):
    audio, s_r = librosa.load(wav_path, sr=sample_rate, res_type='polyphase')

    #if augument:
    #    if random.random()<0.5:
    #        audio = randomstretch(audio)

    #    if random.random()<0.5:
    #        audio = addnoise(audio)

    audio = preprocessing.minmax_scale(audio, axis=0)
    audio = librosa.effects.preemphasis(audio)

    spec = librosa.feature.melspectrogram(y=audio, sr=s_r, n_fft=fft_len, hop_length=hop_len, n_mels=featurelen, fmax=8000)  
    spec = librosa.power_to_db(spec)
    #spec = librosa.amplitude_to_db(spec)

    spec = (spec - spec.mean()) / (spec.std() + my_eps)
    spec = torch.FloatTensor(spec)
    #if augument and random.random()<0.5:
    #    spec = spec_augment(spec)

    return spec

class SpeechDataset(Dataset):
    def __init__(self, index_path, augumentation = False, prefix_path="../datasets/cv-corpus-17.0-delta-2024-03-15/fa/", prefix_path2="clips_wav/"):
        
        self.Raw = False
        with open(prefix_path + index_path,encoding='utf_8_sig') as f:
            lines = f.readlines()

        self.idx  = []
        for x in lines:
            _, path, _, sentence = x.strip().split("\t")[0:4]
            path = path.strip().split(".")[0]
            path = prefix_path + prefix_path2 + path + ".wav"


            if os.path.exists(path):
                line = []
                line.append(path)
                char_indx = uyghur_latin.encode(sentence)
                line.append(char_indx)
                self.idx.append(line)

        self.augument = augumentation
    
    def __getitem__(self, index):
        wav_path, char_index = self.idx[index]
        x = melfuture(wav_path, self.augument)
        return x, char_index, wav_path

    def __len__(self):
        return len(self.idx)
 
def _collate_fn(batch):
    input_lens  = [sample[0].size(1) for sample in batch]
    target_lens = [len(sample[1]) for sample in batch]
    
    inputs = torch.zeros(len(batch), batch[0][0].size(0), max(input_lens) ,dtype=torch.float32)
    targets = torch.zeros(len(batch), max(target_lens),dtype=torch.long).fill_(uyghur_latin.pad_idx)

    target_lens = torch.IntTensor(target_lens)
    input_lens = torch.IntTensor(input_lens)
    paths = []
    for x, sample in enumerate(batch):
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        targets[x][:len(target)] = torch.LongTensor(target)
        paths.append(sample[2])
    return inputs, targets, input_lens, target_lens, paths


class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


# The following code is from: http://hetland.org/coding/python/levenshtein.py
def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def wer(s1, src):
    sw = src.split()
    return levenshtein(s1.split(),sw), len(sw)

def cer(s1, src):
    return levenshtein(s1,src),len(src)

def cer_wer(preds, targets):
    err_c, lettercnt, err_w, wordcnt = 0,0,0,0
    for pred, target in zip(preds, targets):
        c_er, c_cnt = cer(pred, target)
        w_er, w_cnt = wer(pred, target)
        err_c     += c_er
        lettercnt += c_cnt
        wordcnt   += w_cnt
        err_w     += w_er

    return err_c, lettercnt, err_w, wordcnt
