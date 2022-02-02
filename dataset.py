import os
import torch
from glob import glob
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data.dataset import Dataset
import torchaudio
from audio_augmentations import *
from time import time
from utils import *
from scipy import signal
import numpy as np
import pywt
from skimage.restoration import denoise_wavelet
import random
from audiomentations import *

class small_dataset(Dataset):
    def __init__(self, root_dir, mode, split_mode, train):
        self.root_dir = root_dir
        self.class_list = sorted(os.listdir(root_dir))
        self.train = train
        self.x = []
        self.y = []
        self.mode = mode
        self.split_mode = split_mode
        count = 0
        self.temp = []
        for item in self.class_list:
            item_dir = os.path.join(self.root_dir, item)
            x_list = sorted(glob(item_dir + '/*.wav'))
            y_list = [item]*len(x_list)
            for x, y in zip(x_list, y_list):
                if int(x.split('_')[-2]) < 10:
                    if count == 0:
                        audio = torchaudio.load(x)[0]
                        self.x = (audio - torch.mean(audio)) / torch.std(audio)
                    else:
                        audio = torchaudio.load(x)[0]
                        self.x = torch.vstack((self.x, (audio - torch.mean(audio)) / torch.std(audio)))
                        
                    if self.split_mode != 'random':
                        self.y = self.y + [(y, x.split('_')[-1].split('.')[0])]
                    else:
                        if self.mode == 'human':
                            self.y = self.y + [x.split('_')[-1].split('.')[0]]
                        elif self.mode == 'word':
                            self.y = self.y + [y]
                    count += 1
                    self.temp.append(x)
        self.labels = sorted(list(set(data for data in self.y)))
        self.sr = torchaudio.load(x)[1]
        self.new_sr = 2000
        self.resample = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self.new_sr, lowpass_filter_width=10)
        self.x = self.resample(self.x)
        self.x = torchaudio.functional.highpass_biquad(self.x, self.new_sr, 30)
        self.transforms_aug = Compose([
                                        Shift(min_fraction=-0.3, max_fraction=0.3, rollover=False, fade=True, p=0.5),
                                        Gain(min_gain_in_db=-5, max_gain_in_db=5, p=0.5),
                                    ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.train:
            temp = torch.from_numpy(self.transforms_aug(self.x[idx].numpy(), sample_rate=self.new_sr))
            return torch.reshape(temp, [1, temp.shape[0]]), self.y[idx], self.sr 
        else:
            return torch.reshape(self.x[idx], [1, self.x[idx].shape[0]]), self.y[idx], self.sr

    def remove(self):
        temp = []
        for item in self.y:
            if self.mode == 'word':
                temp.append(item[0])
            elif self.mode == 'human':
                temp.append(item[1])
        if self.mode == 'word':
            self.labels = sorted(list(set(data[0] for data in self.y)))
        elif self.mode =='human':
            self.labels = sorted(list(set(data[1] for data in self.y)))
        self.y = temp

    def to_index(self):
        self.y = [label_to_index(self.labels, item) for item in self.y]
        return self.labels

