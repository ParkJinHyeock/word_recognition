#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:03:43 2021

@author: pjh
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import os
import torchaudio
import torch
from audiomentations import *
#from audio_augmentations import *


def csv_to_wav(data, file_name):

    for i, item in enumerate(data['Model']):
        if item == 'Sample Interval':
            base = i
    
    data_part = data.iloc[base+10:,:]['TBS1072C']
    np_data = data_part.to_numpy()
    sr = int(1/float(data['TBS1072C'][base]))
    np_data = np_data.astype(np.float32)
    np_data = np_data - np.mean(np_data)
    np_data = np_data / np.max(np.abs(np_data))
    base_name = os.path.basename(file_name).split('.')[0] 
    np_data = librosa.resample(np_data, sr, 2000)
    
    
    plt.figure(figsize= (4, 5))
    np_data = librosa.resample(np_data, sr, 2000)
    plt.subplot(411)
    librosa.display.waveplot(np_data, sr=2000)
    plt.title('raw_wav')
    
    torch_data = torch.unsqueeze(torch.from_numpy(np_data),0)
    resample = torchaudio.transforms.Resample(orig_freq=2000, new_freq=2000)
    torch_data =  resample(torchaudio.functional.highpass_biquad(torch_data, 2000, 30))
    np_data_2 = torch_data.squeeze().numpy()
    plt.subplot(412)
    librosa.display.waveplot(np_data_2, sr=2000)
    plt.title('filtering')

    plt.subplot(413)
    D = librosa.stft(np_data, n_fft=128, hop_length=32)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(D_db, y_axis='log', sr=2000)
    plt.title('spec_2000_resample')

    plt.subplot(414)
    D = librosa.stft(np_data_2, n_fft=128, hop_length=32)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(D_db, y_axis='log', sr=2000)
    plt.title('filtering resample')

    plt.tight_layout() 
    plt.savefig(base_name + '_wav.png', dpi = 400)
    plt.clf()

    plt.margins(0) 
    #plt.show()
    plt.savefig(base_name + '.jpg', dpi = 400, pad_inches=0)
    plt.clf()
    
    
    
def wav_to_spec(file_name):
    base_name = os.path.basename(file_name).split('.')[0] 
    np_data, sr = librosa.load(file_name)
    np_data = np_data.astype(np.float32)

    plt.figure(figsize= (4, 5))
    np_data = librosa.resample(np_data, sr, 2000)
    plt.subplot(411)
    librosa.display.waveplot(np_data, sr=2000)
    plt.title('raw_wav')
    
    torch_data = torch.unsqueeze(torch.from_numpy(np_data),0)
    resample = torchaudio.transforms.Resample(orig_freq=2000, new_freq=2000)
    torch_data =  resample(torchaudio.functional.highpass_biquad(torch_data, 2000, 30))
    np_data_2 = torch_data.squeeze().numpy()
    plt.subplot(412)
    librosa.display.waveplot(np_data_2, sr=2000)
    plt.title('filtering')


    
    plt.subplot(413)
    D = librosa.stft(np_data, n_fft=128, hop_length=32)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(D_db, y_axis='log', sr=2000)
    plt.title('spec_2000_resample')

    plt.subplot(414)
    D = librosa.stft(np_data_2, n_fft=128, hop_length=32)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(D_db, y_axis='log', sr=2000)
    plt.title('filtering resample')

    plt.tight_layout() 
    plt.savefig(base_name + '_wav.png', dpi = 400)
    plt.clf()

def augmentation(file_name):
    np_data, sr = librosa.load(file_name)
    np_data = np_data.astype(np.float32)
    np_data = librosa.resample(np_data, sr, 2000)
    shift = Shift(min_fraction=-0.3, max_fraction=0.3, rollover=False, fade=True, p=1)
    stretch = TimeStretch(min_rate=1.3, max_rate=1.3, p=1)
    gain = Gain(min_gain_in_db=3, max_gain_in_db=3, p=1)
    shifted = shift(np_data, 2000)
    gained = gain(np_data, 2000)


    plt.subplot(311)
    D = librosa.stft(np_data, n_fft=128, hop_length=32)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(D_db, y_axis='log', sr=1000)

    plt.subplot(312)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    D_db[10:20,:] = 0
    librosa.display.specshow(D_db, y_axis='log', sr=1000)

    plt.subplot(313)
    D_db = np.log(np.abs(D) + 1e-8)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    D_db[:,30:35] = 0
    librosa.display.specshow(D_db, y_axis='log', sr=1000)

    
    plt.tight_layout()
    plt.savefig('./' + 'augment_wav.png', dpi = 400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='Hz1000.CSV')
    args = parser.parse_args() 
    file_name = args.file_name
    if file_name is None:
        print("type your file_name")
    elif file_name.endswith('.csv') or file_name.endswith('.CSV'):        
        data_1 = pd.read_csv(file_name)
        csv_to_wav(data_1, file_name)
    else:
        wav_to_spec(file_name)
#        augmentation(file_name)