import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import os
from dataset import SubsetSC, small_dataset
from models import *
from utils import *
import argparse
import matplotlib.pyplot as plt

def get_audio(path):
    audio, sr = torchaudio.load(path)
    audio = (audio - torch.mean(audio)) / torch.std(torch.abs(audio))
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=1000)
    audio = resample(audio)
    audio = torch.unsqueeze(audio, 0).to(device)
    spec = torchaudio.transforms.Spectrogram(n_fft=128, hop_length=32).to(device)
    db = torchaudio.transforms.AmplitudeToDB()
    data = spec(audio)
    data = db(data)
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='2D')
parser.add_argument('--mode', type=str, default='word')
parser.add_argument('--model_type', type=str, default='CNN_TD')
parser.add_argument('--path', type=str, default='./data_reco')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = args.model_type
mode = args.mode
model_base = args.model
labels = ['fast', 'library', 'pause', 'play', 'previous', 'repeat', 'resume', 'show', 'shuffle', 'skip this', 'slow']



if mode == 'word':
    if model_type == 'CNN_TD':
        model = CNN_TD(num_classes=len(labels))
    if model_type == 'Spinal':
        model = SpinalVGG(num_classes=len(labels))
    if model_type == 'Marble':
        model = MarbleNet(num_classes=len(labels))

else:
    if model_type == 'CNN_TD':
        model = CNN_TD(num_classes=len(labels))
    if model_type == 'Spinal':
        model = SpinalVGG(num_classes=len(labels))
    if model_type == 'Marble':
        model = MarbleNet(num_classes=len(labels))
model.load_state_dict(torch.load(f'./saved_model/{mode}_{model_base}_{model_type}.pth'))
model.eval()
model.to(device)

data = get_audio(args.path)
output = model(data)
pred = get_likely_index(output)
print(labels[pred])
