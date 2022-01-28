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
from glob import glob
def get_audio(path, args):
    count = 0
    class_list = sorted(os.listdir(path))
    data_y = []
    temp = []
    for item in class_list:
        item_dir = os.path.join(path, item)
        x_list = sorted(glob(item_dir + '/*.wav'))
        y_list = [item]*len(x_list)
        for x, y in zip(x_list, y_list):
            if int(x.split('_')[-2]) < 10:
                if x.split('_')[-1] in [f'{args.test}.wav']:
                    if count == 0:
                        audio = torchaudio.load(x)[0]
                        data_x = (audio - torch.mean(audio)) / torch.std(audio)
                    else:
                        audio = torchaudio.load(x)[0]
                        data_x = torch.vstack((data_x, (audio - torch.mean(audio)) / torch.std(audio)))
                    data_y = data_y + [y]
                    count += 1
                    temp.append(x)
    import pdb; pdb.set_trace()
    resample = torchaudio.transforms.Resample(orig_freq=torchaudio.load(x)[1], new_freq=1000)
    data_x = resample(data_x).to(device)
    spec = torchaudio.transforms.Spectrogram(n_fft=128, hop_length=32).to(device)
    data_x_2 = spec(data_x)
    data_x_2 = torch.log(data_x_2 + 1e-8)

    data_x = torchaudio.functional.highpass_biquad(data_x, 1000, 30)
    data_x = torch.unsqueeze(data_x, 1).to(device)
    data_x = torch.cat([data_x[1::2,...]])
    labels = sorted(list(set(data for data in data_y)))
    data_y = [label_to_index(labels, item) for item in data_y]
    data_y = data_y[1::2]
    return data_x, data_x_2, data_y


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='2D')
parser.add_argument('--mode', type=str, default='word')
parser.add_argument('--model_type', type=str, default='CNN_TD')
parser.add_argument('--test', type=str, default='run')
parser.add_argument('--path', type=str, default='./data_reco')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = args.model_type
mode = args.mode
model_base = args.model
labels = ['fast', 'library', 'pause', 'play', 'previous', 'repeat', 'resume', 'show', 'shuffle', 'skip this', 'slow']


if args.model == '2D':
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
    model.load_state_dict(torch.load(f'./saved_model/{mode}_2D_{model_type}_{args.test}.pth'))
    model.load_state_dict(torch.load(f'./saved_model/{mode}_2D_{model_type}_run.pth'))
    model.eval()
    model.to(device)

elif args.model == '1D':
    model = sample_NET(n_input=1, n_output=len(labels), batch_size=10)
    # model.load_state_dict(torch.load(f'./saved_model/{mode}_1D_{model_type}_{args.test}_mic_2.pth'))
    model.load_state_dict(torch.load(f'./saved_model/{mode}_1D_{model_type}_run_mic_2.pth'))

    model.eval()
    model.to(device)

data_x, data_x_2, data_y = get_audio(args.path, args)
if args.model == '1D':
    output = model(data_x).cpu()
else:
    output = model(data_x_2).cpu()
pred = get_likely_index(output)
# pred = get_likely_index(output_2)



import seaborn as sn
import pandas as pd
# labels = sorted(list(set(data.item() for data in data_y)))
confusion = confusion_matrix(data_y, pred)
confusion = np.round(confusion / np.sum(confusion, axis=1), decimals=2)
df_cm = pd.DataFrame(confusion, index=labels, columns=labels)
df_cm.to_csv('./word.csv')
label_ = [data.item() for data in data_y]
accuracy = np.sum(np.asarray(label_)  == pred.numpy()) / np.sum(len(np.asarray(label_)))
print(accuracy)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
sn.heatmap(df_cm, cmap='pink_r')
plt.tight_layout() 
plt.savefig(f'./{args.test}.png', dpi=400)
