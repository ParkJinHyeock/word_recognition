import torch
torch.manual_seed(215)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
np.random.seed(214)

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
import os
from dataset import SubsetSC, small_dataset
from models import *
from utils import *
import torchvision.models as models
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random
random.seed(213)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='2D')
parser.add_argument('--mode', type=str, default='word')
parser.add_argument('--path', type=str, default='./data_reco')
parser.add_argument('--split_mode', type=str, default='random')
args = parser.parse_args()

mode = args.mode
split_mode = args.split_mode
model_base = args.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

dataset = small_dataset(args.path, mode, split_mode)

validation_split = 0.2
dataset_size = len(dataset)
shuffle_dataset = True

train_indices = []
val_indices = []

if split_mode == 'random':
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]


else:
    indices = list(range(dataset_size))
    if split_mode == 'human':
        label_list = [item[1][1] for item in dataset]
    elif split_mode == 'word':
        label_list = [item[1][0] for item in dataset]
    elif split_mode == 'human_word' or split_mode == 'cross':
        label_list = [item[1] for item in dataset]
    
    if split_mode != 'cross':
        splited = train_test_split(indices, label_list, test_size=validation_split, stratify=label_list)
        train_indices = splited[0]
        val_indices = splited[1]
    else:
        if mode == 'human':
            word_list = list(set([item[0] for item in label_list]))
            pick  = random.choices(word_list, k=int(len(word_list)*validation_split))
            for i, item in enumerate(label_list):
                if item[0] in pick:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
        else:
            human_list = list(set([item[1] for item in label_list]))
            pick  = random.choices(human_list, k=int(len(human_list)*validation_split))        
            for i, item in enumerate(label_list):
                if item[1] in pick:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
    dataset.remove()

labels = dataset.to_index()
print(labels)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

batch_size = 8
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                           collate_fn=collate_fn, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                           collate_fn=collate_fn, sampler=valid_sampler)

sample_rate = dataset[0][2]
new_sample_rate = 1000
waveform = dataset[0][0]
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

if model_base == '2D':
    spec = torchaudio.transforms.Spectrogram(n_fft=128).to(device)
    data = next(iter(train_loader))
    db = torchaudio.transforms.AmplitudeToDB()
    m = db(spec(data[0].cuda())).mean(axis=0)
    s = db(spec(data[0].cuda())).std(axis=0)
    transformed = torchaudio.transforms.Spectrogram(n_fft=128)(waveform)
    model = FC_Net(n_input=transformed.shape[0], n_output=len(labels), batch_size=batch_size)

elif model_base == '1D':
    transformed = transform(waveform)
    model = sample_NET(n_input=transformed.shape[0], n_output=len(labels), batch_size=batch_size)


model.to(device)
model.train()
n = count_parameters(model)
print("Number of parameters: %s" % n)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train(model, epoch, log_interval):
    pbar = tqdm(train_loader)
    model.train()
    correct = 0
    losses = []
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)

        if model_base =='2D':
            data = spec(data)
            data = db(data)
            data = (data - m)/s

        output = model(data)
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
        l2_lambda = 0.0001
        l2_reg = torch.tensor(0.).cuda()
        for param in model.parameters():
            l2_reg += torch.norm(param)
        try:
            loss = F.nll_loss(output.squeeze(), target)
        except:
            import pdb; pdb.set_trace()
        loss += l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # # print training stats
        # if batch_idx % log_interval == 0:
        #     print(f"\nTrain Epoch: {epoch} [{batch_idx * len(data)}/{len(train_indices))} ({100. * batch_idx / len(train_indices):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
    print(f"\nTrain Epoch: {epoch}\tAccuracy: {correct}/{len(train_indices)} ({100. * correct / len(train_indices):.0f}%)\n")
    # count number of correct predictions
    accuracy = 100. * correct / (len(train_loader.dataset)*(1-validation_split))
    return sum(losses)/len(losses), accuracy



def test(model, epoch, test_loader_):
    model.eval()
    correct = 0
    pbar = tqdm(test_loader_)
    empty_pred = torch.empty((0),device='cuda')
    empty_target = torch.empty((0),device='cuda')
    losses = []
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)     

        if model_base == '2D':
            data = spec(data)
            data = db(data)
            data = (data - m)/s

        output = model(data)    
        pred = get_likely_index(output)
        empty_pred = torch.cat([empty_pred, pred])
        empty_target = torch.cat([empty_target, target])
        correct += number_of_correct(pred, target)
        loss = F.nll_loss(output, target)
        losses.append(loss.item())        
        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_indices)} ({100. * correct / len(val_indices):.0f}%)\n")
    accuracy = 100. * correct / len(val_indices)
    return  sum(losses)/len(losses), accuracy, empty_pred, empty_target

log_interval = 300
n_epoch = 100

pbar_update = 1 / (len(train_loader) + len(test_loader))

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
max_patient = 10
writer = SummaryWriter()

with tqdm(total=n_epoch) as pbar:
    max_acc = 0
    early_count = 0
    max_epoch = 0
    target = 0
    pred = 0
    for epoch in range(1, n_epoch + 1):
        loss_train, accuracy_train = train(model, epoch, log_interval)
        loss_test, accuracy_test, total_pred, total_target = test(model, epoch, test_loader)
        if max_acc < accuracy_test:
            max_acc = accuracy_test
            early_count = 0
            max_epoch = epoch
            target = total_target
            pred = total_pred
        else:
            early_count += 1
            if early_count == max_patient:
                break
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Loss/test',loss_test, epoch)
        writer.add_scalar('Accuracy/train', accuracy_train, epoch)
        writer.add_scalar('Accuracy/test', accuracy_test, epoch)
        scheduler.step()

print(f'\n Max Test Accuracy is {max_acc} when epoch is {max_epoch}') 
target = target.cpu().numpy()
pred = pred.cpu().numpy()
import seaborn as sn
import pandas as pd
confusion = confusion_matrix(target, pred)
df_cm = pd.DataFrame(confusion, index=dataset.labels, columns=dataset.labels)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()