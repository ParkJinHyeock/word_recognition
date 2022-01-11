import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import os
# CUBLAS_WORKSPACE_CONFIG=4096
    


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

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
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T

def seed_all(seed: int = 1930):

    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(
        seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
seed_all(seed=217)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='2D')
parser.add_argument('--mode', type=str, default='word')
parser.add_argument('--model_type', type=str, default='CNN_TD')
parser.add_argument('--path', type=str, default='./data_reco')
parser.add_argument('--split_mode', type=str, default='random')
parser.add_argument('--use_mel', type=bool, default=False)
parser.add_argument('--save_path', type=str, default='save_image')
parser.add_argument('--number', type=str, default='0')
parser.add_argument('--save', type=str, default='')
parser.add_argument('--seen', type=str, default='')

args = parser.parse_args()

seen = bool(args.seen)
is_save = bool(args.save)
mode = args.mode
split_mode = args.split_mode
model_base = args.model
use_mel = args.use_mel
number = int(args.number)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = args.model_type

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

dataset = small_dataset(args.path, mode, split_mode, train=True)
dataset_test = small_dataset(args.path, mode, split_mode, train=False)

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
        label_list = sorted([item[1][1] for item in dataset])
    elif split_mode == 'word':
        label_list = sorted([item[1][0] for item in dataset])
    elif split_mode == 'human_word' or split_mode == 'cross' or split_mode == 'huristic_cross':
        label_list = sorted([item[1] for item in dataset])
    if 'cross' not in split_mode:
        splited = train_test_split(indices, label_list, test_size=validation_split, stratify=label_list)
        train_indices = splited[0]
        val_indices = splited[1]
    elif split_mode == 'huristic_cross':
        if mode == 'human':
            word_list = sorted(list(set([item[0] for item in label_list])))
            pick = word_list[number]
            for i, item in enumerate(label_list):
                if item[0] in pick:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
        else:
            human_list = sorted(list(set([item[1] for item in label_list])))
            pick = [human_list[number]]
            # pick  = [human_list[number], human_list[number+1]]     
            print(f'picked_human is {pick}')
            for i, item in enumerate(label_list):
                if item[1] in pick:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
            if seen:
                train_indices = train_indices + val_indices[::4]
                val_indices  = list(set(val_indices) - set(val_indices[::4]))
    else:
        if mode == 'human':
            word_list = sorted(list(set([item[0] for item in label_list])))
            pick  = random.choices(word_list, k=int(len(word_list)*validation_split))
            for i, item in enumerate(label_list):
                if item[0] in pick:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
        else:
            human_list = sorted(list(set([item[1] for item in label_list])))
            pick  = random.choices(human_list, k=int(len(human_list)*validation_split))        
            for i, item in enumerate(label_list):
                if item[1] in pick:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
    dataset.remove()
    dataset_test.remove()

labels = dataset.to_index()
labels_test = dataset_test.to_index()
print(labels)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


batch_size = 10
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                           collate_fn=collate_fn, sampler=train_sampler, num_workers=0, worker_init_fn=seed_worker)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False,
                                           collate_fn=collate_fn, sampler=valid_sampler, num_workers=0, worker_init_fn=seed_worker)

sample_rate = dataset[0][2]
new_sample_rate = 1000
waveform = dataset[0][0]
# transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

if model_base == '2D':
    spec = torchaudio.transforms.Spectrogram(n_fft=128, hop_length=32).to(device)
    data = next(iter(train_loader))
    db = torchaudio.transforms.AmplitudeToDB()
    if use_mel:
        mel = torchaudio.transforms.MelScale(n_mels=65, sample_rate=new_sample_rate).to(device)
        m = mel(db(spec(data[0].to(device)))).mean(axis=0)
        s = mel(db(spec(data[0].to(device)))).std(axis=0)
        
        # mel = torchaudio.transforms.MFCC(sample_rate=1000, n_mfcc=20).to(device)
        # m = mel(data[0].to(device)).mean(axis=0)
        # s = mel(data[0].to(device)).std(axis=0)
    else:
        m = db(spec(data[0].to(device))).mean(axis=0)
        s = db(spec(data[0].to(device))).std(axis=0)

    # model = FC_Net(n_input=transformed.shape[0], n_output=len(labels), batch_size=batch_size)
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

elif model_base == '1D':
    model = sample_NET(n_input=waveform.shape[0], n_output=len(labels), batch_size=batch_size)


model.to(device)
model.train()
n = count_parameters(model)
print("Number of parameters: %s" % n)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

freq_masking = T.FrequencyMasking(freq_mask_param=5)
time_masking = T.TimeMasking(time_mask_param=3)

def train(model, epoch, log_interval, scheduler):
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
            if use_mel:
                data = mel(data)
            # data = (data - m)/s
            # data = freq_masking(data)
            # data = time_masking(data)

        output = model(data)
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
        l2_lambda = 0.001
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = F.nll_loss(output, target)
        # loss += l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    n_pred = 0
    for (data, target) in pbar:
        data = data.to(device)
        target = target.to(device)     

        if model_base == '2D':
            data = spec(data)
            data = db(data)
            if use_mel:
                data = mel(data)
            # data = (data - m)/s



        output = model(data)    
        pred = get_likely_index(output)
        n_pred += len(pred)
        empty_pred = torch.cat([empty_pred, pred])
        empty_target = torch.cat([empty_target, target])
        correct += number_of_correct(pred, target)
        loss = F.nll_loss(output, target)
        losses.append(loss.item())        
        # update progress bar
        pbar.update(pbar_update)
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{n_pred} ({100. * correct / n_pred:.0f}%)\n")
    accuracy = 100. * correct / n_pred
    return  sum(losses)/len(losses), accuracy, empty_pred, empty_target

log_interval = 300
n_epoch = 40
pbar_update = 1 / (len(train_loader) + len(test_loader))
if is_save:
    torch.save(model.state_dict(), f'./saved_model/{mode}_{model_base}_{model_type}.pth')
# The transform needs to live on the same device as the model and the data.
max_patient = 10
writer = SummaryWriter(log_dir=mode, filename_suffix=f'{model_base}_{model_type}')

with tqdm(total=n_epoch) as pbar:
    max_acc = 0
    early_count = 0
    max_epoch = 0
    target = 0
    pred = 0
    for epoch in range(1, n_epoch + 1):
        loss_train, accuracy_train = train(model, epoch, log_interval, scheduler)
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
        print(epoch)
        writer.add_scalar(f'Loss/train', loss_train, global_step=epoch)
        writer.add_scalar(f'Loss/test',loss_test, global_step=epoch)
        writer.add_scalar(f'Accuracy/train', accuracy_train, global_step=epoch)
        writer.add_scalar(f'Accuracy/test', accuracy_test, global_step=epoch)
        scheduler.step()

print(f'\n Max Test Accuracy is {max_acc} when epoch is {max_epoch}') 
target = target.cpu().numpy()
pred = pred.cpu().numpy()
# for (data, target) in test_loader:
#     data = data.to(device)
#     target = target.to(device)     

#     if model_base == '2D':
#         data = spec(data)
#         data = db(data)
#         if use_mel:
#             data = mel(data)
#         # data = (data - m)/s

#     output = model(data)    
import seaborn as sn
import pandas as pd
writer.flush()
writer.close()
if is_save:
    torch.save(model.state_dict(), f'./saved_model/{mode}_{model_base}_{model_type}.pth')


confusion = confusion_matrix(target, pred)
confusion = np.round(confusion / np.sum(confusion, axis=1), decimals=2)
df_cm = pd.DataFrame(confusion, index=dataset.labels, columns=dataset.labels)
df_cm.to_csv('./word.csv')

# sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, cmap='pink_r')
plt.rcParams.update({'font.family':'Arial'})
# sn.heatmap(df_cm, annot=True, cmap = plt.cm.Blues, annot_kws={"size": 11}, fmt=".2f") # font size
from datetime import datetime
plt.tight_layout() 
plt.savefig(f'./{args.save_path}/{args.path}_{args.model}_{args.mode}_{args.split_mode}_{datetime.now().hour}_{datetime.now().minute}_{number}_{max_acc}.png', dpi=400)


