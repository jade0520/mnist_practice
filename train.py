import os
import time
import yaml
import random
import shutil
import argparse
import datetime
import editdistance
import scipy.signal
import numpy as np 

# torch 관련
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from model_rnnt.model import Resnet
from model_rnnt.data_loader_deepspeech import SpectrogramDataset, AudioDataLoader, AttrDict

def train(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    total_num = 0

    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, targets = data

        inputs = inputs.to(device) # (batch_size, time, freq)
        targets = targets.type(torch.FloatTensor)
        targets = targets.to(device)
        logits = model(inputs, targets)

        loss = criterion(logits, targets)

        logits  = logits.cpu().detach().numpy()
        labels = targets.cpu().detach().numpy()
        preds = logits > 0.6
        batch_acc = (labels == preds).mean()

        total_acc += batch_acc
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('{} train_batch: {:4d}/{:4d}, train_loss: {:.4f}, train_acc: {:.4f}, train_time: {:.2f}'
                  .format(datetime.datetime.now(), i, total_batch_num, loss.item(), batch_acc, time.time() - start_time))
            start_time = time.time()
    
    train_loss = total_loss / total_batch_num
    train_acc = total_acc / total_batch_num

    return train_loss, train_acc

def evaluation(model, val_loader, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    total_num = 0

    start_time = time.time()
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data
            
            inputs = inputs.to(device) # (batch_size, time, freq)
            targets = targets.type(torch.FloatTensor)
            targets = targets.to(device)
            logits = model(inputs, targets)
            
            logits  = logits.cpu().detach().numpy()
            labels = targets.cpu().detach().numpy()
            preds = logits > 0.6
            batch_acc = (labels == preds).mean()

            total_acc += batch_acc
            

    val_acc = total_acc / total_batch_num

    return val_acc

def main():

    with open("./train.txt", "w") as f:
       
        f.write('\n')
        f.write('\n')
        f.write("학습 시작")
        f.write('\n')

    yaml_name = "./data/Two_Pass.yaml"
    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
    #-------------------------- Model Initialize --------------------------   
    las_model = Resnet().to(device)
    las_model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/dacon/MNIST_2/plz_load/model_end.pth"))
    las_model = nn.DataParallel(las_model).to(device)
    #-------------------------- Loss Initialize ---------------------------
    las_criterion = nn.BCELoss()
    #las_criterion = LabelSmoothingLoss(num_classes=config.model.vocab_size, ignore_index=0, smoothing=0.1, reduction='sum').to(device)
    
    #-------------------- Model Pararllel & Optimizer ---------------------
    las_optimizer = optim.Adam(las_model.module.parameters(), 
                                lr=config.optim.lr,
                                weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(las_optimizer, milestones=[30,80], gamma=0.5)
    #-------------------------- Data load ---------------------------------
    #train dataset
    train_dataset = SpectrogramDataset("./data/train.csv",
                                       feature_type="config.audio_data.type", 
                                       normalize=True, 
                                       spec_augment=False)

    train_loader = AudioDataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=80,
                                    drop_last=True)
    
    #val dataset
    val_dataset = SpectrogramDataset("./data/val.csv", 
                                     feature_type="config.audio_data.type",
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                 shuffle=True,
                                 num_workers=config.data.num_workers,
                                 batch_size=20,
                                 drop_last=True)
    
    print(" ")
    print("las_only 를 학습합니다.")
    print(" ")

    pre_acc = 0.706
    pre_test_loss = 100000
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        for param_group in las_optimizer.param_groups:
            print("lr = ", param_group['lr'])
        
        print('{} 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()
        train_loss, train_acc = train(las_model, train_loader, las_optimizer, las_criterion, device)
        train_total_time = time.time() - train_time
        print('{} Epoch {} (Training) Loss {:.4f}, ACC {:.4f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, train_loss, train_acc, train_total_time))
        
        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        val_acc = evaluation(las_model, val_loader, device)
        eval_total_time = time.time() - eval_time
        print('{} Epoch {} (val) ACC {:.4f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, val_acc, eval_total_time))
        
        #scheduler.step()
        
        with open("./train.txt", "a") as ff:
            ff.write('Epoch %d (Training) Loss %0.4f Acc %0.4f time %0.4f' % (epoch+1, train_loss, train_acc, train_total_time))
            ff.write('\n')
            ff.write('Epoch %d (val) Acc %0.4f time %0.4f ' % (epoch+1, val_acc, eval_total_time))
            ff.write('\n')
            ff.write('\n')

        if pre_acc < val_acc:
            print("best model을 저장하였습니다.")
            torch.save(las_model.module.state_dict(), "./plz_load/model.pth")
            pre_acc = val_acc

        torch.save(las_model.module.state_dict(), "./plz_load/model_end.pth")
        
                

if __name__ == '__main__':
    main()
