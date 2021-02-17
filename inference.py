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

def evaluation(model, val_loader, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    total_num = 0

    count = 50000

    start_time = time.time()
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data
            
            inputs = inputs.to(device) # (batch_size, time, freq)
            logits = model(inputs, None)
            
            logits  = logits.cpu().detach().numpy()           
            logits = np.where(logits>0.5, 1, 0)

            with open("./sample_submission.csv", "a") as f:
                f.write(str(count))
                f.write(",")

                for aaa, ii in enumerate(logits[0]):
                    
                    if aaa == 25:
                        f.write(str(ii))
                        f.write("\n")
                    else:
                        f.write(str(ii))
                        f.write(",")
     
                count += 1
                print(count)
    

    return 

def main():
    

    with open("./sample_submission.csv", "w") as f:
        f.write("index,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z")
        f.write('\n')

    yaml_name = "./data/Two_Pass.yaml"
    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    cuda = torch.cuda.is_available()
    #device = torch.device('cuda' if cuda else 'cpu')
    device = torch.device('cpu')
    #-------------------------- Model Initialize --------------------------   
    las_model = Resnet()
    las_model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/dacon/MNIST_2/plz_load/model.pth"))
    
    las_model = las_model.to(device)
    
    #val dataset
    val_dataset = SpectrogramDataset("./data/inference.csv", 
                                     feature_type="config.audio_data.type",
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                 shuffle=False,
                                 num_workers=config.data.num_workers,
                                 batch_size=1,
                                 drop_last=False)
    
    print(" ")
    print("inference 를 학습합니다.")
    print(" ")

   
    print('{} 평가 시작'.format(datetime.datetime.now()))
    eval_time = time.time()
    evaluation(las_model, val_loader, device)
    
if __name__ == '__main__':
    main()