import math
import os
import time
from matplotlib import pyplot as plt
import pandas as pd
import librosa.display, librosa
import numpy as np
import scipy.signal
import soundfile as sf
import sox
import torch
import csv
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
import matplotlib
from PIL import Image
import torchvision.transforms as transforms
import cv2

windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann,
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value

class AudioParser(object):
    def parse_transcript(self, transcript_path):
        raise NotImplementedError

    def parse_audio(self, audio_path):
        raise NotImplementedError

class SpectrogramParser(AudioParser):
    def __init__(self, feature_type, normalize, spec_augment):
        super(SpectrogramParser, self).__init__()
        self.normalize = normalize
        self.feature_type = feature_type
        self.spec_augment = spec_augment

    def parse_audio(self, audio_path):
       
       # print(audio_path+"에서 데이터를 불러 오는 중")
        image = cv2.imread(audio_path, cv2.IMREAD_UNCHANGED)
        # Resize and Denoise 
	## confg로 옮기기
        #Lap_ksize = 3
        #Gau_ksize = 13

        res = cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
       # res = image	

        blur = cv2.GaussianBlur(res,(13,13),0)
        edge = cv2.Laplacian(blur,cv2.CV_8U,ksize = 3)

        DenoisedImg = res + edge
 #       cv2.imwrite('./data/denoisedData/'+ audio_path[-9:],DenoisedImg)
       # print("변환된 데이터 저장 중 :"+'./data/denoisedData/'+audio_path[-9:])
	
        image = (DenoisedImg/255).astype('float')
        img_t = torch.FloatTensor(image)

        img_t = img_t.unsqueeze(0)
               
        '''
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        

        if self.spec_augment:
            spect = spec_augment(spect)

        if False:
            path = './test_img'
            os.makedirs(path, exist_ok=True)
            matplotlib.image.imsave('./test_img/'+ audio_path[50:-4] +'name.png', spect)
        '''
        return img_t

    def parse_transcript(self, transcript_path):
        raise NotImplementedError

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, manifest_filepath, feature_type, normalize, spec_augment):
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        
        self.ids = ids
        self.size = len(ids)
               
        super(SpectrogramDataset, self).__init__(feature_type, normalize, spec_augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
    
        transcript = self.parse_transcript(transcript_path)
        spect = self.parse_audio(audio_path)
            
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as f:
            transcript = f.read()
            transcript = transcript.strip().split(",")[:-1]
            transcript = list(map(int, transcript))
     
        return transcript

    def __len__(self):
        return self.size

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])
    
    def target_length_(p):
        return len(p[1])
    
    batch_size = len(batch)
    batch_img = torch.zeros(batch_size, batch[0][0].size(1), batch[0][0].size(2))
    
    max_target_size = len(batch[0][1])
    #batch_script = torch.zeros(batch_size, batch[0][0].size(1), batch[0][0].size(2))
    
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0].squeeze()
        target = torch.tensor(sample[1])
        seq_length = tensor.size(0)

        batch_img[x].copy_(tensor)
        targets[x].copy_(torch.LongTensor(target))
    
    return batch_img, targets
    
#torch.Size([16, 1192, 240])
#torch.Size([16, 31])
#[919, 474, 649, 365, 544, 225, 407, 590, 627, 468, 473, 450, 304, 436, 406, 1192]
#[24, 16, 15, 12, 31, 10, 18, 16, 21, 17, 15, 20, 11, 14, 13, 17]

