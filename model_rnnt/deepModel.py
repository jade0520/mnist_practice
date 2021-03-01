import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, autograd
import math
from dropblock import DropBlock2D

class Resnet(nn.Module):
    def __init__(self, ):
        super(Resnet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        # 128 -> 256 
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256)
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256)
        )

        self.conv10_ = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256)
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256)
        )
        
        # 256 -> 512 
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512)
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512)
        )

        self.conv13_ = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512)
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512)
        )

        #########더 깊게
        # 512 -> 1024 

        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu7_ = nn.ReLU()        
        self.relu8 = nn.ReLU()
        self.relu9 = nn.ReLU()
        self.relu10 = nn.ReLU()
        self.relu10_ = nn.ReLU()        
        self.relu11 = nn.ReLU()
        self.relu12 = nn.ReLU()
        self.relu13 = nn.ReLU()
        self.relu14 = nn.ReLU()

        self.AvgPool = nn.AvgPool2d(2, stride=2)
        self.AvgPool1 = nn.AvgPool2d(2, stride=2)
        self.AvgPool2 = nn.AvgPool2d(2, stride=2)
        self.AvgPool3 = nn.AvgPool2d(2, stride=2)
        self.AvgPool4 = nn.AvgPool2d(2, stride=2)

        self.dropout = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.3)
       
       
        # DropBlocks
        self.dropout_2d = DropBlock2D(block_size = 3, drop_prob =0.3) # nn.Dropout2d(p=0.3)
        self.dropout_2d1 = DropBlock2D(block_size = 3, drop_prob =0.3) # nn.Dropout2d(p=0.3)  
        self.dropout_2d2 = DropBlock2D(block_size = 3, drop_prob =0.3) # nn.Dropout2d(p=0.3)
        self.dropout_2d3 = DropBlock2D(block_size = 3, drop_prob =0.3) # nn.Dropout2d(p=0.3) 
        self.dropout_2d4 = DropBlock2D(block_size = 3, drop_prob =0.3) # nn.Dropout2d(p=0.3) 
        """ 
        self.dropout_2d = nn.Dropout2d(p=0.3)
        self.dropout_2d1 = nn.Dropout2d(p=0.3)
        self.dropout_2d2 = nn.Dropout2d(p=0.3)
        self.dropout_2d3 = nn.Dropout2d(p=0.3)
        self.dropout_2d4 = nn.Dropout2d(p=0.3)
        """        

       # inputSize = 512 # upscaled size : original size = 256
       # inSize = inputSize*256 # 131072
        self.linear = nn.Sequential(
            nn.Linear(131072, 256),
            nn.BatchNorm1d(256)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )

        self.projection = nn.Linear(256, 26)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        inputs = inputs.unsqueeze(1)
        
        #first 드랍 아웃 추가해야함
        x = self.conv1(inputs)
        x_res = x
        
        x = self.relu(x)
        x = self.conv2(x)
        x = x + x_res

        x = self.relu1(x)
        x = self.AvgPool(x)
        x = self.dropout_2d(x)

        #second
        x = self.conv3(x)
        x_res = x

        x = self.relu2(x)
        x = self.conv4(x)

        x = self.relu3(x)
        x = self.conv5(x)
        x = x + x_res

        x = self.relu4(x)
        x = self.AvgPool1(x)
        x = self.dropout_2d1(x)

        #third
        x = self.conv6(x)
        x_res = x

        x = self.relu5(x)
        x = self.conv7(x)

        x = self.relu5(x)
        x = self.conv8(x)
        x = x + x_res

        x = self.relu6(x)
        x = self.AvgPool2(x)
        x = self.dropout_2d2(x)

        #fourth
        x = self.conv9(x)
        x_res = x
        
        x = self.relu7(x)
        x = self.conv10(x)
        
        # 추가
        x = self.relu7_(x)
        x = self.conv10_(x)   

        x = self.relu8(x)
        x = self.conv11(x)
        x = x + x_res

        x = self.relu9(x)
        x = self.AvgPool3(x)       
        x = self.dropout_2d3(x)
 
        #fiveth
        x = self.conv12(x)
        x_res = x
        
        x = self.relu10(x)
        x = self.conv13(x)

        # 추가
        x = self.relu10_(x)
        x = self.conv13_(x)        
        
        x = self.relu11(x)
        x = self.conv14(x)
        x = x + x_res

        x = self.relu12(x)
        x = self.AvgPool4(x)
        x = self.dropout_2d4(x)

        x = torch.flatten(x, 1)
        
       # print("flattend shape :",x.shape)       
        #sixth
        x = self.linear(x)
        x_res = x

        x = self.relu13(x)
        x = self.dropout(x)
        
        x = self.linear1(x)
        x = x + x_res
        x = self.relu14(x)
        x = self.dropout1(x)

        x = self.projection(x)
        x = self.sigmoid(x)
        # projection layer 추가해야됨

        return x
