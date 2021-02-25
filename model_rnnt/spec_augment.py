import random
import os
import time
import cv2
import math
import numpy as np

class spec_augment:
    def __init__(self):
        self.rotation = random.randrange(0 , 271, 90)
        self.wave_x = random.randrange(0,21)
        self.wave_y = random.randrange(0,21)

    def rotating(self,img):
        if self.rotation == 90: 
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def warping(self,Target_img):

        rows, cols = Target_img.shape 
        warpImg = np.zeros(Target_img.shape, dtype=Target_img.dtype)
 
        for i in range(rows): 
            for j in range(cols): 
                offset_x = int(self.wave_x * math.sin(2 * 3.14 * i / 150)) 
                offset_y = int(self.wave_y * math.cos(2 * 3.14 * j / 150)) 
                if i+offset_y < rows and j+offset_x < cols: 
                    warpImg[i,j] = Target_img[(i+offset_y)%rows,(j+offset_x)%cols] 
                else: 
                    warpImg[i,j] = 0 
        return warpImg

    def augment(self,Target_img):

        return self.warping(self.rotating(Target_img))
