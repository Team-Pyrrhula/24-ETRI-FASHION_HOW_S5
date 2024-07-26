import cv2
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class ETRI_Dataset():
    def __init__(self,
                config=None,
                train_mode:bool=True,
                transform=None,
                types:str='train',
                ):
        
        if types == 'train':
            self.df = pd.read_csv(config.TRAIN_DF)
            self.image_path = 'train'
        elif types == 'val':
            self.df = pd.read_csv(config.VAL_DF)
            self.image_path = 'val'

        self.label_1 = self.df[config.DF_INFO['label_1']].values
        self.label_2 = self.df[config.DF_INFO['label_2']].values
        self.label_3 = self.df[config.DF_INFO['label_3']].values
        self.images = self.df[config.DF_INFO['path']].values

        self.config = config
        self.train_mode = train_mode
        self.transform = transform

    def __len__(self):
        return (len(self.df))
    
    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.config.DATA_PATH, self.image_path, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.train_mode:
            label_1 = self.label_1[idx]
            label_2 = self.label_2[idx]
            label_3 = self.label_3[idx]
            return (image, label_1, label_2, label_3)
        
        return (image)
