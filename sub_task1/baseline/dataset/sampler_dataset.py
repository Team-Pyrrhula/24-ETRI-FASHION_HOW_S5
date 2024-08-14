import cv2
import os
import pandas as pd
from torch.utils.data import Dataset

class Sampler_Dataset(Dataset):
    def __init__(self,
                df=pd.DataFrame,
                label_type:str=None,
                config=None,
                train_mode:bool=True,
                transform=None,
                types:str='train',
                ):
        
        self.df = df
        self.label_type = label_type
        if types == 'train':
            self.image_path = 'train'
        elif types == 'val':
            self.image_path = 'val'

        if train_mode:
            self.label = self.df[label_type].values
        self.images = self.df[config.INFO['path']].values

        self.config = config
        self.train_mode = train_mode
        self.transform = transform
        self.types = types

    def __len__(self):
        return (len(self.df))
    
    def __p_label__(self):
        print(self.label_type)
    
    def __getitem__(self, idx):
        if self.train_mode == 'test':
            image = cv2.imread(os.path.join(self.image_path, self.images[idx]))
        else:
            image = cv2.imread(os.path.join(self.config.DATA_PATH, self.image_path, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.train_mode:
            label = self.label[idx]
            return (image, label)
        
        return (image)
