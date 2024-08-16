import cv2
import os
import pandas as pd

class ETRI_Dataset_color():
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
        elif types == 'test':
            self.df = pd.read_csv(config.TEST_DF)
            self.image_path = '/aif/Dataset/test'

        if train_mode:
            self.label = self.df[config.INFO['label']].values
        self.images = self.df[config.INFO['path']].values

        self.config = config
        self.train_mode = train_mode
        self.transform = transform
        self.types = types

    def __len__(self):
        return (len(self.df))
    
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
