import cv2
import os
import pandas as pd

from transform import BaseAug

class ETRI_Dataset_color():
    def __init__(self,
                config=None,
                train_mode:bool=True,
                transform=None,
                types:str='train',
                remgb:bool=False,
                crop:bool=False,
                mixup:bool=False,
                ):
        if remgb:
            if types == 'train':
                self.df = pd.read_csv(config.TRAIN_DF)
                self.image_path = './'
        else:
            if types == 'train':
                self.df = pd.read_csv(config.TRAIN_DF)
                self.image_path = 'train'
            elif types == 'val':
                self.df = pd.read_csv(config.VAL_DF)
                self.image_path = 'val'

        if types == 'test':
            self.df = pd.read_csv(config.TEST_DF)
            self.image_path = '/aif/Dataset/test'

        if train_mode:
            self.label = self.df[config.INFO['label']].values
        self.images = self.df[config.INFO['path']].values

        self.crop = crop
        if crop:
            self.xmins = self.df['BBox_xmin'].values
            self.ymins = self.df['BBox_ymin'].values
            self.xmaxs = self.df['BBox_xmax'].values
            self.ymaxs = self.df['BBox_ymax'].values

        self.config = config
        self.train_mode = train_mode
        self.transform = transform
        self.mixup_tranasform = BaseAug(config.RESIZE)
        self.types = types
        self.mixup = mixup
    def __len__(self):
        return (len(self.df))
    
    def __getitem__(self, idx):
        #load image
        if self.train_mode == 'test':
            image = cv2.imread(os.path.join(self.image_path, self.images[idx]))
        else:
            image = cv2.imread(os.path.join(self.config.DATA_PATH, self.image_path, self.images[idx]))
        if self.config.IMG_TYPE == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.config.IMG_TYPE =='HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.config.IMG_TYPE == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        #crop
        if self.crop:
            xy_minmax = (self.xmins[idx], self.ymins[idx], self.xmaxs[idx], self.ymaxs[idx])
        else:
            xy_minmax = None

        #transform setting
        if self.transform is not None:
            if self.mixup:
                origin_image = self.mixup_tranasform(image, xy_minmax)
            else:
                origin_image = []

            # if class_aug
            if self.types == 'train' and self.config.CLASS_AUG :
                image = self.transform(image, self.label[idx], xy_minmax)
            else:
                image = self.transform(image, xy_minmax)

        if self.train_mode:
            label = self.label[idx]
            return (image, label, origin_image)
        return (image)
