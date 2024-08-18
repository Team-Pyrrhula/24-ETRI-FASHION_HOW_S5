import cv2
import pandas as pd
import os
from torch.utils.data import Dataset

class MAE_Dataset(Dataset):
    def __init__(self,
                 df=pd.DataFrame,
                 config=None,
                 transform=None,
                 types:str='train'
                 ):
        self.df = df   
        self.img_path = df['image_name'].values
        self.transform = transform
        self.config = config

    def __len__(self):
        return (len(self.df))
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.config.DATA_PATH, self.img_path[idx])
        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        return (image)