import os
import torch
import json
import pandas as pd
from datetime import datetime
from pprint import pprint

class MAEConfig():
    def __init__(self,
                 fine_tune=False,
                 base_path:str='./',
                 data_path_name:str='../data',
                 train_csv:str='mae_train.csv',
                 seed:int=42,
                 encoder:str='fastvit_t8.apple_in1k',
                 epochs:int=50,
                 num_workers:int = 2,
                 train_batch_size:int = 16,
                 val_batch_size:int = 128,
                 lr:float = 0.0001,
                 criterion:str = 'MSELoss',
                 optimizer:str = 'Adam',
                 resize:int = 224,
                 per_iter:float = 0.3,
                 save_path:str = 'save',
                 num_heads:int = 8,
                 hidden_dim:int = 512,
                 num_layer:int = 8,
                 deep_head:bool=True,
                 ):
        self.BASE_PATH = base_path
        self.DATA_PATH = os.path.join(base_path, data_path_name)

        self.TRAIN_DF = os.path.join(self.DATA_PATH, train_csv)

        self.SEED = seed
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ENCODER = encoder
        self.EPOCHS = epochs
        self.NUM_WORKERS = num_workers
        self.TRAIN_BATCH_SIZE = train_batch_size
        self.VAL_BATCH_SIZE = val_batch_size
        self.LR = lr
        self.RESIZE = resize
        if not fine_tune:
            self.TRAIN_METRICS_ITER = int((len(pd.read_csv(self.TRAIN_DF)) // self.TRAIN_BATCH_SIZE) * per_iter)
        self.OPTIMIZER = optimizer
        self.CRITERION = criterion

        #decdoer setting
        self.NUM_HEADS = num_heads
        self.HIDDEN_DIM = hidden_dim
        self.NUM_LAYERS = num_layer
        self.DEEP_HEAD = deep_head

        #save info
        self.SAVE_PATH = save_path

        now = datetime.now()
        self.TIME = now.strftime("%Y_%m_%d_%H_%M")

    def save_to_json(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        config_dict['DEVICE'] = str(self.DEVICE)

        json_save_path = os.path.join(self.SAVE_PATH, 'mae_config', self.ENCODER)
        os.makedirs(json_save_path, exist_ok=True)

        #set name
        json_save_name = os.path.join(json_save_path,self.TIME + '.json' )

        with open(json_save_name, 'w') as json_file:
            json.dump(config_dict, json_file, indent=4)
            
        return (json_save_name)
    
    def print_config(self):
        print("="*100)
        print("CONFIG SETTING !!!")
        print("="*100)
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        pprint(config_dict)
        print("="*100)

