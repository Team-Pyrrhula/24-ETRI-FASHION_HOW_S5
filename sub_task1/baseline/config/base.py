import torch
import os
import json
import pandas as pd
from datetime import datetime
from pprint import pprint

class BaseConfig():
    def __init__(self,
                info:dict={
                    'path': 'image_name',
                    'label_1_num': 6,
                    'label_2_num': 5,
                    'label_3_num': 3,
                    'label_1' : 'Daily',
                    'label_2' : 'Gender',
                    'label_3' : 'Embellishment'
                },
                base_path:str='./',
                data_path_name:str='../data',
                train_csv:str='Fashion-How24_sub1_train.csv',
                val_csv:str='Fashion-How24_sub1_val.csv',
                seed:int=42,
                model:str='mobilenetv4_conv_small.e2400_r224_in1k',
                epochs:int=50,
                num_workers:int = 2,
                train_batch_size:int = 16,
                val_batch_size:int = 128,
                lr:float = 0.0001,
                criterion:str = 'CrossEntropyLoss',
                optimizer:str = 'Adam',
                scheduler:str = 'StepLR',
                resize:int = 224,
                per_iter:float = 0.3,
                val_metric:str = 'acc',
                save_path:str = 'save',
                model_save_type:str = 'origin',
                train_sampler:bool = False,
                val_sampler:bool = False,
                sampler_type:str='p',
                deep_head:bool=True,
                weight_sampler:bool=False,
                weight_loss:list=[1.0, 1.0, 1.0],
                crop:bool=False,
                ):
        # info
        self.BASE_PATH = base_path
        self.DATA_PATH = os.path.join(base_path, data_path_name)

        self.TRAIN_DF = os.path.join(self.DATA_PATH, train_csv)
        self.VAL_DF = os.path.join(self.DATA_PATH, val_csv)

        self.INFO = info

        #train setting
        self.SEED = seed
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL = model
        self.EPOCHS = epochs
        self.NUM_WORKERS = num_workers
        self.TRAIN_BATCH_SIZE = train_batch_size
        self.VAL_BATCH_SIZE = val_batch_size
        self.LR = lr
        self.RESIZE = resize
        self.TRAIN_METRICS_ITER = int((len(pd.read_csv(self.TRAIN_DF)) // self.TRAIN_BATCH_SIZE) * per_iter)
        self.VAL_METRICS_ITER = int((len(pd.read_csv(self.VAL_DF)) // self.VAL_BATCH_SIZE) * per_iter)
        self.OPTIMIZER = optimizer
        self.CRITERION = criterion
        self.SCHEDULER = scheduler
        self.SCHEDULER_STEP_SIZE = 5
        self.SCHEDULER_GAMMA = 0.1 
        self.VAL_METRIC = val_metric

        #mae 
        self.DEEP_HEAD = deep_head

        #sampler
        self.TRAIN_SAMPLER = train_sampler
        self.VAL_SAMPLER = val_sampler
        self.SAMPLER_TYPE = sampler_type
        self.WEIGHT_SAMPLER =weight_sampler

        #loss weight
        self.WEIGHT_LOSS = weight_loss

        #save info
        self.SAVE_PATH = save_path
        self.MODEL_SAVE_TYPE = model_save_type

        self.CROP = crop

        now = datetime.now()
        self.TIME = now.strftime("%Y_%m_%d_%H_%M")

    def save_to_json(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        config_dict['DEVICE'] = str(self.DEVICE)

        #path check
        json_save_path = os.path.join(self.SAVE_PATH, 'config', self.MODEL)
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

class Inference_BaseConfig():
    def __init__(self,
            info:dict={
                'path': 'image_name',
                'label_1_num': 6,
                'label_2_num': 5,
                'label_3_num': 3,
                'label_1' : 'Daily',
                'label_2' : 'Gender',
                'label_3' : 'Embellishment'
            },
            base_path:str='./',
            data_path_name:str='../data',
            model:str='mobilenetv4_conv_small.e2400_r224_in1k',
            model_path:str='./save/models',
            save_path:str='./save',
            resize:int=224,
            deep_head:bool=True
            ):
        
        self.INFO = info
        self.BASE_PATH = base_path
        self.DATA_PATH = os.path.join(base_path, data_path_name)

        self.TEST_DF = '/aif/Dataset/Fashion-How24_sub1_test.csv'
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL = model
        self.MODEL_PATH = model_path
        self.SAVE_PATH = save_path
        self.RESIZE = resize

        self.DEEP_HEAD = deep_head

        now = datetime.now()
        self.TIME = now.strftime("%Y_%m_%d_%H_%M")