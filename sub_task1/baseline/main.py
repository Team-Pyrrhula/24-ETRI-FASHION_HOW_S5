import logging
# 모든 로깅 출력을 비활성화
logging.disable(logging.CRITICAL)

from utils import parser_arguments, seed_everything, etri_sampler, extract_final_conv
from loose import create_criterion
from optimizer import create_optimizer

from config import BaseConfig, MAEConfig
from transform import BaseAug, CustomAug
from dataset import ETRI_Dataset, Sampler_Dataset
from model import ETRI_model, ETRI_MAE_model, MAE_Model
from torch.utils.data import DataLoader, WeightedRandomSampler
from train import train_run, sampler_train_run

import wandb
import pandas as pd
import torch
import timm
import numpy as np
from importlib import import_module

def compute_sample_weights(labels):
    label_counts = np.sum(labels, axis=0)
    class_weights = 1.0 / (label_counts + 1e-5)  # 0으로 나누는 것을 방지
    sample_weights = np.sum(labels * class_weights, axis=1)
    return sample_weights

# 1. 로그 스케일
def calculate_weights_log(accuracies):
    return list(1 / np.log(np.array(accuracies) + 1.01))

# 2. 선형 스케일링
def calculate_weights_linear(accuracies):
    weights = 1 - np.array(accuracies)
    return list(weights / np.sum(weights))

# 3. 최대값 제한
def calculate_weights_capped(accuracies, max_weight=5):
    weights = 1 / (np.array(accuracies) + 1e-5)
    weights = np.minimum(weights, max_weight)
    return list(weights / np.sum(weights))

def main():
    args = parser_arguments()

    # classifier deeper
    if args.mae_head == 'deep':
        deep_head = True
    else:
        deep_head = False
    
    if args.mae_finetune:
        mae_config = MAEConfig(
            fine_tune=True,
            encoder=args.model,
            deep_head=deep_head,
        )
    # config setting
    config = BaseConfig(
        base_path=args.base_path,
        seed=args.seed,
        model=args.model,
        epochs=args.epochs,
        num_workers=args.num_workers,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        resize=args.resize,
        criterion=args.criterion,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        per_iter=args.per_iter,
        save_path=args.save_path,
        model_save_type=args.model_save_type,
        val_metric=args.val_metric,
        train_sampler=args.train_sampler,
        val_sampler=args.val_sampler,
        sampler_type=args.sampler_type,
        deep_head=deep_head,
        weight_sampler=args.weight_sampler,
        weight_loss=[args.daily_weight, args.gender_weight, args.embel_weight]

    )
    #fix seed
    seed_everything(config.SEED)

    # class weight
    if (args.class_weight):

        daily_accuracies = [0.01, 0.77, 0.46, 0.47, 0.40, 0.94]
        gender_accuracies = [0.18, 0.47, 0.50, 0.78, 0.56]
        embel_accuracies = [0.75, 0.48, 0.59]

        if args.weight_type == 'log':
            daily_weight = torch.FloatTensor(calculate_weights_log(daily_accuracies))
            gender_weight = torch.FloatTensor(calculate_weights_log(gender_accuracies))
            embel_weight = torch.FloatTensor(calculate_weights_log(embel_accuracies))
        elif args.weigt_type == 'linear':
            daily_weight = torch.FloatTensor(calculate_weights_linear(daily_accuracies))
            gender_weight = torch.FloatTensor(calculate_weights_linear(gender_accuracies))
            embel_weight = torch.FloatTensor(calculate_weights_linear(embel_accuracies))
        else:
            daily_weight = torch.FloatTensor(calculate_weights_capped(daily_accuracies))
            gender_weight = torch.FloatTensor(calculate_weights_capped(gender_accuracies))
            embel_weight = torch.FloatTensor(calculate_weights_capped(embel_accuracies))

    else:
        daily_weight = None
        gender_weight = None
        embel_weight = None
    
    config.CLASS_WEIGHT = args.class_weight
    config.WEIGHT_TYPE = args.weight_type

    #wandb project logging
    if (args.wandb):
        #project name & run name setting
        run_name = f'{config.MODEL}_{config.TIME}'
        wandb.init(project=args.project_name, name=run_name)

        #wandb config save
        wandb_config = {key: value for key, value in config.__dict__.items() if not key.startswith('_')}
        wandb.config.update(wandb_config)

    train_transform = CustomAug(config.RESIZE)
    val_transform = BaseAug(config.RESIZE)

    #Make dataset (smapler 기준으로)
    if config.TRAIN_SAMPLER:
        train_df = pd.read_csv(config.TRAIN_DF)
        daily_df, gender_df, embel_df = etri_sampler(df=train_df, types=config.SAMPLER_TYPE)
        train_dataset_dict = {
            'daily' : Sampler_Dataset(daily_df, label_type='Daily', config=config, train_mode=True, transform=train_transform, types='train'),
            'gender' : Sampler_Dataset(gender_df, label_type='Gender', config=config, train_mode=True, transform=train_transform, types='train'),
            'embel' : Sampler_Dataset(embel_df, label_type='Embellishment', config=config, train_mode=True, transform=train_transform, types='train'),
        }
    else:
        train_dataset_dict = {
            'all' : ETRI_Dataset(config=config, train_mode=True, transform=train_transform, types='train')
        }

    if config.VAL_SAMPLER:
        val_df = pd.read_csv(config.VAL_DF)
        daily_val_df, gender_val_df, embel_val_df = etri_sampler(df=val_df, types=config.SAMPLER_TYPE)
        val_dataset_dict = {
            'daily' : Sampler_Dataset(daily_val_df, label_type='Daily', config=config, train_mode=True, transform=val_transform, types='val'),
            'gender' : Sampler_Dataset(gender_val_df, label_type='Gender', config=config, train_mode=True, transform=val_transform, types='val'),
            'embel': Sampler_Dataset(embel_val_df, label_type='Embellishment', config=config, train_mode=True, transform=val_transform, types='val'),
        }
    else:
        val_dataset_dict = {
            'all' : ETRI_Dataset(config=config, train_mode=True, transform=val_transform, types='val')
        }

    #weight_sampler
    if not config.TRAIN_SAMPLER and config.WEIGHT_SAMPLER:
        df = pd.read_csv(config.TRAIN_DF)
        labels = df[[config.INFO['label_1'], config.INFO['label_2'], config.INFO['label_3']]].values.tolist()

        weights = compute_sample_weights(labels)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        sampler = None

    config.save_to_json()
    config.print_config()

    #Make Loader
    train_dataloader_dict, val_dataloader_dict = {}, {}
    for key, dataset in train_dataset_dict.items():
        train_dataloader_dict[key] = DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS, sampler=sampler)
    for key, dataset in val_dataset_dict.items():
        val_dataloader_dict[key] = DataLoader(dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)

    if args.mae_finetune:
        mae_model = MAE_Model(mae_config).to(config.DEVICE)

        #load pretrained 
        mae_model.load_state_dict(torch.load(args.mae_pretrined_model_path, map_location=config.DEVICE))
        encoder = mae_model.encoder
        final_conv =  extract_final_conv(config).to(config.DEVICE)

        #remove not need
        del mae_model
        
        model = ETRI_MAE_model(config, encoder, final_conv, freeze=args.mae_freeze).to(config.DEVICE)

    else:
        model = ETRI_model(config).to(config.DEVICE)

    criterion = {
        'daily' : create_criterion(config.CRITERION, weight=daily_weight).to(config.DEVICE),
        'gender' :create_criterion(config.CRITERION, weight=gender_weight).to(config.DEVICE),
        'embel' : create_criterion(config.CRITERION, weight=embel_weight).to(config.DEVICE)
    }
    optimizer = create_optimizer(
                config.OPTIMIZER,
                params = model.parameters(),
                lr = config.LR
                )
    scheduler= getattr(import_module("torch.optim.lr_scheduler"), config.SCHEDULER)
    scheduler = scheduler(
        optimizer,
        step_size=config.SCHEDULER_STEP_SIZE, 
        gamma=config.SCHEDULER_GAMMA,
    )
    
    #sampler train 
    if config.TRAIN_SAMPLER:
        logs = sampler_train_run(model, train_dataloader_dict, val_dataloader_dict, criterion, optimizer, scheduler, config, args, wandb)
    else:
        logs = train_run(model, train_dataloader_dict['all'], val_dataloader_dict, criterion, optimizer, scheduler, config, args, wandb)

    best_epoch, best_metrics = logs['best_epoch'], logs['best_val_metric']
    print(f'BEST SCORE -> {best_epoch} : {best_metrics}')
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()