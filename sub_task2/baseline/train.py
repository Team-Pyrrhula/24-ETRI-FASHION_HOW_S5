import logging
# 모든 로깅 출력을 비활성화
logging.disable(logging.CRITICAL)

from utils import parser_arguments, model_save, save2img, seed_everything, calculate_mean_std
from config import BaseConfig
from transform import BaseAug, CustomAug, NewCustomAug
from dataset import ETRI_Dataset_color
from model import ETRI_model_color
from loose import create_criterion
from optimizer import create_optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
from importlib import import_module
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from pprint import pprint
import os
import wandb

def train_run(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        args
        ):
    
    label_decoder = {
                    0:'Red',
                    1:'Coral',
                    2:'Orange',
                    3:'Pink',
                    4:'Purple',
                    5:'Brown',
                    6:'Beige',
                    7:'Ivory',
                    8:'Yellow',
                    9:'Mustard',
                    10:'Skyblue',
                    11:'Royalblue',
                    12:'Navy',
                    13:'Green',
                    14:'Khaki',
                    15:'White',
                    16:'Gray',
                    17:'Black',
    }

    epochs = config.EPOCHS
    best_val_metric = 0

    print("+++ TRAIN START +++")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        count = 0 

        for imgs, label in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            imgs, label = imgs.to(config.DEVICE), label.to(config.DEVICE)
            optimizer.zero_grad()

            if args.save_img:
                save2img(imgs.cpu(), epoch, save_path=config.SAVE_PATH)

            out = model(imgs.float())
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1

            ## logging
            if count % config.TRAIN_METRICS_ITER == 0:
                print(f'TRAIN LOSSES : {train_loss / (count * config.TRAIN_BATCH_SIZE):.4f}')

        epoch_loss = train_loss / len(train_loader)
        print(f'EPOCH[{epoch+1}] MEAN LOSS : {epoch_loss:.4f}')

        #wandb logging save
        if args.wandb:
            wandb.log({
                "Train Mean Loss" : epoch_loss,
            })

        scheduler.step() # update scheduler

        #val
        print(f"+++ [EPOCH : {epoch + 1}] - VAL START +++")
        val_loss = 0
        val_true, val_pred = [], []
        model.eval()
        if args.model_half:
            model.half()
        with torch.no_grad():
            for imgs, label in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
                imgs, label = imgs.to(config.DEVICE), label.to(config.DEVICE)
                if args.model_half:
                    outs = model(imgs.half())
                else:
                    outs = model(imgs.float())
                preds = torch.argmax(F.softmax(outs, dim=1), dim=1)

                val_true.extend(label.cpu().numpy())
                val_pred.extend(preds.cpu().numpy())

                loss = criterion(outs, label)
                val_loss += loss.item()

            epoch_val_loss = val_loss / len(val_loader)

            #confusion matrix
            cm = confusion_matrix(val_true, val_pred)
            acsa = []
            for i in range(cm.shape[0]):    
                if i >= 18:
                    continue
                # 각 클래스의 TP와 전체 샘플 수를 사용하여 정확도 계산
                accuracy = cm[i, i] / np.sum(cm[i, :])
                classes = label_decoder[i]
                acsa.append(accuracy)
                print(f'Accuracy for Class {classes}: {accuracy:.2f}')

            metrics = {
            'val_loss' : epoch_val_loss,
            'val_acc' : accuracy_score(val_true, val_pred),
            'val_f1' : f1_score(val_true, val_pred, average='macro', zero_division=1),
            'val_ACSA': np.mean(acsa)
            }
                
            # wandb logging
            if args.wandb:
                wandb.log(metrics)

            print('++++++ VAL METRICS ++++++')
            pprint(metrics)

            # save val metric setting
            if (config.VAL_METRIC == 'f1'):
                val_metric = metrics['val_f1']
            elif (config.VAL_METRIC == 'acc'):
                val_metric = metrics['val_acc']
            print("+"*100)

            #save model by val metrics
            save_path = os.path.join(config.SAVE_PATH, 'model', config.MODEL, config.TIME)
            os.makedirs(save_path, exist_ok=True)
            # if (val_metric  > best_val_metric):
            model_save(config, save_path, model, epoch, val_metric)
            print(f'Save Model {val_metric:.4f} in : {save_path}')
            best_val_metric = val_metric
        if args.model_half:
            model.float()

def main():
    args = parser_arguments()

    # config setting
    config = BaseConfig(
        base_path=args.base_path,
        train_csv=args.train_df,
        val_csv=args.val_df,
        seed=args.seed,
        model=args.model,
        pretrain=bool(args.pretrain),
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
        crop=bool(args.crop),
        remgb=bool(args.remgb),
        sampler=bool(args.sampler),
        img_type=args.img_type,
    )
    seed_everything(config.SEED)

    if (args.model_half):
        config.MODEL_HALF = True

    if (args.wandb):
        #project name & run name setting
        run_name = f'{config.MODEL}_{config.TIME}'
        wandb.init(project=args.project_name, name=run_name)

        #wandb config save
        wandb_config = {key: value for key, value in config.__dict__.items() if not key.startswith('_')}
        wandb.config.update(wandb_config)

    if args.mean_std:
        train_mean, train_std = calculate_mean_std(config, config.IMG_TYPE)
        print(f'train mean : {train_mean}\ntrain std : {train_std}')
    else:
        #for only RGB -> ImageNet dataset mean, std
        train_mean, train_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    config.TRAIN_MEAN = train_mean
    config.TRAIN_STD = train_std

    if args.custom_aug:
        train_transform = NewCustomAug()
        config.CUSTOM_AUG = True
    else:
        train_transform = BaseAug(mean=train_mean, std=train_std)
        config.CUSTOM_AUG = False
        
    val_transform = BaseAug(mean=train_mean, std=train_std)

    train_dataset = ETRI_Dataset_color(config=config, train_mode=True, transform=train_transform, types='train', remgb=config.REMGB, crop=config.CROP)
    val_dataset = ETRI_Dataset_color(config=config, train_mode=True, transform=val_transform, types='val', remgb=False, crop=False)

    if args.sampler:
        class_counts = train_dataset.df['Color'].value_counts().sort_index().values
        class_weights = 1. / class_counts

        sample_weights = class_weights[train_dataset.df['Color'].values]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        print(f'Sampler weight : {sample_weights}')
    else:
        sampler = None
    
    class_weights = [
        0.0435,  # Red
        0.0988,  # Coral
        0.1075,  # Orange
        0.0494,  # Pink
        0.0653,  # Purple
        0.0677,  # Brown
        0.0761,  # Beige
        0.0690,  # Ivory
        0.0609,  # Yellow
        0.0988,  # Mustard
        0.0475,  # Skyblue
        0.0571,  # Royalblue
        0.0571,  # Navy
        0.0487,  # Green
        0.0589,  # Khaki
        0.0537,  # White
        0.0463,  # Gray
        0.0481   # Black
    ] 
    """
    가장 좋았던 모델의 정확성의 역수를 취했음
    가중치를 계산하는 일반적인 방법은 각 클래스의 정확도의 역수를 사용하는 것입니다. 이렇게 하면 정확도가 낮은 클래스에 더 높은 가중치가 부여됩니다. 구체적인 단계는 다음과 같습니다:

    각 클래스의 정확도의 역수를 계산합니다.
    모든 역수의 합을 구합니다.
    각 클래스의 가중치를 정규화하기 위해, 각 역수를 총합으로 나눕니다.
    """

    # class_distributions_dict = dict(sorted(train_dataset.df['Color'].value_counts().to_dict().items()))
    # class_distributions = list( v for k,v in class_distributions_dict.items())
    # total_samples = sum(class_distributions)
    
    # class_weights = [total_samples / (len(class_distributions) * c) for c in class_distributions]
    # #if model output size is 19
    class_weights.append(0.00000001)
    # print(class_weights)
    config.CLASS_WEIGHT = class_weights
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
    
    config.save_to_json()
    config.print_config()

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = ETRI_model_color(config).to(config.DEVICE)

    criterion = create_criterion(config.CRITERION, weight=class_weights).to(config.DEVICE)
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
    
    train_run(model, train_loader, val_loader, criterion, optimizer, scheduler, config, args)
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()