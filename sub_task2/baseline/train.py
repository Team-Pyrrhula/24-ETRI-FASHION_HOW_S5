import logging
# 모든 로깅 출력을 비활성화
logging.disable(logging.CRITICAL)

from utils import parser_arguments, model_save, save2img
from config import BaseConfig
from transform import BaseAug, CustomAug
from dataset import ETRI_Dataset_color
from model import ETRI_model_color
from loose import create_criterion
from optimizer import create_optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
from importlib import import_module
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
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

            out = model(imgs)
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
        val_acc, val_f1 = [], []

        model.eval()
        with torch.no_grad():
            for imgs, label in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
                imgs, label = imgs.to(config.DEVICE), label.to(config.DEVICE)
                
                outs = model(imgs)
                preds = torch.argmax(F.softmax(outs, dim=1), dim=1)
                val_acc.append(accuracy_score(label.cpu().numpy(), preds.cpu().numpy()))
                val_f1.append(f1_score(label.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1))

                loss = criterion(outs, label)
                val_loss += loss.item()

            epoch_val_loss = val_loss / len(val_loader)

            metrics = {
            'val_loss' : epoch_val_loss,
            'val_acc' : np.mean(val_acc),
            'val_f1' : np.mean(val_f1),
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

def main():
    args = parser_arguments()
    
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
        val_metric=args.val_metric
    )
    config.save_to_json()
    config.print_config()

    if (args.wandb):
        #project name & run name setting
        run_name = f'{config.MODEL}_{config.TIME}'
        wandb.init(project=args.project_name, name=run_name)

        #wandb config save
        wandb_config = {key: value for key, value in config.__dict__.items() if not key.startswith('_')}
        wandb.config.update(wandb_config)

    train_transform = CustomAug()
    val_transform = BaseAug()

    train_dataset = ETRI_Dataset_color(config=config, train_mode=True, transform=train_transform, types='train')
    val_dataset = ETRI_Dataset_color(config=config, train_mode=True, transform=val_transform, types='val')

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = ETRI_model_color(config).to(config.DEVICE)

    criterion = create_criterion(config.CRITERION).to(config.DEVICE)
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