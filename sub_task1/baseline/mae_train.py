import logging
# 모든 로깅 출력을 비활성화
logging.disable(logging.CRITICAL)

from model import MAE_Model
from config import MAEConfig
from transform import BaseAug
from dataset import MAE_Dataset
from utils import seed_everything, mae_parser_arguments, save2img
from torch.utils.data import DataLoader
from loose import create_criterion
from optimizer import create_optimizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import wandb
import pandas as pd
import torch
import os

def train():
    args = mae_parser_arguments()

    config = MAEConfig(
        base_path=args.base_path,
        seed=args.seed,
        encoder=args.encoder,
        epochs=args.epochs,
        num_workers=args.num_workers,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        resize=args.resize,
        criterion=args.criterion,
        optimizer=args.optimizer,
        per_iter=args.per_iter,
        save_path=args.save_path,
    )
    config.save_to_json()
    config.print_config()

    #fix seed
    seed_everything(config.SEED)

    if (args.wandb):
        #project name & run name setting
        run_name = f'{config.ENCODER}_{config.TIME}'
        wandb.init(project=args.project_name, name=run_name)

        #wandb config save
        wandb_config = {key: value for key, value in config.__dict__.items() if not key.startswith('_')}
        wandb.config.update(wandb_config)

    train_transform = BaseAug()
    val_transform = BaseAug()

    df = pd.read_csv(config.TRAIN_DF)
    ## split val df 
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=config.SEED)

    train_dataset = MAE_Dataset(train_df, config, train_transform, types='train')
    val_dataset = MAE_Dataset(val_df, config, val_transform, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = MAE_Model(config).to(config.DEVICE)
    criterion = create_criterion(config.CRITERION).to(config.DEVICE)
    optimizer = create_optimizer(
                config.OPTIMIZER,
                params = model.parameters(),
                lr = config.LR
                )
    ## train loops
    epochs = config.EPOCHS
    best_val_metric = 999
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        count = 0

        for imgs in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            imgs = imgs.to(config.DEVICE)
            optimizer.zero_grad()

            re_imgs = model(imgs)
            loss = criterion(re_imgs, imgs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1

            ## logging
            if count % config.TRAIN_METRICS_ITER == 0:
                print(f'TRAIN LOSSES : {train_loss / (count * config.TRAIN_BATCH_SIZE):.4f}')

        epoch_train_loss = train_loss / len(train_dataloader)
        print(f'EPOCH[{epoch+1}] MEAN LOSS : {epoch_train_loss:.4f}')

        #val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs in tqdm(val_dataloader,desc=f'Epoch {epoch + 1}/{epochs}'):
                imgs = imgs. to(config.DEVICE)

                val_imgs = model(imgs)
                loss = criterion(val_imgs, imgs)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_dataloader)
        print(f'EPOCH[{epoch+1}] VAL MEAN LOSS : {epoch_val_loss:.4f}')

        #save best model
        if (epoch_val_loss < best_val_metric):
            save_path = os.path.join(config.SAVE_PATH, 'mae',  config.ENCODER, config.TIME)
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_{epoch_val_loss:.4f}.pt'))
            best_val_metric = epoch_val_loss
            
        if args.wandb:
            wandb.log(
                {
                    'Train Mean Loss' : epoch_train_loss,
                    'Val Mean Loss' : epoch_val_loss,
                }
            )
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    train()