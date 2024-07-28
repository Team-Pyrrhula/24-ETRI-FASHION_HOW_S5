import logging
# 모든 로깅 출력을 비활성화
logging.disable(logging.CRITICAL)

from utils import parser_arguments
from config import BaseConfig
from transform import BaseAug
from dataset import ETRI_Dataset
from model import ETRI_model
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from pprint import pprint
import os

def train_run(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        ):
    
    epochs = config.EPOCHS
    best_val_metric = 0

    print("+++ TRAIN START +++")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        daily_loss = 0.0
        gender_loss = 0.0
        embel_loss = 0.0

        count = 0 

        for imgs, l1, l2, l3 in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            imgs, l1, l2, l3 = imgs.to(config.DEVICE), l1.to(config.DEVICE), l2.to(config.DEVICE), l3.to(config.DEVICE)
            optimizer.zero_grad()

            out_daily, out_gender, out_embel = model(imgs)
            loss_daily = criterion(out_daily, l1)
            loss_gender = criterion(out_gender, l2)
            loss_embel = criterion(out_embel, l3)
            loss = loss_daily + loss_gender + loss_embel

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            daily_loss += loss_daily.item()
            gender_loss += loss_gender.item()
            embel_loss += loss_embel.item()

            count += 1

            ## logging
            if count % config.TRAIN_METRICS_ITER == 0:
                print(f'TRAIN LOSSES : {train_loss / (count * config.TRAIN_BATCH_SIZE):.4f}')

        epoch_loss = train_loss / len(train_loader)
        epoch_daily_loss = daily_loss / len(train_loader)
        epoch_gender_loss = gender_loss / len(train_loader)
        epoch_embel_loss = embel_loss / len(train_loader)

        print(f'EPOCH[{epoch+1}] MEAN LOSS : {epoch_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN DAILY LOSS : {epoch_daily_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN GENDER LOSS : {epoch_gender_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN embel LOSS : {epoch_embel_loss:.4f}')

        scheduler.step() # update scheduler

        #val
        print(f"+++ [EPOCH : {epoch + 1}] - VAL START +++")
        val_loss = 0
        val_daily_loss = 0.0
        val_gender_loss = 0.0
        val_embel_loss = 0.0

        val_acc = {
            'daily' : [],
            'gender' : [],
            'embel' : [],
        }

        val_f1 = {
            'daily' : [],
            'gender' : [],
            'embel' : [],
        }

        model.eval()
        with torch.no_grad():
            for imgs, l1, l2, l3 in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
                imgs, l1, l2, l3 = imgs.to(config.DEVICE), l1.to(config.DEVICE), l2.to(config.DEVICE), l3.to(config.DEVICE)
                
                out_daily, out_gender, out_embel = model(imgs)

                daily_pred = torch.argmax(F.softmax(out_daily, dim=1), dim=1)
                gender_pred = torch.argmax(F.softmax(out_gender, dim=1), dim=1)
                embel_pred = torch.argmax(F.softmax(out_embel, dim=1), dim=1)

                val_acc['daily'].append(accuracy_score(l1.cpu().numpy(), daily_pred.cpu().numpy()))
                val_acc['gender'].append(accuracy_score(l2.cpu().numpy(), gender_pred.cpu().numpy()))
                val_acc['embel'].append(accuracy_score(l3.cpu().numpy(), embel_pred.cpu().numpy()))

                val_f1['daily'].append(f1_score(l1.cpu().numpy(), daily_pred.cpu().numpy(), average='macro', zero_division=1))
                val_f1['gender'].append(f1_score(l2.cpu().numpy(), gender_pred.cpu().numpy(), average='macro', zero_division=1))
                val_f1['embel'].append(f1_score(l3.cpu().numpy(), embel_pred.cpu().numpy(), average='macro', zero_division=1))

                loss_daily = criterion(out_daily, l1)
                loss_gender = criterion(out_gender, l2)
                loss_embel = criterion(out_embel, l3)
                loss = loss_daily + loss_gender + loss_embel

                val_loss += loss.item()
                val_daily_loss += loss_daily.item()
                val_gender_loss += loss_gender.item()
                val_embel_loss += loss_embel.item()

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_daily_loss = val_daily_loss / len(val_loader)
            epoch_val_gender_loss = val_gender_loss / len(val_loader)
            epoch_val_embel_loss = val_embel_loss / len(val_loader)

            metrics = {
            'loss' : epoch_val_loss,
            'daily_acc' : np.mean(val_acc['daily']),
            'gender_acc': np.mean(val_acc['gender']),
            'embel_acc': np.mean(val_acc['embel']),
            'daily_f1' : np.mean(val_f1['daily']),
            'gender_f1' : np.mean(val_f1['gender']),
            'embel_f1' : np.mean(val_f1['embel']),
            'daily_loss' : epoch_val_daily_loss,
            'gender_loss' : epoch_val_gender_loss,
            'embel_loss' : epoch_val_embel_loss,
            }

            print('++++++ VAL METRICS ++++++')
            pprint(metrics)
            val_metric = (metrics['daily_acc'] + metrics['gender_acc'] + metrics['embel_acc']) / 3
            print(f"Val MEAN METRIC : {val_metric}")

            print("+"*100)

            save_path = os.path.join(config.SAVE_PATH, 'model', config.MODEL, config.TIME)
            os.makedirs(save_path, exist_ok=True)
            if (val_metric  > best_val_metric):
                torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_{val_metric:.4f}.pth'))
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
        per_iter=args.per_iter,
        save_path=args.save_path
    )
    config.save_to_json()
    config.print_config()

    transform = BaseAug()
    train_dataset = ETRI_Dataset(config=config, train_mode=True, transform=transform, types='train')
    val_dataset = ETRI_Dataset(config=config, train_mode=True, transform=transform, types='val')

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = ETRI_model(config).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_run(model, train_loader, val_loader, criterion, optimizer, scheduler, config)

if __name__ == '__main__':
    main()