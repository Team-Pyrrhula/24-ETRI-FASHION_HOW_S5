from utils import model_save, save2img
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from pprint import pprint
from pprint import pprint
import os

label_decoder = {
    'l1' : 'daily',
    'l2' : 'gender',
    'l3' : 'embel',
}

def train_run(
        model,
        train_loader,
        val_dataloader_dict,
        criterion,
        optimizer,
        scheduler,
        config,
        args,
        wandb
        ):
    
    epochs = config.EPOCHS
    best_log = {
        'best_val_metric' : 0,
        'best_epoch' : 0,
    }
    print("+++ TRAIN START +++")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        daily_loss = 0.0
        gender_loss = 0.0
        embel_loss = 0.0

        count = 0 
        train_true = {
            'l1':[],
            'l2':[],
            'l3':[],
        }
        train_pred = {
            'l1':[],
            'l2':[],
            'l3':[],
        }

        for imgs, l1, l2, l3 in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            imgs, l1, l2, l3 = imgs.to(config.DEVICE), l1.to(config.DEVICE), l2.to(config.DEVICE), l3.to(config.DEVICE)
            optimizer.zero_grad()

            if args.save_img:
                save2img(imgs.cpu(), epoch, save_path=config.SAVE_PATH)

            out_daily, out_gender, out_embel = model(imgs)
            loss_daily = criterion(out_daily, l1)
            loss_gender = criterion(out_gender, l2)
            loss_embel = criterion(out_embel, l3)
            loss = (loss_daily * config.WEIGHT_LOSS[0]) + (loss_gender * config.WEIGHT_LOSS[1]) + (loss_embel * config.WEIGHT_LOSS[2])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            daily_loss += loss_daily.item()
            gender_loss += loss_gender.item()
            embel_loss += loss_embel.item()

            l1_preds = torch.argmax(F.softmax(out_daily, dim=1), dim=1)
            l2_preds = torch.argmax(F.softmax(out_gender, dim=1), dim=1)
            l3_preds = torch.argmax(F.softmax(out_embel, dim=1), dim=1)
            train_pred['l1'].extend(l1_preds.cpu().numpy())
            train_pred['l2'].extend(l2_preds.cpu().numpy())
            train_pred['l3'].extend(l3_preds.cpu().numpy())
            train_true['l1'].extend(l1.cpu().numpy())
            train_true['l2'].extend(l2.cpu().numpy())
            train_true['l3'].extend(l3.cpu().numpy())

            count += 1

            ## logging
            if count % config.TRAIN_METRICS_ITER == 0:
                print(f'TRAIN LOSSES : {train_loss / (count * config.TRAIN_BATCH_SIZE):.4f}')
        #calculate confusion matrix
        for key in ['l1', 'l2', 'l3']:
            t_cm = confusion_matrix(train_true[key], train_pred[key])
            t_acsa =[]
            for i in range(t_cm.shape[0]):
                denominator = np.sum(t_cm[i, :])

                if denominator == 0:
                    accuracy = 0  
                else:
                    accuracy = t_cm[i, i] / denominator

                classes = label_decoder[key]
                t_acsa.append(accuracy)
                print(f'TRAIN Accuracy for Class {classes}_{i}: {accuracy:.2f}')

        epoch_loss = train_loss / len(train_loader)
        epoch_daily_loss = daily_loss / len(train_loader)
        epoch_gender_loss = gender_loss / len(train_loader)
        epoch_embel_loss = embel_loss / len(train_loader)

        print(f'EPOCH[{epoch+1}] MEAN LOSS : {epoch_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN DAILY LOSS : {epoch_daily_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN GENDER LOSS : {epoch_gender_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN embel LOSS : {epoch_embel_loss:.4f}')

        #wandb logging save
        if args.wandb:
            wandb.log({
                "Train Mean Loss" : epoch_loss,
                "Train Mean Daily Loss" : epoch_daily_loss,
                "Train Mean Gender Loss" : epoch_gender_loss,
                "Train Mean Embel Loss" : epoch_embel_loss,
            })

        scheduler.step() # update scheduler
        #val
        ## 1개라면 일반 val
        if len(val_dataloader_dict) == 1:
            metrics, best_log = val_run(model, val_dataloader_dict['all'], criterion, config, epoch, best_log)
        ## 1개보다 많다면 sampler_val
        else:
            metrics, best_log = sampler_val_run(model, val_dataloader_dict, criterion, config, epoch, best_log)
        #val score wnadb_log

        if args.wandb:
            wandb.log(metrics)

    return (best_log)

def sampler_train_run(
        model,
        train_dataloader_dict,
        val_dataloader_dict,
        criterion,
        optimizer,
        scheduler,
        config,
        args,
        wandb):
    
    epochs = config.EPOCHS
    best_log = {
        'best_val_metric' : 0,
        'best_epoch' : 0,
    }
    print("+++ SAMPLER TRAIN START +++")
    for epoch in range(epochs):
        model.train()
        
        losses = {
            'total': 0.0,
            'daily': 0.0,
            'gender': 0.0,
            'embel': 0.0,
        }
        lens = 0
        for key, train_loader in train_dataloader_dict.items():
            lens += len(train_loader)
            print(f'++++ {key}_train ing ... ++++')
            for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

                optimizer.zero_grad()
                if key == 'daily':
                    out, _, _ = model(imgs)
                elif key == 'gender':
                    _, out, _ = model(imgs)
                else:
                    _, _, out = model(imgs)

                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                losses['total'] += loss.item()
                losses[key] += loss.item()
        
        epoch_loss = losses['total'] / lens
        epoch_daily_loss = losses['daily'] / len(train_dataloader_dict['daily'])
        epoch_gender_loss = losses['gender'] / len(train_dataloader_dict['gender'])
        epoch_embel_loss = losses['embel'] / len(train_dataloader_dict['embel'])

        print(f'EPOCH[{epoch+1}] MEAN LOSS : {epoch_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN DAILY LOSS : {epoch_daily_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN GENDER LOSS : {epoch_gender_loss:.4f}')
        print(f'EPOCH[{epoch+1}] MEAN EMBEL LOSS : {epoch_embel_loss:.4f}')

        if args.wandb:
            wandb.log({
                "Train Mean Loss" : epoch_loss,
                "Train Mean Daily Loss" : epoch_daily_loss,
                "Train Mean Gender Loss" : epoch_gender_loss,
                "Train Mean Embel Loss" : epoch_embel_loss,
            })
        scheduler.step()

        ## 1개라면 일반 val
        if len(val_dataloader_dict) == 1:
            metrics, best_log = val_run(model, val_dataloader_dict['all'], criterion, config, epoch, best_log)
        ## 1개보다 많다면 sampler_val
        else:
            metrics, best_log = sampler_val_run(model, val_dataloader_dict, criterion, config, epoch, best_log)
        #val score wnadb_log

        if args.wandb:
            wandb.log(metrics)
    return (best_log)

def val_run(model,
            val_loader,
            criterion,
            config,
            epoch,
            best_log,
        ):
    print(f"+++ [EPOCH : {epoch + 1}] - VAL START +++")

    losses = {
        'val_loss':0,
        'val_daily_loss':0.0,
        'val_gender_loss':0.0,
        'val_embel_loss':0.0,
    }
    val_true, val_pred = {'l1':[], 'l2':[], 'l3':[]}, {'l1':[], 'l2':[], 'l3':[]}

    model.eval()
    with torch.no_grad():
        for imgs, l1, l2, l3 in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}', leave=False):
            imgs, l1, l2, l3 = imgs.to(config.DEVICE), l1.to(config.DEVICE), l2.to(config.DEVICE), l3.to(config.DEVICE)
            
            out_daily, out_gender, out_embel = model(imgs)

            daily_pred = torch.argmax(F.softmax(out_daily, dim=1), dim=1)
            gender_pred = torch.argmax(F.softmax(out_gender, dim=1), dim=1)
            embel_pred = torch.argmax(F.softmax(out_embel, dim=1), dim=1)

            #extend pred & label
            val_pred['l1'].extend(daily_pred.cpu().numpy())
            val_pred['l2'].extend(gender_pred.cpu().numpy())
            val_pred['l3'].extend(embel_pred.cpu().numpy())
            val_true['l1'].extend(l1.cpu().numpy())
            val_true['l2'].extend(l2.cpu().numpy())
            val_true['l3'].extend(l3.cpu().numpy())

            loss_daily = criterion(out_daily, l1)
            loss_gender = criterion(out_gender, l2)
            loss_embel = criterion(out_embel, l3)
            loss = loss_daily + loss_gender + loss_embel

            losses['val_loss'] += loss.item()
            losses['val_daily_loss'] += loss_daily.item()
            losses['val_gender_loss'] += loss_gender.item()
            losses['val_embel_loss'] += loss_embel.item()

        epoch_val_loss = losses['val_loss'] / len(val_loader)
        epoch_val_daily_loss = losses['val_daily_loss'] / len(val_loader)
        epoch_val_gender_loss = losses['val_gender_loss'] / len(val_loader)
        epoch_val_embel_loss = losses['val_embel_loss'] / len(val_loader)

        metrics = {
        'val_loss' : epoch_val_loss,
        'val_daily_acc' : accuracy_score(val_true['l1'], val_pred['l1']),
        'val_gender_acc': accuracy_score(val_true['l2'], val_pred['l2']),
        'val_embel_acc': accuracy_score(val_true['l3'], val_pred['l3']),
        'val_daily_f1' : f1_score(val_true['l1'], val_pred['l1'], average='macro', zero_division=1),
        'val_gender_f1' : f1_score(val_true['l2'], val_pred['l2'], average='macro', zero_division=1),
        'val_embel_f1' : f1_score(val_true['l3'], val_pred['l3'], average='macro', zero_division=1),
        'val_daily_loss' : epoch_val_daily_loss,
        'val_gender_loss' : epoch_val_gender_loss,
        'val_embel_loss' : epoch_val_embel_loss,
        }

        print("+++++ VAL CM +++++")
        for key in ['l1', 'l2', 'l3']:
            cm = confusion_matrix(val_true[key], val_pred[key])
            acsa = []
            for i in range(cm.shape[0]):
                denominator = np.sum(cm[i, :])
                if denominator == 0:
                    accuracy = 0  
                else:
                    accuracy = cm[i, i] / denominator
                acsa.append(accuracy)
                classes = label_decoder[key]
                print(f'Accuracy for Class {classes}_{i} : {accuracy:.2f}')

        print('++++++ VAL METRICS ++++++')
        pprint(metrics)
        val_acc_mean = (metrics['val_daily_acc'] + metrics['val_gender_acc'] + metrics['val_embel_acc']) / 3
        val_f1_mean =  (metrics['val_daily_f1'] + metrics['val_gender_f1'] + metrics['val_embel_f1']) / 3
        
        print(f'Val F1 mean : {val_f1_mean}')
        print(f'Val ACC mean : {val_acc_mean}')

        # save val metric setting
        if (config.VAL_METRIC == 'f1'):
            val_metric = val_f1_mean
        elif (config.VAL_METRIC == 'acc'):
            val_metric = val_acc_mean

        #append acc, f1 score
        metrics['val_metric_f1'] = val_f1_mean
        metrics['val_metric_acc'] = val_acc_mean

        print("+"*100)
        #save model by val metrics
        save_path = os.path.join(config.SAVE_PATH, 'model', config.MODEL, config.TIME)
        os.makedirs(save_path, exist_ok=True)
        model_save(config, save_path, model, epoch, val_metric)

        #best logging
        if (val_metric  > best_log['best_val_metric']):
            best_log['best_val_metric'] = val_metric
            best_log['best_epoch'] = epoch

        #save print
        print(f'Save Model {val_metric:.4f} in : {save_path}')

    return (metrics, best_log)

def sampler_val_run(model,
                    val_dataloader_dict,
                    criterion,
                    config,
                    epoch,
                    best_log):
    
    print(f"+++ [EPOCH : {epoch + 1}] - VAL START +++")
    losses = {
        'total':0.0,
        'daily':0.0,
        'gender':0.0,
        'embel':0.0,
    }
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
    lens = 0
    model.eval()
    with torch.no_grad():
        for key, val_loader in val_dataloader_dict.items():
            lens += len(val_loader)
            for imgs, labels in tqdm(val_loader, desc=f'{key}_validation_loops', leave=False):
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

                if key == 'daily':
                    out, _, _ = model(imgs)
                elif key == 'gender':
                    _, out, _ = model(imgs)
                else:
                    _, _, out = model(imgs)

                pred = torch.argmax(F.softmax(out, dim=1), dim=1)
                
                val_acc[key].append(accuracy_score(labels.cpu().numpy(), pred.cpu().numpy()))
                val_f1[key].append(f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=1))

                loss = criterion(out, labels)
                losses[key] += loss.item()
                losses['total'] += loss.item()

    #cal score
    epoch_val_loss = losses['total'] / lens
    epoch_daily_loss = losses['daily'] / len(val_dataloader_dict['daily'])
    epoch_gender_loss = losses['gender'] / len(val_dataloader_dict['gender'])
    epoch_embel_loss = losses['embel'] / len(val_dataloader_dict['embel'])

    metrics = {
        'val_loss' : epoch_val_loss,
        'val_daily_acc' : np.mean(val_acc['daily']),
        'val_gender_acc': np.mean(val_acc['gender']),
        'val_embel_acc': np.mean(val_acc['embel']),
        'val_daily_f1' : np.mean(val_f1['daily']),
        'val_gender_f1' : np.mean(val_f1['gender']),
        'val_embel_f1' : np.mean(val_f1['embel']),
        'val_daily_loss' : epoch_daily_loss,
        'val_gender_loss' : epoch_gender_loss,
        'val_embel_loss' : epoch_embel_loss,
        }
    
    print('++++++ VAL METRICS ++++++')
    pprint(metrics)
    val_acc_mean = (metrics['val_daily_acc'] + metrics['val_gender_acc'] + metrics['val_embel_acc']) / 3
    val_f1_mean =  (metrics['val_daily_f1'] + metrics['val_gender_f1'] + metrics['val_embel_f1']) / 3
    
    print(f'Val F1 mean : {val_f1_mean}')
    print(f'Val ACC mean : {val_acc_mean}')

    # save val metric setting
    if (config.VAL_METRIC == 'f1'):
        val_metric = val_f1_mean
    elif (config.VAL_METRIC == 'acc'):
        val_metric = val_acc_mean

    #append acc, f1 score
    metrics['val_metric_f1'] = val_f1_mean
    metrics['val_metric_acc'] = val_acc_mean

    print("+"*100)
    #save model by val metrics
    save_path = os.path.join(config.SAVE_PATH, 'model', config.MODEL, config.TIME)
    os.makedirs(save_path, exist_ok=True)
    model_save(config, save_path, model, epoch, val_metric)

    #best logging
    if (val_metric  > best_log['best_val_metric']):
        best_log['best_val_metric'] = val_metric
        best_log['best_epoch'] = epoch

    #save print
    print(f'Save Model {val_metric:.4f} in : {save_path}')

    return (metrics, best_log)