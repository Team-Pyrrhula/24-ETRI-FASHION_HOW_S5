import torch
import os

def model_save(config, save_path, model, epoch, val_metric):
    if config.MODEL_SAVE_TYPE == 'script':
        m = torch.jit.script(model)
        torch.jit.save(m, os.path.join(save_path, f'{epoch}_{val_metric:.4f}.pt'))
    elif config.MODEL_SAVE_TYPE == 'origin':
        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_{val_metric:.4f}.pt'))
    