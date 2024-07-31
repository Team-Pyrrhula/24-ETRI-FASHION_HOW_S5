from model import ETRI_model
from utils import val_parser_arguments
from config import Inference_BaseConfig
from dataset import ETRI_Dataset
from transform import BaseAug

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch

def main():
    args = val_parser_arguments()

    config = Inference_BaseConfig(
        model = args.model,
        model_path=args.model_path,
        save_path=args.save_path,
        resize=args.resize
    )

    transform = BaseAug(resize=config.RESIZE)
    test_dataset = ETRI_Dataset(config, train_mode=False, transform=transform, types='test')
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    model = ETRI_model(config).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH))

    daily_preds = []
    gender_preds = []
    embel_preds = []
    
    model.eval()
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(config.DEVICE)

            out_daily, out_gender, out_embel = model(imgs)
            daily_pred = torch.argmax(F.softmax(out_daily, dim=1), dim=1).tolist()
            gender_pred = torch.argmax(F.softmax(out_gender, dim=1), dim=1).tolist()
            embel_pred = torch.argmax(F.softmax(out_embel, dim=1), dim=1).tolist()

            daily_preds.extend(daily_pred)
            gender_preds.extend(gender_pred)
            embel_preds.extend(embel_pred)

    print(daily_preds)
    print(gender_preds)
    print(embel_preds)


if __name__ == '__main__':
    main()