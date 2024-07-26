from utils import parser_arguments
from config import BaseConfig
from dataset import ETRI_Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

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

    transform =  A.Compose(
            [
                A.Resize(config.RESIZE, config.RESIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    train_dataset = ETRI_Dataset(config=config, train_mode=True, transform=transform, types='train')
    val_dataset = ETRI_Dataset(config=config, train_mode=True, transform=transform, types='val')

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE)

if __name__ == '__main__':
    main()