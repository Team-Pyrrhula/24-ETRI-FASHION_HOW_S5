import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

class CustomAugV2:
    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = A.Compose(
            [
                A.Resize(resize, resize),
                A.Normalize(mean=mean, std=std),
                A.OneOf([
                    A.HorizontalFlip(),  # 좌우 반전
                    A.VerticalFlip(), # 상하 반전
                    A.NoOp(), # Nope
                ], p=1.0),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, xy_minmax=None):
        # crop
        if xy_minmax is not None:
            crop = A.OneOf([
                A.Crop(x_min=xy_minmax[0], y_min=xy_minmax[1], x_max=xy_minmax[2], y_max=xy_minmax[3]),
            ], p=0.3)
            image = crop(image=image)["image"]
        return self.transform(image=image)["image"]