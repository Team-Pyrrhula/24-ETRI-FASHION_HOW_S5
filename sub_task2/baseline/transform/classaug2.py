"""
 0: 0.87
 1: 0.41
 2: 0.39
 3: 0.72
 4: 0.64
 5: 0.59
 6: 0.50
 7: 0.49
 8: 0.65
 9: 0.30
 10: 0.77
 11: 0.58
 12: 0.65
 13: 0.75
 14: 0.67
 15: 0.67
 16: 0.73
 17: 0.75

 현재 가장 좋은 성능의 모델 class 별 acc 
 단순한 접근으로 63 스코어 니까 63 아래의 클래스는 추가적인 어그멘 테이션 수행

 ## 수가 적은 class에만적용해볼까

 -> 1, 2, 9 
"""

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import random

class ClassAug2:
    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), target_class=[1,2,9]):
        #target key setting
        self.target_class = target_class
        self.general_transform = A.Compose(
            [
                A.Resize(resize, resize),
                A.Normalize(mean=mean, std=std),
                A.OneOf([
                    A.HorizontalFlip(),  # 좌우 반전
                    A.VerticalFlip(), # 상하 반전
                    A.NoOp(), # Nope
                ], p=1.0), 
                # 색변환
                #A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.CoarseDropout(
                    max_holes=8,  # 제거할 사각형의 개수
                    max_height=40,  # 최대 사각형 높이
                     max_width=40,  # 최대 사각형 너비
                    fill_value=0,  # 사각형을 채울 색상 (0: 검은색)
                    p=0.5  # 50% 확률로 Cutout 적용
                 ),
                ToTensorV2(),
            ]
        )
        self.class_transforms = A.Compose([
            A.OneOf([
                    A.HorizontalFlip(),  # 좌우 반전
                    A.VerticalFlip(), # 상하 반전
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
                    A.NoOp(),
            ], p=1.0),
            A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
                    A.RandomGamma(gamma_limit=(90, 110)),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                    A.Equalize(),
                    A.NoOp()
            ], p=1.0),
            A.OneOf([
                    A.ElasticTransform(),
                    A.GridDistortion(),
                    A.OpticalDistortion(),
                    A.GaussNoise(var_limit=(5.0, 20.0)),
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
                    A.NoOp(),
            ], p=1.0),
            A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.NoOp(),
            ], p=0.5),
            A.CoarseDropout(max_holes=12, max_height=40, max_width=40, fill_value=0, p=0.5),
            A.Resize(resize, resize),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, image, label, xy_minmax=None):
        # crop
        if xy_minmax is not None:
            crop = A.OneOf([
                A.Crop(x_min=xy_minmax[0], y_min=xy_minmax[1], x_max=xy_minmax[2], y_max=xy_minmax[3]),
                A.NoOp(),
            ])
            image = crop(image=image)["image"]

        # class 별 aug 수행
        if label in self.target_class:
            #수행할 어그 선택
            image = self.class_transforms(image=image)["image"]
        else:
            image = self.general_transform(image=image)["image"]
        return image