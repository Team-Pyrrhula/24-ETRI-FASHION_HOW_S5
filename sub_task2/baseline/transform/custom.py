import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

"""
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Shift, Scale, Rotate
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),                   # Elastic 변형
    A.GridDistortion(p=0.5),                                                        # Grid 왜곡
    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),                 # Optical 왜곡
    A.RandomBrightnessContrast(p=0.2),                                              # 밝기 및 대비 조정
    A.HorizontalFlip(p=0.5),                                                        # 좌우 반전
    A.VerticalFlip(p=0.1),                                                          # 상하 반전
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),             # 정규화
    ToTensorV2()                                                                    # Tensor 변환
])
"""

class CustomAug:
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

    def __call__(self, image, xy_minmax=None):
        # crop
        if xy_minmax is not None:
            crop = A.OneOf([
                A.Crop(x_min=xy_minmax[0], y_min=xy_minmax[1], x_max=xy_minmax[2], y_max=xy_minmax[3]),
                A.NoOp(),
            ])
            image = crop(image=image)["image"]
        return self.transform(image=image)["image"]