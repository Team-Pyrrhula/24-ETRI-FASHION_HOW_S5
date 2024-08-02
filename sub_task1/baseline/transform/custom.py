import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

class CustomAug:
    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = A.Compose(
            [
                A.Resize(resize, resize),
                A.Normalize(mean=mean, std=std),
                A.OneOf([
                    A.HorizontalFlip(p=1.0),  # 좌우 반전
                    A.VerticalFlip(p=1.0),  # 상하 반전
                ], p=0.5), 
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

    def __call__(self, image):
        image = np.ascontiguousarray(image)
        return self.transform(image=image)["image"]