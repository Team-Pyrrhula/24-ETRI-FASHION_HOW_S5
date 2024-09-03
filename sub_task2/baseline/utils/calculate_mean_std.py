import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def transfer_color_space(image, color_space: str = 'RGB'):
    if color_space == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype(np.float32)
        image[:, :, 0] = image[:, :, 0] / 179.0
        image[:, :, 1] = image[:, :, 1] / 255.0
        image[:, :, 2] = image[:, :, 2] / 255.0

    elif color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = image.astype(np.float32)
        image[:, :, 0] = image[:, :, 0] / 100.0
        image[:, :, 1] = (image[:, :, 1] + 128) / 255.0
        image[:, :, 2] = (image[:, :, 2] + 128) / 255.0
    else:
        raise ValueError(f"지원하지 않는 색상 공간입니다: {color_space}")
    return image

def calculate_mean_std(config, color_space:str='RGB'):
    train_imgs = pd.read_csv(config.TRAIN_DF)['image_name'].values
    train_path = os.path.join(config.DATA_PATH, 'train')

    train_channel_sum = np.zeros(3, dtype=np.float64)
    train_channel_sum_squared = np.zeros(3, dtype=np.float64)
    train_pixel_count = 0

    # train 데이터셋에 대해 평균과 표준 편차 계산
    print('Train 데이터셋의 평균 및 표준 편차 계산 중...')
    for img in tqdm(train_imgs):
        image = cv2.imread(os.path.join(train_path, img))
        image = transfer_color_space(image, color_space)

        train_channel_sum += np.sum(image, axis=(0, 1))
        train_channel_sum_squared += np.sum(np.square(image), axis=(0, 1))
        train_pixel_count += image.shape[0] * image.shape[1]
    
    train_channel_mean = train_channel_sum / train_pixel_count
    train_channel_std = np.sqrt(train_channel_sum_squared / train_pixel_count - np.square(train_channel_mean))

    return train_channel_mean.tolist(), train_channel_std.tolist()