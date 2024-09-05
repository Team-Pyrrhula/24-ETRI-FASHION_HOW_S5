from config import BaseConfig
from dataset import ETRI_Dataset_color
from transform import CustomAug, ClassAug
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

label_decoder = {
                0:'Red',
                1:'Coral',
                2:'Orange',
                3:'Pink',
                4:'Purple',
                5:'Brown',
                6:'Beige',
                7:'Ivory',
                8:'Yellow',
                9:'Mustard',
                10:'Skyblue',
                11:'Royalblue',
                12:'Navy',
                13:'Green',
                14:'Khaki',
                15:'White',
                16:'Gray',
                17:'Black',
}
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

# 예시: mean과 std는 정규화에 사용된 값
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 정규화된 이미지 반정규화

config = BaseConfig(train_csv='Fashion-How24_sub2_tain_remgb_clean.csv')
config.CLASS_AUG = 1
config.REMGB = 1
config.CROP = 1


transform = ClassAug(224)
dataset = ETRI_Dataset_color(config, True, transform=transform, types='train', remgb=1, crop=1)
train_loader = DataLoader(dataset, batch_size=100)

for img, label in train_loader:
    #imgs = denormalize(img[0], mean, std)
    imgs = img[0]
    print(imgs)
    plt.figure(figsize=(5, 5))
    plt.imshow(imgs.permute(1, 2, 0).numpy())
    plt.title(f'label : {label_decoder[label.tolist()[0]]}')
    plt.axis('off')
    plt.show()