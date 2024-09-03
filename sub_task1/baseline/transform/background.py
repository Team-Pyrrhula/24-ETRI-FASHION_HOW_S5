import numpy as np
from rembg import remove
from PIL import Image
import albumentations as A

class BackgroundRemover(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(BackgroundRemover, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # img를 PIL 이미지로 변환
        pil_img = Image.fromarray(img)
        
        # 배경 제거
        pil_img_no_bg = remove(pil_img)

        #rgb로 변경
        pil_img_no_bg_rgb = pil_img_no_bg.convert("RGB")
        
        # PIL 이미지를 다시 넘파이 배열로 변환
        img_no_bg = np.array(pil_img_no_bg_rgb)
        
        return img_no_bg
