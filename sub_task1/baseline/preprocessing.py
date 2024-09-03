from rembg import remove
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import multiprocessing as mp

# 단일 이미지에 대해 배경 제거 후 저장하는 함수
def remove_background(input_img:str, output_img:str):
    image = Image.open(input_img)
    output = remove(image)
    output = output.convert("RGB")
    output.save(output_img)

# 병렬 처리를 위한 개별 작업 단위를 수행하는 함수
def process_image(data_tuple):
    input_img, output_img = data_tuple
    remove_background(input_img, output_img)

# 메인 전처리 함수, 멀티프로세싱 지원
def preprocessing(target_csv:str, data_path:str="../data/"):
    df = pd.read_csv(os.path.join(data_path, target_csv))
    
    folder_name = target_csv.split("_")[2].split(".")[0]
    print(folder_name)
    
    image_path = df['image_name'].values

    # 모든 입력 이미지 경로와 출력 이미지 경로를 튜플로 묶어 리스트로 생성
    data_list = []
    for ip in image_path:
        output_folder = os.path.join(data_path, folder_name + '_remgb', ip.split('/')[0])
        os.makedirs(output_folder, exist_ok=True)
        
        input_img = os.path.join(data_path, folder_name, ip)
        output_img = os.path.join(output_folder, ip.split("/")[-1].split(".")[-2] + '_remgb.jpg')

        data_list.append((input_img, output_img))

    # 멀티프로세싱을 사용하여 병렬 처리
    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, data_list), total=len(data_list)))

def make_csv(target_csv:str, data_path:str):
    origin_df = pd.read_csv(os.path.join(data_path, target_csv))
    types = target_csv.split("_")[2].split(".")[0]

    #preprocessing
    new_df = origin_df.copy()
    new_df['image_name'] = f'{types}_remgb/' + new_df["image_name"].str.split(".").str[0] + '_remgb.jpg'

    origin_df['image_name'] = f'{types}/' + origin_df['image_name']

    # merge
    merged_df = pd.concat([origin_df, new_df], axis=0)

    # save
    merged_df.to_csv(os.path.join(data_path, f"Fashion-How24_sub1_{types}_all.csv"), index=False)
    new_df.to_csv(os.path.join(data_path, f"Fashion-How24_sub1_{types}_remgb.csv"), index=False)


if __name__ == '__main__':
    target_csv = "Fashion-How24_sub1_val.csv"
    data_path = "../data/"

    # preprocessing(target_csv, data_path)
    make_csv(target_csv, data_path)