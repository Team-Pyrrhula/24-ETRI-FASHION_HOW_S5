from __future__ import annotations

# torch
import torch

# bulit-in library
import os
import yaml
import random
import argparse

# external library
import numpy as np
import pandas as pd


def set_seed(seed: int=2024) -> None:
    """실험 재현을 위한 시드를 설정할 때 사용합니다.

    Args:
        seed (int, optional): 시드로 사용할 값. Defaults to 2024.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_udevice() -> torch.device:
    """
    학습 및 추론 시 사용가능한 장치를 찾아 반환합니다.

    Returns:
        torch.device: 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()

    else:    
        device = torch.device('cpu')

    print('Using device: {}'.format(device))

    if torch.cuda.is_available():
        print('# of GPU: {}'.format(num_gpu))

    return device


def str2bool(v: str | bool) -> bool:
    """
    string을 bool로 변환합니다.

    Args:
        v (str | bool): 터미널에서 입력된 실험 인자입니다.

    Raises:
        argparse.ArgumentTypeError: string, bool이 아닌 type이 입력되는 경우 예외 처리합니다.

    Returns:
        bool: True 또는 False를 반환합니다.
    """
    if isinstance(v, bool): 
        return v 
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def load_cfg(cfg_path: str) -> dict:
    """
    실험에 사용할 설정값들을 불러올 때 사용합니다.
    ArgParser가 아닌 yaml 파일 기반으로 실험하기 위해 추가한 함수입니다.

    Args:
        cfg_name: config 파일의 경로

    Return:
        cfg_dict: 실험에 사용할 설정값들이 담긴 dictionary
    """
    with open(cfg_path, encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    return cfg_dict    


def save_results(exp_name: str, score: float, task_id: int, 
                 mode: str = 'eval', save_dir: str = 'results') -> None:
    """실험 결과를 csv 파일에 누적합니다.

    exp_name    | task1 | task2 | task3 | task4 | task5 | task6
    01_baseline | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx
    02_test     | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx
    ...

    Args:
        exp_name (str): 실험 이름입니다.
        score (float): 실험 점수입니다.
        task_id (int): 평가 데이터의 번호입니다.
        mode (str, optional): 평가 종류입니다. eval 또는 test를 사용합니다. Defaults to 'eval'.
        save_dir (str, optional): csv 파일을 저장할 디렉토리의 이름입니다. Defaults to 'results'.
    """
    # 현재 경로를 기준으로 디렉토리를 생성
    dir_path = os.path.join(os.getcwd(), save_dir)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # 필요한 변수 생성
    file_path = os.path.join(dir_path, mode + '.csv')
    
    # 실험 결과를 저장할 DataFrame 불러오기
    df = pd.DataFrame()
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)

    else:
        df = pd.DataFrame(index=[exp_name],
                          columns=[f'task{i}' for i in range(1, 6 + 1)])

    # 실험명을 기반으로 원하는 위치에 score 기록하기
    df.loc[exp_name, f'task{task_id}'] = score

    # 저장
    df.to_csv(file_path)
        

# Test code
if __name__ == "__main__":
    # load_cfg test
    print(load_cfg("./cfgs/00_baseline.yaml"))