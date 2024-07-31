# torch
import torch

# bulit-in library
import os
import random

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


def save_results(exp_name: str, score: float, tr_task_id: str, tt_task_id: str,
                 mode: str = 'eval', save_path: str = './results') -> None:
    """실험 결과를 csv 파일에 누적합니다.

    exp_name    | task1 | task2 | task3 | task4 | task5 | task6
    01_baseline | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx
    02_test     | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx  | 0.xx
    ...

    Args:
        exp_name (str): 실험 이름입니다.
        score (float): 실험 점수입니다.
        tr_task_id (str): 평가에 사용한 모델이 마지막으로 학습한 데이터의 번호입니다.
        tt_task_id (str): 현재 평가에 사용하고 있는 데이터의 번호입니다.
        mode (str, optional): 평가 종류입니다. eval 또는 test를 사용합니다. Defaults to 'eval'.
        save_path (str, optional): csv 파일을 저장할 디렉토리의 경로입니다. Defaults to './results'.
    """
    # 필요한 변수 생성
    file_path = os.path.join(save_path, mode + '.csv')
    if mode == 'eval':
        exp_name = f'{exp_name}@task{tr_task_id}' # 01_baseline@task1, ...

    # 실험 결과를 저장할 DataFrame 불러오기
    df = pd.DataFrame()
    if os.path.exists(file_path):df = pd.read_csv(file_path, index_col=0)
    else: df = pd.DataFrame(index=[exp_name], columns=[f'task{i}' for i in range(1, 6 + 1)])

    # 실험명을 기반으로 원하는 위치에 score 기록하기
    df.loc[exp_name, f'task{tt_task_id}'] = score

    # 저장
    df.to_csv(file_path)
        
