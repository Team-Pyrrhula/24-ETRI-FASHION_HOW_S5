# external library
import pandas as pd
import matplotlib.pyplot as plt

def visualization(df: pd.DataFrame) -> None:
    """continual learning 실험 결과를 시각화할 때 사용합니다.

    Args:
        df (pd.DataFrame): 시각화 대상 데이터입니다.
    """
    # 실험 제목 추출
    exp_name = df.iloc[0]['Unnamed: 0'].split("@")[0]

    # task1 ~ 6의 성능 변화 추이를 확인하기 위한 figure 생성
    fig, ax = plt.subplots(1, 6, figsize=(30, 4))
    fig.suptitle(exp_name, fontsize=14)

    # 시각화
    for i in range(6):
        # 기본 세팅
        ax[i].set_title(f"task{i + 1}")

        ax[i].set_xlabel("task number")
        ax[i].set_ylabel("WKT score")
        
        # 해당 task의 값을 가져옴
        exp_results = df[f'task{i + 1}'].values

        # 유의미한 값만 추출
        if i > 0:
            paddings = [0] * i
            paddings.extend(exp_results[i:])
            exp_results = paddings

        # x축 값 설정
        task_numbers = [1, 2, 3, 4, 5, 6]

        # 가져온 값을 plot
        ax[i].plot(task_numbers, exp_results, linestyle='--', marker='o')

        # task만 학습했을 때, 모든 task를 학습했을 때 수치를 그래프에 표시
        only_task_res = round(exp_results[i], 3)
        after_cl_res = round(exp_results[-1], 3)

        if i < 5:
            ax[i].annotate(only_task_res, ((i + 1) - 0.25, exp_results[i] + 0.05),
                           fontsize=14, fontweight='bold', color='blue')
        
        text = str(after_cl_res) + "\n" + "(" + \
                str(round((after_cl_res - only_task_res), 3)) + ")"
        ax[i].annotate(text, (6 - 0.5, exp_results[-1] + 0.05),
                       fontsize=14, fontweight='bold', color='red')

        # ticks(눈금) 통일
        ax[i].set_yticks([n / 10 for n in range(0, 11)])