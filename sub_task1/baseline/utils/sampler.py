import pandas as pd
from sklearn.utils import resample

def etri_sampler(df:pd.DataFrame, sampling_columns:list=['Daily', 'Gender', 'Embellishment'], types:str='m'):
    """_summary_
        dataset의 라벨별로 샘플링하여 set을 구성 (under sampling)
    Args:
        df (pd.DataFrame): train dataset 원본
        sampling_columns (list): under sampling 할 columns
        types (str): under sampling 할 방법 [m, p]
            - m (min) : 가장 적은 class에 맞춰 under sampling
            - p (percentage) : 전체 class에서 차지 하는 비율에 따라서 
                            under sampling( * [100% - 차지하는 비율])을 적용
    Retruns:
        (daily, gender, embel) : 각 columns 마다 sampling 되어진 Dataframe 
    """

    for col in sampling_columns:
        data = df[col]
        data_count = data.value_counts().to_dict()
        data_frame = pd.DataFrame()

        # types = min인 경우
        if types == 'm':
            min_len = data_count[min(data_count)]

            for key in data_count.keys():
                target = df[df[col] == key]
                under_sampling = resample(target, replace=False, n_samples=min_len)
                data_frame = pd.concat([data_frame, under_sampling])

        #types = p인 경우
        elif types == 'p':
            total = sum(data_count.values())
            
            for key in data_count.keys():
                p = 1 - (data_count[key] / total)
                p_len = int(data_count[key] * p)
                target = df[df[col] == key]
                under_sampling = resample(target, replace=False, n_samples=p_len)
                data_frame = pd.concat([data_frame, under_sampling])

        if col == 'Daily':
            daily = data_frame
        elif col == 'Gender':
            gender = data_frame
        elif col == 'Embellishment':
            embel = data_frame
            
    return (daily, gender, embel)