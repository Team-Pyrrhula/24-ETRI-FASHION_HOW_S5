import pandas as pd
from sklearn.utils import resample


def sampler(df:pd.DataFrame):
    data = df['Color']
    data_count = data.value_counts().to_dict()
    data_frame = pd.DataFrame()

    total = sum(data_count.values())
    for key in data_count.keys():
        p = 1 - (data_count[key] / total)
        p_len = int(data_count[key] * p)
        target = df[df['Color'] == key]
        under_sampling = resample(target, replace=False, n_samples=p_len)
        data_frame = pd.concat([data_frame, under_sampling])

    return (data_frame)