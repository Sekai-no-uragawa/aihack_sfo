import numpy as np
import pandas as pd
import ast

from statsmodels.tsa.seasonal import STL
from scipy.signal import find_peaks, argrelmax, argrelmin
from scipy.spatial.distance import euclidean, correlation, cosine
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import argrelmin


def calc_distance_corr(row):
    euc = euclidean(row['data'], row['data_2'])
    cos = cosine(row['data'], row['data_2'])
    corr = correlation(row['data'], row['data_2'])
    
    return euc, cos, corr


def rr_span_trend(x: np.array):
    
    X = pd.DataFrame(x, columns=['val'], index=pd.date_range("2022-01-01", periods=240, freq="H"))
    res = STL(X).fit()
    trend = res.trend.values
    
    peaks, h_peaks = find_peaks(x, height=trend, distance=10)
    dips, _ = find_peaks(-x, height=-trend, distance=10)
    
    heights = h_peaks['peak_heights']
    X_rr = X.query('val in @heights')
    X_rr['delta'] = ([(X_rr.index[i] - X_rr.index[i-1]).total_seconds() / 60  if i != 0 else 0 
                      for i in range(len(X_rr.index))])
    rr = X_rr['delta'].values[1:]
    
    length = len(peaks) if len(peaks) <= len(dips) else len(dips)
    span = np.array([x[peaks][i] - x[dips][i] for i in range(length)])
    
    return rr, span, trend


def apply_rr_span_trend(row):

    res_1 = rr_span_trend(row['data'])
    res_2 = rr_span_trend(row['data_2'])
       
    return res_1[0], res_1[1], res_1[2], res_2[0], res_2[1], res_2[2]


def count_spasms(x: np.array):
    
    diff = np.diff(x[1:-2])
    cnt = len(argrelmin(diff)[0])
    
    return cnt


def fourier_index(x):
    N = 240
    yf = 2.0/N * np.abs(fft(x)[0 : N//2])
    indices = argrelmax(2.0 / N * np.abs(yf[0 : N//2]), order=4)
    
    return yf[indices]


def df_preparation(df: pd.DataFrame):
    df.rename(columns={'Filename': 'id', 
        'Test_index': 'q_group',
        'Presentation': 'q_nrepeat',
        'Class_label': 'label',
    }, inplace=True)
    df.rename(str.lower, axis='columns', inplace=True)

    df.data = df.data.apply(ast.literal_eval)
    df.data_2 = df.data_2.apply(ast.literal_eval)

    df.data = df.data.apply(lambda row: np.array(row))
    df.data_2 = df.data_2.apply(lambda row: np.array(row))

    df2 = df.copy()

    names1 = [f'v{i}' for i in range(240)]
    names2 = [f'vv{i}' for i in range(240)]
    df2[names1] = pd.DataFrame(df2.data.tolist(), index= df2.index)
    df2[names2] = pd.DataFrame(df2.data_2.tolist(), index= df2.index)

    numerical = names1 + names2
    df2 = df2.dropna().reset_index(drop=True)
    df2 = df2.drop(columns=numerical, errors='ignore')

    df2[['euc', 'cos', 'corr']] = df2.apply(calc_distance_corr, result_type='expand', axis=1)

    indices = df2.groupby(['id', 'question']).indices
    for k, ind in indices.items():
    
        scl_1 = MinMaxScaler()
        stacked_1 = (np.hstack(df2['data'].loc[ind].values)).reshape(-1, 1)
        scl_1.fit(stacked_1)
        
        scl_2 = MinMaxScaler()
        stacked_2 = (np.hstack(df2['data_2'].loc[ind].values)).reshape(-1, 1)
        scl_2.fit(stacked_2)
    
        for i in ind:
            df2.at[i, 'data'] = scl_1.transform(df2['data'].loc[i].reshape(-1, 1)).reshape(1, -1)[0]
            df2.at[i, 'data_2'] = scl_2.transform(df2['data_2'].loc[i].reshape(-1, 1)).reshape(1, -1)[0]

    df2[['rr_1', 'span_1', 'trend_1', 'rr_2', 'span_2', 'trend_2']] = df2.apply(
        apply_rr_span_trend,
        result_type='expand',
        axis=1)

    df2['t1_std'] = df2['trend_1'].apply(lambda x: np.std(x))
    df2['t2_std'] = df2['trend_2'].apply(lambda x: np.std(x))
    df2['t1_mean'] = df2['trend_1'].apply(lambda x: np.mean(x))
    df2['t2_mean'] = df2['trend_2'].apply(lambda x: np.mean(x))
    df2['rr1_min'] = df2['rr_1'].apply(lambda x: np.min(x))
    df2['rr2_min'] = df2['rr_2'].apply(lambda x: np.min(x) if len(x) != 0 else 0)
    df2['rr1_max'] = df2['rr_1'].apply(lambda x: np.max(x))
    df2['rr2_max'] = df2['rr_2'].apply(lambda x: np.max(x) if len(x) != 0 else 0)
    df2['rr1_std'] = df2['rr_1'].apply(lambda x: np.std(x))
    df2['rr2_std'] = df2['rr_2'].apply(lambda x: np.std(x) if len(x) != 0 else 0)
    df2['s1_std'] = df2['span_1'].apply(lambda x: np.std(x))
    df2['s2_std'] = df2['span_2'].apply(lambda x: np.std(x) if len(x) != 0 else 0)

    df2['spasms_1'] = df2['span_1'].apply(count_spasms)
    df2['spasms_2'] = df2['span_2'].apply(count_spasms)
    df2['spasms_1'] = df2['spasms_1'].replace([8, 7, 6], 0)
    df2['spasms_2'] = df2['spasms_2'].replace([8, 7, 6], 0)
    df2['spasms'] = (df2['spasms_2'] + df2['spasms_1']) // 2
    df2.drop(columns=['spasms_1', 'spasms_2'], inplace=True)

    df2['fourier_1'] = df2['data'].apply(fourier_index)
    df2['fourier_2'] = df2['data_2'].apply(fourier_index)
    l1 = []
    l2 = []

    for v in df2['fourier_1'].values:
        l1.append(len(v))
        
    for v in df2['fourier_2'].values:
        l2.append(len(v))
        
    df2['fourier_1'] = df2['fourier_1'].apply(lambda x: np.pad(x, (0, max(l1)-len(x)), 'constant'))
    df2['fourier_2'] = df2['fourier_2'].apply(lambda x: np.pad(x, (0, max(l2)-len(x)), 'constant'))

    names1 = [f'f{i}' for i in range(max(l1))]
    names2 = [f'ff{i}' for i in range(max(l2))]

    df2[names1] = pd.DataFrame(df2['fourier_1'].tolist(), index=df2.index)
    df2[names2] = pd.DataFrame(df2['fourier_2'].tolist(), index=df2.index)

    df2.fillna(0, inplace=True)

    train = df2.drop(
        columns=[
            'id', 'q_group', 'q_nrepeat',
            'question', 'data', 'data_2',
            'fourier_1', 'fourier_2', 'rr_1',
            'span_1', 'trend_1', 'rr_2',
            'span_2', 'trend_2'
        ], errors='ignore').reset_index(drop=True)

    #train.drop(columns=['corr', 'euc'], errors='ignore', inplace=True)

    return train