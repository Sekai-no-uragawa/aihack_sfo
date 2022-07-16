import numpy as np
import pandas as pd
import ast

from statsmodels.tsa.seasonal import STL
from functools import partialmethod

from sklearn.preprocessing import MinMaxScaler
from scipy.fft import fft
from scipy.signal import argrelextrema



pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=3)
pd.options.display.max_rows = 10
#FUNCTIONS
def det_outliers(df, columns):
    '''Рассчитывает и возвращает границы выбросов по формуле +/- 3 стандартных отклонения от среднего.
    В случае, если с какой-то стороны выбросов по данному критерию нет, возвращает минимум/максимум соответственно '''
    outliers = {}
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        left = mean - 3 * std
        right = mean + 3 * std
        ratio = len(df[(df[col] < left) | (df[col] > right)]) / len(df) * 100
        if ratio > 0:
            outliers[col] = (round(ratio, 2), round(left, 2) if left > round(df[col].min(), 2) else 0,
                            round(right, 2) if right < df[col].max() else round(df[col].max(), 2))
    return outliers

def rr_span_trend(x: np.array):

    X = pd.DataFrame(x, columns=['val'], index=pd.date_range(
        "2022-01-01", periods=240, freq="H"))
    res = STL(X).fit()
    trend = res.trend.values

    return trend
def apply_rr_span_trend(row):

    res_1 = rr_span_trend(row['Data'])
    res_2 = rr_span_trend(row['Data_2'])

    return res_1, res_2

def fourierindex(x):
        N = 240
        yf = 2.0/N * np.abs(fft(np.array(x))[0:N//2])
        indexes = argrelextrema(2.0/N * np.abs(yf[0:N//2]), np.greater, order = 10)
        return yf[indexes]

def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    
    df.Data = df.Data.apply(ast.literal_eval)
    df.Data_2 = df.Data_2.apply(ast.literal_eval)

    df.Data = df.Data.apply(np.array)
    df.Data_2 = df.Data_2.apply(np.array)

    names1 = [f'v{i}' for i in range(240)]
    names2 = [f'vv{i}' for i in range(240)]
    numerical = names1 + names2
    df[names1] = pd.DataFrame(df.Data.tolist(), index= df.index)
    df[names2] = pd.DataFrame(df.Data_2.tolist(), index= df.index)

    df = df.dropna().reset_index(drop=True)

    indices = df.groupby(['Filename', 'Question']).indices

    for k, ind in indices.items():
        scl_1 = MinMaxScaler()
        stacked_1 = np.hstack(df['Data'].loc[ind].values).reshape(-1, 1)
        scl_1.fit(stacked_1)
        
        scl_2 = MinMaxScaler()
        stacked_2 = np.hstack(df['Data_2'].loc[ind].values).reshape(-1, 1)
        scl_2.fit(stacked_2)
        
        for i in ind:
            df.at[i, 'Data'] = (scl_1.transform(df['Data'].loc[ind].loc[i].reshape(-1, 1))).reshape(1, -1)[0]
            df.at[i, 'Data_2'] = (scl_2.transform(df['Data_2'].loc[ind].loc[i].reshape(-1, 1))).reshape(1, -1)[0]


    df = df.drop(columns = numerical)

    df[names1] = pd.DataFrame(df.Data.tolist(), index= df.index)
    df[names2] = pd.DataFrame(df.Data_2.tolist(), index= df.index)

    outliers = det_outliers(df, numerical)

    d1_bottom = []
    d1_top = []
    d2_bottom = []
    d2_top = []

    for k, v in outliers.items():
        if 'vv' in k:
            d2_top.append(v[2])
            d2_bottom.append(v[1])
        else:
            d1_top.append(v[2])
            d1_bottom.append(v[1])

    def has_outliers(row):
        data1_out = any(row[names1].values > np.mean(d1_top)) or any(
            row[names1].values < np.mean(d1_bottom))
        data2_out = any(row[names2].values > np.mean(d2_top)) or any(
            row[names2].values < np.mean(d2_bottom))
        if data1_out or data2_out:
            return True
        else:
            return False
            
    df['outliers'] = df.apply(has_outliers, axis=1)

    df = df[df.outliers == False]

    df[['trend_1', 'trend_2']] = df.apply(apply_rr_span_trend,
                                        result_type='expand',
                                        axis=1,
                                        )

    df['t1_var'] = df['trend_1'].apply(lambda x: np.var(x))
    df['t2_var'] = df['trend_2'].apply(lambda x: np.var(x))
    df['t1_mean'] = df['trend_1'].apply(lambda x: np.mean(x))
    df['t2_mean'] = df['trend_2'].apply(lambda x: np.mean(x))

    df['fourier1'] = df['Data'].apply(lambda x: fourierindex(x) )
    df['fourier2'] = df['Data_2'].apply(lambda x: fourierindex(x))

    max_fourier1 = 0
    for i in df.fourier1:
        if len(i)>max_fourier1:
            max_fourier1 = len(i)

    max_fourier2 = 0
    for i in df.fourier2:
        if len(i)>max_fourier2:
            max_fourier2 = len(i)

    df['fourier1'] = df['fourier1'].apply(lambda x: np.pad(x, (0, max_fourier1-len(x)), 'constant'))
    df['fourier2'] = df['fourier2'].apply(lambda x: np.pad(x, (0, max_fourier2-len(x)), 'constant'))

    fourier_names1 = [f'f{i}' for i in range(len(df['fourier1'][0]))]
    fourier_names2 = [f'ff{i}' for i in range(len(df['fourier2'][0]))]

    df[fourier_names1] = pd.DataFrame(df.fourier1.tolist(), index= df.index)
    df[fourier_names2] = pd.DataFrame(df.fourier2.tolist(), index= df.index)

    temp = pd.DataFrame()
    t_names1 = [f't{i}' for i in range(len(df['trend_1'][0]))]
    t_names2 = [f'tt{i}' for i in range(len(df['trend_2'][0]))]
    temp[t_names1] = pd.DataFrame(df.trend_1.tolist(), index= df.index)
    temp[t_names2] = pd.DataFrame(df.trend_2.tolist(), index= df.index)

    cols_diff = [str(i+'df_t') for i in t_names1+t_names2]
    difference1 = temp[t_names1].apply(pd.Series.pct_change)
    difference1 = difference1.apply(lambda x: x.fillna(x.median(), axis = 0))
    difference2 = temp[t_names2].apply(pd.Series.pct_change)
    difference2 = difference2.apply(lambda x: x.fillna(x.median(), axis = 0))
    difference = pd.concat([difference1, difference2], axis = 1)
    difference.columns = cols_diff

    diff_trend1 = df[names1 + ['t1_mean'] ].apply(lambda x: x  - x['t1_mean'], axis = 1)
    diff_trend1 = diff_trend1.drop(columns = ['t1_mean'])
    diff_trend2 = df[names2 + ['t2_mean'] ].apply(lambda x: x  - x['t2_mean'], axis = 1)
    diff_trend2 = diff_trend2.drop(columns = ['t2_mean'])
    diff_trend = pd.concat([diff_trend1, diff_trend2], axis = 1)
    cols_diff_trend = [str(i+'_df_tr') for i in numerical]
    diff_trend.columns = cols_diff_trend
    
    final = df[['Class_label','t1_mean', 't2_mean']]
    final = pd.concat([final, diff_trend], axis = 1)
    
    return final