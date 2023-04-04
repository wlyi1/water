import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import july
from july.utils import date_range
import pandas as pd
from pycaret.classification import *

def ol_over(param, ol, station, start = '2021-11-01', end = '2022-10-31', maps = 'golden'):
    st.header(ol)
    dates = date_range(start, end)
    dfs = df1[df1['Station'] == station]
    if param == 'pH':
        datas = dfs.resample('D')['pH'].apply(lambda x: (x>14).sum()).values
    elif param == 'DO':
        datas = dfs.resample('D')['DO'].apply(lambda x: (x>15).sum()).values
    elif param == 'NH4':
        datas = dfs.resample('D')['NH4'].apply(lambda x: (x>1000).sum()).values
    else:
        datas = dfs.resample('D')['NO3'].apply(lambda x: (x>1000).sum()).values
    fig, axs = plt.subplots(figsize=(32, 16))
    july.heatmap(dates, datas, month_grid=True, value_label=True, colorbar=True, title=f'Daily Sent Data {ol}', cmap='golden', ax = axs)
    return st.pyplot(fig)

def ol_on(ol, station, start = '2021-11-01', end = '2022-10-31', maps = 'golden'):
    st.header(ol)
    dates = date_range(start, end)
    dfs = df1[df1['Station'] == station]
    datas = dfs.resample('D')['DO'].count().values
    fig, axs = plt.subplots(figsize=(32, 16))
    july.heatmap(dates, datas, month_grid=True, value_label=True, colorbar=True, title=f'Daily Sent Data {ol}', cmap='golden', ax = axs)
    return st.pyplot(fig)

def ol_fluc(option, ol, station, model, start = '2021-11-01', end = '2022-10-31', maps = 'golden'):
    st.header(ol)
    dates = date_range(start, end)
    dfs = df1[df1['Station'] == station]
    df_ml = data_ml(dfs, option)
    py_df = predict_model(model, df_ml)
    fig, axs = plt.subplots(figsize=(32, 16))
    july.heatmap(dates, py_df['prediction_label'].sort_index(), month_grid=True, value_label=True, colorbar=True, title=f'Daily Sent Data {ol}', cmap='golden', ax = axs)
    return st.pyplot(fig)

def data_ml(ml_df, param):
    mean_data = list(ml_df[param].groupby(ml_df.index).mean().values)
    max_data = list(ml_df[param].groupby(ml_df.index).max().values)
    min_data = list(ml_df[param].groupby(ml_df.index).min().values)
    std_data = list(ml_df[param].groupby(ml_df.index).std().values)
    std_grad = list(ml_df[param].groupby(ml_df.index).apply(np.gradient).explode().groupby(ml_df.index).std().values)
    data = pd.DataFrame({'mean_data': mean_data, 'max_data': max_data, 'min_data': min_data, 'std_data' : std_data, 'std_grad' : std_grad})
    return data


saved_model = load_model('logreg')


#option
col1, col2 = st.columns(2)
option = col1.selectbox('Pilih Parameter', ('pH', 'DO', 'NH4', 'NO3'))
kasus = col2.selectbox('Pilih Kasus', ('Sent', 'Over', 'Fluc'))

## Create data
df = pd.read_csv('data_21_2022.csv')
df1 = df.copy()
df1['logDate'] = pd.to_datetime(df['logDate'])
df1 = df1.set_index('logDate')
df1 = df1.fillna(0)

st.title(f"📊 {kasus} Data ONLIMO")

ol = ['KLHK41', 'KLHK42', 'KLHK43', 'KLHK44', 'KLHK45', 'KLHK46', 'KLHK48', 'KLHK49', 'KLHK50', 'KLHK51', 'KLHK52', 'KLHK53', 'KLHK54', 'KLHK55', 'KLHK56', 'KLHK57', 'KLHK58', 'KLHK59', 'KLHK60', 'KLHK61']
ids = [x for x in range(11,31)]

for i,j in zip(ol, ids):
    if kasus == 'Over':
        ol_over(option, i, j)
    elif kasus == 'Sent':
        ol_on(i,j)
    else:
        ol_fluc(option, i,j, saved_model)
