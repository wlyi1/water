import streamlit as st
import snowflake.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import datetime
from numpy import load
import plotly.express as px

st.write('test ss')

@st.cache_resource
def init_connection():
    return snowflake.connector.connect(
        **st.secrets["snowflake"], client_session_keep_alive=True
    )

conn = init_connection()

@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

rows = run_query('select * from data21 where station = 11')

df = pd.DataFrame(rows, columns = ['Station', 'pH', 'DO', 'Temp', 'NH4', 'NO3', 'COD', 'BOD', 'logDate', 'logTime'])
st.write(df)
st.write(df.dtypes)

df.index = pd.DatetimeIndex(df['logDate'] + ' ' + df['logTime'])
df = df.drop_duplicates(subset=['logTime', 'logDate'], keep='last')
df = df.drop(columns=['logTime', 'logDate'])

new_index = pd.date_range(df.index[0].date(), df.index[len(df.index)-1].date() + timedelta(days=1), 
                            freq = 'H', normalize=True, inclusive = 'left')
df = df.reindex(new_index)
for i in np.unique(df.index.date).astype('str'):
    df.loc[i] = df.loc[i].fillna(method='ffill').fillna(method='bfill') 

df = df.fillna(0)
    
def chart(d0,d1,param):
    dfs = df.loc[d0 : d1, param]
    dfs.drop(dfs.tail(1).index,inplace=True)
    fig = px.line(dfs, x=[i+1 for i in range(int(len(dfs)/4))], y=[dfs.loc[dfs.index.hour.isin([0,1,2,3,4,5])], 
                                                    dfs.loc[dfs.index.hour.isin([6,7,8,9,10,11])], 
                                                    dfs.loc[dfs.index.hour.isin([12,13,14,15,16,17])],
                                                    dfs.loc[dfs.index.hour.isin([18,19,20,21,22,23])]],
                                                    color_discrete_sequence = px.colors.qualitative.Plotly, markers=False)
    series_names = ['Subuh', 'Pagi', 'Siang', 'Malam'] 
    for idx, name in enumerate(series_names):
        fig.data[idx].name = name
        fig.data[idx].hovertemplate = name

    
    return fig

st.title('Perbandingan Data Parameter Periode Waktu (6H)')

head1, head2, head3, head4 = st.columns(4)
ID_choice = head1.selectbox('Stasiun', [11,12,13,14,15,16,17])

d1 = head2.date_input('Tanggal Awal', datetime.date(2021,9,21))
d2 = head3.date_input('Tanggal Akhir' , datetime.date(2021,9,22))
param = head4.selectbox('Parameter:', ('pH', 'DO', 'COD', 'BOD', 'NH4', 'NO3', 'Temp'))

#import data from SQL Server

komp = chart(d1, d2, param)
st.plotly_chart(komp, theme='streamlit')