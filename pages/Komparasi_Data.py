import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.widgets import CheckButtons
from datetime import datetime, timedelta, date
import datetime
import pyodbc
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from numpy import load
import plotly.express as px

class Komparasi():
    def __init__(self, engine, stasiun, d0, d1, param):
        self.engine = engine
        self.stasiun = stasiun
        self.d0 = d0
        self.d1 = d1
        self.param = param
        
    def gen_data(self):
        query = f"""select pH, DO, Temp,NH4,NO3,COD,BOD,TSS,logTime,logDate from data where Station={self.stasiun} order by logDate,logTime"""
        df = pd.read_sql(query, self.engine)
        df.index = pd.DatetimeIndex(df['logDate'] + ' ' + df['logTime'])
        df = df.drop_duplicates(subset=['logTime', 'logDate'], keep='last')
        df = df.drop(columns=['logTime', 'logDate'])

        new_index = pd.date_range(df.index[0].date(), df.index[len(df.index)-1].date() + timedelta(days=1), 
                                  freq = 'H', normalize=True, inclusive = 'left')
        df = df.reindex(new_index)
        for i in np.unique(df.index.date).astype('str'):
            df.loc[i] = df.loc[i].fillna(method='ffill').fillna(method='bfill') 

        self.df = df.fillna(0)
        #return self.df
    
    
    def chart(self):
        dfs = self.df.loc[self.d0 : self.d1, self.param]
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

files_id = pd.read_csv('id_stasiun.csv')

data_24 = load('jam_24.npy', allow_pickle = True)
df_nan = pd.read_csv('df_nan.csv')
index = np.arange(1,25)
df_test = pd.read_csv('testml.csv')

head1, head2, head3, head4 = st.columns(4)
ID_choice = head1.selectbox('Stasiun', files_id['CODE'])
ID = files_id[files_id['CODE']==ID_choice].index.values + 11

d1 = head2.date_input('Tanggal Awal', datetime.date(2021,9,21))
d2 = head3.date_input('Tanggal Akhir' , datetime.date(2021,9,22))
param = head4.selectbox('Parameter:', ('pH', 'DO', 'COD', 'BOD', 'NH4', 'NO3', 'Temp'))

#import data from SQL Server
conn_str = 'DRIVER={SQL Server};server=DESKTOP-ECB4MMH\SQLEXPRESS;Database=awrl;Trusted_Connection=yes;'
con_url = URL.create('mssql+pyodbc', query={'odbc_connect': conn_str})
engine = create_engine(con_url)

komp = Komparasi(engine, ID[0], d1, d2, param)
komp.gen_data()
st.plotly_chart(komp.chart(), theme='streamlit')
   

    
    
    
    
    
    
    