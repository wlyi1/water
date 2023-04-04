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
import plotly.graph_objects as go
import pickle
from fungsi import *

st.set_page_config(layout='wide')

#hack CSS
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 style='text-align: center;'>Summary Onlimo</h1>", unsafe_allow_html=True)

model = pickle.load(open('model.pkl', 'rb'))

## Create data
df = pd.read_csv('data_21_2022.csv', parse_dates=['logDate'])
df1 = df.copy()
df1 = df1.set_index(['logDate'])

files_id = pd.read_csv('id_stasiun.csv')

#col dates
col1, col2, col3 = st.columns(3)
with col1:
    ID_choice = st.selectbox('Stasiun', files_id['CODE'][:21])
    ID = files_id[files_id['CODE']==ID_choice].index.values + 11
with col2:
    d1 = st.date_input('Tanggal Awal', datetime.date(2021,11,1))
with col3:
    d2 = st.date_input('Tanggal Akhir' , datetime.date(2022,6,30))
st.write(":heavy_minus_sign:" * 67)
df1 = df1.fillna(0)

#Data
df_graph = df1.loc[f'{str(d1)}' : f'{str(d2)}']
df_graph = df_graph.sort_index()
df_graph = df_graph[df_graph['Station'] == float(ID)]

#null
ph_nol = df_graph.loc[df_graph['pH'] == 0].resample('D')['pH'].count().values.sum()
do_nol = df_graph.loc[df_graph['DO'] == 0].resample('D')['DO'].count().values.sum()
nh_nol = df_graph.loc[df_graph['NH4'] == 0].resample('D')['NH4'].count().values.sum()
no_nol = df_graph.loc[df_graph['NO3'] == 0].resample('D')['NO3'].count().values.sum()
bod_nol = df_graph.loc[df_graph['BOD'] == 0].resample('D')['BOD'].count().values.sum()
cod_nol = df_graph.loc[df_graph['COD'] == 0].resample('D')['COD'].count().values.sum()

#over
ph_over = df_graph.loc[df_graph['pH'] >14].resample('D')['pH'].count().values.sum()
do_over = df_graph.loc[df_graph['DO'] > 14].resample('D')['DO'].count().values.sum()
nh_over = df_graph.loc[df_graph['NH4'] > 1000].resample('D')['NH4'].count().values.sum()
no_over = df_graph.loc[df_graph['NO3'] > 1000].resample('D')['NO3'].count().values.sum()
bod_over = df_graph.loc[df_graph['BOD'] > 1000].resample('D')['BOD'].count().values.sum()
cod_over = df_graph.loc[df_graph['COD'] > 1000].resample('D')['COD'].count().values.sum()

#fluctuate
def gen(data, param):
    mean_data = list(data[param].groupby(data.index).mean().values)
    max_data = list(data[param].groupby(data.index).max().values)
    min_data = list(data[param].groupby(data.index).min().values)
    std_data = list(data[param].groupby(data.index).std().values)
    std_grad = list(data[param].groupby(data.index).apply(np.gradient).explode().groupby(data.index).std().values)
    return mean_data, max_data, min_data, std_data, std_data, std_grad

for parameter in ['pH', 'DO', 'NH4', 'NO3', 'COD', 'BOD']:
    klhk = gen(df_graph, parameter)
    df_ml = pd.DataFrame(list(zip(klhk[0], klhk[1], klhk[2], klhk[3], klhk[4])), columns = ['Mean', 'Max', 'Min', 'Std_Data', 'Std_Gradient'])
    status = ml(model, df_ml)
    globals () [f'fluc_{parameter}'] = len(status) - status.sum()

st.markdown("<h1 style='text-align: center;'>Pembacaan Sensor Abnormal</h1>", unsafe_allow_html=True)
#Nul, Over and Fluctuate
null, over, flucs = st.columns(3)
null.metric('Total Nol', ph_nol + do_nol + nh_nol + no_nol + bod_nol + cod_nol)
over.metric('Total Over', ph_over + do_over + nh_over + no_over + bod_over + cod_over)
flucs.metric('Total Fluktuasi', fluc_pH + fluc_DO + fluc_NH4 + fluc_NO3 + fluc_BOD + fluc_COD)

#Bar Chart Horizontal
bar_col1, bar_col2, bar_col3 = st.columns(3)
with bar_col1:
    fig_bar1 = go.Figure(go.Bar(y =[ph_nol, do_nol, nh_nol, no_nol, bod_nol, cod_nol], x=['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD'], orientation = 'v', width = 0.3))
    fig_bar1.update_traces(marker_colorbar_title = {'text' : 'Data 0.00', 'side': 'bottom'}, selector = dict(type='bar'))
    #st.markdown("<h3 style='text-align: center;'>Data 0.00</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_bar1)

with bar_col2:
    fig_bar2 = go.Figure(go.Bar(y =[ph_over, do_over, nh_over, no_over, bod_over, cod_over], x=['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD'], orientation = 'v', width = 0.3))
    #st.markdown("<h3 style='text-align: center;'>Data Over</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_bar2)

with bar_col3:
    fig_bar3 = go.Figure(go.Bar(y =[fluc_pH, fluc_DO, fluc_NH4, fluc_NO3, fluc_BOD, fluc_COD], x=['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD'], orientation = 'v', width = 0.3))
    #st.markdown("<h3 style='text-align: center;'>Data Fluktuasi Ekstrem</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_bar3)



#DataFrame HeatMAp
index = ['Null', 'Over', 'Fluktuasi']
cols = ['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD']
data_heat = {'pH' : [df_graph.loc[df_graph['pH'] == 0].resample('D')['pH'].count().values.sum(), df_graph.loc[df_graph['pH'] >14].resample('D')['pH'].count().values.sum()],
'DO':[df_graph.loc[df_graph['NO3'] == 0].resample('D')['NO3'].count().values.sum(), df_graph.loc[df_graph['NO3'] >14].resample('D')['NO3'].count().values.sum()],
}
df_heat = pd.DataFrame(data_heat)
df_heat.style.background_gradient(cmap='viridis')

#order kerja
st.markdown("<h1 style='text-align: center;'>Order Pekerjaan</h1>", unsafe_allow_html=True)
sensor, logger, sipil, elektrik, intake = st.columns(5)
order = ['Sensor', 'Logger', 'Sipil', 'Elektrik', 'Intake']

#for s,o in zip([sensor, logger, sipil, elektrik, intake], order):
#    with s:
#        st.metric(o, 2, "Rp 1.500.000")




#pie chart anomali and offline

#chart offline
st.markdown("<h1 style='text-align: center;'>Chart Data Online</h1>", unsafe_allow_html=True)
off1, off2, off3 = st.columns(3)
off1.write(" ")
off3.write(" ")
off = list(df_graph.resample('D')['DO'].count().values).count(0)
on = len(df_graph.resample('D')['DO'].count().values)
with off2:
    fig_off = go.Figure(data=[go.Pie(labels = ['Online', 'Offline'], values = [on-off, off], hole = 0.4)])
    st.plotly_chart(fig_off)
#anomali
st.markdown("<h1 style='text-align: center;'>Chart Data Memenuhi Baku Mutu</h1>", unsafe_allow_html=True)
pie_1, pie_2, pie_3 = st.columns(3)
pie_4, pie_5, pie_6 = st.columns(3)

norm_ph = len([x for x in df_graph['pH']])
norm_do = len([x for x in df_graph['DO']])
norm_nh = len([x for x in df_graph['NH4']])
norm_no = len([x for x in df_graph['NO3']])
norm_bod = len([x for x in df_graph['BOD']])
norm_cod = len([x for x in df_graph['COD']])

anom_ph = sum(map(lambda x : x<6 or x>9, [x for x in df_graph['pH']]))
anom_do = sum(map(lambda x : x<3, [x for x in df_graph['DO']]))
anom_nh = sum(map(lambda x : x>0.5, [x for x in df_graph['NH4']]))
anom_no = sum(map(lambda x : x>20, [x for x in df_graph['NO3']]))
anom_bod = sum(map(lambda x : x>12, [x for x in df_graph['BOD']]))
anom_cod = sum(map(lambda x : x>100, [x for x in df_graph['COD']]))

colors = px.colors.qualitative.G10
st.write(type(colors))

for pie, anom, norm, param in zip([pie_1, pie_2, pie_3], [anom_ph, anom_do, anom_nh], [norm_ph, norm_do, norm_nh], ['pH', 'DO', 'NH4']):
    with pie:
        labels = ['Normal', 'Anomaly']
        values = [(norm - anom), anom]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values = values, hole = .5)])
        fig_pie.update_traces(marker=dict(colors = ['#00337C', '#13005A']))
        fig_pie.update_layout(annotations=[dict(text=f'{param}', x=0.5, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig_pie)

for pie, anom, norm, param in zip([pie_4, pie_5, pie_6], [anom_no, anom_bod, anom_cod], [norm_no, norm_bod, norm_cod], ['NO3', 'BOD', 'COD']):
    with pie:
        labels = ['Normal', 'Anomaly']
        values = [(norm - anom), anom]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values = values, hole = .5)])
        fig_pie.update_traces(marker=dict(colors = ['#00337C', '#13005A']))
        fig_pie.update_layout(annotations=[dict(text=f'{param}', x=0.5, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig_pie)

#create parameter graph
st.markdown("<h1 style='text-align: center;'>Grafik Rata-Rata Harian</h1>", unsafe_allow_html=True)
col_ph, col_do = st.columns(2)
col_nh, col_no = st.columns(2)
col_cod, col_bod = st.columns(2)

with col_ph:
    fig = px.line(df_graph.resample('D')['pH'].mean())
    st.plotly_chart(fig)
with col_do:
    fig = px.line(df_graph.resample('D')['DO'].mean())
    st.plotly_chart(fig)

with col_nh:
    fig = px.line(df_graph.resample('D')['NH4'].mean())
    st.plotly_chart(fig)
with col_no:
    fig = px.line(df_graph.resample('D')['NO3'].mean())
    st.plotly_chart(fig)

with col_bod:
    fig = px.line(df_graph.resample('D')['BOD'].mean())
    st.plotly_chart(fig)
with col_cod:
    fig = px.line(df_graph.resample('D')['COD'].mean())
    st.plotly_chart(fig)



st.write(":heavy_minus_sign:" * 67)
#Create Metrics for Every Parameter



