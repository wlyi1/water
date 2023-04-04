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

@st.cache(allow_output_mutation=True)
def service():
    return[]




st.markdown("<h1 style='text-align: center;'>Form Servis Onlimo</h1>", unsafe_allow_html=True)

ol_16 = ['KLHK' + str(x) for x in range(1,17)]
ol_21 = ['KLHK' + str(x) for x in range(41,62)]
ol_12 = ['KLHK' + str(x) for x in range (62,74)]

ol16, ol21, ol12 = st.tabs(['Onlimo 16', 'Onlimo 21', 'Onlimo 12'])

for i,j,k in zip([ol16, ol21, ol12], [ol_16, ol_21, ol_12], ['ol16', 'ol21', 'ol12']):
    with i:
        with st.form(f'Servis Onlimo {k}', clear_on_submit=True):
            nama = st.text_input('Nama : ')
            onlimo = st.selectbox('Lokasi Onlimo', j)
            pekerjaan = st.selectbox('Pilih Pekerjaan', ('Sensor', 'Logger', 'Intake', 'Sipil', 'Elektrik'))
            tanggal = st.date_input('Tanggal Pengerjaan')
            biaya = st.number_input("Biaya yang keluar", value = 1000, format = '%i')
            keterangan = st.text_area('Keterangan')

            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write('Terima Kasih')
                service().append({'Nama': nama, 'Lokasi': onlimo, 'Pekerjaan': pekerjaan, 'Tanggal': tanggal, 'Biaya': biaya, 'Keterangan': keterangan})
                lap_service = pd.DataFrame(service())
                lap_service.tail(1).to_csv(f'service/service.csv', mode='a', index = False, header = False)
             

