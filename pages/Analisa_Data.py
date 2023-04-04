import pandas as pd
from datetime import date
import datetime as dt
import numpy as np
from datetime import datetime
from numpy import load
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score   
from sklearn.model_selection import cross_val_score
import pickle
from fungsi import *
from pycaret.classification import load_model, predict_model
import plotly.express as px
import plotly.graph_objects as go

#load model
model = pickle.load(open('model.pkl', 'rb'))
model_py = load_model('logreg')

def predict(model, input):
    predict_df = predict_model(estimator = model, data = input)
    preds = predict_df['Label'][0]
    return preds

st.markdown("<h1 style='text-align: center;'>Sensor Failure Detection</h>", unsafe_allow_html=True)

#import pre data
files_id = pd.read_csv('id_stasiun.csv')

data_24 = load('jam_24.npy', allow_pickle = True)
df_nan = pd.read_csv('df_nan.csv')
index = np.arange(1,25)
df_test = pd.read_csv('testml.csv')

col_id, col_tgl = st.columns(2)
ID_choice = col_id.selectbox('Stasiun', files_id['CODE'])
ID = files_id[files_id['CODE']==ID_choice].index.values + 11
tgl = col_tgl.date_input('Tanggal' , dt.date(2022,1,2))

#import data from SQL Server
conn_str = 'DRIVER={SQL Server};server=DESKTOP-ECB4MMH\SQLEXPRESS;Database=awrl;Trusted_Connection=yes;'
con_url = URL.create('mssql+pyodbc', query={'odbc_connect': conn_str})
engine = create_engine(con_url)

query = f"""select pH, DO, Cond, Turb, Temp, NH4,NO3,ORP,COD,BOD,TSS,logTime as NH3_N,logDate, datepart(hour, logTime) as logTime 
from data where Station={int(ID)} order by logDate,logTime"""

df = pd.read_sql(query, engine)
df['logDate'] = pd.to_datetime(df['logDate']).dt.date


#drop today data
select_tgl = date.today()
df = df.loc[df['logDate'] != select_tgl]
tanggal = np.unique(df['logDate'].values)
j = len(tanggal)

#array data daily
arr = []

for i in tanggal:
    df_tgl = df.loc[df['logDate'] == i]
    
    if not np.array_equiv(df_tgl['logTime'].values, data_24):
        df_clean = df_tgl.drop_duplicates(subset='logTime', keep='last')
        df_clean = pd.concat([df_clean, df_nan])
        df_clean = df_clean.sort_values(by=['logTime'])
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        df_clean = df_clean.drop_duplicates(subset='logTime', keep='last')
        df_clean.drop(columns=df_clean.columns[-1], axis=1, inplace=True)
        arr_clean = df_clean.to_numpy()
        arr.append(arr_clean)
       
    else:
        arr_clean = df_tgl.to_numpy()
        arr.append(arr_clean)
        
arr = np.asarray(arr)
x_arr = arr[:,:,6].flatten()   

#return 1 dataframe date
cols = df.columns.values.tolist()
len_arr = arr.shape[0]*arr.shape[1]

arr_df = arr.reshape(len_arr,14)
df_con = pd.DataFrame(arr_df, columns = cols)

df_con = df_con.set_index(df_con['logDate'])


@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_con)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD'])
name_param = ['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD']
stat_data = data_anom(df_con, tgl)
posisi_param = [0,1,5,6,9,8]
df.index = df['logDate']

l_tab = [tab1, tab2, tab3, tab4, tab5, tab6]
for i1,j2,z in zip(l_tab, name_param, posisi_param):
    with i1:
        
        st.markdown(f"<h2 style='text-align: center;'>{j2} Sensor Failure Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Using Logistic Regression ML</p>", unsafe_allow_html=True)
         
        df_no = create_ml_data(df, j2)
        df_no.index = df.index.unique()
        predict_df = predict_model(estimator = model_py, data = df_no)
        stats = predict_df['prediction_label'].loc[tgl]

        col_an1, col_an2, col_an3 = st.columns(3)
        col_an2.metric(label = 'Quality Standard', value=24 - stat_data[f'ab_{j2}'])
        col_an3.metric(label = 'Out of Quality Standard', value=stat_data[f'ab_{j2}'])
        if stats == 1:
            col_an1.success(f'Sensor {j2} Normal')
        else:
            col_an1.error(f'Sensor {j2} Failure')
                
        sensor_failure = len(predict_df[(predict_df['prediction_label'] == 0)])
        sensor_normal = len(predict_df[(predict_df['prediction_label'] == 1)])   
        labels = ['Sensor Failure', 'Sensor Normal']
        values = [sensor_failure, sensor_normal]     

        pie_1, pie_2, pie_3 = st.columns(3)
        with pie_3:
            st.subheader(f'Normal and Failure Sensor {j2}')
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values = values, hole = .5)])
            fig_pie.update_layout(annotations=[dict(text=j2, x=0.5, y=0.5, font_size=20, showarrow=False)])
            st.plotly_chart(fig_pie)
        with pie_2:
            st.subheader('Dates of Data Sensor Failure')
            st.write(predict_df[(predict_df['prediction_label'] == 0)].index)
        with pie_1:
            st.subheader('Dataframe')
            st.write(df_con[['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD', 'logTime']].loc[tgl, :])
            st.download_button("Press to Download", csv, "file.csv", "text/csv", key=f'download-csv_{j2}')

       
st.write(":heavy_minus_sign:" * 32)



