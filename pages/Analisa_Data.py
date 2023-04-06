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

@st.cache_resource
def init_connection():
    return snowflake.connector.connect(
        **st.secrets["snowflake"], client_session_keep_alive=True
    )

#conn = init_connection()

@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

st.markdown("<h1 style='text-align: center;'>Sensor Failure Detection</h>", unsafe_allow_html=True)


st.title("Projects")

st.write("You have entered", st.session_state["my_input"])

#st.session_state['df'] = df
#df = st.session_state['df']

st.write(st.session_state.df)
col_id, col_tgl = st.columns(2)
ID = col_id.selectbox('Stasiun', [11,12,13,14,15,16,17,18])
tgl = col_tgl.date_input('Tanggal' , dt.date(2022,1,2))
st.write(ID)
#import data from SQL Server
#conn = init_connection()
#rows = run_query(f'select * from data21 where station = {ID}')

df = pd.DataFrame(rows, columns = ['Station', 'pH', 'DO', 'NH4', 'NO3', 'COD', 'BOD', 'logDate', 'logTime'])
df = df.astype({'logDate':'string', 'logTime': 'string'})
df.index = pd.DatetimeIndex(df['logDate'] + ' ' + df['logTime'])
df = df.drop_duplicates(subset=['logTime', 'logDate'], keep='last')
df = df.drop(columns=['logTime', 'logDate'])

new_index = pd.date_range(df.index[0].date(), df.index[len(df.index)-1].date() + timedelta(days=1), 
                            freq = 'H', normalize=True, inclusive = 'left')
df = df.reindex(new_index)
for i in np.unique(df.index.date).astype('str'):
    df.loc[i] = df.loc[i].fillna(method='ffill').fillna(method='bfill') 

df = df.fillna(0)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD'])
name_param = ['pH', 'DO', 'NH4', 'NO3', 'BOD', 'COD']
stat_data = data_anom(df, tgl)
posisi_param = [0,1,2,3,4,5]
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



