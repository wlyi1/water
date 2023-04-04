import pandas as pd
import datetime
from datetime import datetime as dt
import streamlit as st
import streamlit.components.v1 as components
import pyodbc
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
import matplotlib.pyplot as plt

def chart(ylabel, xlabel, yvalues, xvalues, title=''):
    #create new graph
    
    fig = plt.figure(figsize = (10,7))
    plt.plot(xvalues, yvalues)
    plt.title(title, fontsize = 20, fontweight = 'bold')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return fig

st.title('Data 24 Jam Terakhir')

files_id = pd.read_csv('id_stasiun.csv')
header_1, header_2 = st.columns(2)

st.title('Grafik Parameter 24 Jam Terakhir')
option = header_1.selectbox('Parameter untuk dilihat data dan grafiknya', ('pH', 'DO', 'NH4', 'NO3', 'COD', 'BOD', 'TSS'))
params = ['pH', 'DO', 'NH4', 'NO3', 'COD', 'BOD', 'TSS']

ID_choice = header_2.selectbox('Stasiun', files_id['CODE'])
ID = files_id[files_id['CODE']==ID_choice].index.values + 11



#import data from SQL Server
conn_str = 'DRIVER={SQL Server};server=DESKTOP-ECB4MMH\SQLEXPRESS;Database=awrl;Trusted_Connection=yes;'
con_url = URL.create('mssql+pyodbc', query={'odbc_connect': conn_str})
engine = create_engine(con_url)

query = f"""select pH, DO, Cond, Turb, Temp, NH4,NO3,ORP,COD,BOD,TSS,logTime as NH3_N,logDate, datepart(hour, logTime) as logTime 
from data where Station={int(ID)} order by logDate,logTime"""

df = pd.read_sql(query, engine, coerce_float=False)
#f['logDate'] = pd.to_datetime(df['logDate']).dt.date


df['tgl'] = pd.to_datetime(df['logDate'] + ' ' + df['NH3_N'])
#st.write(df[-24:])



#empty chart
c1, c2 = st.columns(2)
with c1:
    if 'space_initial' not in st.session_state:
        st.session_state.space_initial = st.empty()
with c2:
    if 'space_initial_2' not in st.session_state:
        st.session_state.space_initial_2 = st.empty()

#chart parameter
for par in params:
    globals()[f'{par}'] = chart(f'{par}', 'Date', df[-24:][f'{par}'], df[-24:]['tgl'], title= f'Grafik {par}')
    globals()[f'data_{par}'] = df[-24:][['tgl', f'{par}']]
    
st.session_state.space_initial.write(globals()[f'{option}'])
st.session_state.space_initial_2.write(globals()[f'data_{option}'])
    
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


#input action from PJ
@st.cache(allow_output_mutation=True)
def get_data_input():
    return[]

@st.cache(allow_output_mutation=True)
def get_data_output():
    return[]

@st.cache(allow_output_mutation=True)
def anomali():
    return[]


st.header('Take Action :smiley:')
with st.form("Form  ", clear_on_submit=True):
    st.write("Hallo Engineer MA")
    tindakan = st.multiselect('apa yang sudah dilakukan untuk data anomali ini ? 😀', ['Mencelupkan sensor ke air bersih', 'Refill solution', 'Hapus data kalibrasi', 'Kalibrasi ulang'])
    catatan = st.text_area('Keterangan ? ')
    hasil = st.radio('Bagaimana hasilnya?', ('Pembacaan sensor normal', 'Pembacaan sensor masih anomali'))

   # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write('Terimakasih 👌')
        anomali().append({'Stasiun': ID_choice, 'Tanggal':datetime.datetime.now(), 'Tindakan': tindakan, 'Catatan': catatan, 'Hasil': hasil})
        lap_anomali = pd.DataFrame(anomali())
        lap_anomali.tail(1).to_csv(f'anomali/anomali_{ID_choice}.csv', mode='a', index = False, header = False)
        st.write(tindakan, catatan, hasil)
             

report_anomali = pd.read_csv(f'anomali/anomali_{ID_choice}.csv', names=['Stasiun', 'Tanggal', 'Tindakan', 'Catatan', 'Hasil'], index_col=False)
#report_output = pd.read_csv(f'log_hasil_{ID_choice}.csv', names=['Tanggal', 'Hasil'])

    
st.subheader('Datalog Aksi')
st.write(report_anomali)

