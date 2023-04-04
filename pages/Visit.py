import pandas as pd
import datetime
from datetime import datetime as dt
import streamlit as st
        
@st.cache(allow_output_mutation=True)
def kunjungan():
    return[]

@st.cache(allow_output_mutation=True)
def kalibrasi():
    return[]

st.title('Form Kunjungan Kerja MA 👋')
files_id = pd.read_csv('id_stasiun.csv')
params = ['pH', 'DO', 'NH4', 'NO3', 'COD', 'BOD', 'TSS']

tab1, tab2, tab3 = st.tabs(['Kalibrasi', 'Kunjungan ONLIMO', 'Kunjungan Swasta' ,])

with tab2:
    st.header('Form pengisian hasil kunjungan')
    with st.form("Form Pengisian Hasil Kunjungan "):
        st.write("Form Pengisian Hasil Kunjungan 😀")
        nama_tab2 = st.text_input("Nama: ")
        stasiun_tab2 = st.selectbox('Stasiun: ', files_id['CODE'])
        tgl_in = st.date_input('Tanggal masalah terjadi: ')
        masalah_tab2 = st.text_area('Apa masalahnya? ')
        tgl_out = st.date_input('Tanggal kunjungan: ?')
        tindakan_tab2 = st.text_area('Apa tindakannya? ')

    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write('Terimakasih 👌')
            kunjungan().append({'Nama': nama_tab2, 'Stasiun': stasiun_tab2, 'Tgl_in': tgl_in, 'Masalah': masalah_tab2, 'Tgl_out': tgl_out, 'Tindakan':tindakan_tab2})
            lap_kunjungan = pd.DataFrame(kunjungan())
            lap_kunjungan.tail(1).to_csv(f'kunjungan_{stasiun_tab2}.csv', mode='a', index = False, header = False)
             
with tab1:
    st.header('Form pengisian hasil kalibrasi')
    with st.form("Form Kalibrasi"):
        st.write("Form Hasil Kalibrasi 😄")
        name = st.text_input("Nama: ")
        tgl = st.date_input('Tanggal kalibrasi: ')
        station = st.selectbox('Stasiun: ', files_id['CODE'])
        for pr in params[:4]:
            globals() [f'{pr}'] = st.multiselect(f'Sensor {pr}', ['Kalibrasi ulang', 'Ganti electrode', 'Ganti tip'], key=f'{pr}')
        opm = st.multiselect('Tindakan OPM', ['Kalibrasi ulang', 'Ganti UV-Lamp', 'Ganti board'])
        kendala = st.text_area('Apakah ada kendala? ')


    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write('Terimakasih 👌')
            kalibrasi().append({'Nama': name, 'Stasiun': station, 'Tgl': tgl, 'pH': pH, 'DO': DO, 'NH4': NH4, 'NO3': NO3, 'OPM': opm, 'Kendala':kendala})
            lap_kalibrasi = pd.DataFrame(kalibrasi())
            lap_kalibrasi.tail(1).to_csv(f'kalibrasi_{station}.csv', mode='a', index = False, header = False)
            
with tab3:
    st.header('Form pengisian hasil kunjungan')
    with st.form("Form kunjungan swasta "):
        st.write("Form Pengisian Hasil Kunjungan 😀")
        nama_tab3 = st.text_input("Nama: ")
        pabrik = st.text_input("Perusahaan: ")
        tgl_datang = st.date_input('Tanggal kunjungan: ')
        keterangan = st.text_area('Kegiatan ? ')
 
    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write('Terimakasih 👌')
            kunjungan().append({'Nama': nama_tab3, 'Perusahaan': pabrik, 'Kedatangan': tgl_datang, 'Keterangan': keterangan})
            lap_kunjungan = pd.DataFrame(kunjungan())
            lap_kunjungan.tail(1).to_csv(f'kunjungan_{pabrik}.csv', mode='a', index = False, header = False)
             

            
            
        
        
        
        
        
        
        
        