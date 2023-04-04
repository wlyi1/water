import streamlit as st
import snowflake.connector
import pandas as pd

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
        return cur.fetch().items

rows = run_query('select * from data21 where station = 11')

st.write(pd.DataFrame(rows))
#columns = ['Station', 'pH', 'DO', 'Temp', 'NH4', 'NO3', 'COD', 'BOD', 'LogDate', 'LogTIme']