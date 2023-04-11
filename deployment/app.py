import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih halaman:', ('EDA', 'Predict a Player'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()