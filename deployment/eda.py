import streamlit as st
import pandas as pd
import seaborn as sns
import calendar
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title='Customer Churn',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():

    # Membuat title
    st.title('Churn Prediction')
    st.write('Page ini dibuat oleh Bagus Tirta Aji Hakim')
    '''
    Customer churn adalah kehilangan pelanggan dari suatu bisnis. Churn dihitung dari berapa banyak pelanggan meninggalkan bisnis Anda dalam waktu tertentu. 
    Customer churn penting diketahui bisnis karena merupakan gambaran kesuksesan suatu bisnis dalam mempertahankan pelanggan.
    '''
    st.markdown('---')

    # Membuat Sub Headrer
    # st.subheader('EDA untuk Analisa Churn Dataset')

    # Menambahkan Gambar
    # image = Image.open('score.png')
    # st.image(image, caption='FIFA 2022')

    # Menambahkan Deskripsi
    # st.write('# Halo')
    # st.write('## Halo')
    # st.write('### Halo')

    # Membuat Garis Lurus
    # st.markdown('---')

    # Magic Syntax
    '''
    Pada page kali ini, penulis akan melakukan eksplorasi sederhana.
    Dataset yang digunakan adalah Churn.
    Dataset ini berasal dari GitHub.
    '''

    # Show DataFrame
    data = pd.read_csv('churn.csv')
    st.write('Berikut adalah tampilan dataset yang akan digunakan:')
    data_duplicate = data.copy()

    # Preprocessing
    # merubah tipe data object menjadi datetime
    data_duplicate["joining_date"] = pd.to_datetime(
        data_duplicate["joining_date"])
    data_duplicate["last_visit_time"] = pd.to_datetime(
        data_duplicate["last_visit_time"])

    # mengambil nilai spesifik seperti jam, tahun, dan bulan
    data_duplicate['last_visit_time'] = pd.to_datetime(
        data_duplicate['last_visit_time'], format='%H:%M').dt.hour
    data_duplicate['joining_year'] = pd.DatetimeIndex(
        data_duplicate['joining_date']).year
    data_duplicate['joining_month'] = pd.DatetimeIndex(
        data_duplicate['joining_date']).month

    # merubah bulan yang nilainya berbentuk angka menjadi sebuah string, contoh: 1 menjadi Jan (January)
    data_duplicate['joining_month'] = data_duplicate['joining_month'].apply(
        lambda x: calendar.month_abbr[x])

    # mengubah integer menjadi object, karena joining year termasuk sebagai data_duplicate kategorik
    data_duplicate['joining_year'] = data_duplicate['joining_year'].astype(str)

    # melakukan dropping kolom yang tidak digunakan dan melakukan rename column agar mudah diimplementasikan
    data_duplicate.drop(['user_id'], axis=1, inplace=True)
    data_duplicate.drop(['joining_date'], axis=1, inplace=True)
    data_duplicate.rename(columns={'churn_risk_score': 'churn'}, inplace=True)

    st.dataframe(data_duplicate)

    st.markdown('---')
    st.write('## EDA Sederhana pada Dataset:')

    st.write('#### Age')
    fig = plt.figure(figsize=(15, 5))
    sns.countplot(data=data_duplicate, x='age', hue='churn')
    st.pyplot(fig)
    st.write('Umur customer terdistribusi secara merata dari umur 10 tahun hingga 64 tahun dimana tingkat churn tertinggi terjadi pada customer yang berumur 38 tahun.')

    st.write('#### Status Complaint terhadap Churn')
    fig = plt.figure(figsize=(15, 5))
    sns.countplot(data=data_duplicate, x='complaint_status', hue='churn')
    st.pyplot(fig)
    '''
    Berdasarkan grafik, peneliti dapat menyimpulkan jika status complaint yang Not Applicable lebih mendominasi dibandingkan status complaint lainnya. 
    Hal ini tentunya menjadi sebuah pekerjaan rumah bagi perusahaan untuk memperbaiki layanannya agar bagaimana complaint yang disampaikan oleh customer dapat terselesaikan dengan baik.
    Mengapa? karena jika dilihat kembali pada status complaint yang Not Applicable juga memiliki tingkat churn yang lebih tinggi dibandingkan status complaint yang lain.
    '''

    st.write('#### Feedback terhadap Churn')
    fig = plt.figure(figsize=(20, 5))
    sns.countplot(data=data_duplicate, x='feedback', hue='churn')
    st.pyplot(fig)
    '''
    Kemudian ternyata, potensi customer yang akan churn akan banyak terjadi jika terjadi beberapa hal berikut:

    1. Dari website perusahaan yang mungkin sulit digunakan oleh customer untuk melakukan pembelian atau bisa jadi tampilan website yang sudah jadul sehingga mengurangi minat customer untuk tidak belanja di perusahaan tersebut.
    2. Buruknya customer service dalam melayani customer, hal ini berhubungan dengan status complaint Not Applicable yang berarti complaint yang disampaikan oleh customer tidak tersampaikan dengan baik oleh customer service
    3. Banyaknya tampilan iklan pada website yang mana juga berhubungan dengan poin nomor 1
    4. Kualitas produk yang buruk sehingga perusahaan perlu melakukan evaluasi kembali mengenai produk yang telah dirilis.
    '''

    st.markdown('---')

    # Membuat Histogram Berdasakan Input User
    st.write(
        '#### Anda dapat melihat distirbusi data berdasarkan kolom yang anda pilih')
    pilihan = st.selectbox('Pilih column: ', ('age', 'last_visit_time', 'days_since_last_login',
                           'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'))
    fig = plt.figure(figsize=(15, 5))
    sns.histplot(data=data_duplicate, x=pilihan, bins=30, kde=True)
    st.pyplot(fig)


# agar bisa di run sendiri, tanpa dari main
if __name__ == '__main__':
    run()
