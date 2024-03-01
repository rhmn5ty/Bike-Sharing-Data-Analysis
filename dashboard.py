import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler\

sns.set(style='darkgrid')

day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")
df = day_df.merge(hour_df, on='dteday', how='inner', suffixes=('_d', '_h'))

df.sort_values(by="dteday", inplace=True)
df.reset_index(inplace=True)

df["dteday"] = pd.to_datetime(df["dteday"])
 

min_date = df["dteday"].min()
max_date = df["dteday"].max()
 
with st.sidebar:
    st.header("Rahman Satya's Final Project")
    # Menambahkan logo perusahaan
    st.image("Logo-Satya.png")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = df[(df["dteday"] >= str(start_date)) & 
                (df["dteday"] <= str(end_date))]

st.header('Bike Dashboard :bike:')
st.subheader('Belajar Analisis Data dengan Python')

st.subheader('Graph 1 - Clustering Data Waktu per Jam Pengguna Bike Sharing Berdasarkan Suhu dan Pengguna Kasual')


# Pilih fitur untuk clustering
features = main_df[['temp_h', 'atemp_h', 'hum_h', 'windspeed_h', 'casual_h', 'registered_h', 'cnt_h']]

# Normalisasi data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Gunakan K-Means untuk clustering
kmeans = KMeans(n_clusters=4, random_state=42)
main_df['cluster'] = kmeans.fit_predict(features_scaled)

# Visualisasi hasil clustering menggunakan scatter plot
# Asumsi kita ingin melihat hubungan antara 'temp' dan 'casual' dengan pewarnaan berdasarkan cluster
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=main_df, x='temp_h', y='cnt_h', hue='cluster', palette='viridis', alpha=0.5)

ax.set_title('Clustering Data Waktu per Jam Pengguna Bike Sharing Berdasarkan Suhu dan Pengguna Kasual', color='white')
ax.set_xlabel('Temperatur', color='white')
ax.set_ylabel('Jumlah Pengguna Kasual', color='white')
ax.tick_params(colors='white', which='both')

fig.patch.set_facecolor('#303030')
ax.set_facecolor('#505050')

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Cluster')
# Menampilkan plot
st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    st.subheader('Graph 2 - Pola Penggunaan Bike Sharing Berdasarkan Jam dalam Sehari')

    # Konversi 'dteday' ke datetime dan ekstraksi informasi hari kerja
    main_df['dteday'] = pd.to_datetime(main_df['dteday'])
    main_df['is_weekend'] = main_df['weekday_d'].apply(lambda x: 0 if x in [1, 2, 3, 4, 5] else 1)

    # Menggunakan seaborn untuk membuat Line Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=main_df, x='hr', y='cnt_h', hue='is_weekend', estimator=np.mean, ci=None, ax=ax)

    # Menyesuaikan label dan judul
    ax.set_title('Pola Penggunaan Bike Sharing Berdasarkan Jam dalam Sehari', color='white')
    ax.set_xlabel('Jam', color='white')
    ax.set_ylabel('Rata-rata Jumlah Penggunaan', color='white')
    ax.tick_params(colors='white', which='both')

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Hari Kerja', 'Akhir Pekan'], title='Tipe Hari')

    fig.patch.set_facecolor('#303030')
    ax.set_facecolor('#505050')

    st.pyplot(fig)
    

with col2:


    st.subheader('Graph 3 - Pengaruh Musim terhadap Jumlah Pengguna Terdaftar dan Kasual')


    # Mengagregasi data
    usage_by_season = main_df.groupby('season_d').agg({'casual_d':'mean', 'registered_d':'mean'}).reset_index()

    # Mengubah data agar sesuai format untuk visualisasi side-by-side bars
    # Melelehkan DataFrame agar 'casual' dan 'registered' menjadi dua baris terpisah untuk setiap 'season'
    usage_melted = pd.melt(usage_by_season, id_vars=['season_d'], value_vars=['casual_d', 'registered_d'], 
                        var_name='User Type', value_name='Average Usage')

    # Visualisasi data dengan sns.barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=usage_melted, x='season_d', y='Average Usage', hue='User Type', palette=['blue', 'red'])

    ax.set_title('Pengaruh Musim terhadap Jumlah Pengguna Terdaftar dan Kasual', color='white')
    ax.set_xlabel('Musim', color='white')
    ax.set_ylabel('Rata-Rata Jumlah Penggunaan', color='white')
    ax.set_xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])  # Ganti label sesuai data Anda


    ax.tick_params(colors='white', which='both')
    fig.patch.set_facecolor('#303030')
    ax.set_facecolor('#505050')

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, title='Tipe Pengguna')

    st.pyplot(fig)

st.subheader('Graph 4 - Pengaruh Kondisi Cuaca terhadap Jumlah Pengguna Kasual dan Terdaftar')


# Visualisasi pengaruh weathersit terhadap jumlah pengguna kasual dan terdaftar
fig, ax = plt.subplots(figsize=(14, 7))

# Scatter plot untuk 
df_weather1 = main_df[main_df['weathersit_d'] == 1]
sns.scatterplot(data=df_weather1, x='casual_d', y='registered_d', color='green', alpha=0.5, label='Clear')

# Scatter plot untuk 
df_weather2 = main_df[main_df['weathersit_d'] == 2]
sns.scatterplot(data=df_weather2, x='casual_d', y='registered_d', color='yellow', alpha=0.5, label='Mist/Cloudy')

# Scatter plot untuk 
df_weather3 = main_df[main_df['weathersit_d'] == 3]
sns.scatterplot(data=df_weather3, x='casual_d', y='registered_d', color='orange', alpha=0.5, label='Light Snow/Rain')

# Scatter plot untuk 
df_weather4 = main_df[main_df['weathersit_d'] == 4]
sns.scatterplot(data=df_weather4, x='casual_d', y='registered_d', color='red', alpha=0.5, label='Heavy Rain/Snow/Fog')

# Memberikan judul dan label yang sesuai
ax.set_title('Pengaruh Kondisi Cuaca terhadap Jumlah Pengguna Kasual dan Terdaftar', color='white')
ax.set_xlabel('Jumlah Pengguna Kasual', color='white')
ax.set_ylabel('Jumlah Pengguna Terdaftar', color='white')
ax.tick_params(colors='white', which='both')

fig.patch.set_facecolor('#303030')
ax.set_facecolor('#505050')

st.pyplot(fig)