import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat data
day_df = pd.read_csv('day.csv')
day_df.info()
hour_df = pd.read_csv('hour.csv')
hour_df.info()

# Konversi kolom tanggal
datetime_columns = ["dteday"]
for column in datetime_columns:
    day_df[column] = pd.to_datetime(day_df[column])
    hour_df[column] = pd.to_datetime(hour_df[column])

# Filter data untuk tahun 2012
day_df_2012 = day_df[day_df['yr'] == 1]

# Mengelompokkan berdasarkan musim
seasonal_rentals = day_df_2012.groupby('season').agg(
    avg_rentals=('cnt', 'mean'),
    total_rentals=('cnt', 'sum')
).reset_index()

# Mengelompokkan berdasarkan kondisi cuaca
weather_rentals = day_df_2012.groupby('weathersit').agg(
    avg_rentals=('cnt', 'mean'),
    total_rentals=('cnt', 'sum')
).reset_index()

# Filter data untuk musim panas 2011
summer_months = [6, 7, 8]  # bulan Juni, Juli, dan Agustus
hour_df_summer_2011 = hour_df[(hour_df['yr'] == 0) & (hour_df['mnth'].isin(summer_months))]

# Kolom untuk menandakan hari kerja atau akhir pekan
hour_df_summer_2011['is_weekend'] = hour_df_summer_2011['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Menghitung rata-rata penyewaan berdasarkan hari kerja dan akhir pekan
rentals_by_day_type = hour_df_summer_2011.groupby('is_weekend').agg(
    avg_rentals=('cnt', 'mean'),
    total_rentals=('cnt', 'sum')
).reset_index()

# Mengganti nilai is_weekend dengan label
rentals_by_day_type['is_weekend'] = rentals_by_day_type['is_weekend'].map({0: 'Hari Kerja', 1: 'Akhir Pekan'})

# Judul aplikasi
st.title("Analisis Penyewaan Sepeda Capital Bikeshare")

# Visualisasi Rata-Rata Penyewaan Berdasarkan Musim
st.subheader('Rata-Rata Penyewaan Sepeda Berdasarkan Musim (2012)')
fig, ax = plt.subplots()
sns.barplot(x='season', y='avg_rentals', data=seasonal_rentals, palette='viridis', ax=ax)
ax.set_xlabel('Musim')
ax.set_ylabel('Rata-Rata Penyewaan')
ax.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
st.pyplot(fig)

# Visualisasi Rata-Rata Penyewaan Berdasarkan Kondisi Cuaca
st.subheader('Rata-Rata Penyewaan Sepeda Berdasarkan Kondisi Cuaca (2012)')
fig, ax = plt.subplots()
sns.barplot(x='weathersit', y='avg_rentals', data=weather_rentals, palette='Set2', ax=ax)
ax.set_xlabel('Kondisi Cuaca')
ax.set_ylabel('Rata-Rata Penyewaan')
ax.set_xticklabels(['Clear', 'Mist + Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'])
st.pyplot(fig)

# Visualisasi Heatmap dari Matriks Korelasi
correlation_matrix = day_df_2012[['temp', 'hum', 'windspeed', 'cnt']].corr()
st.subheader('Matriks Korelasi antara Variabel Humidity, Temperature, Wind Speed, dan Penyewaan Sepeda (2012)')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
            xticklabels=['Temperature', 'Humidity', 'Wind Speed', 'Bike Rentals'],
            yticklabels=['Temperature', 'Humidity', 'Wind Speed', 'Bike Rentals'], ax=ax)
plt.title('Matriks Korelasi')
st.pyplot(fig)

# Visualisasi Scatter Plot antara Suhu dan Penyewaan Sepeda
st.subheader('Scatter Plot antara Suhu dan Penyewaan Sepeda dengan Garis Regresi')
fig, ax = plt.subplots()
sns.regplot(x='temp', y='cnt', data=day_df_2012, scatter_kws={'s': 10}, line_kws={"color": "red"}, ax=ax)
ax.set_title('Scatter Plot antara Suhu dan Penyewaan Sepeda Harian dengan Garis Regresi')
ax.set_xlabel('Suhu (Ternormalisasi)')
ax.set_ylabel('Jumlah Penyewaan Sepeda')
st.pyplot(fig)

# Visualisasi Rata-Rata Penyewaan Berdasarkan Hari Kerja dan Akhir Pekan
st.subheader('Rata-Rata Penyewaan Sepeda: Hari Kerja vs Akhir Pekan (Musim Panas 2011)')
fig, ax = plt.subplots()
sns.barplot(x='is_weekend', y='avg_rentals', data=rentals_by_day_type, palette=["#FF9999", "#FF4C4C"], ax=ax)
ax.set_xlabel('Tipe Hari')
ax.set_ylabel('Rata-Rata Penyewaan')
st.pyplot(fig)

# Visualisasi Total Penyewaan Berdasarkan Hari Kerja dan Akhir Pekan
st.subheader('Total Penyewaan Sepeda: Hari Kerja vs Akhir Pekan (Musim Panas 2011)')
fig, ax = plt.subplots()
sns.barplot(x='is_weekend', y='total_rentals', data=rentals_by_day_type, palette=["#FF9999", "#FF4C4C"], ax=ax)
ax.set_xlabel('Tipe Hari')
ax.set_ylabel('Total Penyewaan')
st.pyplot(fig)

# Visualisasi Trend Penyewaan Sepeda Sepanjang Hari: Hari Kerja vs. Akhir Pekan
hourly_rentals = hour_df_summer_2011.groupby(['hr', 'is_weekend']).agg(avg_rentals=('cnt', 'mean')).reset_index()
st.subheader('Trend Penyewaan Sepeda Sepanjang Hari: Hari Kerja vs. Akhir Pekan (Musim Panas 2011)')
fig, ax = plt.subplots(figsize=(14, 7))
sns.lineplot(data=hourly_rentals, x='hr', y='avg_rentals', hue='is_weekend', marker='o', ax=ax)
ax.set_xlabel('Jam')
ax.set_ylabel('Rata-Rata Penyewaan')
ax.set_title('Trend Penyewaan Sepeda Sepanjang Hari: Hari Kerja vs. Akhir Pekan (Musim Panas 2011)')
plt.xticks(range(0, 24))
plt.grid()
st.pyplot(fig)
