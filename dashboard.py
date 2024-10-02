import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load datasets directly
day_df = pd.read_csv('day.csv')
hour_df = pd.read_csv('hour.csv')

# Title of the app
st.title("Dashboard Analisis Penyewaan Sepeda berdasarkan Musim, Cuaca, dan Hari Kerja")

# Sidebar menu
st.sidebar.header("Pengaturan")
option = st.sidebar.selectbox(
    "Pilih Analisis",
    ["Rata-Rata Penyewaan Berdasarkan Musim", "Rata-Rata Penyewaan Berdasarkan Cuaca", "Analisis Time Series", "Penyewaan Berdasarkan Hari Kerja/Akhir Pekan"]
)

# Question 1: Analysis for daily data
if option in ["Rata-Rata Penyewaan Berdasarkan Musim", "Rata-Rata Penyewaan Berdasarkan Cuaca", "Analisis Time Series"]:
    # Filter data for 2012
    day_df_2012 = day_df[day_df['yr'] == 1]

    # Label mapping for season and weather
    season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    day_df_2012['season'] = day_df_2012['season'].map(season_labels)

    weather_labels = {1: 'Clear', 2: 'Mist + Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    day_df_2012['weathersit'] = day_df_2012['weathersit'].map(weather_labels)

    # Group by season and weather
    seasonal_rentals = day_df_2012.groupby('season').agg(
        avg_rentals=('cnt', 'mean'),
        total_rentals=('cnt', 'sum')
    ).reset_index()

    weather_rentals = day_df_2012.groupby('weathersit').agg(
        avg_rentals=('cnt', 'mean'),
        total_rentals=('cnt', 'sum')
    ).reset_index()

    # Option for Season-based analysis
    if option == "Rata-Rata Penyewaan Berdasarkan Musim":
        st.subheader("Rata-Rata Penyewaan Berdasarkan Musim (2012)")
        st.write(seasonal_rentals)

        # Visualization
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(x='season', y='avg_rentals', data=seasonal_rentals, palette='viridis', ax=ax1)
        ax1.set_title('Rata-Rata Penyewaan Berdasarkan Musim (2012)')
        ax1.set_xlabel('Musim')
        ax1.set_ylabel('Rata-Rata Penyewaan')
        st.pyplot(fig1)

    # Option for Weather-based analysis
    elif option == "Rata-Rata Penyewaan Berdasarkan Cuaca":
        st.subheader("Rata-Rata Penyewaan Berdasarkan Kondisi Cuaca (2012)")
        st.write(weather_rentals)

        # Visualization
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.barplot(x='weathersit', y='avg_rentals', data=weather_rentals, palette='Set2', ax=ax2)
        ax2.set_title('Rata-Rata Penyewaan Berdasarkan Kondisi Cuaca (2012)')
        ax2.set_xlabel('Kondisi Cuaca')
        ax2.set_ylabel('Rata-Rata Penyewaan')
        st.pyplot(fig2)

    # Option for Time Series analysis
    elif option == "Analisis Time Series":
        st.subheader("Analisis Time Series Penyewaan Sepeda Harian (2012)")

        # Group by date and calculate total rentals per day
        daily_rentals = day_df_2012.groupby('dteday')['cnt'].sum().reset_index()
        daily_rentals['dteday'] = pd.to_datetime(daily_rentals['dteday'])
        daily_rentals.set_index('dteday', inplace=True)

        # Plot total daily rentals
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        ax3.plot(daily_rentals.index, daily_rentals['cnt'], label='Total Penyewaan', color='blue')
        ax3.set_title('Total Penyewaan Sepeda Harian (2012)')
        ax3.set_xlabel('Tanggal')
        ax3.set_ylabel('Jumlah Penyewaan')
        ax3.legend()
        ax3.grid()
        st.pyplot(fig3)

        # Decompose Time Series
        st.subheader("Decomposition Time Series (Additive Model)")
        decomposition = sm.tsa.seasonal_decompose(daily_rentals['cnt'], model='additive')
        fig4 = decomposition.plot()
        fig4.set_size_inches(14, 10)
        st.pyplot(fig4)

        # Autocorrelation Analysis
        st.subheader("Autokorelasi dan Partial Autokorelasi")
        fig5, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 6))

        plot_acf(daily_rentals['cnt'], lags=30, ax=ax_acf)
        ax_acf.set_title('Autokorelasi (ACF)')

        plot_pacf(daily_rentals['cnt'], lags=30, ax=ax_pacf)
        ax_pacf.set_title('Partial Autokorelasi (PACF)')
        st.pyplot(fig5)

# Question 2: Analysis for hourly data
if option == "Penyewaan Berdasarkan Hari Kerja/Akhir Pekan":
    st.subheader("Penyewaan Berdasarkan Hari Kerja/Akhir Pekan (Musim Panas 2011)")

    # Filter for summer 2011 (June, July, August)
    summer_months = [6, 7, 8]
    hour_df_summer_2011 = hour_df[(hour_df['yr'] == 0) & (hour_df['mnth'].isin(summer_months))]

    # Create a column for weekend/weekday
    hour_df_summer_2011['is_weekend'] = hour_df_summer_2011['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Group by weekend/weekday
    rentals_by_day_type = hour_df_summer_2011.groupby('is_weekend').agg(
        avg_rentals=('cnt', 'mean'),
        total_rentals=('cnt', 'sum')
    ).reset_index()

    rentals_by_day_type['is_weekend'] = rentals_by_day_type['is_weekend'].map({0: 'Hari Kerja', 1: 'Akhir Pekan'})
    st.write(rentals_by_day_type)

    # Visualization for rentals by day type
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='is_weekend', y='avg_rentals', data=rentals_by_day_type, palette=["#FF9999", "#FF4C4C"], ax=ax6)
    ax6.set_title('Rata-Rata Penyewaan Sepeda: Hari Kerja vs Akhir Pekan (2011)')
    ax6.set_xlabel('Tipe Hari')
    ax6.set_ylabel('Rata-Rata Penyewaan')
    st.pyplot(fig6)
