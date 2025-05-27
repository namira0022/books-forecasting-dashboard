import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.impute import SimpleImputer

# Load dataset (pastikan file ada di folder yang sama atau sesuaikan path)
df = pd.read_csv('Books_Data_Clean.csv')

# Data cleansing dan imputasi
df_clean = df.copy()
df_clean = df_clean.dropna(subset=['Publishing Year', 'units sold'])
df_clean['Author_Rating'] = pd.to_numeric(df_clean['Author_Rating'], errors='coerce')

imputer = SimpleImputer(strategy='mean')
cols_to_impute = ['Author_Rating', 'Book_average_rating']
for col in cols_to_impute:
    if col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            if df_clean[col].notna().any():
                df_clean[col] = imputer.fit_transform(df_clean[[col]]).flatten()

df_clean = df_clean.drop_duplicates()

# Time series data
ts_data = df_clean.groupby('Publishing Year')['units sold'].sum().sort_index()

# ADF Test
adf_result = adfuller(ts_data)
p_value = adf_result[1]

# Differencing jika perlu
d = 0
if p_value >= 0.05:
    d = 1
    ts_used = ts_data.diff().dropna()
else:
    ts_used = ts_data

# ARIMA modeling default p=1, d berdasarkan ADF test, q=1
p = 1
q = 1
model = ARIMA(ts_data, order=(p,d,q))
model_fit = model.fit()

# Residuals dan forecasting
residuals = model_fit.resid
forecast_steps = 5
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = np.arange(ts_data.index.max()+1, ts_data.index.max()+1+forecast_steps)
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
conf_int_df = forecast.conf_int()
conf_int_df.index = forecast_index

# Streamlit UI
st.title("Forecasting Units Sold Buku dengan ARIMA")

st.subheader("Statistik Deskriptif Data")
st.dataframe(df_clean[['Publishing Year', 'units sold']].describe())

st.subheader("Hasil Augmented Dickey-Fuller Test")
st.write(f"ADF Statistic: {adf_result[0]:.4f}")
st.write(f"p-value: {p_value:.4f}")
if d == 1:
    st.warning("Data tidak stasioner. Melakukan differencing sekali (d=1).")
else:
    st.success("Data stasioner. Bisa langsung modeling ARIMA.")

st.subheader("Plot Trend Units Sold per Publishing Year")
fig1, ax1 = plt.subplots()
sns.lineplot(data=ts_data, ax=ax1)
ax1.set_xlabel('Publishing Year')
ax1.set_ylabel('Units Sold')
st.pyplot(fig1)

st.subheader("Plot ACF dan PACF")
fig2, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(ts_data, lags=10, ax=axes[0])
plot_pacf(ts_data, lags=10, ax=axes[1])
st.pyplot(fig2)

st.subheader("Residuals dari Model ARIMA")
fig3, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].plot(residuals)
axes[0].set_title('Residuals')
sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_title('Distribusi Residuals')
st.pyplot(fig3)

st.subheader("Forecast 5 Tahun ke Depan")
fig4, ax4 = plt.subplots()
ax4.plot(ts_data, label='Historical')
ax4.plot(forecast_series, label='Forecast')
ax4.fill_between(forecast_series.index, conf_int_df['lower units sold'], conf_int_df['upper units sold'], color='pink', alpha=0.3)
ax4.set_xlabel('Publishing Year')
ax4.set_ylabel('Units Sold')
ax4.legend()
st.pyplot(fig4)

# Rekomendasi
st.subheader("Rekomendasi berdasarkan Forecasting")
if forecast_series.mean() > ts_data.mean():
    st.success("Perkiraan units sold meningkat. Rekomendasi: Perkuat stok buku dan optimalkan pemasaran.")
else:
    st.warning("Perkiraan units sold menurun atau stagnan. Rekomendasi: Evaluasi strategi pemasaran, diversifikasi produk, atau inovasi.")

