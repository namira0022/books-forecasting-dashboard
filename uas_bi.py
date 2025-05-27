import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.impute import SimpleImputer

# Untuk Google Colab mount drive (jika dataset dari Google Drive)
# Jika ingin baca dari Google Drive, ganti path sesuai lokasi dataset di Drive
# Contoh:
# path = '/content/drive/MyDrive/Books_Data_Clean.csv'
# df = pd.read_csv(path)

# Namun karena kamu sudah upload file di /mnt/data, kita pakai itu langsung
df = pd.read_csv('Books_Data_Clean.csv')

# Step 2: Data Cleansing
print("Data Awal:")
print(df.head())
print("\nInfo Data:")
print(df.info())
print("\nMissing Values per Kolom:")
print(df.isnull().sum())

# Handling missing values:
# Kita fokus pada variabel utama untuk forecasting: 'Publishing Year' dan 'units sold'
# Cek apakah ada missing pada kolom tsb:
df_clean = df.copy()

# Drop baris yang 'Publishing Year' atau 'units sold' kosong karena ini penting
df_clean = df_clean.dropna(subset=['Publishing Year', 'units sold'])

# Jika ada kolom lain penting bisa di-impute atau drop sesuai kebutuhan
# Misal kita imputasi Author_Rating & Book_average_rating yang mungkin penting

# Convert 'Author_Rating' to numeric, coercing errors to NaN
# This handles cases where 'Author_Rating' might contain non-numeric strings like 'Novice'
# If 'Novice' should correspond to a specific numerical value, you'd need to map it first.
# Here, we assume non-numeric entries should be treated as missing for imputation.
df_clean['Author_Rating'] = pd.to_numeric(df_clean['Author_Rating'], errors='coerce')

imputer = SimpleImputer(strategy='mean')
cols_to_impute = ['Author_Rating', 'Book_average_rating']

print("\nInfo Data Sebelum Imputasi:")
print(df_clean.info())

for col in cols_to_impute:
    if col in df_clean.columns:
        # Check if the column is numeric before applying mean imputation
        # Note: pd.to_numeric with errors='coerce' already handles non-numeric
        # by turning them into NaN, which is numeric (float).
        # So, this check is mostly redundant after the pd.to_numeric step for Author_Rating.
        # For Book_average_rating, it ensures it was numeric to begin with or became so.
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # --- START OF CHANGES ---
            # Check if the column has any non-null values before imputing
            if df_clean[col].notna().any():
                 # SimpleImputer returns a numpy array, which pandas will handle upon assignment
                 df_clean[col] = imputer.fit_transform(df_clean[[col]]).flatten() # Use flatten() to get 1D array
            else:
                # Handle case where the column is all NaNs
                print(f"Warning: Column '{col}' contains only missing values after cleanup. Skipping mean imputation.")
            # --- END OF CHANGES ---
        else:
             # This block might be hit if Book_average_rating wasn't numeric and imputation was skipped
             print(f"Warning: Column '{col}' is not numeric. Skipping mean imputation for this column.")

print("\nInfo Data Setelah Imputasi:")
print(df_clean.info())

# Cek duplikat
print(f"\nJumlah data sebelum drop duplicates: {df_clean.shape[0]}")
df_clean = df_clean.drop_duplicates()
print(f"Jumlah data setelah drop duplicates: {df_clean.shape[0]}")

# Step 3: Analisis Deskriptif
print("\nStatistik Deskriptif untuk variabel numerik utama:")
print(df_clean[['Publishing Year', 'units sold']].describe())

# Visualisasi trend units sold per tahun (time series)
plt.figure(figsize=(12,5))
sns.lineplot(data=df_clean.groupby('Publishing Year')['units sold'].sum().reset_index(), x='Publishing Year', y='units sold')
plt.title('Trend Units Sold per Publishing Year')
plt.xlabel('Publishing Year')
plt.ylabel('Total Units Sold')
plt.show()

# Step 4: Diagnostic Analysis
# Ubah data menjadi time series indexed by Publishing Year
ts_data = df_clean.groupby('Publishing Year')['units sold'].sum()
ts_data = ts_data.sort_index()

# Plot time series
plt.figure(figsize=(12,5))
plt.plot(ts_data)
plt.title('Time Series Units Sold per Publishing Year')
plt.xlabel('Year')
plt.ylabel('Units Sold')
plt.show()

# Cek stasioneritas dengan Augmented Dickey-Fuller Test
adf_result = adfuller(ts_data)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
if adf_result[1] < 0.05:
    print("Data stasioner, bisa lanjut modeling ARIMA langsung.")
else:
    print("Data tidak stasioner, perlu differencing.")

# Plot ACF dan PACF untuk identifikasi parameter ARIMA
fig, axes = plt.subplots(1,2, figsize=(15,4))
plot_acf(ts_data, lags=10, ax=axes[0])
plot_pacf(ts_data, lags=10, ax=axes[1])
plt.show()

# Jika data tidak stasioner, lakukan differencing 1 kali
if adf_result[1] >= 0.05:
    ts_diff = ts_data.diff().dropna()
    plt.figure(figsize=(12,5))
    plt.plot(ts_diff)
    plt.title('Differenced Time Series')
    plt.show()
else:
    ts_diff = ts_data

# Step 5: Modeling ARIMA
# Tentukan order (p,d,q) berdasarkan ACF PACF dan ADF test
# Misal p=1, d=1 (jika perlu differencing), q=1 sebagai contoh
p = 1
d = 1 if adf_result[1] >= 0.05 else 0
q = 1

# Ensure ts_data index is compatible for modeling (e.g., DatetimeIndex or integer)
# If Publishing Year is integer, ARIMA model in statsmodels can handle it.
# If it represents years, integer index is appropriate.

model = ARIMA(ts_data, order=(p,d,q))
model_fit = model.fit()

print(model_fit.summary())

# Diagnostic plot residual
residuals = model_fit.resid
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(residuals)
plt.title('Residuals dari Model ARIMA')
plt.subplot(122)
sns.histplot(residuals, kde=True)
plt.title('Distribusi Residuals')
plt.show()

# Step 6: Forecasting
# Forecast 5 tahun ke depan (misal 5 tahun)
forecast_steps = 5
forecast = model_fit.get_forecast(steps=forecast_steps)
# The forecast index should continue from the last index of ts_data
# If ts_data index is integer (years), this is correct.
forecast_index = np.arange(ts_data.index.max()+1, ts_data.index.max()+1+forecast_steps)
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

# Visualisasi forecasting
plt.figure(figsize=(12,5))
plt.plot(ts_data, label='Historical')
plt.plot(forecast_series, label='Forecast')
# The confidence intervals also need to be aligned with the forecast_index
conf_int_df = forecast.conf_int()
conf_int_df.index = forecast_index # Align index
plt.fill_between(forecast_series.index, conf_int_df['lower units sold'], conf_int_df['upper units sold'], color='pink', alpha=0.3)
plt.title('Forecast Units Sold per Publishing Year')
plt.xlabel('Publishing Year')
plt.ylabel('Units Sold')
plt.legend()
plt.show()

# Step 7: Prescriptive Analytics
print("Rekomendasi berdasarkan hasil forecasting:")
if forecast_series.mean() > ts_data.mean():
    print("- Perkiraan units sold meningkat. Rekomendasi: Perkuat stok buku dan optimalkan pemasaran untuk memanfaatkan peningkatan permintaan.")
else:
    print("- Perkiraan units sold menurun atau stagnan. Rekomendasi: Lakukan evaluasi strategi pemasaran, diversifikasi produk atau inovasi untuk menarik pembeli baru.")

# Step 8: Dashboard sederhana
import matplotlib.dates as mdates

def dashboard():
    fig, axs = plt.subplots(3,1, figsize=(14,15))

    # 1. Trend units sold per year
    axs[0].plot(ts_data)
    axs[0].set_title('Trend Units Sold per Publishing Year')
    axs[0].set_xlabel('Publishing Year')
    axs[0].set_ylabel('Units Sold')

    # 2. Residual plot
    axs[1].plot(residuals)
    axs[1].set_title('Residuals ARIMA Model')
    axs[1].set_xlabel('Publishing Year')
    # Ensure residual index aligns with the historical data index for plotting
    axs[1].set_xticks(ts_data.index) # Use historical years for x-ticks

    # 3. Forecast plot
    axs[2].plot(ts_data, label='Historical')
    axs[2].plot(forecast_series, label='Forecast')
    # Ensure confidence intervals are aligned with the forecast_series index
    conf_int_df = forecast.conf_int()
    conf_int_df.index = forecast_series.index # Align index for plotting
    plt.fill_between(forecast_series.index, conf_int_df['lower units sold'], conf_int_df['upper units sold'], color='pink', alpha=0.3)
    axs[2].set_title('Forecast Units Sold per Publishing Year')
    axs[2].set_xlabel('Publishing Year')
    axs[2].set_ylabel('Units Sold')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

dashboard()


