import pandas as pd

# Membaca dataset Excel
df = pd.read_excel("dataset_curah_hujan_bmkg_2024_2026.xlsx")

# Menampilkan seluruh dataset
print(df)

# Menampilkan informasi dataset
print("\nInformasi Dataset:")
print(df.info())

# Menampilkan statistik data
print("\nStatistik Curah Hujan:")
print(df["rainfall_mm"].describe())

from sklearn.preprocessing import MinMaxScaler

# Mengambil kolom curah hujan
data_hujan = df[["rainfall_mm"]].values

# Membuat scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalisasi data
data_normalisasi = scaler.fit_transform(data_hujan)

print("\nHasil Normalisasi:")
print(data_normalisasi)