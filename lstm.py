import tensorflow as tf
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("Matplotlib berhasil diimpor!")

import pandas as pd
import numpy as np

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
import numpy as np

# Mengambil data curah hujan
data_hujan = df[["rainfall_mm"]].values

# Membagi data: 80% training dan 20% testing
train_size = int(len(data_hujan) * 0.8)

data_train = data_hujan[:train_size]

# Normalisasi berdasarkan data training saja
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

data_normalisasi = scaler.transform(data_hujan)

# Membuat sequence
time_step = 3

X_train, y_train = [], []
X_test, y_test = [], []

for i in range(len(data_normalisasi) - time_step):
    X = data_normalisasi[i:i + time_step]
    y = data_normalisasi[i + time_step]

    if i + time_step < train_size:
        X_train.append(X)
        y_train.append(y)
    else:
        X_test.append(X)
        y_test.append(y)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("\nPembagian Data:")
print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))
print("Bentuk X_train:", X_train.shape)
print("Bentuk X_test:", X_test.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Membangun model LSTM
model = Sequential()

# Menambahkan layer LSTM
model.add(LSTM(16, input_shape=(3, 1)))

# Menambahkan layer output
model.add(Dense(1))

# Mengatur proses pembelajaran
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Menampilkan arsitektur model
model.summary()


# Melatih model LSTM
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=4,
    shuffle=False,
    verbose=1
)

print("\nTraining selesai!")


# Melakukan prediksi pada data testing
prediksi = model.predict(X_test)

# Mengembalikan hasil normalisasi ke satuan mm
prediksi_mm = scaler.inverse_transform(prediksi)
aktual_mm = scaler.inverse_transform(y_test)

# Menampilkan hasil prediksi dan aktual
print("\nHasil Pengujian Model LSTM:")

for i in range(len(prediksi_mm)):
    print(
        f"Data {i+1}: "
        f"Prediksi = {prediksi_mm[i][0]:.2f} mm, "
        f"Aktual = {aktual_mm[i][0]:.2f} mm"
    )

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Menghitung nilai evaluasi model
mae = mean_absolute_error(aktual_mm, prediksi_mm)
rmse = np.sqrt(mean_squared_error(aktual_mm, prediksi_mm))

print("\nEvaluasi Model LSTM:")
print(f"MAE  : {mae:.2f} mm")
print(f"RMSE : {rmse:.2f} mm")

# Membuat grafik perbandingan aktual dan prediksi

periode_test = df["Periode"].iloc[-len(aktual_mm):]

plt.figure(figsize=(10, 5))

plt.plot(
    periode_test,
    aktual_mm.flatten(),
    marker="o",
    label="Aktual"
)

plt.plot(
    periode_test,
    prediksi_mm.flatten(),
    marker="o",
    label="Prediksi LSTM"
)

plt.title("Perbandingan Curah Hujan Aktual dan Prediksi LSTM")
plt.xlabel("Periode")
plt.ylabel("Curah Hujan (mm)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prediksi curah hujan bulan Juli 2026

data_terakhir = data_normalisasi[-time_step:].reshape(1, time_step, 1)

prediksi_bulan_berikutnya = model.predict(data_terakhir)

prediksi_juli_mm = scaler.inverse_transform(prediksi_bulan_berikutnya)

print("\nPrediksi Curah Hujan Bulan Juli 2026:")
print(f"{prediksi_juli_mm[0][0]:.2f} mm")

# Grafik data historis curah hujan BMKG

plt.figure(figsize=(12, 5))

plt.plot(
    df["Periode"],
    df["rainfall_mm"],
    marker="o",
    label="Curah Hujan Aktual"
)

plt.title("Data Historis Curah Hujan BMKG")
plt.xlabel("Periode")
plt.ylabel("Curah Hujan (mm)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()