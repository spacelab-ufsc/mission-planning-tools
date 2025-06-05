import pandas as pd
import numpy as np
import os
import glob
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Caminho da pasta com os arquivos CSV
caminho_pasta = "/home/rebecca/python"  # ajuste conforme necessário
arquivos_csv = glob.glob(os.path.join(caminho_pasta, "*.csv"))

# Lista para armazenar todos os DataFrames
dfs = []

# Carrega todos os arquivos CSV da pasta
for arquivo in arquivos_csv:
    print(f"Lendo {arquivo}")
    df_temp = pd.read_csv(arquivo)
    dfs.append(df_temp)

# Concatena todos os DataFrames em um só
df = pd.concat(dfs, ignore_index=True)

# Converte tempo para ano decimal (feature importante)
df['Tempo (UTC)'] = pd.to_datetime(df['Tempo (UTC)'], errors='coerce')
df = df.dropna(subset=["Tempo (UTC)"])  # Remove linhas com datas inválidas

df['yeardec'] = df['Tempo (UTC)'].apply(
    lambda dt: dt.year + (dt.timetuple().tm_yday + dt.hour / 24) / 365.25
)

# Verifica se colunas esperadas existem
colunas_necessarias = ['lat', 'lon', 'alt_km', 'East (G)', 'North (G)', 'Down (G)']
for col in colunas_necessarias:
    if col not in df.columns:
        raise ValueError(f"Coluna obrigatória '{col}' não encontrada no DataFrame!")

# Define entradas e saídas
X = df[['lat', 'lon', 'alt_km', 'yeardec']].values
y = df[['East (G)', 'North (G)', 'Down (G)']].values

# Normaliza os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Modelo com Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    Dense(3)  # Saída com 3 valores: East, North, Down
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Treinamento
print("Treinando modelo...")
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Avaliação
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro quadrático médio (normalizado): {mse:.6f}")

# Função para prever o campo magnético com NN
def prever_campo_magnetico(lat, lon, alt_km, yeardec):
    entrada = np.array([[lat, lon, alt_km, yeardec]])
    entrada_scaled = scaler_X.transform(entrada)
    pred_scaled = model.predict(entrada_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)

    return {
        "East (G)": pred[0][0],
        "North (G)": pred[0][1],
        "Down (G)": pred[0][2]
    }

# Interface com usuário
print("\n--- Previsão do Campo Magnético com IA ---")
try:
    lat = float(input("Latitude (graus): "))
    lon = float(input("Longitude (graus): "))
    alt_km = float(input("Altitude (km): "))
    data_input = input("Data e hora (YYYY-MM-DD HH:MM:SS): ")

    timestamp = datetime.datetime.strptime(data_input, "%Y-%m-%d %H:%M:%S")
    yeardec = timestamp.year + (timestamp.timetuple().tm_yday + timestamp.hour / 24) / 365.25

    resultado = prever_campo_magnetico(lat, lon, alt_km, yeardec)
    print("\nPrevisão do campo magnético:")
    print(f"East  (G): {resultado['East (G)']:.6f}")
    print(f"North (G): {resultado['North (G)']:.6f}")
    print(f"Down  (G): {resultado['Down (G)']:.6f}")

except Exception as e:
    print(f"Erro: {e}")

