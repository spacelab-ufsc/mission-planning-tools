import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import os
import re

# Configuração do caminho dos dados
caminho_pasta = "/home/rebecca/python"

# 1. Pré-processamento dos dados com extração de parâmetros da TLE
def load_and_preprocess_data(csv_folder):
    all_data = []
    
    for filename in os.listdir(csv_folder):
        if filename.startswith('campo_magnetico_') and filename.endswith('.csv'):
            # Extrai NORAD ID do nome do arquivo
            norad_id = filename.split('_')[2].split('.')[0]
            df = pd.read_csv(os.path.join(csv_folder, filename))
            
            # Encontra o arquivo TLE correspondente
            tle_file = os.path.join(csv_folder, f"tle_{norad_id}.txt")
            if os.path.exists(tle_file):
                with open(tle_file, 'r') as f:
                    tle_lines = f.readlines()
                    if len(tle_lines) >= 2:
                        line1 = tle_lines[0].strip()
                        line2 = tle_lines[1].strip()
                        
                        # Extrai parâmetros da TLE
                        tle_params = extract_tle_features(line1, line2)
            
            for idx, row in df.iterrows():
                time = datetime.datetime.fromisoformat(row['Tempo (UTC)'])
                
                # Adiciona todos os dados
                data_point = {
                    'timestamp': time.timestamp(),
                    'day_of_year': time.timetuple().tm_yday,
                    'hour_of_day': time.hour,
                    'east_g': row['East (G)'],
                    'north_g': row['North (G)'],
                    'down_g': row['Down (G)'],
                    'norad_id': norad_id
                }
                
                # Adiciona parâmetros da TLE se disponíveis
                if 'tle_params' in locals():
                    data_point.update(tle_params)
                
                all_data.append(data_point)
    
    return pd.DataFrame(all_data)

# Função para extrair características da TLE
def extract_tle_features(line1, line2):
    try:
        parts1 = line1.split()
        parts2 = line2.split()
        
        return {
            'inclination': float(parts2[2]),
            'raan': float(parts2[3]),  # Right Ascension of the Ascending Node
            'eccentricity': float(f"0.{parts2[4]}"),  # Formato 0000000 para 0.0000000
            'arg_perigee': float(parts2[5]),  # Argument of Perigee
            'mean_anomaly': float(parts2[6]),
            'mean_motion': float(parts2[7][:11]),
            'epoch_year': float(parts1[3][:2]),  # Ano da época
            'epoch_day': float(parts1[3][2:]),   # Dia da época
            'bstar': float(f"{parts1[4][0]}.{parts1[4][1:]}e{parts1[4][-1]}")  # Coef. arrasto
        }
    except Exception as e:
        print(f"Erro ao extrair TLE: {e}")
        return {}

# 2. Modelo de IA aprimorado
def build_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(3)  # Saída para East, North, Down (G)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

# 3. Previsão direta sem propagação orbital
def predict_magnetic_field(model, scaler, tle_line1, tle_line2, target_datetime):
    # Extrai características da TLE
    tle_params = extract_tle_features(tle_line1, tle_line2)
    
    # Características temporais
    features = {
        'timestamp': target_datetime.timestamp(),
        'day_of_year': target_datetime.timetuple().tm_yday,
        'hour_of_day': target_datetime.hour
    }
    
    # Combina todos os parâmetros
    features.update(tle_params)
    
    # Converte para array no formato correto
    feature_names = ['timestamp', 'day_of_year', 'hour_of_day', 
                    'inclination', 'raan', 'eccentricity', 
                    'arg_perigee', 'mean_anomaly', 'mean_motion',
                    'epoch_year', 'epoch_day', 'bstar']
    
    feature_values = [features.get(k, 0) for k in feature_names]
    
    # Normaliza e faz a previsão
    features_normalized = scaler.transform([feature_values])
    prediction = model.predict(features_normalized)[0]
    
    return {
        'datetime': target_datetime,
        'East (G)': prediction[0],
        'North (G)': prediction[1],
        'Down (G)': prediction[2]
    }

# 4. Previsão para série temporal
def predict_time_series(model, scaler, tle_line1, tle_line2, start_datetime, hours=24, step_hours=1):
    predictions = []
    
    for i in range(0, hours, step_hours):
        current_time = start_datetime + datetime.timedelta(hours=i)
        pred = predict_magnetic_field(model, scaler, tle_line1, tle_line2, current_time)
        predictions.append(pred)
    
    return pd.DataFrame(predictions)

# 5. Visualização dos resultados
def plot_predictions(df):
    plt.figure(figsize=(15, 10))
    
    # Componente Leste
    plt.subplot(3, 1, 1)
    plt.plot(df['datetime'], df['East (G)'], 'b-', linewidth=2)
    plt.title('Componente Leste do Campo Magnético', fontsize=12)
    plt.ylabel('Gauss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Componente Norte
    plt.subplot(3, 1, 2)
    plt.plot(df['datetime'], df['North (G)'], 'r-', linewidth=2)
    plt.title('Componente Norte do Campo Magnético', fontsize=12)
    plt.ylabel('Gauss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Componente Vertical
    plt.subplot(3, 1, 3)
    plt.plot(df['datetime'], df['Down (G)'], 'g-', linewidth=2)
    plt.title('Componente Vertical do Campo Magnético', fontsize=12)
    plt.ylabel('Gauss')
    plt.xlabel('Data e Hora')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# 6. Função para salvar o modelo e scaler
def save_model_artifacts(model, scaler, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    model.save(os.path.join(folder, 'modelo_campo_magnetico.h5'))
    
    import joblib
    joblib.dump(scaler, os.path.join(folder, 'scaler_campo_magnetico.pkl'))
    
    print(f"Modelo e scaler salvos em {folder}")

# Fluxo principal
if __name__ == "__main__":
    # 1. Carregar e preparar dados
    print("Carregando e processando dados...")
    data = load_and_preprocess_data(caminho_pasta)
    
    if data.empty:
        print("Nenhum dado encontrado. Verifique o caminho da pasta.")
        exit()
    
    # 2. Preparar features e targets
    feature_columns = ['timestamp', 'day_of_year', 'hour_of_day',
                      'inclination', 'raan', 'eccentricity',
                      'arg_perigee', 'mean_anomaly', 'mean_motion',
                      'epoch_year', 'epoch_day', 'bstar']
    
    # Preenche valores NaN com 0 (simplificação - pode ser melhorado)
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    
    features = data[feature_columns]
    targets = data[['east_g', 'north_g', 'down_g']]
    
    # 3. Normalizar dados
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    y = targets.values
    
    # 4. Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Construir e treinar modelo
    print("\nTreinando modelo...")
    model = build_model((X_train.shape[1],))
    
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # 6. Salvar modelo treinado
    save_model_artifacts(model, scaler, os.path.join(caminho_pasta, 'modelo_salvo'))
    
    # 7. Interface com usuário
    print("\n" + "="*50)
    print("Sistema de Previsão de Campo Magnético para Satélites")
    print("="*50 + "\n")
    
    print("Insira os dados do satélite:")
    line1 = input("Linha 1 da TLE: ").strip()
    line2 = input("Linha 2 da TLE: ").strip()
    
    print("\nInsira a data e hora inicial para previsão:")
    date_str = input("Data (YYYY-MM-DD): ").strip()
    time_str = input("Hora (HH:MM): ").strip()
    
    try:
        start_datetime = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        print("Formato de data/hora inválido. Usando data atual.")
        start_datetime = datetime.datetime.now()
    
    hours = int(input("\nPeríodo de previsão (horas): ") or 24)
    step = int(input("Intervalo entre previsões (horas): ") or 1)
    
    # 8. Fazer previsões
    print("\nGerando previsões...")
    predictions = predict_time_series(model, scaler, line1, line2, start_datetime, hours, step)
    
    # 9. Mostrar resultados
    print("\nPrimeiras previsões:")
    print(predictions.head())
    
    # 10. Plotar gráficos
    plot_predictions(predictions)
    
    # 11. Salvar resultados
    output_file = os.path.join(caminho_pasta, 'previsoes_campo_magnetico.csv')
    predictions.to_csv(output_file, index=False)
    print(f"\nPrevisões salvas em: {output_file}")
