import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import sqlalchemy

# Configurações iniciais
st.set_page_config(page_title="Clima e Previsões", layout="wide", page_icon="🌤️")

# Função para conectar ao banco de dados
def conectar_banco():
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        return engine
    except Exception as err:
        st.error(f"Erro ao conectar ao banco de dados: {err}")
        return None

# Caminho para o arquivo .pkl
model_pkl_file = "pesoLSTM.pkl"

# Função para salvar o modelo em um arquivo .pkl
def salvar_modelo(model, scaler, arquivo_pkl):
    with open(arquivo_pkl, 'wb') as file:
        pickle.dump({'model': model, 'scaler': scaler}, file)

# Função para carregar o modelo de um arquivo .pkl
def carregar_modelo(arquivo_pkl):
    with open(arquivo_pkl, 'rb') as file:
        return pickle.load(file)

# Sidebar - Configurações
st.sidebar.title("Configurações")
st.sidebar.markdown("### Personalize sua visualização de dados")

# Carregar dados do banco de dados
engine = conectar_banco()
if engine:
    query = "SELECT * FROM castanhal"
    try:
        data = pd.read_sql(query, engine)
        # Verificar e corrigir a coluna "Data"
        if "Data" in data.columns:
            data["Data"] = pd.to_datetime(data["Data"], format="%Y/%m/%d", errors="coerce")
            data = data.dropna(subset=["Data"])
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        data = None
else:
    data = None

if data is not None and not data.empty:
    st.title("🌤️ Dados Meteorológicos - Castanhal")
    st.markdown("Explore os dados meteorológicos com previsões baseadas em redes neurais LSTM.")
    st.subheader("Tabela de Dados")
    linhas = st.slider("Escolha o número de linhas para exibir:", min_value=5, max_value=len(data), value=10)
    st.dataframe(data.head(linhas), use_container_width=True)

    # Pré-processamento dos dados
    weather_data = data[['Data', 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)']]
    weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] = weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].str.replace(',', '.')
    weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] = pd.to_numeric(weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'], errors='coerce')
    weather_data = weather_data.dropna(subset=['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'])

    # Ordenar os dados por data
    weather_data = weather_data.sort_values(by='Data')

    data = weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 100
    X, y = create_dataset(data_scaled, time_step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Verificar se o modelo já foi salvo
    if os.path.exists(model_pkl_file):
        st.write("Carregando modelo salvo...")
        model_data = carregar_modelo(model_pkl_file)
        model = model_data['model']
        scaler = model_data['scaler']
    else:
        st.write("Treinando modelo LSTM...")
        model = Sequential()
        model.add(Bidirectional(LSTM(units=120, return_sequences=True), input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, return_sequences=False))
        model.add(Dropout(0.4))
        model.add(Dense(units=1))
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=70, batch_size=32, verbose=1)
        salvar_modelo(model, scaler, model_pkl_file)
        st.write("Modelo treinado e salvo com sucesso!")

    # Previsões
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform([y_test])

    mse_lstm = mean_squared_error(y_test_rescaled[0], y_pred_rescaled)
    st.write(f'Mean Squared Error (MSE) - LSTM: {mse_lstm}')

    # Gráfico de previsão
    st.subheader("Gráfico de Temperatura Prevista vs Real")
    fig = plt.figure(figsize=(10, 5))
    plt.plot(y_test_rescaled[0], label='Temperatura Real', color='blue')
    plt.plot(y_pred_rescaled, label='Temperatura Prevista (LSTM)', color='red', linestyle='--')
    plt.title('Comparação entre Temperatura Real e Previsões LSTM')
    plt.xlabel('Índice de Teste')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    st.pyplot(fig)

    # Previsão de amanhã
    last_data = data_scaled[-time_step:]
    last_data = last_data.reshape(1, time_step, 1)
    tomorrow_prediction_scaled = model.predict(last_data)
    tomorrow_prediction = scaler.inverse_transform(tomorrow_prediction_scaled)
    st.write(f'Previsão de temperatura para amanhã: {tomorrow_prediction[0][0]:.2f}°C')

else:
    st.warning("Nenhum dado disponível para exibição.")
