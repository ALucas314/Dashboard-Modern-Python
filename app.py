import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlalchemy
import plotly.graph_objects as go
import requests  # Adicionado para fazer requisições à API

# Configurações iniciais
st.set_page_config(page_title="Clima e Previsões", layout="wide", page_icon="🌤️")

# Função para carregar dados do banco de dados
@st.cache_data
def carregar_dados():
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        query = "SELECT * FROM view_clima_completo"
        data = pd.read_sql(query, engine)
        # Verificar e corrigir a coluna "Data"
        if "Data" in data.columns:
            data["Data"] = pd.to_datetime(data["Data"], format="%Y/%m/%d", errors="coerce")
            data = data.dropna(subset=["Data"])
        return data
    except Exception as err:
        st.error(f"Erro ao carregar os dados: {err}")
        return None

# Função para consultar a média da temperatura
@st.cache_data
def consultar_media_temperatura_mes(mes, ano):
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        query = f"SELECT CalcularMediaTemperaturaMes({mes}, {ano}) AS MediaTemperatura;"
        result = pd.read_sql(query, engine)
        return result['MediaTemperatura'].iloc[0]
    except Exception as err:
        st.error(f"Erro ao consultar a média da temperatura: {err}")
        return None

# Função para consultar a média de umidade por dia
@st.cache_data
def consultar_media_umidade_por_dia():
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        query = """
            SELECT dh.Data, AVG(u.`UMIDADE RELATIVA DO AR, HORARIA (%)`) AS umidade_media
            FROM Umidade u
            JOIN data_hora dh ON u.data_hora_id_data_hora = dh.id_data_hora
            GROUP BY dh.Data;
        """
        result = pd.read_sql(query, engine)
        return result
    except Exception as err:
        st.error(f"Erro ao consultar a média de umidade por dia: {err}")
        return None


# Função para carregar dados da tabela historico_clima
@st.cache_data
def carregar_historico_clima():
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        query = "SELECT * FROM historico_clima;"
        data = pd.read_sql(query, engine)
        return data
    except Exception as err:
        st.error(f"Erro ao carregar os dados da tabela historico_clima: {err}")
        return None

# Função para carregar o modelo de um arquivo .pkl
@st.cache_resource
def carregar_modelo(arquivo_pkl):
    if os.path.exists(arquivo_pkl):
        with open(arquivo_pkl, 'rb') as file:
            return pickle.load(file)
    return None

# Função para buscar dados da API
@st.cache_data
def buscar_dados_api():
    try:
        api_key = "cd4c426f74a94faf95b50704250503"
        cidade = "Castanhal"
        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={cidade}&days=7&aqi=no&alerts=no"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Ajustando a temperatura para corresponder à API do Google
            temperatura_atual_api = data['current']['temp_c']
            temperatura_ajustada = temperatura_atual_api + 1.19  # Aplicando o offset
            
            # Atualizando o valor de temperatura no JSON retornado
            data['current']['temp_c'] = temperatura_ajustada
            
            # Ajustando também as previsões futuras
            for day in data['forecast']['forecastday']:
                day['day']['avgtemp_c'] += 1.19
                for hour in day['hour']:
                    hour['temp_c'] += 1.19
            
            return data
        else:
            st.error(f"Erro ao buscar dados da API: {response.status_code}")
            return None
    except Exception as err:
        st.error(f"Erro ao buscar dados da API: {err}")
        return None

# Sidebar - Configurações
st.sidebar.title("Configurações")
st.sidebar.markdown("### Personalize sua visualização de dados")

# Adicionando a nova aba na sidebar para dados da API
with st.sidebar.expander("Dados sobre o clima - API"):
    st.markdown("### Dados da API")
    if st.button("Carregar Dados da API"):
        dados_api = buscar_dados_api()
        if dados_api:
            st.session_state['dados_api'] = dados_api
            st.success("Dados da API carregados com sucesso!")
        else:
            st.warning("Não foi possível carregar os dados da API.")

# Carregar dados do banco de dados
data = carregar_dados()

if data is not None and not data.empty:
    st.title("🌤️ Dados Meteorológicos - Castanhal")
    st.markdown("Explore os dados meteorológicos com previsões baseadas em redes neurais LSTM.")

    # Sidebar - Escolher o número de linhas para exibir
    linhas = st.sidebar.slider("Escolha o número de linhas para exibir:", min_value=5, max_value=len(data), value=10)

    # Exibir tabela de dados
    st.subheader("Tabela de Dados")
    st.dataframe(data.head(linhas), use_container_width=True)

    # Pré-processamento dos dados
    weather_data = data[['Data', 'Temperatura']]

    # Corrigir os valores da coluna "Temperatura" para float
    weather_data['Temperatura'] = weather_data['Temperatura'].replace({',': '.'}, regex=True).astype(float)
    weather_data = weather_data.dropna(subset=['Temperatura'])

    # Ordenar os dados por data
    weather_data = weather_data.sort_values(by='Data')

    data = weather_data['Temperatura'].values.reshape(-1, 1)
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
    model_data = carregar_modelo('pesoLSTM.pkl')
    if model_data:
        st.sidebar.write("Carregando modelo salvo...")
        model = model_data['model']
        scaler = model_data['scaler']
    else:
        st.sidebar.write("Modelo não encontrado. Treine o modelo primeiro.")
        st.stop()

    # Previsões
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform([y_test])

    # Consultar a média da temperatura para o mês de janeiro de 2025
    mes = 1
    ano = 2025
    media_temperatura = consultar_media_temperatura_mes(mes, ano)

    if media_temperatura is not None:
        st.sidebar.write(f'Média da temperatura para {mes}/{ano}: {media_temperatura:.2f}°C')
    else:
        st.sidebar.write("Não foi possível obter a média da temperatura.")

    # Cálculo do MSE (Erro Quadrático Médio)
    mse_lstm = mean_squared_error(y_test_rescaled[0], y_pred_rescaled)
    st.sidebar.write(f'Erro quadrático médio (MSE) - LSTM: {mse_lstm:.4f}')

    # Previsão para amanhã (próxima temperatura)
    last_data = data_scaled[-time_step:].reshape(1, time_step, 1)
    prediction = model.predict(last_data)
    prediction_rescaled = scaler.inverse_transform(prediction)
    st.sidebar.write(f'Previsão de Temperatura para Amanhã: {prediction_rescaled[0][0]:.2f}°C')

    # Sidebar - Média de umidade por dia
    st.sidebar.markdown("### Média de Umidade por Dia")

    # Botão expansível
    with st.sidebar.expander("Ver Média de Umidade por Dia"):
        media_umidade_por_dia = consultar_media_umidade_por_dia()
        if media_umidade_por_dia is not None:
            st.dataframe(media_umidade_por_dia, use_container_width=True)
        else:
            st.warning("Não foi possível carregar a média de umidade por dia.")

    # Sidebar - Ativar/Desativar as linhas de Temperatura Real e Previsões LSTM
    st.sidebar.markdown("### Comparação entre Temperatura Real e Previsões LSTM")
    show_real = st.sidebar.checkbox("Mostrar Temperatura Real", value=True)
    show_pred = st.sidebar.checkbox("Mostrar Temperatura Prevista (LSTM)", value=True)

    # Sidebar - Visualizar Dados Atuais
    st.sidebar.markdown("### Visualizar Dados Históricos do Clima")
    show_dados_atuais = st.sidebar.checkbox("Mostrar Dados do Histórico", value=False)

    # Exibir dados da tabela historico_clima se o botão estiver ativado
    if show_dados_atuais:
        st.markdown("### Dados Históricos do Clima")
        historico_clima = carregar_historico_clima()
        if historico_clima is not None and not historico_clima.empty:
            st.dataframe(historico_clima, use_container_width=True)
        else:
            st.warning("Não foi possível carregar os dados históricos do clima.")

    # Layout dos gráficos
    st.subheader("Visualizações Gráficas")

    # Dividindo a tela em 2 colunas para os gráficos lado a lado
    col1, col2 = st.columns(2)

    # Gráfico de Temperatura ao Longo do Tempo com Média
    with col2:
        st.markdown("### Temperatura ao Longo do Tempo com Média", unsafe_allow_html=True)
        fig_temp_media = go.Figure()

        # Traço da temperatura
        fig_temp_media.add_trace(go.Scatter(
            x=weather_data['Data'], 
            y=weather_data['Temperatura'], 
            mode='lines', 
            name='Temperatura', 
            line=dict(color='Lightblue')
        ))

        # Linha de média
        fig_temp_media.add_hline(
            y=media_temperatura, 
            line=dict(color='Yellow', dash='dot')
        )

        # Adicionando a anotação acima do gráfico
        fig_temp_media.update_layout(
            title="Temperatura ao Longo do Tempo com Média", 
            xaxis_title="Data", 
            yaxis_title="Temperatura (°C)", 
            template="plotly_dark",
            margin=dict(t=100, b=50),  # Ajusta a margem superior para dar mais espaço
            annotations=[
                dict(
                    x=0.5,  # Coloca o texto no meio do gráfico
                    y=1.1,  # Coloca o texto fora da área do gráfico, acima
                    text=f"Média: {media_temperatura:.2f}°C", 
                    showarrow=False,  # Não desenha uma seta
                    font=dict(size=16, color='white'),  # Fonte do texto
                    xref="paper",  # Usando o sistema de coordenadas 'paper', que não está restrito ao gráfico
                    yref="paper",  # Usando o sistema de coordenadas 'paper'
                    align="center"  # Centraliza o texto
                )
            ]
        )

        st.plotly_chart(fig_temp_media)

    # Gráfico de Temperatura ao Longo do Tempo
    with col1:
        st.markdown("### Temperatura ao Longo do Tempo", unsafe_allow_html=True)
        fig_temp = go.Figure()
        days = weather_data['Data'].dt.day
        fig_temp.add_trace(go.Scatter(
            x=days, y=weather_data['Temperatura'], mode='lines', name='Temperatura', line=dict(color='Lightblue')
        ))
        fig_temp.update_layout(
            title="Temperatura ao Longo do Tempo",
            xaxis_title="Dia",
            yaxis_title="Temperatura (°C)",
            template="plotly_dark",
            margin=dict(t=100, b=50)
        )
        st.plotly_chart(fig_temp)

    # Gráfico de Temperatura Real vs Previsões LSTM
    if show_real and show_pred:
        st.markdown("### Temperatura Real vs Previsões LSTM")
        fig = go.Figure()

        # Temperatura Real
        fig.add_trace(go.Scatter(x=np.arange(len(y_test_rescaled[0])), y=y_test_rescaled[0], mode='lines', name="Temperatura Real"))

        # Previsões LSTM
        fig.add_trace(go.Scatter(x=np.arange(len(y_pred_rescaled)), y=y_pred_rescaled.flatten(), mode='lines', name="Previsões LSTM"))

        # Layout
        fig.update_layout(
            title="Temperatura Real vs Previsões LSTM",
            xaxis_title="Amostras",
            yaxis_title="Temperatura (°C)",
            template="plotly_dark",
            legend=dict(
                orientation='h',  # Coloca a legenda na horizontal
                yanchor='top',    # Anexa a legenda no topo do gráfico
                y=-0.2,           # Ajuste a posição para baixo (pode ser alterado para o seu gosto)
                xanchor='center', # Alinha ao centro
                x=0.5             # Posiciona a legenda no centro horizontalmente
            )
        )

        st.plotly_chart(fig)

    # Gráfico de Temperatura Real
    elif show_real:
        st.markdown("### Temperatura Real")
        fig_real = go.Figure()
        fig_real.add_trace(go.Scatter(x=np.arange(len(y_test_rescaled[0])), y=y_test_rescaled[0], mode='lines', name="Temperatura Real"))
        fig_real.update_layout(
            title="Temperatura Real",
            xaxis_title="Amostras",
            yaxis_title="Temperatura (°C)",
            template="plotly_dark",
            legend=dict(
                orientation='h',  # Coloca a legenda na horizontal
                yanchor='top',    # Anexa a legenda no topo do gráfico
                y=-0.2,           # Ajuste a posição para baixo (pode ser alterado para o seu gosto)
                xanchor='center', # Alinha ao centro
                x=0.5             # Posiciona a legenda no centro horizontalmente
            )
        )
        st.plotly_chart(fig_real)

    # Gráfico de Previsões LSTM
    elif show_pred:
        st.markdown("### Previsões LSTM")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=np.arange(len(y_pred_rescaled)), y=y_pred_rescaled.flatten(), mode='lines', name="Previsões LSTM"))
        fig_pred.update_layout(
            title="Previsões LSTM",
            xaxis_title="Amostras",
            yaxis_title="Temperatura (°C)",
            template="plotly_dark",
            legend=dict(
                orientation='h',  # Coloca a legenda na horizontal
                yanchor='top',    # Anexa a legenda no topo do gráfico
                y=-0.2,           # Ajuste a posição para baixo (pode ser alterado para o seu gosto)
                xanchor='center', # Alinha ao centro
                x=0.5             # Posiciona a legenda no centro horizontalmente
            )
        )
        st.plotly_chart(fig_pred)

    # Verificando se os dados da API estão carregados
    if 'dados_api' in st.session_state and st.session_state['dados_api']:
        dados_api = st.session_state['dados_api']
        
        # Criando um gráfico de barras comparando as previsões da API
        st.markdown("### Previsões da API")
        fig_comparacao = go.Figure()

        # Previsões da API
        datas_api = [pd.to_datetime(day['date']) for day in dados_api['forecast']['forecastday']]
        temperaturas_api = [day['day']['avgtemp_c'] for day in dados_api['forecast']['forecastday']]  # Já ajustado na função buscar_dados_api

        # Previsões LSTM (usando as previsões já carregadas)
        datas_lstm = weather_data['Data'][-len(y_pred_rescaled):]  # Ajuste para corresponder ao período da API
        temperaturas_lstm = y_pred_rescaled.flatten()

        # Adicionando as previsões da API
        fig_comparacao.add_trace(go.Bar(
            x=datas_api,
            y=temperaturas_api,
            name="Previsões da API (Ajustadas)",
            marker_color='Lightblue'
        ))

        # Layout do gráfico
        fig_comparacao.update_layout(
            title="Previsões da API para os determinados dias:",
            xaxis_title="Data",
            yaxis_title="Temperatura (°C)",
            template="plotly_dark",
            barmode='group'  # Barras agrupadas
        )
        st.plotly_chart(fig_comparacao)

        # Gráfico da temperatura atual em Castanhal
        st.markdown("### Temperatura atual em Castanhal com base no sensor da API:")
        fig_temperatura_atual = go.Figure(go.Indicator(
            mode="gauge+number",
            value=dados_api['current']['temp_c'],  # Usando o valor ajustado
            title={'text': "Temperatura Atual (°C)"},
            gauge={'axis': {'range': [None, 40]},  # Define o range da temperatura
            'bar': {'color': "aqua"},  # Cor da barra do gráfico
            'steps': [
                {'range': [0, 10], 'color': "green"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 30], 'color': "orange"},
                {'range': [30, 40], 'color': "red"}
            ],
        },
        number={'font': {'size': 60}},  # Ajuste de tamanho da fonte do número

        ))
        fig_temperatura_atual.update_layout(
            template="plotly_dark",
            autosize=True,  # Garante que o gráfico se ajuste ao espaço disponível
            margin=dict(l=0, r=0, t=30, b=30),  # Ajuste as margens conforme necessário
            height=420  # Pode ser ajustado conforme a necessidade
        )
        st.plotly_chart(fig_temperatura_atual)

else:
    st.warning("Não foi possível carregar os dados ou os dados estão vazios.")
