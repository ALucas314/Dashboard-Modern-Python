import os
import pickle
import pandas as pd # type: ignore
import numpy as np # type: ignore
import streamlit as st # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import sqlalchemy # type: ignore
import plotly.graph_objects as go # type: ignore # type: ignore
import requests
import json

# Configurações iniciais
st.set_page_config(page_title="Clima e Previsões", layout="wide", page_icon="🌤️")

# Função para carregar animações Lottie
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Função para carregar dados do banco de dados
@st.cache_data
def carregar_dados():
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        query = "SELECT * FROM view_clima_completo"
        data = pd.read_sql(query, engine)
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

# Função para realizar a busca de texto completo no banco de dados
@st.cache_data
def full_text_search(search_term, table_name="dados_atuais", mode="NATURAL LANGUAGE"):
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        if mode.upper() == "NATURAL LANGUAGE":
            query = f"""
                SELECT *
                FROM {table_name}
                WHERE MATCH(descricao_dia, descricao_madrugada, descricao_manha, descricao_tarde, descricao_noite, previsao)
                AGAINST('{search_term}' IN NATURAL LANGUAGE MODE);
            """
        elif mode.upper() == "BOOLEAN":
            query = f"""
                SELECT *
                FROM {table_name}
                WHERE MATCH(descricao_dia, descricao_madrugada, descricao_manha, descricao_tarde, descricao_noite, previsao)
                AGAINST('{search_term}' IN BOOLEAN MODE);
            """
        else:
            st.error("Modo de busca inválido. Use 'NATURAL LANGUAGE' ou 'BOOLEAN'.")
            return None

        result = pd.read_sql(query, engine)
        return result
    except Exception as err:
        st.error(f"Erro ao realizar a busca Full-Text: {err}")
        return None

# Componente Full Text Search
def full_text_search_component():
    st.sidebar.markdown("### Pesquisa Full-Text")
    search_term = st.sidebar.text_input("Digite uma palavra para buscar no histórico:")

    if search_term:
        st.sidebar.write(f"Buscando por: **{search_term}**")
        
        # Selecionar a tabela para busca
        table_name = st.sidebar.selectbox("Selecione a tabela para busca:", ["dados_atuais", "historico_clima"])
        
        # Selecionar o modo de busca
        mode = st.sidebar.selectbox("Selecione o modo de busca:", ["NATURAL LANGUAGE", "BOOLEAN"])
        
        # Realizar a busca
        filtered_data = full_text_search(search_term, table_name=table_name, mode=mode)
        
        if filtered_data is not None and not filtered_data.empty:
            st.sidebar.write(f"**{len(filtered_data)}** resultados encontrados:")
            st.sidebar.dataframe(filtered_data)
        else:
            st.sidebar.warning("Nenhum resultado encontrado.")
    else:
        st.sidebar.write("Digite uma palavra para buscar no histórico.")

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
            temperatura_atual_api = data['current']['temp_c']
            temperatura_ajustada = temperatura_atual_api + 1.19
            data['current']['temp_c'] = temperatura_ajustada
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

# Função para criar o dataset para o modelo LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Função para exibir o gráfico de temperatura ao longo do tempo
def plot_temperatura_ao_longo_do_tempo(weather_data):
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=weather_data['Data'], 
        y=weather_data['Temperatura'], 
        mode='lines', 
        name='Temperatura', 
        line=dict(color='Lightblue')
    ))
    fig_temp.update_layout(
        title="Temperatura ao Longo do Tempo",
        xaxis_title="Dias",
        yaxis_title="Temperatura (°C)",    
        template="plotly_dark",
        margin=dict(t=100, b=50),
        xaxis=dict(
            tickformat="%b %d", 
            tickangle=0,         
            nticks=10            
        )
    )
    return fig_temp

# Função para exibir o gráfico de temperatura ao longo do tempo com média
def plot_temperatura_com_media(weather_data, media_temperatura):
    fig_temp_media = go.Figure()
    fig_temp_media.add_trace(go.Scatter(
        x=weather_data['Data'], 
        y=weather_data['Temperatura'], 
        mode='lines', 
        name='Temperatura', 
        line=dict(color='Lightblue')
    ))
    fig_temp_media.add_hline(
        y=media_temperatura, 
        line=dict(color='Yellow', dash='dot')
    )
    fig_temp_media.update_layout(
        title="Temperatura ao Longo do Tempo com Média", 
        xaxis_title="Dias", 
        yaxis_title="Temperatura (°C)", 
        template="plotly_dark",
        margin=dict(t=100, b=50),
        xaxis=dict(
            tickformat="%b %d",  
            tickangle=0,         
            nticks=10            
        ),
        annotations=[
            dict(
                x=0.5,  
                y=1.1,  
                text=f"Média: {media_temperatura:.2f}°C", 
                showarrow=False,  
                font=dict(size=16, color='white'),  
                xref="paper",  
                yref="paper",  
                align="center"  
            )
        ]
    )
    return fig_temp_media

# Função para exibir o gráfico de temperatura real vs previsões LSTM
def plot_temperatura_real_vs_previsao(y_test_rescaled, y_pred_rescaled, days):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days[:len(y_test_rescaled[0])],  
        y=y_test_rescaled[0],
        mode='lines',
        name="Temperatura Real"
    ))
    fig.add_trace(go.Scatter(
        x=days[:len(y_pred_rescaled)],  
        y=y_pred_rescaled.flatten(),
        mode='lines',
        name="Previsões LSTM"
    ))
    fig.update_layout(
        title="Temperatura Real vs Previsões LSTM",
        xaxis=dict(
            title="Dias",
            tickmode="array",
            tickvals=list(range(0, len(days), 5)),  
            ticktext=[days[i] for i in range(0, len(days), 5)]  
        ),
        yaxis_title="Temperatura (°C)",
        template="plotly_dark",
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    return fig

# Função para exibir o gráfico de temperatura real
def plot_temperatura_real(y_test_rescaled, days):
    fig_real = go.Figure()
    fig_real.add_trace(go.Scatter(
        x=days[:len(y_test_rescaled[0])],  
        y=y_test_rescaled[0],
        mode='lines',
        name="Temperatura Real"
    ))
    fig_real.update_layout(
        title="Temperatura Real",
        xaxis=dict(
            title="Dias",
            tickmode="array",
            tickvals=list(range(0, len(days), 5)),  
            ticktext=[days[i] for i in range(0, len(days), 5)]  
        ),
        yaxis_title="Temperatura (°C)",
        template="plotly_dark",
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    return fig_real

# Função para exibir o gráfico de previsões LSTM
def plot_previsoes_lstm(y_pred_rescaled, days):
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=days[:len(y_pred_rescaled)],  
        y=y_pred_rescaled.flatten(),
        mode='lines',
        name="Previsões LSTM"
    ))
    fig_pred.update_layout(
        title="Previsões LSTM",
        xaxis=dict(
            title="Dias",
            tickmode="array",
            tickvals=list(range(0, len(days), 5)),  
            ticktext=[days[i] for i in range(0, len(days), 5)]  
        ),
        yaxis_title="Temperatura (°C)",
        template="plotly_dark",
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    return fig_pred

# Função para exibir o gráfico de previsões da API
def plot_previsoes_api(dados_api, y_pred_rescaled, weather_data):
    fig_comparacao = go.Figure()
    datas_api = [pd.to_datetime(day['date']) for day in dados_api['forecast']['forecastday']]
    temperaturas_api = [day['day']['avgtemp_c'] for day in dados_api['forecast']['forecastday']]
    datas_lstm = weather_data['Data'][-len(y_pred_rescaled):]  
    temperaturas_lstm = y_pred_rescaled.flatten()
    fig_comparacao.add_trace(go.Bar(
        x=datas_api,
        y=temperaturas_api,
        name="Previsões da API (Ajustadas)",
        marker_color='Lightblue'
    ))
    fig_comparacao.update_layout(
        title="Previsões da API para os determinados dias:",
        xaxis_title="Data",
        yaxis_title="Temperatura (°C)",
        template="plotly_dark",
        barmode='group'  
    )
    return fig_comparacao

# Função para exibir o gráfico de temperatura atual
def plot_temperatura_atual(dados_api):
    fig_temperatura_atual = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dados_api['current']['temp_c'],  
        title={'text': "Temperatura Atual (°C)"},
        gauge={'axis': {'range': [None, 40]},  
        'bar': {'color': "aqua"},  
        'steps': [
            {'range': [0, 10], 'color': "green"},
            {'range': [10, 20], 'color': "yellow"},
            {'range': [20, 30], 'color': "orange"},
            {'range': [30, 40], 'color': "red"}
        ],
    },
    number={'font': {'size': 60}},  
    ))
    fig_temperatura_atual.update_layout(
        template="plotly_dark",
        autosize=True,  
        margin=dict(l=0, r=0, t=30, b=30),  
        height=420  
    )
    return fig_temperatura_atual

# Função principal
def main():
    with st.spinner("Carregando dados e preparando o site..."):
        st.sidebar.title("Configurações")
        st.sidebar.markdown("### Personalize sua visualização de dados")

        # Adicionar o componente Full-Text Search
        full_text_search_component()

        # Adicionar o componente de busca de dados da API
        st.sidebar.markdown("### Buscar Dados da API")
        if st.sidebar.button("Buscar Dados da API"):
            dados_api = buscar_dados_api()
            if dados_api:
                st.session_state['dados_api'] = dados_api
                st.sidebar.success("Dados da API carregados com sucesso!")
            else:
                st.sidebar.error("Erro ao buscar dados da API.")

        # Restante do código...
        data = carregar_dados()

        if data is not None and not data.empty:
            st.title("🌤️ Dados Meteorológicos - Castanhal")
            st.markdown("Explore os dados meteorológicos com previsões baseadas em redes neurais LSTM.")

            linhas = st.sidebar.slider("Escolha o número de linhas para exibir:", min_value=5, max_value=len(data), value=10)

            st.subheader("Tabela de Dados")
            st.dataframe(data.head(linhas), use_container_width=True)

            weather_data = data[['Data', 'Temperatura']]
            weather_data['Temperatura'] = weather_data['Temperatura'].replace({',': '.'}, regex=True).astype(float)
            weather_data = weather_data.dropna(subset=['Temperatura'])
            weather_data = weather_data.sort_values(by='Data')

            data_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(weather_data['Temperatura'].values.reshape(-1, 1))

            time_step = 100
            X, y = create_dataset(data_scaled, time_step)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model_data = carregar_modelo('../models/pesoLSTM.pkl')
            if model_data:
                st.sidebar.write("Carregando modelo salvo...")
                model = model_data['model']
                scaler = model_data['scaler']
            else:
                st.sidebar.write("Modelo não encontrado. Treine o modelo primeiro.")
                st.stop()

            y_pred = model.predict(X_test)
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform([y_test])

            mes = 1
            ano = 2025
            media_temperatura = consultar_media_temperatura_mes(mes, ano)

            if media_temperatura is not None:
                st.sidebar.write(f'Média da temperatura para {mes}/{ano}: {media_temperatura:.2f}°C')
            else:
                st.sidebar.write("Não foi possível obter a média da temperatura.")

            mse_lstm = mean_squared_error(y_test_rescaled[0], y_pred_rescaled)
            st.sidebar.write(f'Erro quadrático médio (MSE) - LSTM: {mse_lstm:.4f}')

            last_data = data_scaled[-time_step:].reshape(1, time_step, 1)
            prediction = model.predict(last_data)
            prediction_rescaled = scaler.inverse_transform(prediction)
            st.sidebar.write(f'Previsão de Temperatura para Amanhã: {prediction_rescaled[0][0]:.2f}°C')

            st.sidebar.markdown("### Média de Umidade por Dia")
            with st.sidebar.expander("Ver Média de Umidade por Dia"):
                media_umidade_por_dia = consultar_media_umidade_por_dia()
                if media_umidade_por_dia is not None:
                    st.dataframe(media_umidade_por_dia, use_container_width=True)
                else:
                    st.warning("Não foi possível carregar a média de umidade por dia.")

            st.sidebar.markdown("### Comparação entre Temperatura Real e Previsões LSTM")
            show_real = st.sidebar.checkbox("Mostrar Temperatura Real", value=True)
            show_pred = st.sidebar.checkbox("Mostrar Temperatura Prevista (LSTM)", value=True)

            st.sidebar.markdown("### Visualizar Dados Históricos do Clima")
            show_dados_atuais = st.sidebar.checkbox("Mostrar Dados do Histórico", value=False)

            if show_dados_atuais:
                st.markdown("### Dados Históricos do Clima")
                historico_clima = carregar_historico_clima()
                if historico_clima is not None and not historico_clima.empty:
                    st.dataframe(historico_clima, use_container_width=True)
                else:
                    st.warning("Não foi possível carregar os dados históricos do clima.")

            st.subheader("Visualizações Gráficas")
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(plot_temperatura_ao_longo_do_tempo(weather_data))

            with col2:
                st.plotly_chart(plot_temperatura_com_media(weather_data, media_temperatura))

            if show_real and show_pred:
                days = [f"{day} Jan" for day in range(1, 32)] + [f"{day} Fev" for day in range(1, 29)]
                st.plotly_chart(plot_temperatura_real_vs_previsao(y_test_rescaled, y_pred_rescaled, days))

            elif show_real:
                days = [f"{day} Jan" for day in range(1, 32)] + [f"{day} Fev" for day in range(1, 29)]
                st.plotly_chart(plot_temperatura_real(y_test_rescaled, days))

            elif show_pred:
                days = [f"{day} Jan" for day in range(1, 32)] + [f"{day} Fev" for day in range(1, 29)]
                st.plotly_chart(plot_previsoes_lstm(y_pred_rescaled, days))

            if 'dados_api' in st.session_state and st.session_state['dados_api']:
                dados_api = st.session_state['dados_api']
                st.plotly_chart(plot_previsoes_api(dados_api, y_pred_rescaled, weather_data))
                st.plotly_chart(plot_temperatura_atual(dados_api))

        else:
            st.warning("Não foi possível carregar os dados ou os dados estão vazos.")

if __name__ == "__main__":
    main()
