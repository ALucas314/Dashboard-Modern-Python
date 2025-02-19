import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(layout="wide")

# Função para carregar e processar a base de dados selecionada
def load_data(selected_database):
    try:
        if selected_database == "GlobalWeatherRepository.csv":
            df = pd.read_csv("GlobalWeatherRepository.csv")
            df["last_updated"] = pd.to_datetime(df["last_updated"])
            df["Date"] = df["last_updated"].dt.date
            df["Month"] = df["last_updated"].dt.to_period("M").astype(str)
        elif selected_database == "weather_2025.csv":
            df = pd.read_csv("weather_2025.csv")
            df.rename(columns={
                "DATA (YYYY-MM-DD)": "Date",
                "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temperature_celsius",
                "UMIDADE RELATIVA DO AR, HORARIA (%)": "humidity",
                "VENTO, VELOCIDADE HORARIA (m/s)": "wind_speed",
                "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precipitation"
            }, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Month"] = df["Date"].dt.to_period("M").astype(str)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

# Sidebar para seleção da base de dados
st.sidebar.title("Configurações")
selected_database = st.sidebar.selectbox(
    "Selecione a Base de Dados",
    ["GlobalWeatherRepository.csv", "weather_2025.csv"]
)

# Carregar a base de dados selecionada
df = load_data(selected_database)

if not df.empty:
    # Filtro para selecionar um mês específico
    month = st.sidebar.selectbox("Selecione o Mês", df["Month"].unique())
    df_filtered = df[df["Month"] == month]

    if df_filtered.empty:
        st.warning(f"Não há dados disponíveis para o mês selecionado: {month}")
    else:
        # Sidebar para selecionar localização (apenas para GlobalWeatherRepository.csv)
        if selected_database == "GlobalWeatherRepository.csv":
            selected_location = st.sidebar.selectbox("Selecione a Localização", df["location_name"].unique())
            location_data = df[df["location_name"] == selected_location]
            if not location_data.empty:
                st.sidebar.write(f"**Última atualização para {selected_location}:**")
                st.sidebar.write(location_data.iloc[-1][["last_updated", "temperature_celsius", "condition_text"]].to_dict())

        # Layout de colunas
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # Visualização 1: Temperatura média diária
        if "temperature_celsius" in df_filtered.columns:
            df_daily_avg = df_filtered.groupby("Date")["temperature_celsius"].mean().reset_index()
            fig_actual_temp = px.line(df_daily_avg, x="Date", y="temperature_celsius", 
                                      title=f"Temperatura Média por Dia ({month})",
                                      labels={"temperature_celsius": "Temperatura (°C)", "Date": "Data"})
            col1.plotly_chart(fig_actual_temp, use_container_width=True)
        else:
            col1.warning("Dados de temperatura não disponíveis.")

        # Visualização 2: Umidade média diária
        if "humidity" in df_filtered.columns:
            df_humidity_avg = df_filtered.groupby("Date")["humidity"].mean().reset_index()
            fig_humidity_avg = px.bar(df_humidity_avg, x="Date", y="humidity",
                                      title=f"Umidade Média por Dia ({month})",
                                      labels={"humidity": "Umidade (%)", "Date": "Data"})
            col2.plotly_chart(fig_humidity_avg, use_container_width=True)
        else:
            col2.warning("Dados de umidade não disponíveis.")

        # Visualização 3: Precipitação total por dia (apenas para weather_2000.csv)
        if selected_database == "weather_2025.csv" and "precipitation" in df_filtered.columns:
            df_precipitation = df_filtered.groupby("Date")["precipitation"].sum().reset_index()
            fig_precipitation = px.bar(df_precipitation, x="Date", y="precipitation",
                                       title=f"Precipitação Total por Dia ({month})",
                                       labels={"precipitation": "Precipitação (mm)", "Date": "Data"})
            col3.plotly_chart(fig_precipitation, use_container_width=True)
        elif selected_database == "weather_2025.csv":
            col3.warning("Dados de precipitação não disponíveis.")

        # Visualização 4: Distribuição da velocidade do vento
        if "wind_speed" in df_filtered.columns:
            fig_wind_distribution = px.histogram(df_filtered, x="wind_speed", nbins=10,
                                                  title=f"Distribuição da Velocidade do Vento ({month})",
                                                  labels={"wind_speed": "Velocidade do Vento (m/s)"})
            col4.plotly_chart(fig_wind_distribution, use_container_width=True)
        else:
            col4.warning("Dados de velocidade do vento não disponíveis.")
else:
    st.error("A base de dados está vazia ou não foi carregada corretamente. Verifique os arquivos.")
