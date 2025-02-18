import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(layout="wide")

# Carregar a base de dados
df = pd.read_csv("temps.csv")

# Criar coluna de data combinando ano, mês e dia
df["Date"] = pd.to_datetime(df[["year", "month", "day"]])

# Ordenar por data
df = df.sort_values("Date")

# Criar uma coluna para identificação do mês-ano
df["Month"] = df["Date"].apply(lambda x: str(x.year) + "-" + str(x.month))

# Filtro para selecionar um mês específico
month = st.sidebar.selectbox("Mês", df["Month"].unique())
df_filtered = df[df["Month"] == month]

# Layout de colunas
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Visualização 1: Temperatura real por dia
fig_actual_temp = px.line(df_filtered, x="Date", y="actual", 
                          title="Temperatura Real por Dia")
col1.plotly_chart(fig_actual_temp, use_container_width=True)

# Visualização 2: Comparação entre temperaturas previstas e reais
fig_temp_comparison = px.line(
    df_filtered, x="Date", y=["temp_2", "temp_1", "actual"],
    title="Comparação de Temperaturas (Previstas e Reais)",
    labels={"value": "Temperatura", "variable": "Tipo"}
)
col2.plotly_chart(fig_temp_comparison, use_container_width=True)

# Visualização 3: Temperatura média semanal
df_week = df_filtered.groupby("week")[["average"]].mean().reset_index()
fig_week_avg = px.bar(df_week, x="week", y="average", 
                      title="Temperatura Média Semanal")
col3.plotly_chart(fig_week_avg, use_container_width=True)

# Visualização 4: Contribuição por amigo
fig_friend = px.pie(df_filtered, values="friend", names="week", 
                    title="Contribuição de Amigos por Semana")
col4.plotly_chart(fig_friend, use_container_width=True)
