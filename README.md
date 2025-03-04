### 🛠 Tutorial de Instalação do Ambiente

# Siga os passos abaixo para configurar o ambiente e executar o projeto:

---

#### 1️⃣ **Criar um Ambiente Virtual**

```bash
# No terminal, crie o ambiente virtual:
python -m venv venv

# Ative o ambiente virtual:
# No Linux/macOS:
source venv/bin/activate

# No Windows:
.\venv\Scripts\activate
```

---

#### 2️⃣ **Instalar as Dependências**

```bash
# Certifique-se de que o ambiente virtual está ativo.
# Em seguida, instale as dependências do projeto:
pip install pandas numpy matplotlib streamlit scikit-learn tensorflow sqlalchemy plotly
pip install mysql-connector-python
```

---

#### 3️⃣ **Treinar o Modelo LSTM**

```bash
# Execute o script que treina o modelo e salva os pesos:
streamlit run modelo_lstm.py
```

# Aguarde a página do Streamlit abrir no navegador.
# No terminal, observe a mensagem de confirmação:
# "Modelo treinado e salvo com sucesso!"

# Verifique se o arquivo `pesoLSTM.pkl` foi criado na pasta do projeto.

---

#### 4️⃣ **Executar o Dashboard**

```bash
# Agora execute o aplicativo principal para visualizar os resultados:
streamlit run app.py
```

# O site será aberto automaticamente no navegador, exibindo as visualizações e análises.

---

🎉 **Pronto!** Agora você pode explorar o projeto e interagir com os gráficos e a interface. 🚀
