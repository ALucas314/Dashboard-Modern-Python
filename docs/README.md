# 🛠 Dashboard Project 🔗

## 🛠 Tools and Skills 🔗

<table>
  <tr>
    <td align="center">
      <a href="#"><img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Python"></a>
    </td>
    <td align="center">
      <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"></a>
    </td>
    <td align="center">
      <a href="#"><img src="https://img.shields.io/badge/Git-F05032.svg?style=for-the-badge&logo=Git&logoColor=white" alt="Git"></a>
    </td>
    <td align="center">
      <a href="#"><img src="https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub"></a>
    </td>
    <td align="center">
      <a href="#"><img src="https://img.shields.io/badge/Visual%20Studio%20Code-007ACC.svg?style=for-the-badge&logo=Visual-Studio-Code&logoColor=white" alt="VSCode"></a>
    </td>
  </tr>
</table>

---

<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
    <img src="/static/img 1.png" alt="Exemplo imagem" style="width: 96%; height: auto;">
    <img src="/static/img 2.png" alt="Exemplo imagem" style="width: 96%; height: auto;">
</div>
<br>

---


## Ajustes e melhorias

O projeto foi capaz de realizar as seguintes tarefas:

- [x] **Animações simples**  
- [x] **Responsividade**  
- [x] **Conexão com o banco de dados**  
- [x] **Treinamento da rede neural LSTM**  
- [x] **Carregamento do peso da rede neural LSTM**  
- [x] **Exibição de gráficos sobre o clima**  

> O projeto utiliza **Python** e a biblioteca **Streamlit** para construir a interface interativa. Ferramentas como **Git**, **GitHub** e **Visual Studio Code** foram utilizadas no desenvolvimento.

---

## 🚀 Tecnologias Utilizadas

- **Python**: Linguagem principal para desenvolvimento do backend e análise de dados.
- **Streamlit**: Framework para criação de interfaces web interativas.
- **Git & GitHub**: Controle de versão e hospedagem do código.
- **Visual Studio Code**: Editor de código principal para desenvolvimento.

---

## 📊 Visualizações

As imagens acima mostram diferentes aspectos do projeto, incluindo gráficos interativos, visualizações de dados e a interface do usuário. O design foi pensado para ser **moderno** e **responsivo**, garantindo uma boa experiência em diferentes dispositivos.

---

## 📂 Estrutura do Projeto

### Tutorial de instalação do ambiente

1. Crie um ambiente virtual Python:
   ```bash
   python -m venv venv
   source venv/bin/activate

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
