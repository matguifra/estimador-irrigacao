import streamlit as st

from utils import load_data  # Importando do nosso módulo

st.set_page_config(page_title="Home", page_icon="🌾")
st.title("🌾 Estimador de Irrigação")
st.markdown(
    """
    Bem-vindo ao Estimador de Irrigação!
    Este aplicativo foi desenvolvido para ajudar agricultores e profissionais do setor agrícola a estimar a quantidade ideal de irrigação necessária para diferentes culturas com base em variáveis como nutrientes do solo, temperatura, umidade e pH.

    ### Funcionalidades:
    - **Exploração dos Dados**: Analise estatísticas descritivas, distribuições e correlações entre as variáveis.
    - **Modelagem e Previsão**: Treine um modelo de aprendizado de máquina para prever a irrigação necessária com base nas características fornecidas.

    ### Como Usar:
    1. Navegue até a seção "Exploração dos Dados" para entender melhor o conjunto de dados.
    2. Vá para "Modelagem e Previsão" para configurar o modelo e fazer previsões.

    Aproveite a experiência e otimize sua gestão de irrigação!
    """
)


# Carrega os dados e salva na sessão
df = load_data()
if "df_agricola" not in st.session_state:
    st.session_state["df_agricola"] = df

st.write("Dados carregados e prontos!")
