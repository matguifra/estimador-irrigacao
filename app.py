import streamlit as st

from utils import load_data  # Importando do nosso módulo

st.set_page_config(page_title="Home", page_icon="🌾")
st.title("🌾 Estimador de Irrigação")

# Carrega os dados e salva na sessão
df = load_data()
if "df_agricola" not in st.session_state:
    st.session_state["df_agricola"] = df

st.write("Dados carregados e prontos!")
