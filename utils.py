import pandas as pd
import streamlit as st


# Carregamento dos dados com cache para performance
@st.cache_data
def load_data():
    df = pd.read_csv("produtos_agricolas.csv")
    return df
