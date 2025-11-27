import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st

from utils import load_data

st.set_page_config(page_title="Exploração dos Dados")
# Se os dados não foram carregados por algum motivo
if "df_agricola" not in st.session_state:
    df = load_data()  # Carrega os dados
    st.session_state["df_agricola"] = df  # Salva na sessão
else:  # se já foram carregados
    df = st.session_state["df_agricola"]  # Puxa os dados da sessão

st.title("Exploração dos Dados")  # Título
# --- DESCRIÇÃO GERAL DOS DADOS ---
col1, col2, col3 = st.columns(3)
col1.metric("Número de registros", df.shape[0])
col2.metric("Número de características", df.shape[1])
col3.metric("Número de culturas", df["crop"].nunique())
st.header("Descrição Geral dos Dados")  # Título da seção
st.dataframe(df.describe())  # Estatísticas descritivas
st.subheader("Uma amostra dos dados")  # Subtítulo
st.dataframe(df.head())  # Mostra as primeiras linhas

# --- DISTRIBUIÇÃO DA IRRIGAÇÃO (TARGET) ---
st.header("Distribuição da Irrigação")  # Título da seção
fig, ax = plt.subplots(figsize=(6, 5))  # Cria a figura
sns.histplot(df["irrigation"], kde=True, ax=ax, color="green")  # type: ignore
ax.set(xlabel="Irrigação em mm", ylabel="Frequência")  # Rótulos dos eixos
st.pyplot(fig)  # Mostra o gráfico

# --- BOXPLOT DAS VARIÁVEIS NUMÉRICAS ---
st.header("Boxplots das Variáveis Numéricas")  # Título da seção
# Colunas numéricas disponíveis para seleção
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
box_y_axis = st.selectbox(
    "Selecione a variável para o eixo Y:",
    options=numeric_cols,
    index=6,  # Padrão: irrigation (índice 6 da lista)
)
# Criação do Boxplot
fig_box = px.box(
    df,
    y=box_y_axis,
    title=f"Boxplot de {box_y_axis}",
    points="outliers",  # Mostra os outliers
)
# Mostra o Boxplot
st.plotly_chart(fig_box, use_container_width=True)

# --- CORRELAÇÕES ---
st.header("Matriz de Correlação")  # Título da seção
corr = df.select_dtypes(include=["number"]).corr()  # Cálculo da correlação
fig, ax = plt.subplots(figsize=(12, 8))  # Cria a figura
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)  # Cria o heatmap
st.pyplot(fig)  # Mostra o gráfico

# --- SCATTER PLOT INTERATIVO ---
# Título da seção
st.header("Scatter Plot Interativo")
st.write(
    "Selecione uma ou mais culturas para visualizar a relação entre duas variáveis."
)
# Lista de culturas disponíveis para filtro
all_crops = df["crop"].unique().tolist()
# Seleção das culturas para visualização
selected_crops = st.multiselect(
    "Selecione as culturas para visualizar", options=all_crops, default=all_crops[:1]
)
# Filtra o DataFrame com base nas culturas selecionadas
df_filtered = df[df["crop"].isin(selected_crops)]

# Seleção dos Eixos X e Y
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox(
        "Eixo X (Horizontal):",
        options=numeric_cols,
        index=3,  # Padrão: temperature (índice 3 da lista)
    )
with col2:
    y_axis = st.selectbox(
        "Eixo Y (Vertical):",
        options=numeric_cols,
        index=6,  # Padrão: irrigation (índice 6 da lista)
    )

# Criação do Gráfico com os dados filtrados
fig = px.scatter(
    df_filtered,
    x=x_axis,
    y=y_axis,
    color="crop",  # Cores diferentes para cada cultura selecionada
    hover_name="crop",  # Mostra o nome da cultura ao passar o mouse
    title=f"Relação: {x_axis} vs {y_axis}",
    height=500,
)

# Ajuste fino: aumenta o tamanho dos pontos para melhor visualização
fig.update_traces(marker=dict(size=10, opacity=0.7))

st.plotly_chart(fig, use_container_width=True)
