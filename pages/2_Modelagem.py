import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import load_data

st.set_page_config(page_title="Modelagem e Previsão", layout="wide")
st.title("Modelagem e Estimador de Irrigação")

# Caso os dados não tenham sido carregados por algum motivo
if "df_agricola" not in st.session_state:
    df = load_data()  # Carrega os dados
    st.session_state["df_agricola"] = df  # Salva na sessão
else:  # se já foram carregados
    df = st.session_state["df_agricola"]  # Puxa os dados da sessão

# --- 1. Feature Engineering ---
# Definição das Variáveis
target = "irrigation"
X = df.drop(target, axis=1)
y = df[target]

# O que é categórico e numérico
categorical_features = ["crop"]
numerical_features = ["N", "P", "K", "temperature", "humidity", "ph"]

# Configuração do Pipeline de encoding
preprocessor = ColumnTransformer(
    transformers=[
        # Pega a coluna 'crop', transforma em números (0s e 1s)
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        # Pega as numéricas e deixa passar direto ('passthrough')
        ("num", "passthrough", numerical_features),
    ]
)

# Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=X["crop"]
)

# --- 2. BARRA LATERAL (CONFIGURAÇÃO) ---
# Configuração do modelo na barra lateral
st.sidebar.header("Configuração do Modelo")
# Número de árvores
n_estimators = st.sidebar.slider("Número de Árvores", 50, 500, 100)
# Profundidade máxima
max_depth = st.sidebar.slider("Profundidade Máxima", 2, 20, 10)

# --- 3. LÓGICA DE TREINAMENTO ---
# Ao clicar no botão
if st.sidebar.button("🚀 Treinar Modelo"):
    # Criação do Pipeline completo
    model_pipeline = Pipeline(
        steps=[
            # Primeiro o pré-processador (OneHotEncoder)
            ("preprocessor", preprocessor),
            (
                # Depois o modelo Random Forest Regressor
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                ),
            ),
        ]
    )

    with st.spinner("Treinando Pipeline (Encoder + Random Forest)..."):
        # Treina e gera previsões para teste
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        # --- SALVANDO TUDO NO SESSION_STATE ---
        st.session_state["trained_model"] = model_pipeline
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred

        # Salvando métricas
        st.session_state["metrics"] = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        # Calculando Importância das Features
        rf_model = model_pipeline.named_steps["regressor"]
        encoder = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"]

        # Nomes das features por reversão do OneHotEncoder
        cat_names = encoder.get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate([cat_names, numerical_features])

        # DataFrame de importância das Features ordenado
        importance_df = (
            pd.DataFrame(
                {
                    "Feature": all_feature_names,
                    "Importance": rf_model.feature_importances_,
                }
            )
            .sort_values(by="Importance", ascending=False)
            .head(10)
        )

        # Salvando o DataFrame de importância na sessão
        st.session_state["importance_df"] = importance_df
        # Mensagem de sucesso
        st.success("Modelo Treinado e Salvo na Memória!")

# --- 4. EXIBIÇÃO DOS RESULTADOS (AVALIAÇÃO) ---
# Se o modelo foi treinado e as métricas produzidas
if "trained_model" in st.session_state and "metrics" in st.session_state:
    st.divider()
    st.header("Avaliação do Modelo")

    metrics = st.session_state["metrics"]  # Puxa as métricas salvas

    # Mostra as métricas em 3 colunas
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['r2']:.4f}")
    col2.metric("MAE (Erro Médio)", f"{metrics['mae']:.2f}")
    col3.metric("RMSE", f"{metrics['rmse']:.2f}")

    # Cria as duas colunas para os gráficos
    col_chart1, col_chart2 = st.columns(2)

    # Gráfico de Dispersão Real vs Predito
    with col_chart1:
        st.subheader("Real vs Predito")
        y_test_saved = st.session_state["y_test"]
        y_pred_saved = st.session_state["y_pred"]

        # Criação do gráfico de dispersão
        fig_real_pred = px.scatter(
            x=y_test_saved,
            y=y_pred_saved,
            labels={"x": "Valor Real", "y": "Valor Predito"},
            title="Dispersão Real x Predito",
        )
        # Adiciona a linha y=x para referência
        fig_real_pred.add_shape(
            type="line",
            line=dict(dash="dash", color="gray"),
            x0=y_test_saved.min(),
            y0=y_test_saved.min(),
            x1=y_test_saved.max(),
            y1=y_test_saved.max(),
        )
        # Mostra o gráfico
        st.plotly_chart(fig_real_pred, use_container_width=True)

    # Gráfico de Importância das Features
    with col_chart2:
        st.subheader("Top 10 Variáveis Importantes")
        imp_df = st.session_state["importance_df"]

        # Criação do gráfico de barras horizontais
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Influência das Features",
            text="Importance",  # Define qual coluna será o texto
        )
        # Formatação manual do texto (3 casas decimais)
        fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))

        # Mostra o gráfico
        st.plotly_chart(fig_imp, use_container_width=True)

# --- 5. SIMULADOR DE PREVISÃO ---
st.divider()
st.header("Simulador de Irrigação")

# Se o modelo não foi treinado ainda
if "trained_model" not in st.session_state:
    st.info("👈 Treine o modelo na barra lateral para habilitar o simulador.")
# Se o modelo foi treinado
else:
    st.markdown(
        "Insira as condições do solo e ambiente para prever a necessidade de água."
    )
    # Formulário de entrada de dados para previsão
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            n_input = st.number_input("Nitrogênio (N)", 0, 200, int(df["N"].mean()))
            p_input = st.number_input("Fósforo (P)", 0, 200, int(df["P"].mean()))
        with col2:
            k_input = st.number_input("Potássio (K)", 0, 200, int(df["K"].mean()))
            temp_input = st.number_input(
                "Temperatura (°C)", 0.0, 60.0, df["temperature"].mean()
            )
        with col3:
            hum_input = st.number_input(
                "Umidade (%)", 0.0, 100.0, df["humidity"].mean()
            )
            ph_input = st.number_input("pH do Solo", 0.0, 14.0, df["ph"].mean())
        # Seleção da cultura
        crop_list = sorted(df["crop"].unique())
        crop_input = st.selectbox("Cultura (Crop)", options=crop_list)
        # Botão de submissão do formulário
        submit_btn = st.form_submit_button("Calcular Irrigação Necessária")
    # Ao submeter o formulário
    if submit_btn:
        # Cria o DataFrame de entrada para previsão
        input_data = pd.DataFrame(
            {
                "N": [n_input],
                "P": [p_input],
                "K": [k_input],
                "temperature": [temp_input],
                "humidity": [hum_input],
                "ph": [ph_input],
                "crop": [crop_input],
            }
        )
        # Puxa o modelo treinado da sessão
        model = st.session_state["trained_model"]
        # Gera a previsão
        prediction = model.predict(input_data)[0]
        # Mostra o resultado
        st.success(f"💧 Previsão de Irrigação: **{prediction:.2f}mm**")
