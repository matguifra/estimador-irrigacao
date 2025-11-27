# Estimador de Irrigação Agrícola

Bem-vindo ao repositório do Estimador de Irrigação Agrícola! Este projeto tem como objetivo desenvolver um modelo preditivo para estimar a quantidade de irrigação necessária para diferentes culturas agrícolas com base em variáveis ambientais e climáticas.

## Requisitos

1. Tenha certeza de ter as seguintes bibliotecas instaladas:
    - Streamlit
    - Pandas
    - Numpy
    - Scikit-learn
    - Plotly
    - Matplotlib
    - Seaborn

2. Com o terminal aberto na pasta do projeto, rode o comando:
    ```bash
    streamlit run Home.py
    ```

## Estrutura
- `Home.py`: Página inicial do aplicativo Streamlit.
- `pages/1_Exploração.py`: Página para exploração dos dados.
- `pages/2_Modelagem.py`: Página para modelagem e avaliação do modelo.
- `utils.py`: Funções utilitárias para carregar e processar os dados.
- `produtos_agricolas.csv`: Conjunto de dados utilizado para treinamento e avaliação do modelo.
