import streamlit as st
import numpy as np
import torch
import urllib
import pandas as pd
import plotly.express as px
import ast
from data_preparation_1 import df_preparation
from data_preparation_2 import pipeline
import pickle


#@st.experimental_memo
def load_file(uploaded_file):
    try:
        df=pd.read_csv(uploaded_file, error_bad_lines=True, warn_bad_lines=False)
    except:
        #try:
            df = pd.read_excel(uploaded_file)
        #except:      
        #    df=pd.DataFrame()
            
    return df


@st.experimental_memo
def load_model_knn():
    url = 'https://github.com/Sekai-no-uragawa/aihack_sfo/raw/master/models/KNN_model.pkl'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

@st.experimental_memo
def load_model_rfc():
    url = 'https://github.com/Sekai-no-uragawa/aihack_sfo/raw/master/models/RFC_model.pkl'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

@st.experimental_memo
def load_model_knn2():
    url = 'https://github.com/Sekai-no-uragawa/aihack_sfo/raw/master/models/2_KNN_model.pkl'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def plot_upload_data(df, n):
    data = ast.literal_eval(df.Data[n])
    data1 = ast.literal_eval(df.Data_2[n])

    fig = px.line(x = range(len(data)), y= data, width=800, height=300)
    fig1 = px.line(x = range(len(data1)), y= data1, width=800, height=300)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title='Фотоплетизмограмма',
        xaxis_title="Время",
        yaxis_title="?",
    )
    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title='Пьезоплетизмограмма',
        xaxis_title="Время",
        yaxis_title="?",
    )

    return fig, fig1


def main_page():
    st.title('Классификация уровня стресса человека по датчику плетизмограммы')
    st.subheader('По зарегистрированным при помощи датчиков полиграфа (пьезо и фото плетизмограмма) реакции на вопрос определяется уровень стресса человека.')
    uploaded_file = st.file_uploader('Choose a file')
    is_upload = False
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        with st.expander(f"Просмотреть загруженные данные"):
            df
        is_upload = True
    else:
        is_upload = False
    
    if is_upload:
        options = st.multiselect(
        'Выбери номера строк для анализа',
        df.index.tolist())


        start = st.button('PRESS')
        if start:
            for d in options:
                with st.expander(f"Посмотреть график"):
                    fig, fig1 = plot_upload_data(df, d)
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(fig1, use_container_width=True)

                df2 = df.copy()
                df_prep_1 = df_preparation(df)
                df_prep_1

                df_prep_2 = pipeline(df2)
                df_prep_2

                model_knn = load_model_knn()
                model_rfc = load_model_rfc()
                model_kn2 = load_model_knn2()
                #pred_knn = model_knn.predict(df_prep_2)
                #pred_knn
                #pred_rfc = model_rfc.predict(df_prep_2)
                #pred_rfc
                pred_knn2 = model_kn2.predict(df_prep_1.drop(['label'], axis=1))
                pred_knn2

                st.write(f'Уровень стресса для строки №{d}: {pred_knn2[d]}')




def sidebar():
    page_names_to_funcs = {
        "Страница Пользователя": main_page,
    }

    #selected_page = st.sidebar.selectbox("Выбрать страницу", page_names_to_funcs.keys())
    #page_names_to_funcs[selected_page]()

    st.sidebar.markdown(
        '''
        Web-сервис, основанный на алгоритмах машинного обучения. Данный сервис позволяет определить уровень стресса при реакции на стимул в виде вопроса.

        \n
        ___
        Developed by team **fit_predict**\n
        2022 г.
        '''
    )


if __name__ == '__main__':
    st.set_page_config(
        page_title="True/False",
        page_icon="🍫",
        layout="wide",
    )

    # STYLES
    #with open('data/style.css') as f:
    #    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
    
    sidebar()
    main_page()