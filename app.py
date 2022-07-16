import streamlit as st
import numpy as np
import torch
import urllib
import pandas as pd
import plotly.express as px
import ast


@st.experimental_memo
def load_file(uploaded_file):
    try:
        df=pd.read_csv(uploaded_file, error_bad_lines=True, warn_bad_lines=False)
    except:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception:
            st.write(Exception)
            df=pd.DataFrame()

    return df

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

        st.write('You selected:', options)


        start = st.button('PRESS')
        if start:
            for d in options:
                with st.expander(f"Посмотреть график"):
                    fig, fig1 = plot_upload_data(df, d)
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(fig1, use_container_width=True)

                st.write(f'Уровень стресса для строки №{d}: {np.random.randint(0,3)}')




def sidebar():
    page_names_to_funcs = {
        "Страница Пользователя": main_page,
    }

    selected_page = st.sidebar.selectbox("Выбрать страницу", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

    st.sidebar.markdown(
        '''
        Автоматическая детекция уровня стресса по данным с датчиков

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