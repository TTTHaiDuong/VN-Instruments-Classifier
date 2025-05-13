import streamlit as st
import base64
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()



def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

title_style = """
<style>
.title {
    color: red;
    text-align: center;
    font-size: 60px;
    white-space: nowrap;    
    font-family: 'Brush Script MT', cursive;
}
</style>
"""

result_style = """
<style>
.result {
    color: darkgreen;
    text-align: center;
    font-size: 36px;
    font-family: 'Brush Script MT', cursive;
}
</style>
"""

def create_data_input(filename_list, label_list, list_data, index_previous):
    data = []
    for file, label in zip(filename_list, label_list):
        data.append([file, label])
    datapoint = pd.DataFrame(
        data=np.array(data),
        index=range(index_previous + 1, index_previous + len(data) + 1),
        columns=list_data.columns
    )
    return datapoint

def save_prediction(excel_url, list_data, data_point):
    list_data = pd.concat((list_data, data_point), axis=0)
    list_data.to_excel(excel_url)
    print("Saved prediction successfully!")
