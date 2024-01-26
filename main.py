import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def data_loading(uploaded_file):

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        
        # To read file as string:
        string_data = stringio.read()
        
        # Can be used wherever a "file-like" object is accepted:
        return pd.read_csv(uploaded_file, sep=';')

uploaded_file = st.sidebar.file_uploader('Upload your csv here')
if uploaded_file is not None:
    raw = data_loading(uploaded_file)
    st.dataframe(raw)

    buffer = StringIO()
    raw.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    target = st.sidebar.selectbox('Target', raw.columns, placeholder="Choose the target")
    features = [feature for feature in raw.columns if feature != target]
    eda_feature = st.sidebar.selectbox('EDA', raw.columns, placeholder="Choose a feature")
    #if type(raw[eda_feature][0]) in [int,float]:
    fig = plt.figure()
    sns.boxplot(data=raw,y=eda_feature,hue=target, showfliers = False)
    st.pyplot(fig)
