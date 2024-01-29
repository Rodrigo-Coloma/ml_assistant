import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def data_loading(uploaded_file,separator):
    if len(separator) > 0:
        raw = pd.read_csv(uploaded_file,sep=separator)
    else:
        raw = pd.read_csv(uploaded_file)
    st.dataframe(raw)
    st.dataframe(pd.DataFrame({"name": raw.columns, "non-nulls": len(raw)-raw.isnull().sum().values,
                                "nulls": raw.isnull().sum().values, "type": raw.dtypes.values}))
    return raw

uploaded_file = st.sidebar.file_uploader('Upload your csv here')
separator = st.sidebar.text_input('Separator',placeholder=',')

if uploaded_file is not None:
    st.session_state.raw = data_loading(uploaded_file, separator)
    
    target = st.sidebar.selectbox('Target', st.session_state.raw.columns, placeholder="Choose the target")
    features = [feature for feature in st.session_state.raw.columns if feature is not target]
    eda_feature = st.sidebar.selectbox('EDA', st.session_state.raw.columns, placeholder="Choose a feature")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.figure()
    st.write(str(is_numeric_dtype(st.session_state.raw[eda_feature])))
    if is_numeric_dtype(st.session_state.raw[eda_feature]) and len(list(st.session_state.raw[eda_feature].unique())) > 15:
        sns.boxplot(data=st.session_state.raw,y=eda_feature,hue=target, showfliers = False)
        st.pyplot()
    else:
        sns.countplot(data=st.session_state.raw,y=eda_feature,hue=target)
        st.pyplot()
        if st.sidebar.button("One Hot Encode"):
            if 'to_dummy' not in st.session_state:
                st.session_state.to_dummy = ['eda_feature']
            else:
                st.session_state.to_dummy.append('eda_feature')

target_features = st.sidebar.multiselect('Selected Features',features)



if st.sidebar.button('Filter and transform'):
    st.session_state.data = st.session_state.raw[target_features].dropna()
    to_dummy = [feature for feature in target_features if len(list(st.session_state.raw[feature].unique())) < 15]
    for feature in to_dummy:
        dummies_df = pd.get_dummies(st.session_state.data[feature], prefix=feature, drop_first=True)
        st.session_state.data= pd.concat([st.session_state.data, dummies_df],axis=1).drop(feature, axis=1)
    st.dataframe(st.session_state.data)
    st.dataframe(pd.DataFrame({"name": st.session_state.data.columns, "non-nulls": len(st.session_state.data)-st.session_state.data.isnull().sum().values, "nulls": st.session_state.data.isnull().sum().values, "type": st.session_state.data.dtypes.values}))
        