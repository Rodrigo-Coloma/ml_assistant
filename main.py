import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, explained_variance_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI
from dotenv import dotenv_values
import pandas as pd
from pandas.api.types import is_numeric_dtype
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
import json
import time
import os

#Project creation
def create_project(project_name):
    os.mkdir('./projects/' + project_name)

#Project loading
def load_project(project_name):
    pass
# Data loading
#@st.cache_data
def data_loading():
    if len(separator) > 0:
        st.session_state.raw = pd.read_csv(uploaded_file,sep=separator)
    else:
        st.session_state.raw = pd.read_csv(uploaded_file)
    if st.session_state.raw.columns[0] == 'Unnamed: 0':
        st.session_state.raw = st.session_state.raw.iloc[:,1:]
    return st.session_state.raw
        
#EDA
def eda(eda_feature):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.figure()
    if st.session_state.approach == 'classifier'    :
        if is_numeric_dtype(st.session_state.raw[eda_feature]) and len(list(st.session_state.raw[eda_feature].unique())) > 15:
            sns.boxplot(data=st.session_state.raw,y=eda_feature,hue=st.session_state.target, showfliers = False)           
        else:
            sns.countplot(data=st.session_state.raw,y=eda_feature,hue=st.session_state.target)            
    else:
        if is_numeric_dtype(st.session_state.raw[eda_feature]) and len(list(st.session_state.raw[eda_feature].unique())) > 15:
            sns.scatterplot(data=st.session_state.raw,x=eda_feature,y=st.session_state.target)           
        else:
            sns.boxplot(data=st.session_state.raw,y=st.session_state.target,hue=eda_feature, showfliers = False)
    st.pyplot()

#Filter the table and transform the data to numeric
def filter_tarnsform(df,selected_features,target):
    st.session_state.data = df[selected_features].dropna()
    to_dummy = [feature for feature in selected_features if len(st.session_state.raw[feature].unique()) < 15]
    for feature in to_dummy:
        dummies_df = pd.get_dummies(st.session_state.data[feature], prefix=feature, drop_first=True)
        st.session_state.data = pd.concat([st.session_state.data, dummies_df],axis=1).drop(feature, axis=1)   
    st.session_state.data = pd.concat([st.session_state.data.select_dtypes(exclude=['object']),df[target]], axis=1).dropna()
    return st.session_state.data


#Model selection prompt
#@st.cache_data
def model_selection(data,target):
    
    try:
        st.session_state.api_key = dotenv_values('./.env')['GPTAPIKEY']
    except:
        st.session_state.api_key = st.secrets["GPTAPIKEY"]
    st.session_state.client = OpenAI(api_key=st.session_state.api_key)

    completion = st.session_state.client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature= 0.3,
        messages=[{"role": "system", "content": f"For the given dataset choose the top 7 sklearn models to create a {st.session_state.approach} model for {target} as well as the method to instance the model, the recomended scaler if any for each model and the required import"},
                {"role": "user", "content": f'''Given the dataset below which shows the first 100 rows of a dataset with a total of {len(data.index)}, create a JSON object which enumerates a set of 7 child objects.                       
                        Each child object has four properties named "model", "method","scaler" and "import". The 7 child objects are the 7 top models of sklearn to create a {st.session_state.approach} for {target}.
                        For each child object assign to the property named "model name" to the models name, "method" to the sklearn library method used to invoke the model, "Scaler" the 
                        recomended scaler if any for the model and dataset, and "import" the python script used to import the required final method from the library.
                        ''' + '''The resulting JSON object should be in this format: [{"model":"string","method":"string","scaler": "string","import": "string"}].\n\n
                        The dataset:\n''' +
                        f'''{str(data.head(100))}\n\n
                        The JSON object:\n\n'''}])
    
    return pd.DataFrame(json.loads(completion.choices[0].message.content))

#Running every recommended model
#@st.cache_data
def model_testing(data,target, approach, models):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    X= data[data.columns.drop(target)]
    if approach == 'classifier' and len(data[target].unique()) == 2 :
        y = pd.get_dummies(data[target],drop_first=True)
        models['AUC'] = None
        models['score'] = None
    else:
        y = data[target]
        models['rmse'] = None
        models['r2_score'] = None
        models['explained_variance'] = None
    models['training_time'] = None
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
    fig = plt.figure()
    for i in models.index:
        try:
            exec(models.loc[i,'import'])
            if models.loc[i,['scaler']][0] is not None and models.loc[i,['scaler']][0].split('.')[-1].strip('()') in ['StandardScaler', 'RobustScaler', 'MinMaxScaler']:
                scaler = eval(f"{models.loc[i,['scaler']][0].split('.')[-1].strip('()')}()")
                X_train_i = scaler.fit_transform(X_train)
                X_test_i = scaler.fit_transform(X_test)
            else:
                X_train_i, X_test_i = X_train, X_test
            start = time.time()
            model = eval(f"{models.loc[i,['method']][0].split('.')[-1].strip('()')}().fit(X_train_i,y_train)")
            models.loc[i,'training_time'] = time.time() - start
            if st.session_state.approach == 'classifier':
                models.loc[i,'score'] = model.score(X_test_i, y_test)
                try:
                    models.loc[i,'AUC'] = roc_auc_score(y_test,model.predict_proba(X_test_i)[:, 1])
                    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_i)[:, 1])
                    plt.plot(fpr, tpr, label=models.loc[i,'model'])
                except:
                    pass
            else:
                models.loc[i,'rmse'] = mean_squared_error(y_test,model.predict(X_test_i),squared=False)
                models.loc[i,'r2_score'] = r2_score(y_test,model.predict(X_test_i))
                models.loc[i,'explained_variance'] = explained_variance_score(y_test,model.predict(X_test_i))
        except:
            st.text(f"{models.loc[i,'model'][0]} could not be tested")
    if st.session_state.approach == 'classifier':
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        st.pyplot(fig)
    else:
        sns.scatterplot(data=st.session_state.models,y='rmse',x='r2_score')
        st.pyplot()

def grid_search(test_model, models, data, minutes, approach, scaler):
    i = models.index[models['model'] == test_model].to_list()[0]
    test_model_df = models.loc[models['model'] == test_model,:]
    combinations = max(int(minutes * 60 / (models.loc[i,'training_time'] + 1)),4)
    exec(models.loc[i,'import'])
    model = eval(f"{list(test_model_df['method'])[0].split('.')[-1].strip('()')}()")
    hyperparams = str(inspect.signature(model.__init__))
    hyperparams = [h.split('=')[0] for h in hyperparams.split(', ')[1:]]
    st.write(str(combinations))
    response = st.session_state.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {
        "role": "system",
        "content": "For a certain machine learning model and a number of fits. Create a parameter grid to perform a GridSearch wich approximately the given number of fits. Answer should consist only of the requested JSON object."
        },
        {
        "role": "user",
        "content": "Create a GridSearch for a RandomForestClassifier model. This gridsearch MUST be around 30 fits. Create a JSON object which contains 1 child object with as many properties as hyperparameters should be included in the GridSearch JSON. the product of the number of values in each property of the child object MUST be around or less than 30"
        },
        {
        "role": "assistant",
        "content": "{\n  \"param_grid\": {\n    \"n_estimators\": [10, 20],\n    \"max_depth\": [None, 5],\n    \"min_samples_split\": [2, 5],\n    \"min_samples_leaf\": [1, 2],\n    \"max_features\": [\"auto\", \"sqrt\"]\n  }\n}"
        },
        {
        "role": "user",
        "content": f"Create a GridSearch for a {test_model} {approach} model. This gridsearch MUST be around {combinations} fits. Create a JSON object which contains 1 child object with as many properties as hyperparameters should be included in the GridSearch JSON."# The product of the number of values in each property of the child object MUST be around or less than {combinations}"
        }
        ],
        temperature=1.4,
        max_tokens=1770,    
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    param_grid = eval(response.choices[0].message.content.split('grid":')[-1].strip('}'))
    st.write(param_grid)
    param_grid2 = param_grid.copy()
    for key in param_grid.keys():
        if key not in hyperparams:
            del param_grid2[key]
    st.write(param_grid2)
    
    X= data[data.columns.drop(st.session_state.target)]
    if approach == 'Classifier':
        y = pd.get_dummies(data[st.session_state.target], drop_first=True, prefix=st.session_state.target)
    else:
        y = data[st.session_state.target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
    st.write(f"{list(test_model_df['method'])[0].split('.')[-1].strip('()')}()")
    st.text(hyperparams)
    test_model_df.loc[i,'scaler'] = scaler
    if scaler is not None:
        scaler = eval(scaler + '()')
        X = scaler.fit_transform(X)
    grid_search = GridSearchCV(model, param_grid2, cv=5)
    start = time.time()
    grid_search.fit(X,y)
    st.write(f"Best Parameters: {grid_search.best_params_} Best Score: {grid_search.best_score_} Execution time: {time.time() - start}")
    best_model = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    best_model.fit(X_train,y_train)

    if st.session_state.approach == 'classifier':
        test_model_df.loc[i,'score'] = best_model.score(X_test, y_test)
    else:
        test_model_df.loc[i,'rmse'] = mean_squared_error(y_test,best_model.predict(X_test),squared=False)
        test_model_df.loc[i,'r2_score'] = r2_score(y_test,best_model.predict(X_test))
        test_model_df.loc[i,'explained_variance'] = explained_variance_score(y_test,best_model.predict(X_test))
    #test_model_df['trained_model'] = None
    #test_model_df.loc[i,'trained_model'][0] = best_model
    st.dataframe(test_model_df)
    try:
        test_model_df.loc[i,'AUC'] = roc_auc_score(y_test,best_model.predict_proba(X_test)[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
        fig = plt.figure()
        plt.plot(fpr, tpr, label=models.loc[i,'model'])
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        st.pyplot(fig)
    except:
        pass
    

    return best_model, test_model_df
#
def save_model(model_name, trained_model, model_df, selected_features):
    model_df['model'] = model_name
    model_df['hyperparameters'] = str(trained_model.get_params())
    model_df['features'] = str(selected_features)
    
    st.dataframe(model_df)
    
    if "my_models" not in st.session_state:
        st.session_state.my_models = model_df
    else:
        st.session_state.my_models = pd.concat([st.session_state.my_models, model_df],axis=0).reset_index(drop=True)

#def plot_results(my_models,approach)

st.components.v1.html('<h2 style="text-align: center;">A.I.A.M.A.</h2>', width=None, height=50, scrolling=False)

# We choose the step (page) to work on
if "step" not in st.session_state:
    st.session_state.step = 'Projects'
    st.session_state.steps = ['Projects', 'Data Loading','EDA and Feature Selection', 'Model Selection', 'Model Testing']
st.session_state.step = st.sidebar.selectbox('Step:',st.session_state.steps, st.session_state.steps.index(st.session_state.step))

#Project Management
if st.session_state.step == 'Projects':
    status = st.selectbox('Create or Load', ['Create', 'Load'], 0)
    if status == 'Create':
        st.session_state.project = st.text_input('Porject Name').replace(' ','_')
        if st.button('Create', type='primary'):
            try:
                create_project(st.session_state.project)
            except:
                st.text("Could not create the project. Project name already exists or contains invalid characters")
    if status == 'Load':
        st.session_state.project = st.selectbox('Select Project', os.listdir('./projects/'))
        if st.button('Load'):
            load_project(st.session_state.project)
    







#Data loading
if st.session_state.step == 'Data Loading' and "project" in st.session_state:
    uploaded_file = st.sidebar.file_uploader('Upload your csv here')
    separator = st.sidebar.text_input('Separator',placeholder=',')
    if uploaded_file is not None:
        st.session_state.raw = data_loading()
    # Choosing target and approach
    if "raw" in st.session_state:
        st.session_state.target = st.sidebar.selectbox('Target', st.session_state.raw.columns, placeholder="Choose the target")
        st.session_state.approach = st.sidebar.selectbox('Approach', ['classifier', 'regressor'], placeholder="Choose the target")
        st.dataframe(st.session_state.raw)
        st.dataframe(pd.DataFrame({"name": st.session_state.raw.columns, "non-nulls": len(st.session_state.raw)-st.session_state.raw.isnull().sum().values,
                                "nulls": st.session_state.raw.isnull().sum().values, "type": st.session_state.raw.dtypes.values, "unique": [len(st.session_state.raw[col].unique()) for col in st.session_state.raw.columns] }))   
    else:
        st.markdown('#### Load a file to work on')
# EDA
elif st.session_state.step == 'EDA and Feature Selection' and 'raw' in st.session_state:   
    st.session_state.features = list(st.session_state.raw.columns)
    st.session_state.features.remove(st.session_state.target)
    eda_feature = st.sidebar.selectbox('EDA', st.session_state.raw.columns, placeholder="Choose a feature")
    eda(eda_feature)
    st.session_state.show_table = st.checkbox('Show table', value=True)
    if st.session_state.show_table:
        st.dataframe(st.session_state.raw)
        st.dataframe(pd.DataFrame({"name": st.session_state.raw.columns, "non-nulls": len(st.session_state.raw)-st.session_state.raw.isnull().sum().values,
                                "nulls": st.session_state.raw.isnull().sum().values, "type": st.session_state.raw.dtypes.values, "unique": [len(st.session_state.raw[col].unique()) for col in st.session_state.raw.columns] }))   
# Feature selection
    st.session_state.selected_features = st.sidebar.multiselect('Selected Features',st.session_state.features)
    if st.sidebar.button('Filter and transform'):
        st.session_state.data = filter_tarnsform(st.session_state.raw,st.session_state.selected_features,st.session_state.target)
    if "data" in st.session_state:
        st.dataframe(st.session_state.data)
        st.dataframe(pd.DataFrame({"name": st.session_state.data.columns, "non-nulls": len(st.session_state.data)-st.session_state.data.isnull().sum().values, "nulls": st.session_state.data.isnull().sum().values, "type": st.session_state.data.dtypes.values, "unique": [len(st.session_state.data[col].unique()) for col in st.session_state.data.columns] }))
elif st.session_state.step == 'Model Selection' and 'data' in st.session_state:
# model recommendation   
    if st.sidebar.button('Recomended models'):
        st.session_state.models = model_selection(st.session_state.data, st.session_state.target)
# model testing
    if "models" in st.session_state:
        if st.sidebar.button('Test Models'):
            model_testing(st.session_state.data,st.session_state.target, st.session_state.approach, st.session_state.models)
    if "models" in st.session_state:
        if st.checkbox('Show recommended models', value=True):    
            st.dataframe(st.session_state.models)
# Gridsearch
elif st.session_state.step == "Model Testing" and "models" in st.session_state:
    test_model = st.sidebar.selectbox(' Test Model ', st.session_state.models['model'], placeholder="Choose the model")
    i = st.session_state.models.index[st.session_state.models['model'] == test_model].to_list()[0]
    scaler = st.sidebar.selectbox("Scaler",[None,"StandardScaler","RobustScaler","MinMaxScaler"])
    minutes = st.sidebar.slider('Minutes', 0.5, 30.0, 0.5)
    if st.sidebar.button('Gridsearch'):
        st.session_state.trained_model, st.session_state.test_model_df = grid_search(test_model,st.session_state.models, st.session_state.data, minutes, st.session_state.approach, scaler)
    if "test_model_df" in st.session_state:
        model_name = st.text_input('model name')
        if st.button('Save model'):
            save_model(model_name, st.session_state.trained_model, st.session_state.test_model_df, st.session_state.selected_features)
    if st.checkbox('My models') and "my_models" in st.session_state:
        st.dataframe(st.session_state.my_models)
else:
    st.write("Please perform previous steps before continuing")
if st.sidebar.button('Next',type='primary'):
    st.session_state.step = st.session_state.steps[1 + st.session_state.steps.index(st.session_state.step)]
    st.rerun()

