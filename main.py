import streamlit as st
from streamlit_ace import st_ace
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, explained_variance_score, mean_squared_error, r2_score, precision_recall_fscore_support, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor, XGBRFClassifier
from openai import OpenAI
from dotenv import dotenv_values
import pandas as pd
from pandas.api.types import is_numeric_dtype
from io import StringIO
import matplotlib.pyplot as plt
import sqlite3 as lite
import seaborn as sns
import numpy as np
import pickle
import umap
import inspect
import json
import time
import os

# OpenAI
def gpt_connect():
    try:
        st.session_state.gpt_key = dotenv_values('./.env')['GPTAPIKEY']
    except:
        st.session_state.gpt_key = st.secrets['GPTAPIKEY']
    st.session_state.client = OpenAI(api_key=st.session_state.gpt_key)

# SQLite connection
def sqlite_connection():
    connection = lite.connect('./database/aiama.db')
    st.session_state.cursor = connection.cursor()
    return connection

# Create folder struccture if it doesn't exist
def folder_management():
    if 'users' not in os.listdir('./'):
        os.mkdir('./users')

def user_create(username,password, password_confirm):
    if password != password_confirm:
        st.write('Passwords do not match')
    else:
        try:
            st.session_state.cursor.execute(f"INSERT INTO users(Username, Password, Registration_Date) VALUES ('{username}','{password}',CURRENT_TIMESTAMP);")
            st.session_state.connection.commit()
            st.session_state.username = username
            st.write('User succesfully created!!')
            st.session_state.projects_df = pd.read_sql(f"SELECT * FROM projects WHERE Owner = '{st.session_state.username}'", st.session_state.connection)
            if username not in os.listdir('./users/'):
                os.mkdir(f'./users/{username}')
            time.sleep(1.2)
            st.session_state.step = 'Projects'
        except:
            st.write('Username already exists or contains invalid characters, please choose a new one')
            time.sleep(2)
        st.rerun()

def user_login(username,password):
    if list(st.session_state.users_df.loc[st.session_state.users_df['Username'] == username,:]['Password'])[0] == password:
        st.session_state.username = username
        st.session_state.projects_df = pd.read_sql(f"SELECT * FROM projects WHERE owner = '{st.session_state.username}'", st.session_state.connection)
        if username not in os.listdir('./users/'):
            os.mkdir(f'./users/{username}')
        st.session_state.step = 'Projects'
        st.rerun()
    else:
        st.write('Incorrect username or password')

#Project creation
def create_project(project_name,user):
    try:
        st.session_state.cursor.execute(f'''INSERT INTO projects(ProjectName, Owner, Target, Approach, CreatedAt, LastOpened)
                                         VALUES ('{project_name}','{st.session_state.username}', NULL, NULL, CURRENT_TIMESTAMP,CURRENT_TIMESTAMP);''')
        st.session_state.connection.commit()
        st.session_state.projects_df = pd.read_sql(f"SELECT * FROM projects WHERE Owner = '{st.session_state.username}'", st.session_state.connection)      
        st.session_state.project = project_name
        try:
            os.mkdir(f'./users/{user}/{project_name}')
        except:
            st.write('Directory existed')
        st.write('Project succesfully created!!')
        time.sleep(1.5)
        st.session_state.step = 'Data Loading'
    except:
        st.write('Project name already exists or contains invalid characters, please choose a new one')
        time.sleep(2)
    st.rerun()

#Project loading
def load_project(project_name,user):
    st.session_state.cursor.execute(f"UPDATE projects set LastOpened = CURRENT_TIMESTAMP WHERE ProjectName = '{project_name}';")
    st.session_state.connection.commit()
    st.session_state.projects_df = pd.read_sql(f"SELECT * FROM projects WHERE Owner = '{st.session_state.username}'", st.session_state.connection)      
    st.session_state.my_models = pd.read_sql(f'''SELECT Name AS name,
                                                Model AS model,
                                                Import AS import,
                                                Scaler AS scaler,
                                                DimensionalityReduction AS dimensionality_reduction,
                                                Rmse AS rmse,
                                                R2Score AS r2_score,
                                                ExplainedVariance AS explained_variance,
                                                AUC,
                                                Accuracy AS accuracy,
                                                Recall AS recall,
                                                Precision AS precision,
                                                F1 AS f1,
                                                TrainingTime AS training_time,
                                                Features AS features,
                                                Hyperparameters AS hyperparameters,
                                                pr.Approach AS approach,
                                                Project AS project,
                                                pr.CreatedAt AS created_at
                                             FROM tested_models tm JOIN projects pr
                                                ON tm.Project = pr.ProjectName
                                             WHERE tm.Owner = '{st.session_state.username}'
                                                AND tm.Project = '{project_name}';''',
                                            st.session_state.connection)
    try:
        os.mkdir(f'./users/{user}/{project_name}')
    except:
        # We try to load the information involved in every step one by one if we succed the app takes us to the next step
        try:
            st.session_state.raw = pd.read_csv(f'./users/{user}/{project_name}/raw.csv').iloc[:,1:]
            st.session_state.target = st.session_state.projects_df.loc[st.session_state.projects_df['ProjectName'] == project_name,'Target'].reset_index(drop=True)[0]
            st.session_state.approach = st.session_state.projects_df.loc[st.session_state.projects_df['ProjectName'] == project_name,'Approach'].reset_index(drop=True)[0]
            try:
                st.session_state.data = pd.read_csv(f'./users/{user}/{project_name}/data.csv').iloc[:,1:] 
                st.session_state.selected_features = list(st.session_state.my_models['features'])[-1].strip(']').strip('[')
                st.session_state.selected_features = [feature.strip('\"') for feature in st.session_state.selected_features.split(', ')]
                try:
                    st.session_state.models = pd.read_csv(f'./users/{user}/{project_name}/recommended.csv') 
                    st.session_state.step = 'Model Testing'
                except:
                    st.session_state.step = 'Model Selection'
            except:
                st.session_state.step = 'EDA and Feature Engineering'
        except:
            st.session_state.step = 'Data Loading' 
    st.rerun()

# Project Update
def update_project():
    st.session_state.cursor.execute(f'''UPDATE projects
                                    SET Target = '{st.session_state.target}', Approach = '{st.session_state.approach}'
                                    WHERE ProjectName = '{st.session_state.project}';''')
    st.session_state.connection.commit()


# Project deletion  
def delete_project(project,owner):
    st.session_state.cursor.execute(f'''DELETE FROM projects WHERE Owner = '{owner}' and ProjectName = '{project}';''', {'owner': owner, 'project': project})
    st.session_state.connection.commit()
    st.session_state.projects_df.drop(st.session_state.projects_df[st.session_state.projects_df['ProjectName'] == project].index, inplace=True)
    st.rerun()
    
# Data loading
def data_loading():
    if len(separator) > 0:
        st.session_state.raw = pd.read_csv(uploaded_file,sep=separator)
    else:
        st.session_state.raw = pd.read_csv(uploaded_file)
    if st.session_state.raw.columns[0] == 'Unnamed: 0':
        st.session_state.raw = st.session_state.raw.iloc[:,1:]
    st.session_state.raw.to_csv(f'./users/{st.session_state.username}/{st.session_state.project}/raw.csv')
    return st.session_state.raw
        
#EDA
def eda(eda_feature):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax= plt.subplots(1,2, figsize = (16,9))
    if st.session_state.approach == 'classifier'    :
        if is_numeric_dtype(st.session_state.raw[eda_feature]) and len(list(st.session_state.raw[eda_feature].unique())) > 15:
            sns.histplot(ax= ax[0],data=st.session_state.raw,x=eda_feature,hue=st.session_state.target)
            sns.boxplot(ax= ax[1],data=st.session_state.raw,y=eda_feature,hue=st.session_state.target, showfliers = False)           
        elif len(list(st.session_state.raw[eda_feature].unique())) < 30:
            sns.histplot(ax= ax[0],data=st.session_state.raw,x=eda_feature)
            sns.countplot(ax= ax[1],data=st.session_state.raw,y=eda_feature,hue=st.session_state.target)            
    else:
        if is_numeric_dtype(st.session_state.raw[eda_feature]) and len(list(st.session_state.raw[eda_feature].unique())) > 15:
            sns.histplot(ax= ax[0],data=st.session_state.raw,x=eda_feature,bins=50)
            sns.scatterplot(ax= ax[1],data=st.session_state.raw,x=eda_feature,y=st.session_state.target)          
        elif len(list(st.session_state.raw[eda_feature].unique())) < 30:
            sns.histplot(ax= ax[0],data=st.session_state.raw,x=eda_feature)
            sns.boxplot(ax= ax[1],data=st.session_state.raw,y=st.session_state.target,hue=eda_feature, showfliers = False)
    st.pyplot(fig)

#Correlations heatmap
def correlations_heatmap():  
    fig = plt.figure(figsize=(16, 8))
    heatmap = sns.heatmap(st.session_state.raw.select_dtypes(exclude= ['object']).corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    st.pyplot(fig)

def eliminate_outliers(feature,ma,mi):
    st.session_state.raw = st.session_state.raw.loc[(st.session_state.raw[feature] > mi) & (st.session_state.raw[feature] < ma)]

def substitute_outliers(feature, ma,mi):
    st.session_state.raw.loc[st.session_state.raw[feature] < mi] = mi
    st.session_state.raw.loc[st.session_state.raw[feature] > ma] = ma

def label_encode(eda_feature, categories):
    st.session_state.raw.loc[~st.session_state.raw[eda_feature].isin(categories),eda_feature]= 0
    for value, category in enumerate(categories):
        st.session_state.raw.loc[st.session_state.raw[eda_feature] == category,eda_feature]= value + 1
    st.session_state.raw[eda_feature] = st.session_state.raw[eda_feature].astype('int64')
    st.rerun()

def feature_creation(feature_name,feature1,operation,feature2):
    if operation == 'multiply':
        st.session_state.raw[feature_name] = st.session_state.raw[feature1] * st.session_state.raw[feature2]
    if operation == 'divide':
        st.session_state.raw[feature_name] = st.session_state.raw[feature1] / st.session_state.raw[feature2]
    st.rerun()

#Filter the table and transform the data to numeric
def filter_transform(df,selected_features,target):
    st.session_state.data = df[selected_features].dropna()
    to_dummy = [feature for feature in selected_features if st.session_state.data[feature].dtype == 'object' and len(st.session_state.raw[feature].unique()) < 15]
    for feature in to_dummy:
        dummies_df = pd.get_dummies(st.session_state.data[feature], prefix=feature, drop_first=True).astype(int)
        st.session_state.data = pd.concat([st.session_state.data, dummies_df],axis=1).drop(feature, axis=1)   
    st.session_state.data = pd.concat([st.session_state.data.select_dtypes(exclude=['object']),df[target]], axis=1).dropna()
    st.session_state.data.to_csv(f'./users/{st.session_state.username}/{st.session_state.project}/data.csv')
    st.session_state.step = 'Model Selection'
    st.rerun()

#Model selection prompt
def model_selection(data,target):  
    while True:
        completion = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature= 0.4,
            response_format={ "type": "json_object" },
            messages=[{"role": "system", "content": f"For the given dataset choose the top 7 s models to create a {st.session_state.approach} model for {target} as well as the method to instance the model, the recomended scaler if any for each model and the required import"},
                    {"role": "user", "content": f'''Given the dataset below which shows the first 100 rows of a dataset with a total of {len(data.index)}, create a JSON object which enumerates a set of 7 child objects.                       
                        Each child object has four properties named "model", "method","scaler" and "import". The 7 child objects are the top 7 models to create a {st.session_state.approach} for {target}.
                        For each child object assign to the property named "model name" to the models name, "method" to the sklearn library method used to invoke the model, "Scaler" the 
                        recomended scaler if any for the model and dataset, and "import" the python script used to import the required final method from the library.
                        ''' + '''The resulting JSON object should be in this format: [{"model":"string","method":"string","scaler": "string","import": "string"}].\n\n
                        The dataset:\n''' +
                        f'''{str(data.head(100))}\n\n
                        The JSON object:\n\n'''}])
        try:
            #st.write(json.loads(completion.choices[0].message.content))
            return pd.DataFrame(json.loads(completion.choices[0].message.content)['models'])
        except:
            st.write('Our assistant is a little too imaginative todaay, lets give him another chance')

#Running every recommended model
def model_testing(data,target, approach, models):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    X= data[data.columns.drop(target)]
    if approach == 'classifier' and len(data[target].unique()) == 2 :
        y = pd.get_dummies(data[target],drop_first=True)
    else:
        y = data[target]
    models['rmse'] = np.nan
    models['r2_score'] = np.nan
    models['explained_variance'] = np.nan
    models['AUC'] = np.nan
    models['accuracy'] = np.nan
    models['recall'] = np.nan
    models['precision'] = np.nan
    models['f1'] = np.nan
    models['training_time'] = np.nan
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
    fig = plt.figure()
    for i in models.index:
        st.text(f"Testing {models.loc[i,'model']}...")
        try:
            exec(models.loc[i,'import']) # here we execute the required imports for each model
            if models.loc[i,['scaler']][0] is not None and models.loc[i,['scaler']][0].split('.')[-1].split(' ')[-1].strip('()') in ['StandardScaler', 'RobustScaler', 'MinMaxScaler']:
                scaler = eval(f"{models.loc[i,['scaler']][0].split('.')[-1].split(' ')[-1].strip('()')}()")
                X_train_i = scaler.fit_transform(X_train)
                X_test_i = scaler.fit_transform(X_test)
            else:
                X_train_i, X_test_i = X_train, X_test
            start = time.time()
            try:
                model = eval(f"{models.loc[i,['method']][0].split('.')[-1].split(' ')[-1].strip('()')}(n_jobs=-1).fit(X_train_i,y_train)")
            except:
                model = eval(f"{models.loc[i,['method']][0].split('.')[-1].split(' ')[-1].strip('()')}().fit(X_train_i,y_train)")
            models.loc[i,'training_time'] = time.time() - start
            y_pred = model.predict(X_test_i)
            if st.session_state.approach == 'classifier':
                models.loc[i,'accuracy'] = model.score(X_test_i, y_test)
                models.loc[i,'precision'], models.loc[i,'recall'], models.loc[i,'f1'], x = precision_recall_fscore_support(y_test,y_pred,average='weighted')
                try:
                    models.loc[i,'AUC'] = roc_auc_score(y_test,model.predict_proba(X_test_i)[:, 1])
                    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_i)[:, 1])
                    plt.plot(fpr, tpr, label=models.loc[i,'model'])
                except:
                    pass
            else:
                models.loc[i,'rmse'] = mean_squared_error(y_test,y_pred,squared=False)
                models.loc[i,'r2_score'] = r2_score(y_test,y_pred)
                models.loc[i,'explained_variance'] = explained_variance_score(y_test,y_pred)
        except:
            st.text(f"{models.loc[i,'model']} could not be tested")
    if st.session_state.approach == 'classifier':
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        st.pyplot(fig)
    else:
        sns.scatterplot(data=st.session_state.models,y='rmse',x='r2_score')
        st.pyplot(fig)
    models.to_csv(f'./users/{st.session_state.username}/{st.session_state.project}/recommended.csv', index=False)
    st.session_state.step = 'Model Testing'
    time.sleep(2)
    st.rerun()

# Grid search function
def grid_search(test_model, models, data, complexity, approach, scaler, dimensionality_reduction, dimensions):
    i = models.index[models['model'] == test_model].to_list()[0]
    st.session_state.test_model_df = models.loc[models['model'] == test_model,:]
    exec(models.loc[i,'import'])
    try:
        model = eval(f"{list(st.session_state.test_model_df['method'])[0].split('.')[-1].split(' ')[-1].strip('()')}(n_jobs=-1)")
    except:
        model = eval(f"{list(st.session_state.test_model_df['method'])[0].split('.')[-1].split(' ')[-1].strip('()')}()")
    hyperparams = str(inspect.signature(model.__init__))
    hyperparams = [h.split('=')[0] for h in hyperparams.split(', ')]
    response = st.session_state.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {
        "role": "system",
        "content": "For a certain machine learning model and a complexity level. Create a parameter grid to perform a GridSearch with the given level of complexity. Answer should consist only of the requested JSON object."
        },
        {
        "role": "user",
        "content": "Create a GridSearch for a RandomForestClassifier model. This gridsearch MUST be extremely simple. Create a JSON object which contains 1 child object with as many properties as hyperparameters should be included in the GridSearch JSON."
        },
        {
        "role": "assistant",
        "content": "{\n  \"param_grid\": {\n    \"n_estimators\": [10, 20],\n    \"max_depth\": [None, 5],\n    \"min_samples_split\": [2, 5],\n    \"min_samples_leaf\": [1, 2],\n    \"max_features\": [\"auto\", \"sqrt\"]\n  }\n}"
        },
        {
        "role": "user",
        "content": f"Create a GridSearch for a {test_model} {approach} model. This gridsearch MUST be {complexity}. Create a JSON object which contains 1 child object with as many properties as hyperparameters should be included in the GridSearch JSON."# The product of the number of values in each property of the child object MUST be around or less than {combinations}"
        }
        ],
        temperature=0.6,
        max_tokens=1770,    
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    param_grid = eval(response.choices[0].message.content.split('grid":')[-1].strip('}'))
    #st.write(param_grid)

    param_grid2 = param_grid.copy()
    for key in param_grid.keys():
        if key not in hyperparams:
            del param_grid2[key]

    X= data[data.columns.drop(st.session_state.target)]
    if approach == 'Classifier' and len(data[st.session_state.target].unique()) == 2:
        y = pd.get_dummies(data[st.session_state.target], drop_first=True, prefix=st.session_state.target)
    else:
        y = data[st.session_state.target]
    #st.write(f"{list(test_model_df['method'])[0].split('.')[-1].strip('()')}()")
    #st.text(hyperparams)
    st.session_state.test_model_df.loc[i,'scaler'] = scaler
    st.session_state.test_model_df.loc[i,'dimensionality_reduction'] = f'{dimensionality_reduction}: {dimensions} dimensions'
    start = time.time()
    if scaler is not None:
        scaler = eval(scaler + '()')
        X = scaler.fit_transform(X)
    if dimensionality_reduction == 'UMAP':
        reducer = umap.UMAP(n_components=dimensions)
        X = reducer.fit_transform(X)
    elif dimensionality_reduction == 'PCA':
        reducer = PCA(n_components=dimensions)
        X = reducer.fit_transform(X)
    try:
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)#,error_score='raise')
        grid_search.fit(X,y)
    except:
        grid_search = GridSearchCV(model, param_grid2, cv=3, n_jobs=-1)#,error_score='raise')
        grid_search.fit(X,y)
    #st.write(f"Best Parameters: {grid_search.best_params_} Best Score: {grid_search.best_score_} Execution time: {time.time() - start}")
    best_model = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
    best_model.fit(X_train,y_train)
    if st.session_state.approach == 'classifier':
        st.session_state.test_model_df.loc[i,'accuracy'] = best_model.score(X_test, y_test)
        models.loc[i,'precision'], models.loc[i,'recall'], models.loc[i,'f1'], x = precision_recall_fscore_support(y_test,best_model.predict(X_test),average='weighted')  
        try:
            st.session_state.test_model_df.loc[i,'AUC'] = roc_auc_score(y_test,best_model.predict_proba(X_test)[:, 1])
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
    else:
        st.session_state.test_model_df.loc[i,'rmse'] = mean_squared_error(y_test,best_model.predict(X_test),squared=False)
        st.session_state.test_model_df.loc[i,'r2_score'] = r2_score(y_test,best_model.predict(X_test))
        st.session_state.test_model_df.loc[i,'explained_variance'] = explained_variance_score(y_test,best_model.predict(X_test))
    
    return best_model, st.session_state.test_model_df
# Saving models
def save_model(model_name, trained_model, model_df, selected_features,dimensionality_reduction, dimensions):
    model_df['name'] = model_name.replace(' ','_')
    model_df['hyperparameters'] = str(trained_model.get_params())
    model_df['features'] = str(selected_features)
    model_df['dimensionality_reduction'] = f'{dimensionality_reduction}: {dimensions} dimensions'
    model_df = model_df.reset_index(drop=True)
    buffer = StringIO()
    model_df.info(buf=buffer)
    s = buffer.getvalue()
    upload_df = model_df.fillna('NULL')
    try:
        st.session_state.cursor.execute(f'''INSERT INTO tested_models(
                                                Name,
                                                Model,
                                                Approach,
                                                Import,
                                                Scaler,
                                                DimensionalityReduction,
                                                Rmse,
                                                R2Score,
                                                ExplainedVariance,
                                                AUC,
                                                Accuracy,
                                                Recall,
                                                Precision,
                                                F1,
                                                TrainingTime,
                                                Features,
                                                Hyperparameters,
                                                Project,
                                                CreatedAt,
                                                Owner)
                                    VALUES('{upload_df.loc[0,'name']}',
                                                '{upload_df.loc[0,'method'].split('.')[-1].split(' ')[-1].strip('()')}',
                                                '{st.session_state.approach}',
                                                '{upload_df.loc[0,'import']}',
                                                '{upload_df.loc[0,'scaler']}',
                                                '{upload_df.loc[0,'dimensionality_reduction']}',
                                                {upload_df.loc[0,'rmse']},
                                                {upload_df.loc[0,'r2_score']},
                                                {upload_df.loc[0,'explained_variance']},
                                                {upload_df.loc[0,'AUC']},
                                                {upload_df.loc[0,'accuracy']},
                                                {upload_df.loc[0,'recall']},
                                                {upload_df.loc[0,'precision']},
                                                {upload_df.loc[0,'f1']},
                                                {upload_df.loc[0,'training_time']},
                                                '{str(upload_df.loc[0,'features']).replace("'",'"')}',
                                                '{str(upload_df.loc[0,'hyperparameters']).replace("'",'"')}',
                                                '{st.session_state.project}',
                                                CURRENT_TIMESTAMP,
                                                '{st.session_state.username}');''')
        st.session_state.connection.commit()
        if "my_models" not in st.session_state:
            st.session_state.my_models = model_df
        else:
            st.session_state.my_models = pd.concat([st.session_state.my_models, model_df],axis=0).reset_index(drop=True)
    except:
        st.write('Model name already in use please choose another one')

# Header
st.components.v1.html('<h2 style="text-align: center;">A.I.A.M.A.</h2>', width=None, height=50, scrolling=False)

# Create the connection with the database and OpenAI
if "connetion" not in st.session_state:    
    st.session_state.connection = sqlite_connection()
if 'client' not in st.session_state:
    gpt_connect()

# Create folders if necessary
if 'folders' not in st.session_state:
    folder_management()
    st.session_state.folders = True

# We choose the step (page) to work on
if "step" not in st.session_state:
    st.session_state.step = 'User Login'
    st.session_state.steps = ['User Login', 'Projects','Data Loading','EDA and Feature Engineering', 'Model Selection', 'Model Testing', 'ChatBot Assistant']
st.session_state.step = st.sidebar.selectbox('Choose step', st.session_state.steps, st.session_state.steps.index(st.session_state.step))

#User Management
if st.session_state.step == 'User Login':
    login_tab, register_tab = st.tabs(['Login','Register'])
    if "username" in st.session_state and st.session_state.username == 'admin':
        st.dataframe(st.session_state.users_df)
    st.session_state.users_df = pd.read_sql("SELECT * FROM users", st.session_state.connection)
    with login_tab:
        username = st.text_input('Username: ',placeholder='your_username')
        password = st.text_input('Password: ',placeholder='your_password',type='password')
        if st.button('Login',type='primary'):
            user_login(username,password)
    with register_tab:
        username = st.text_input('Username: ')
        password = st.text_input('Password: ',type='password')
        password_confirm = st.text_input('Comfirm Password: ',placeholder='Repeat your password',type='password')
        if st.button('Create',type='primary'):
            user_create(username,password,password_confirm)

# Project Management
if st.session_state.step == 'Projects':
    create_tab, load_tab = st.tabs(['Create','Load'])
    if "username" not in st.session_state:
        st.write('Please login to be able to manage your projects')
    else:
        
        with create_tab:
            st.session_state.project = st.text_input('Project Name').replace(' ','_')
            if st.button('Create', type='primary'):
                create_project(st.session_state.project, st.session_state.username)
        
        with load_tab:
            st.dataframe(st.session_state.projects_df)
            st.session_state.project = st.selectbox('Select Project', list(st.session_state.projects_df['ProjectName']))
            if st.button('Load',type='primary'):
                load_project(st.session_state.project, st.session_state.username)
            if st.button('Delete'):
                st.write(st.session_state.project, st.session_state.username)
                delete_project(st.session_state.project, st.session_state.username)

if 'columns_to show' not in st.session_state and 'approach' in st.session_state and 'target' in st.session_state:
    st.session_state.columns_to_show = ['model', 'method', 'scaler', 'dimensionality_reduction']
    if st.session_state.approach == 'regressor':
        st.session_state.columns_to_show += ['rmse', 'r2_score', 'explained_variance', 'training_time']
    elif st.session_state.approach == 'classifier':
        st.session_state.columns_to_show += ['AUC', 'accuracy', 'recall', 'precision', 'f1', 'training_time']

#Data loading
if st.session_state.step == 'Data Loading':
    if "project" not in st.session_state:
        st.write('Before continuing, please create or load a project')
    else:
        uploaded_file = st.sidebar.file_uploader('Upload your csv here')
        separator = st.sidebar.text_input('Separator',placeholder=',')
        if uploaded_file is not None:
                st.session_state.raw = data_loading()
        # Choosing target and approach
        if "raw" in st.session_state:
            if 'target' not in st.session_state:
                target = st.sidebar.selectbox('Target', st.session_state.raw.columns,0, placeholder="Choose the target")
            else:
                target = st.sidebar.selectbox('Target', st.session_state.raw.columns,list(st.session_state.raw.columns).index(st.session_state.target), placeholder="Choose the target")
            if 'approach' not in st.session_state:
                st.session_state.approach = 'classiffier'
            st.session_state.approach = st.sidebar.selectbox('Approach', ['classifier', 'regressor'],1 if st.session_state.approach == 'regressor' else 0, placeholder="Choose the target")
            st.dataframe(st.session_state.raw)
            st.dataframe(pd.DataFrame({"name": st.session_state.raw.columns, "non-nulls": len(st.session_state.raw)-st.session_state.raw.isnull().sum().values,
                                "nulls": st.session_state.raw.isnull().sum().values, "type": st.session_state.raw.dtypes.values, "unique": [len(st.session_state.raw[col].unique()) for col in st.session_state.raw.columns] }))   
            if st.sidebar.button('Next',type='primary'):
                st.session_state.target = target
                update_project()
                st.session_state.step = 'EDA and Feature Engineering'
                st.rerun()        
        else:
            st.markdown('#### Load a file to work on')
# EDA
if st.session_state.step == 'EDA and Feature Engineering':
    eda_tab, table_tab, corr_tab, fe_tab, data_tab = st.tabs(['EDA','Source Table','Correlations', 'Feature Engineering', 'Clean Data'])
    if "raw" not in st.session_state:
        st.write('Before continuing, please load a dataset to work on')
    else:
        st.session_state.features = list(st.session_state.raw.columns)
        st.session_state.features.remove(st.session_state.target)
        with eda_tab:
            eda_feature = st.selectbox('EDA', st.session_state.raw.columns, placeholder="Choose a feature")
            eda(eda_feature)
            col1, col2 = st.columns([0.2,0.8])
            with col1:
                st.dataframe(st.session_state.raw[[eda_feature]].describe())
            with col2:
                if len(st.session_state.raw[eda_feature].unique()) > 15 and st.session_state.raw[eda_feature].dtype != 'object':
                    mi = st.number_input('Minimum',st.session_state.raw[eda_feature].min(),st.session_state.raw[eda_feature].max(),st.session_state.raw[eda_feature].min())
                    ma = st.number_input('Maximum',st.session_state.raw[eda_feature].min(),st.session_state.raw[eda_feature].max(),st.session_state.raw[eda_feature].max())
                    if st.button('Eliminate Outliers', type='primary'):
                        eliminate_outliers(eda_feature,ma,mi)
                        st.rerun()
                    if st.button('Substitute Outliers',):
                        substitute_outliers(eda_feature,ma,mi)
                        st.rerun()
                elif len(st.session_state.raw[eda_feature].unique()) < 15:
                    coll, colr = st.columns([0.25,0.75])
                    with colr:
                        st.write('Label encoding')
                        category_importance = st.multiselect('Choose the categories to be label encoded, the order in which you choose the categories will represent their value when encoded. All non selected categories will be enconded as 0', st.session_state.raw[eda_feature].unique())
                        if st.button('Label encode',type='primary'):
                            label_encode(eda_feature,category_importance)
                    with coll:
                        st.dataframe({category: value + 1  for value, category in enumerate(category_importance)})
            st.markdown('''*Any categorical feature with less than 15 categories that is not label enconded,
                                 will be automatically One Hot Encoded when filter and transform is pressed''')
                
        with table_tab:
            st.dataframe(st.session_state.raw)
            st.dataframe(pd.DataFrame({"name": st.session_state.raw.columns, "non-nulls": len(st.session_state.raw)-st.session_state.raw.isnull().sum().values,
                                "nulls": st.session_state.raw.isnull().sum().values, "type": st.session_state.raw.dtypes.values, "unique": [len(st.session_state.raw[col].unique()) for col in st.session_state.raw.columns] }))   
        with corr_tab:
            correlations_heatmap()

# Feature engineering       
        with data_tab:
            if "data" in st.session_state:
                st.dataframe(st.session_state.data)
                st.dataframe(pd.DataFrame({"name": st.session_state.data.columns, "non-nulls": len(st.session_state.data)-st.session_state.data.isnull().sum().values, "nulls": st.session_state.data.isnull().sum().values, "type": st.session_state.data.dtypes.values, "unique": [len(st.session_state.data[col].unique()) for col in st.session_state.data.columns] }))
            else:
                st.write('Please filter and transform in order to visualize clean data')

        with fe_tab:
            col1, col2 , col3, col4 = st.columns(4)
            with col1:
                feature_name = st.text_input('Feature Name')
            with col2:
                feature1 = st.selectbox('Feature 1', st.session_state.raw.columns)
            with col3:
                operation = st.selectbox('Operation', ['multiply', 'divide'])
            with col4:
                feature2 = st.selectbox('Feature 2', st.session_state.raw.columns)
            if st.button('Create Feature', type='primary'):
                feature_creation(feature_name,feature1,operation,feature2)
            st.write('Here you have a box to create your own features with a bit of Python. Please refer to the dataframe as df:')
            df = st.session_state.raw

            if 'execution' not in st.session_state:
                st.session_state.execution = None
            code = st_ace(language='python', theme='chrome')
            if code != st.session_state.execution:
                try:
                    exec(code)
                    st.session_state.execution = code
                except:
                    st.write('Your code could not be executed please, check it for errors') 
                    time.sleep(60)
                st.rerun()
            st.session_state.raw = df.copy()

# Feature selection
        if "selected_features" not in st.session_state:
            st.session_state.selected_features = st.session_state.features.copy()
        st.session_state.selected_features = st.sidebar.multiselect('Selected Features',st.session_state.features, st.session_state.selected_features)
        if st.sidebar.button('Filter and transform', type='primary'):
            st.session_state.data = filter_transform(st.session_state.raw,st.session_state.selected_features,st.session_state.target)

# model recommendation   
if st.session_state.step == 'Model Selection':
    if 'data' not in st.session_state:
        st.write('Before continuing, please select features and filter data.')
    else:
        if st.sidebar.button('Recomended models'):
            st.session_state.models = model_selection(st.session_state.data, st.session_state.target)

# model testing
        if "models" in st.session_state:
            if st.sidebar.button('Test Models',type='primary'):
                model_testing(st.session_state.data,st.session_state.target, st.session_state.approach, st.session_state.models)
        if "models" in st.session_state:
            if st.checkbox('Show recommended models', value=True):    
                st.dataframe(st.session_state.models)

# Gridsearch
if st.session_state.step == "Model Testing" and "models" in st.session_state:
    test_model = st.sidebar.selectbox(' Test Model ', st.session_state.models['model'], placeholder="Choose the model")
    i = st.session_state.models.index[st.session_state.models['model'] == test_model].to_list()[0]
    scaler = st.sidebar.selectbox("Scaler",[None,"StandardScaler","RobustScaler","MinMaxScaler"])
    dimensionality_reduction = st.sidebar.selectbox("Dimensionality reduction",[None,"UMAP","PCA"])
    if dimensionality_reduction is not None:
        dimensions  = st.sidebar.slider('Number of dimensions', 2, len(st.session_state.data.columns),2)
    else:
        dimensions = len(st.session_state.data.columns)
    complexity = st.sidebar.select_slider('Grid complexity', ['extremely simple', 'very simple', 'simple', 'complex', 'very complex', 'extremely complex'],)
    if st.sidebar.button('Gridsearch'):
        st.session_state.trained_model, st.session_state.test_model_df = grid_search(test_model,st.session_state.models, st.session_state.data, complexity, st.session_state.approach, scaler, dimensionality_reduction,dimensions)
    st.dataframe(st.session_state.test_model_df[st.session_state.columns_to_show])

    
    if "test_model_df" in st.session_state:
        model_name = st.text_input('model name')
        if st.button('Save model'):
            save_model(model_name, st.session_state.trained_model, st.session_state.test_model_df, st.session_state.selected_features, dimensionality_reduction, dimensions)
    if st.checkbox('My models', value = True) and "my_models" in st.session_state:
        st.dataframe(st.session_state.my_models[['name'] + st.session_state.columns_to_show])
    if st.checkbox('Show recommended models', value=True):    
        columns = st.session_state.columns_to_show.copy()
        columns.remove('dimensionality_reduction')
        st.dataframe(st.session_state.models[columns])

# ChatBot Assistant
if st.session_state.step == "ChatBot Assistant" and "raw" in st.session_state:
    assistant_id = "asst_bJ4RIPQgYL4pINAvQOYyjH4h"

    if "start_chat" not in st.session_state:
        st.session_state.start_chat = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    if st.sidebar.button('Start Chat', type='primary'):
        st.session_state.start_chat = True
        thread = st.session_state.client.beta.threads.create()
        st.session_state.thread_id = thread.id
        
    if st.sidebar.button('Clear Chat'):
        st.session_state.start_chat = False
        st.session_state.thread_id = None
        st.session_state.messages = []

    if st.session_state.start_chat:
        if "openai_model" not in st.session_state:
            st.session_state.openai_model = "gpt-3.5-turbo"
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.session_state.client.beta.threads.messages.create(
                    thread_id= st.session_state.thread_id,
                    role= 'user',
                    content = f''' These are th first 50 rows of a dataframe with a total of {len(st.session_state.raw.index)} rows:/n
                                {str(st.session_state.raw.head(50))}/n
                                I am trying to create a {st.session_state.approach} model for {st.session_state.target}/n
                                Can you help me out?'''
                    )
        
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        if prompt := st.chat_input('Can you help me with this dataset?'):

            st.session_state.messages.append({'role' : 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)

            st.session_state.client.beta.threads.messages.create(
                    thread_id= st.session_state.thread_id,
                    role= 'user',
                    content = prompt
                    )
            
            run = st.session_state.client.beta.threads.runs.create(
                        thread_id = st.session_state.thread_id,
                        assistant_id = assistant_id
                        )
            
            while run.status != "completed":
                time.sleep(1)
                run = st.session_state.client.beta.threads.runs.retrieve(
                        thread_id = st.session_state.thread_id,
                        run_id = run.id
                        )
                
            messages = st.session_state.client.beta.threads.messages.list(
                    thread_id = st.session_state.thread_id
                    )
            
            #Proccess and display messages
            assistant_messages_for_run = [
                message for message in messages
                if message.run_id == run.id and message.role == 'assistant'
                ]
        
            for message in assistant_messages_for_run:
                st.session_state.messages.append({'role': 'assistant', 'content': message.content[0].text.value})
                with st.chat_message('Assistant'):
                    st.markdown(message.content[0].text.value)

# Signature
if "project" in st.session_state:
    st.sidebar.write(f'Working on:  {st.session_state.project}')
if "username" in st.session_state:
    st.sidebar.write(f'Logged as:  {st.session_state.username}')
