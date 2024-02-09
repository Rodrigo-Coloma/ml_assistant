import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, explained_variance_score, mean_squared_error, r2_score, precision_recall_fscore_support, f1_score
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
import numpy as np
import pyodbc
import inspect
import json
import time
import os

# Azure connection
#@st.cache_resource
def azure_connection():
    try:
        PASSWORD = dotenv_values('./.env')['AZUREPWD']
    except:
        PASSWORD = st.session_state.api_key = st.secrets["AZUREPWD"]
    SERVER = 'sqlrjcg123.database.windows.net'
    DATABASE = 'Database'
    USERNAME = 'azureuser'
    connectionString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};Trusted_connection=no'
    connection = pyodbc.connect(connectionString)
    st.session_state.cursor = connection.cursor()
    return connection

# Create folder struccture if it doesn't exist
def folder_management():
    if 'users' not in os.listdir('./'):
        os.mkdir('./users')

def user_create(username,password):
    try:
        st.session_state.cursor.execute(f"INSERT INTO mlassistant.users(Username, Password, Registration_Date) VALUES ('{username}','{password}',CURRENT_TIMESTAMP);")
        st.session_state.connection.commit()
        st.session_state.username = username
        st.write('User succesfully created!!')
        st.session_state.projects_df = pd.read_sql(f"SELECT * FROM mlassistant.projects WHERE Owner = '{st.session_state.username}'", st.session_state.connection)
        os.mkdir(f'./users/{username}')
        time.sleep(1.5)
        st.session_state.step = st.session_state.steps[1 + st.session_state.steps.index(st.session_state.step)]
        st.rerun()
    except:
        st.write('Username already exists or contains invalid characters, please choose a new one')

def user_login(username,password):
    if list(st.session_state.users_df.loc[st.session_state.users_df['Username'] == username,:]['Password'])[0] == password:
        st.session_state.username = username
        st.session_state.projects_df = pd.read_sql(f"SELECT * FROM mlassistant.projects WHERE owner = '{st.session_state.username}'", st.session_state.connection)
        if username not in os.listdir('./users/'):
            os.mkdir(f'./users/{username}')
        st.session_state.step = st.session_state.steps[1 + st.session_state.steps.index(st.session_state.step)]
        st.rerun()
    else:
        st.write('Incorrect username or password')




#Project creation
def create_project(project_name,user):
    #try:
        st.session_state.cursor.execute(f'''INSERT INTO mlassistant.projects(ProjectName, Owner, Target, Approach, CreatedAt, LastOpened)
                                         VALUES ('{project_name}','{st.session_state.username}', NULL, NULL, CURRENT_TIMESTAMP,CURRENT_TIMESTAMP);''')
        st.session_state.connection.commit()
        st.session_state.projects_df = pd.read_sql(f"SELECT * FROM mlassistant.projects WHERE Owner = '{st.session_state.username}'", st.session_state.connection)      
        st.session_state.project = project_name
        try:
            os.mkdir(f'./users/{user}/{project_name}')
        except:
            st.write('Directory existed')
        st.write('Project succesfully created!!')
        time.sleep(1.5)
        st.session_state.step = st.session_state.steps[1 + st.session_state.steps.index(st.session_state.step)]
        st.rerun()
    #except:
        st.write('Project name already exists or contains invalid characters, please choose a new one')

#Project loading
def load_project(project_name,user):
    st.session_state.cursor.execute(f"UPDATE mlassistant.projects set LastOpened = CURRENT_TIMESTAMP WHERE ProjectName = '{project_name}';")
    st.session_state.connection.commit()
    st.session_state.projects_df = pd.read_sql(f"SELECT * FROM mlassistant.projects WHERE Owner = '{st.session_state.username}'", st.session_state.connection)      
    st.session_state.my_models = pd.read_sql(f'''SELECT ModelName AS model,
                                                Method AS method,
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
                                             FROM mlassistant.tested_models tm JOIN mlassistant.projects pr
                                                ON tm.Project = pr.ProjectName
                                             WHERE pr.Owner = '{st.session_state.username}'
                                                AND tm.Project = '{project_name}';''',
                                            st.session_state.connection)
    try:
        os.mkdir(f'./users/{user}/{project_name}')
    except:
        try:
            st.session_state.raw = pd.read_csv(f'./users/{user}/{project_name}/raw.csv').iloc[:,1:]
            st.session_state.target = st.session_state.projects_df.loc[st.session_state.projects_df['ProjectName'] == project_name,'Target'].reset_index(drop=True)[0]
            st.session_state.approach = st.session_state.projects_df.loc[st.session_state.projects_df['ProjectName'] == project_name,'Approach'].reset_index(drop=True)[0]
            try:
                st.session_state.data = pd.read_csv(f'./users/{user}/{project_name}/data.csv').iloc[:,1:] 
                try:
                    st.session_state.models = pd.read_csv(f'./users/{user}/{project_name}/recommended.csv').drop('Unnamed: 0', inplace=True) 
                    st.session_state.step = 'Model Testing'
                except:
                    st.session_state.step = 'Model Selection'
            except:
                st.session_state.step = 'EDA and Feature Selection'
        except:
            st.session_state.step = 'Data Loading' 
    st.rerun()
        
    
# Data loading
#@st.cache_data
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
def filter_transform(df,selected_features,target):
    st.session_state.data = df[selected_features].dropna()
    to_dummy = [feature for feature in selected_features if len(st.session_state.raw[feature].unique()) < 15]
    for feature in to_dummy:
        dummies_df = pd.get_dummies(st.session_state.data[feature], prefix=feature, drop_first=True)
        st.session_state.data = pd.concat([st.session_state.data, dummies_df],axis=1).drop(feature, axis=1)   
    st.session_state.data = pd.concat([st.session_state.data.select_dtypes(exclude=['object']),df[target]], axis=1).dropna()
    st.session_state.data.to_csv(f'./users/{st.session_state.username}/{st.session_state.project}/data.csv')
    st.session_state.step = 'Model Selection'
    st.rerun()
    return st.session_state.data


#Model selection prompt
#@st.cache_data
def model_selection(data,target):
    
    try:
        st.session_state.api_key = dotenv_values('./.env')['GPTAPIKEY']
    except:
        st.session_state.api_key = st.secrets['GPTAPIKEY']
    st.session_state.client = OpenAI(api_key=st.session_state.api_key)
    while True:
        completion = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature= 0.4,
            messages=[{"role": "system", "content": f"For the given dataset choose the top 7 sklearn models to create a {st.session_state.approach} model for {target} as well as the method to instance the model, the recomended scaler if any for each model and the required import"},
                    {"role": "user", "content": f'''Given the dataset below which shows the first 100 rows of a dataset with a total of {len(data.index)}, create a JSON object which enumerates a set of 7 child objects.                       
                        Each child object has four properties named "model", "method","scaler" and "import". The 7 child objects are the top 7 models to create a {st.session_state.approach} for {target}.
                        For each child object assign to the property named "model name" to the models name, "method" to the sklearn library method used to invoke the model, "Scaler" the 
                        recomended scaler if any for the model and dataset, and "import" the python script used to import the required final method from the library.
                        ''' + '''The resulting JSON object should be in this format: [{"model":"string","method":"string","scaler": "string","import": "string"}].\n\n
                        The dataset:\n''' +
                        f'''{str(data.head(100))}\n\n
                        The JSON object:\n\n'''}])
        try:
            return pd.DataFrame(json.loads(completion.choices[0].message.content))
        except:
            st.write('Our assistant is a little too imaginative todaay, lets give him another chance')

#Running every recommended model
#@st.cache_data
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
            exec(models.loc[i,'import'])
            if models.loc[i,['scaler']][0] is not None and models.loc[i,['scaler']][0].split('.')[-1].strip('()') in ['StandardScaler', 'RobustScaler', 'MinMaxScaler']:
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
        st.pyplot()
    models.to_csv(f'./users/{st.session_state.username}/{st.session_state.project}/recommended.csv')
    st.session_state.step = 'Model Testing'
    st.rerun()

def grid_search(test_model, models, data, minutes, approach, scaler):
    i = models.index[models['model'] == test_model].to_list()[0]
    test_model_df = models.loc[models['model'] == test_model,:]
    combinations = max(int(minutes * 60 / (models.loc[i,'training_time'] + 1)),4)
    exec(models.loc[i,'import'])
    try:
        model = eval(f"{list(test_model_df['method'])[0].split('.')[-1].split(' ')[-1].strip('()')}(n_jobs=-1)")
    except:
        model = eval(f"{list(test_model_df['method'])[0].split('.')[-1].split(' ')[-1].strip('()')}()")
    hyperparams = str(inspect.signature(model.__init__))
    st.write(hyperparams)
    hyperparams = [h.split('=')[0] for h in hyperparams.split(', ')]
    st.write(hyperparams)
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
        temperature=0.6,
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
    try:
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X,y)
    except:
        grid_search = GridSearchCV(model, param_grid2, cv=5)
        grid_search.fit(X,y)
    start = time.time()
    st.write(f"Best Parameters: {grid_search.best_params_} Best Score: {grid_search.best_score_} Execution time: {time.time() - start}")
    best_model = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    best_model.fit(X_train,y_train)

    if st.session_state.approach == 'classifier':
        test_model_df.loc[i,'accuracy'] = best_model.score(X_test, y_test)
        models.loc[i,'precision'], models.loc[i,'recall'], models.loc[i,'f1'], x = precision_recall_fscore_support(y_test,best_model.predict(X_test),average='weighted')
                
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
    model_df['dimensionality_reduction'] = None
    model_df = model_df.reset_index(drop=True)
    st.dataframe(model_df)
    buffer = StringIO()
    model_df.info(buf=buffer)
    s = buffer.getvalue()
    upload_df = model_df.fillna('NULL')
    try:
        st.session_state.cursor.execute(f'''INSERT INTO mlassistant.tested_models(
                                                ModelName,
                                                Approach,
                                                Method,
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
                                                CreatedAt)
                                    VALUES('{upload_df.loc[0,'model']}',
                                                '{st.session_state.approach}',
                                                '{upload_df.loc[0,'method']}',
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
                                                CURRENT_TIMESTAMP);''')
        st.session_state.connection.commit()
        if "my_models" not in st.session_state:
            st.session_state.my_models = model_df
        else:
            st.session_state.my_models = pd.concat([st.session_state.my_models, model_df],axis=0).reset_index(drop=True)
    except:
        st.write('Model name already in use please choose anotherone')


def update_project():
    st.session_state.cursor.execute(f'''UPDATE mlassistant.projects
                                    SET Target = '{st.session_state.target}', Approach = '{st.session_state.approach}'
                                    WHERE ProjectName = '{st.session_state.project}';''')
    st.session_state.connection.commit()




#def plot_results(my_models,approach)

st.components.v1.html('<h2 style="text-align: center;">A.I.A.M.A.</h2>', width=None, height=50, scrolling=False)


# Create the connection with the database
if "connetion" not in st.session_state:    
    st.session_state.connection = azure_connection()

# Create folders if necessary
folder_management()

# We choose the step (page) to work on
if "step" not in st.session_state:
    st.session_state.step = 'User Login'
    st.session_state.steps = ['User Login', 'Projects','Data Loading','EDA and Feature Selection', 'Model Selection', 'Model Testing']
st.session_state.step = st.sidebar.selectbox('Choose step', st.session_state.steps, st.session_state.steps.index(st.session_state.step))
#We create tabs to navigate through the steps
#tabs = st.tabs(['Projects', 'Data Loading','EDA and Feature Selection', 'Model Selection', 'Model Testing'])

#User Management
if st.session_state.step == 'User Login':
    st.session_state.users_df = pd.read_sql("SELECT * FROM mlassistant.users", st.session_state.connection)
    if "username" in st.session_state and st.session_state.username == 'admin':
        st.dataframe(st.session_state.users_df)
    username = st.text_input('Username: ',placeholder='Your Username')
    password = st.text_input('Password: ',placeholder='Your Password',type='password')
    if st.button('Login',type='primary'):
        user_login(username,password)
    if st.button('Create'):
        user_create(username,password)

    



# Project Management
if st.session_state.step == 'Projects':
    if "username" not in st.session_state:
        st.write('Please login to be able to manage your projects')
    else:
        st.dataframe(st.session_state.projects_df)
        status = st.selectbox('Create or Load', ['Create', 'Load'], 0)
        if status == 'Create':
            st.session_state.project = st.text_input('Porject Name').replace(' ','_')
            if st.button('Create', type='primary'):
                create_project(st.session_state.project, st.session_state.username)
        if status == 'Load':
            st.session_state.project = st.selectbox('Select Project', list(st.session_state.projects_df['ProjectName']))
            if st.button('Load'):
                load_project(st.session_state.project, st.session_state.username)
    
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
            st.session_state.target = st.sidebar.selectbox('Target', st.session_state.raw.columns, placeholder="Choose the target")
            st.session_state.approach = st.sidebar.selectbox('Approach', ['classifier', 'regressor'], placeholder="Choose the target")
            st.dataframe(st.session_state.raw)
            st.dataframe(pd.DataFrame({"name": st.session_state.raw.columns, "non-nulls": len(st.session_state.raw)-st.session_state.raw.isnull().sum().values,
                                "nulls": st.session_state.raw.isnull().sum().values, "type": st.session_state.raw.dtypes.values, "unique": [len(st.session_state.raw[col].unique()) for col in st.session_state.raw.columns] }))   
            if st.sidebar.button('Next',type='primary'):
                update_project()
                st.session_state.step = 'EDA and Feature Selection'
                st.rerun()        
        else:
            st.markdown('#### Load a file to work on')
# EDA
if st.session_state.step == 'EDA and Feature Selection':
    if "raw" not in st.session_state:
        st.write('Before continuing, please load a dataset to work on')
    else:
        st.session_state.features = list(st.session_state.raw.columns)
        st.session_state.features.remove(st.session_state.target)
        eda_feature = st.selectbox('EDA', st.session_state.raw.columns, placeholder="Choose a feature")
        eda(eda_feature)
        st.session_state.show_table = st.checkbox('Show table', value=True)
        if st.session_state.show_table:
            st.dataframe(st.session_state.raw)
            st.dataframe(pd.DataFrame({"name": st.session_state.raw.columns, "non-nulls": len(st.session_state.raw)-st.session_state.raw.isnull().sum().values,
                                "nulls": st.session_state.raw.isnull().sum().values, "type": st.session_state.raw.dtypes.values, "unique": [len(st.session_state.raw[col].unique()) for col in st.session_state.raw.columns] }))   
# Feature selection
        if "selected_features" not in st.session_state:
            st.session_state.selected_features = []
        st.session_state.selected_features = st.sidebar.multiselect('Selected Features',st.session_state.features,st.session_state.selected_features)
        if st.sidebar.button('Filter and transform', type='primary'):
            st.session_state.data = filter_transform(st.session_state.raw,st.session_state.selected_features,st.session_state.target)
        if "data" in st.session_state:
            st.dataframe(st.session_state.data)
            st.dataframe(pd.DataFrame({"name": st.session_state.data.columns, "non-nulls": len(st.session_state.data)-st.session_state.data.isnull().sum().values, "nulls": st.session_state.data.isnull().sum().values, "type": st.session_state.data.dtypes.values, "unique": [len(st.session_state.data[col].unique()) for col in st.session_state.data.columns] }))

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
    minutes = st.sidebar.slider('Minutes', 0.5, 30.0, 0.5)
    if st.sidebar.button('Gridsearch'):
        st.session_state.trained_model, st.session_state.test_model_df = grid_search(test_model,st.session_state.models, st.session_state.data, minutes, st.session_state.approach, scaler)
    if "test_model_df" in st.session_state:
        model_name = st.text_input('model name')
        if st.button('Save model'):
            save_model(model_name, st.session_state.trained_model, st.session_state.test_model_df, st.session_state.selected_features)
    if st.checkbox('My models', value = True) and "my_models" in st.session_state:
        st.dataframe(st.session_state.my_models)
    if st.checkbox('Show recommended models', value=True):    
        st.dataframe(st.session_state.models)

if "project" in st.session_state:
    st.sidebar.write(f'Working on:  {st.session_state.project}')
if "username" in st.session_state:
    st.sidebar.write(f'Logged as:  {st.session_state.username}')
