import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from PIL import Image
import pandas_profiling
import streamlit_pandas_profiling
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import random
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Biomarker Genie',
    layout='wide')

def build_model(df):

    X=df[df.columns[0:-1]]
    Y = df.iloc[:,-1]
    if(agree):
        X= (X - np.min(X))/(np.max(X) - np.min(X))
    if perfil:
        pr = ProfileReport(df, explorative=True,minimal=True,progress_bar=True)
        st.header('**Pandas Profiling Report**')
        st.write('This can take a while, please be patient.')
        st_profile_report(pr)
    if selecty:
        featurelist = st.multiselect('Features to use (pick at least one to fix the ValueError)',X.columns)
        X=df[featurelist]

    st.write('Please ensure that the features and label columns look correct')
    st.write('features (first 5 shown) ')
    st.info(list(X.columns)[:5])
    st.write('label (should be 0 for control, 1 for experimental)')
    st.info(Y.name)

    if(selecty == False):
        with st.echo():
            st.subheader('Correlation Heatmap')
            corr = df.corr()
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            with sns.axes_style("white"):
                f, ax = plt.subplots(figsize=(7, 5))
                ax = sns.heatmap(corr, mask=mask, vmax=1, square=True,cmap="Blues")
            st.pyplot(f)

    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_size)
    plt.clf()
    rf = RandomForestClassifier(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        n_jobs=parameter_n_jobs)

    rf.fit(X_train, y_train)

    st.subheader('Random Forest Metrics Result')
    y_pred=rf.predict(X_test)
    st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    st.write("Precision:",metrics.precision_score(y_test, y_pred))
    st.write("Recall:",metrics.recall_score(y_test, y_pred))
    st.write("F:",metrics.f1_score(y_test, y_pred))
    st.write("Confusion Matrix")
    st.write(metrics.confusion_matrix(y_test, y_pred))
    st.write("ROC AUC based on y_pred and y_test",metrics.roc_auc_score(y_test, y_pred))


    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(rf, X_test, y_test,cmap="Blues")
    st.pyplot(plt)

    feature_imp = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
    st.subheader('Features Ranked by Importance')
    st.write(feature_imp)
    plt.clf()
    sns.barplot(x=feature_imp, y=feature_imp.index)

    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature')
    plt.title("Importance of Each Feature For Classification")
    plt.legend()
    st.pyplot(plt)
    with st.echo():
        classifiers = [LogisticRegression(random_state=1234), GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=1234), RandomForestClassifier(random_state=1234)]
        result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
        for cls in classifiers:
            model = cls.fit(X_train, y_train)
            yproba = model.predict_proba(X_test)[::,1]

            fpr, tpr, _ = roc_curve(y_test,  yproba)
            auc = roc_auc_score(y_test, yproba)

            result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                            'fpr':fpr,
                                            'tpr':tpr,
                                            'auc':auc}, ignore_index=True)

    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve for Five Machine Learning Classifiers', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    st.subheader('ROC Curves')
    st.pyplot(plt)
    st.write("Thank you for using this tool.")
#---------------------------------#
st.write("""
# Biomarker Genie
## Automatic Machine Learning and Feature Selection on Omics Data

""")
#image = Image.open('logo.png')

#st.sidebar.image(image, width=300,output_format='png')

with st.sidebar.header('Upload Data (CSV only)'):
    uploaded_file = st.sidebar.file_uploader("Upload CSV file (features in columns, label in final column, rows are samples)", type=["csv"])
    st.sidebar.write("View/download [example data](https://drive.google.com/file/d/1GsPdKfSpa9wLLPRqK4pF-NbBG8cPjS4G/view?usp=sharing)")


with st.sidebar.header('Adjust Settings'):
    agree = st.sidebar.checkbox('Normalize data?')
    perfil = st.sidebar.checkbox('Create Profile Report? (avoid on large datasets with many features)')
    selecty = st.sidebar.checkbox('Select features manually?')

    split_size = st.sidebar.slider('Percent of Data to use as Testing Data', 0.05, 0.95, 0.5, 0.01)
    parameter_n_estimators = st.sidebar.slider('Number of estimators', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features', options=['auto', 'sqrt', 'log2'])
    parameter_random_state = st.sidebar.slider('Seed number', 0, 1000, 500, 1)
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel', options=[1, -1])

st.subheader('Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('Dataset Preview')
    st.write(df)
    st.balloons()
    build_model(df)
else:
    st.info('Waiting for CSV file...')
    if st.button('Load example data'):
        df = pd.read_csv("GSE137140trimmedData.csv")
        st.write("Example data comes from GSE137140, a microRNA expression profiling dataset of lung cancer patients. 13 features are pre-selected to reduce computing time. Please be aware that the manual feature selection method does not work for the example dataset due to Streamlit structure. You should upload your own data if you want to use manual feature selection.")
        st.write(df)
        st.balloons()
        build_model(df)
