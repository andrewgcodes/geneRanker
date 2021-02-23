import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import numpy as np # linear algebra
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


import random
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Gene Expression Data Biomarker Selection Tool',
    layout='wide')

#---------------------------------#
# Model building
def build_model(df):
    #X = df.iloc[:,:-1] # Using all column except for the last column as X
    X=df[df.columns[0:-1]]
    Y = df.iloc[:,-1] # Selecting the last column as Y

    X= (X - np.min(X))/(np.max(X) - np.min(X))

    st.markdown('Info')
    st.write('features (first 5 shown) ')
    st.info(list(X.columns)[:5])
    st.write('label (should be 0 for control, 1 for experimental)')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_size)

    rf = RandomForestClassifier(n_estimators=parameter_n_estimators)
    rf.fit(X_train, y_train)

    st.subheader('2. Model Performance')
    y_pred=rf.predict(X_test)
    st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))


    feature_list = ["MIMAT0000101", "MIMAT0015072","MIMAT0004597","MIMAT0000425","MIMAT0000065","MIMAT0002835","MIMAT0000067","MIMAT0000681","MIMAT0000691","MIMAT0000762","MIMAT0002174","MIMAT0000703","MIMAT0000231"]

    feature_imp = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
    st.subheader('Features Ranked by Importance')
    st.write(feature_imp)
    #st.subheader('Top')

    #feature_imp = feature_imp[:10]
    #st.write(list(feature_imp.index))
#    selectedColumns =list(feature_imp.index)
    #X =X.loc[selectedColumns]
#    Y=Y.loc[selectedColumns]
#    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=split_size,random_state=1)

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

# Set name of the classifiers as index labels
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

    st.pyplot(plt)
#---------------------------------#
st.write("""
# Gene Expression Data Biomarker Selection Tool

""")

with st.sidebar.header('Upload Data (CSV format)'):
    uploaded_file = st.sidebar.file_uploader("Upload CSV file (features in columns, label in final column, rows are samples)", type=["csv"])


with st.sidebar.header('Adjust Parameters'):
    split_size = st.sidebar.slider('Percent of Data to use as Testing Data', 0.1, 0.9, 0.5, 0.01)
    parameter_n_estimators = st.sidebar.slider('Number of Estimators to Use', 0, 1000, 100, 100)

st.subheader('Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('Input Dataset Preview')
    st.write(df)
    build_model(df)
else:
    st.info('Waiting for CSV file...')
    if st.button('Load example data'):
        df = pd.read_csv("GSE137140trimmedData.csv")
        st.write(df)
        build_model(df)
