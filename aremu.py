import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,mean_absolute_error,confusion_matrix
from sklearn.naive_bayes import MultinomialNB,CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page setup
st.set_page_config(page_title='Development of a Multipath Computational Model for Drug Allergies Detection', layout='wide')
st.title('Development of a Multipath Computational Model for Drug Allergies Detection')
st.subheader('''By :green[Aremu Ayomide Racheal]:female-student:''')
st.caption('''__AUPG/13/0276__ ''')

# st.set_option('deprecation.showPyplotGlobalUse', False)

def build_model(df):
    df['ALLERGY_HISTORY'] = LabelEncoder().fit_transform(df['ALLERGY_HISTORY'])
    df['AGE']= pd.to_numeric(df['AGE'],errors='coerce').astype(int)
    X = pd.get_dummies(df.drop('ALLERGY_HISTORY',axis =1),drop_first=True)
    y = df['ALLERGY_HISTORY']
    st.write('Encoding the categorical variable into dummy variable')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
    st.markdown('**Data Splits** ')
    st.write('Training Set')
    st.info(X_train.shape)
    st.write('Testing Set')
    st.info(X_test.shape)

    st.markdown('**Variable Details**')
    st.write('X Variable')
    st.info(list(X_train.columns))

    st.write('Y Variable')
    st.info(y.name)
    sns.countplot(df,x='ALLERGY_HISTORY',hue='GENDER')
    st.pyplot()

    rf =RandomForestClassifier(n_estimators=parameter_estimator,random_state=101,max_features=parameter_features)
    rf.fit(X_train,y_train)

    st.subheader('Model Performance')
    st.markdown('**Training Set**')
    pred = rf.predict(X_test)

    cm = confusion_matrix(y_test,pred)
    st.write('ConfuSion Matrix')
    st.code(cm)
    fig,ax = plt.subplots(figsize=(7,3))
    # fig.set_figwidth(4)
    # fig.set_figheight(1)
    CCM=ConfusionMatrixDisplay.from_predictions(y_test,pred)

    st.pyplot(fig.show())

    st.info('Classification Report')
    st.code(classification_report(y_test,pred))

    st.info('Feature Importance Report')
    # st.write(rf.feature_importances_)

    st.write('Features above 0.03')
    feat = pd.DataFrame(index=X.columns,data=rf.feature_importances_,columns=['Feature Importance'])
    imp_feats = feat[feat['Feature Importance']>=0.03]
    st.dataframe(imp_feats)

    st.markdown('''Adaboost Algorithm''')
    adb = AdaBoostClassifier(n_estimators=parameter_Adaestimator,learning_rate=parameter_learning,algorithm='SAMME')
    adb.fit(X_train, y_train)

    st.subheader('AdaBoost Model Performance')
    st.markdown('**Training Set**')
    adbpred = adb.predict(X_test)

    adbcm = confusion_matrix(y_test, adbpred)
    st.write('Adaboost ConfuSion Matrix')
    st.code(adbcm)

    st.info('Adaboost Classification Report')
    st.code(classification_report(y_test, adbpred))

    st.write('Adaboost Features above 0.03')
    adbfeat = pd.DataFrame(index=X.columns, data=adb.feature_importances_, columns=['Feature Importance'])
    adbimp_feats = adbfeat[adbfeat['Feature Importance'] >= 0.03]
    st.dataframe(adbimp_feats)

    st.subheader('''CategoricalNB Algorithm''')
    cnb = CategoricalNB()
    cnb.fit(X_train, y_train)
    predNB = cnb.predict(X_test)

    st.info('CategoricalNB Classification Report')
    st.code(classification_report(y_test, predNB))

    st.subheader("""Combining the 3 Models using VotingClassifier""")
    ang = [('RandomForest',rf),('AdaBoost',adb),('CategoricalNB',cnb)]
    parfait = VotingClassifier(estimators=ang,voting=para_vote)
    parfait.fit(X_train, y_train)

    sumprediction = parfait.predict(X_test)
    st.code(classification_report(y_test, sumprediction))
    st.write('ConfuSion Matrix')
    st.code(confusion_matrix(y_test,sumprediction))
    st.write('Mean Absolute Error')
    st.info(mean_absolute_error(y_test,sumprediction))

    st.subheader('Saving the model')
    st.code(joblib.dump(parfait,'AremuRachael.pkl'))




st.write('''
In this Implementation RandomForestClassifier, AdaBoostClassifier and Multinomial Naive Bayes algorithms are combined in this app to build
a multipath drug allergy detection model 
''')

with st.sidebar.header('Upload your Excel data'):
    uploaded_file = st.sidebar.file_uploader('Upload your input Excel file', type=['xlsx'])

with st.sidebar.header('RandomForest Parameters'):
    parameter_estimator = st.sidebar.slider('Number of estimators (n_estimators)',10,900,50,50)
    parameter_features = st.sidebar.select_slider('Max Features (max_features)',options=[None,'sqrt','log2'])


with st.sidebar.header('AdaBoost Parameters'):
    parameter_Adaestimator = st.sidebar.slider('Number of estimators (n_estimators)',50,900,50,50)
    parameter_learning = st.sidebar.slider('Learning_Rate (learning_rate)',1.0,30.0,5.0,5.0)

with st.sidebar.header('VotingClassifier Parameters'):
    para_vote = st.sidebar.select_slider('Voting (voting)',options=['hard','soft'])



st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.markdown('** 1.1. Glimpse of the Dataset** ')
    st.write(df.head())
    build_model(df)

else:
    st.info('Awaiting for Excel file to be uploaded')
    st.button('Press to use Example Dataset')
