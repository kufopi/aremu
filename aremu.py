import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,mean_absolute_error,confusion_matrix,mean_squared_error
from sklearn.naive_bayes import MultinomialNB,CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import joblib





# Page setup
st.set_page_config(page_title='Development of a Multipath Computational Model for Drug Allergies Detection', layout='wide')


left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('drug.jpg', width=350)

st.markdown("<h1 style='text-align: center; color: grey;'>Development of a Multipath Computational Model for Drug Allergies Detection</h1>",unsafe_allow_html=True)
st.subheader('''By :green[Aremu Ayomide Racheal]:female-student:''')
st.write('''**__AUPG/13/0276__** :flag-ng:''')

st.set_option('deprecation.showPyplotGlobalUse', False)


def build_model(df):
    df['ALLERGY_HISTORY'] = LabelEncoder().fit_transform(df['ALLERGY_HISTORY'])
    df['AGE']= pd.to_numeric(df['AGE'],errors='coerce').astype(int)
    X = pd.get_dummies(df.drop('ALLERGY_HISTORY',axis =1),drop_first=True)
    y = df['ALLERGY_HISTORY']
    st.info('1.2. Encoding the categorical variable into dummy variable')
    st.code('''
    X = pd.get_dummies(df.drop('ALLERGY_HISTORY',axis =1),drop_first=True)
    df['ALLERGY_HISTORY'] = LabelEncoder().fit_transform(df['ALLERGY_HISTORY'])
    y = df['ALLERGY_HISTORY']
    ''')
    st.subheader('2. Splitting the dataset into Training and Testing using ratio 7:3')
    st.image('train.png')
    st.code('''
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
    ''')
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
    st.markdown('**Data Splits** ')
    st.write('2.1 Training Set Shape')
    st.info(X_train.shape)
    st.write('2.2 Testing Set Shape')
    st.info(X_test.shape)

    st.markdown('**Variable Details**')
    st.write('2.3 X Variables after encoding')
    st.info(list(X_train.columns))

    st.write('2.4 Y Variable')
    st.info(y.name)
    st.success('Frequency count for the target variable ie Y variable')
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.countplot(df,x='ALLERGY_HISTORY',hue='GENDER')
    st.pyplot(fig.show())

    rf =RandomForestClassifier(n_estimators=parameter_estimator,random_state=101,max_features=parameter_features)
    rf.fit(X_train,y_train)
    st.divider()

    st.subheader('3. RandomForestClassifier Model Performance')
    st.code('''
    rf =RandomForestClassifier(n_estimators=parameter_estimator,random_state=101,max_features=parameter_features)
    rf.fit(X_train,y_train)
    pred = rf.predict(X_test)
    ''')
    pred = rf.predict(X_test)

    cm = confusion_matrix(y_test,pred)
    st.write('3.1 Confusion Matrix')
    st.warning('''
    A confusion matrix is a matrix that summarizes the performance of a machine learning model on a set of test data. It is a means of displaying the number of accurate and inaccurate instances based on the model’s predictions. It is often used to measure the performance of classification models, which aim to predict a categorical label for each input instance.

The matrix displays the number of instances produced by the model on the test data.

True positives (TP): occur when the model accurately predicts a positive data point.\n
True negatives (TN): occur when the model accurately predicts a negative data point. \n
False positives (FP): occur when the model predicts a positive data point incorrectly. \n
False negatives (FN): occur when the model mispredicts a negative data point.
    
    ''')
    st.code( '''confusion_matrix(y_test,pred)''')
    st.code(cm)
    sq,ax = plt.subplots(figsize=(7,3))

    CCM=ConfusionMatrixDisplay.from_predictions(y_test,pred)

    st.pyplot(sq.show())

    st.info('3.2 Classification Report')
    st.code(classification_report(y_test,pred))

    st.info('3.3 Feature Importance Report')
    # st.write(rf.feature_importances_)

    st.write('3.3.1 Features above 0.03')
    feat = pd.DataFrame(index=X.columns,data=rf.feature_importances_,columns=['Feature Importance'])
    imp_feats = feat[feat['Feature Importance']>=0.03]
    st.dataframe(imp_feats)

    st.info('3.4 Decision Tree Visualization')
    st.write('One of the trees')
    st.code('''tree.plot_tree(rf.estimators_[0])''')
    fiw, ax = plt.subplots(figsize=(10, 8))
    tree.plot_tree(rf.estimators_[0],filled=True)
    st.pyplot(fiw.show())
    st.divider()

    #st.markdown('''Adaboost Algorithm''')
    adb = AdaBoostClassifier(n_estimators=parameter_Adaestimator,learning_rate=parameter_learning,algorithm='SAMME')
    adb.fit(X_train, y_train)

    st.subheader('4. AdaBoost Model Performance')
    st.code('''
    adb = AdaBoostClassifier(n_estimators=parameter_Adaestimator,learning_rate=parameter_learning,algorithm='SAMME')
    adb.fit(X_train, y_train)
    adbpred = adb.predict(X_test)
    ''')
    adbpred = adb.predict(X_test)

    adbcm = confusion_matrix(y_test, adbpred)
    st.write('4.1 Adaboost Confusion Matrix')
    st.code(adbcm)
    v=pd.DataFrame(adbcm,index=['ALLERGY_HISTORY_Y','ALLERGY_HISTORY_Y'],columns=['Predicted_Yes','Predicted_No'])
    st.dataframe(v)

    st.info('4.2 Adaboost Classification Report')
    st.code(classification_report(y_test, adbpred))

    st.info('4.3 Adaboost Features above 0.03')
    adbfeat = pd.DataFrame(index=X.columns, data=adb.feature_importances_, columns=['Feature Importance'])
    adbimp_feats = adbfeat[adbfeat['Feature Importance'] >= 0.03]
    st.dataframe(adbimp_feats)

    st.divider()
    st.subheader('''5. CategoricalNaive Bayes Algorithm''')
    cnb = CategoricalNB()
    cnb.fit(X_train, y_train)
    predNB = cnb.predict(X_test)

    st.info('5.1 CategoricalNB Classification Report')
    st.code('''
    cnb = CategoricalNB()
    cnb.fit(X_train, y_train)
    predNB = cnb.predict(X_test)
    ''')
    st.code(classification_report(y_test, predNB))
    prednbcm = confusion_matrix(y_test, predNB)
    st.write('5.2 CategoricalNB Confusion Matrix')
    st.code(prednbcm)

    st.divider()
    url = 'https://www.linkedin.com/pulse/enhancing-predictive-accuracy-voting-classifiers-guide-manoj-s-negi'
    st.subheader(f"""6. Combining the 3 Models using [VotingClassifier]({url})""")
    st.warning('''
    A Voting Classifier is an ensemble learning method that combines the predictions of multiple base estimators (machine learning models) and predicts the class label by taking a vote. It's applicable to both classification and regression problems.''')
    ang = [('RandomForest',rf),('AdaBoost',adb),('CategoricalNB',cnb)]
    parfait = VotingClassifier(estimators=ang,voting=para_vote)
    parfait.fit(X_train, y_train)
    st.code('''
    ang = [('RandomForest',rf),('AdaBoost',adb),('CategoricalNB',cnb)]
    parfait = VotingClassifier(estimators=ang,voting=para_vote)
    parfait.fit(X_train, y_train)
    sumprediction = parfait.predict(X_test)
    ''')

    sumprediction = parfait.predict(X_test)
    st.code(classification_report(y_test, sumprediction))
    st.info('6.1 Confusion Matrix')
    st.code(confusion_matrix(y_test,sumprediction))
    st.info('6.2 Mean Absolute Error for the Combined Algorithm')
    st.info(mean_absolute_error(y_test,sumprediction))

    st.subheader('7. Mean Error Comparison Chart')
    errdata = {
        'Model': ['RandomForest', 'AdaBoost', 'CategoricalNB', 'CombinedModel'],
        'Mean Absolute Error': [mean_absolute_error(y_test, pred), mean_absolute_error(y_test, adbpred),
                                mean_absolute_error(y_test, predNB), mean_absolute_error(y_test, sumprediction)],
        'Root Mean Sq Error': [mean_squared_error(y_test, pred) ** 0.5, mean_squared_error(y_test, adbpred) ** 0.5,
                               mean_squared_error(y_test, predNB) ** 0.5,
                               mean_squared_error(y_test, sumprediction) ** 0.5]
    }
    errdf = pd.DataFrame(errdata)
    st.dataframe(errdf)
    st.info('7.1 Error Chart')
    st.line_chart(errdf, x='Model', y=['Mean Absolute Error', 'Root Mean Sq Error'])


    st.subheader('8. Saving the model')
    st.code(joblib.dump(parfait,'AremuRachael.pkl'))




st.write('''
In this Implementation [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
 [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier) 
 and [Categorical Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html) algorithms are combined in this app to build
a multipath drug allergy detection model 
''')
st.write('	:point_left: Try and adjust the hyperparameters')
st.markdown('''
Python Libraries Used:
* **[Numpy](https://numpy.org/)**,
* **[Pandas](https://pandas.pydata.org/)**,
* **[Scikit-learn](https://scikit-learn.org/)**,
* **Joblib**,
* **[Matplotlib](https://matplotlib.org/)**,
* **[Seaborn](https://seaborn.pydata.org/)**,
* **[Streamlit](https://streamlit.io/)**''')

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






if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader('1. Dataset')
    st.info('** 1.1. Glimpse of the Dataset** ')
    st.write(df.head())
    build_model(df)

else:
    st.info('Awaiting for Excel file to be uploaded')

