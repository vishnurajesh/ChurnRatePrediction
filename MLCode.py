import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import json
from flask import Flask, request
import pandas as pd

import time
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    # Get the file from the request.
    file = request.files["file"]

    # Save the file to the filesystem.
    with open("uploads/" + file.filename, "wb") as f:
        f.write(file.read())

    df_initial = pd.read_csv("uploads/" + file.filename)
    #import dataset

    df = df_initial[df_initial['Year'].isin([2021, 2022])]

    # Filter the rows for "2023"
    df_final = df_initial[df_initial['Year'] == 2023]

    print(df.head())
    print(df_final.head())



    ##### category variables to Integers
    le = LabelEncoder()
    cols = ['Gender', 'Churned', 'Department', 'Job_level', 'Years of service within company', 'Overtime Worked', 'Stock Option', 'Marital Status']
    for i in cols:
        le.fit(df[i])
        df[i] = le.transform(df[i])




    scaler = MinMaxScaler(feature_range=(0, 5))
    col = 'Salary'
    df[col] = df[col].astype(float)
    df[[col]] = scaler.fit_transform(df[[col]])
    df.head()


    df = df.drop(['Year'], axis=1)
    X = df.drop('Churned', axis=1)  # Drop the 'Target' column to get features
    y = df['Churned']



    ########################### Models ########################

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Creating and training the logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = logreg.predict(X_test)
    #print("Y_predict ", y_pred)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
     #   print("Y_test", y_test)

    # Evaluating the model (e.g., accuracy)
    accuracy = logreg.score(X_test, y_test)
    print("Accuracy:", accuracy)


    le = LabelEncoder()
    cols = ['Gender', 'Churned', 'Department', 'Job_level', 'Years of service within company', 'Overtime Worked', 'Stock Option', 'Marital Status']
    for i in cols:
        le.fit(df_final[i])
        df_final[i] = le.transform(df_final[i])


    scaler = MinMaxScaler(feature_range=(0, 5))
    col = 'Salary'
    df_final[col] = df_final[col].astype(float)
    df_final[[col]] = scaler.fit_transform(df_final[[col]])

    df_final = df_final.drop(['Year'], axis=1)
    df_processed = df_final.drop('Churned', axis=1)  # Drop the 'Target' column to get features
    predictions = logreg.predict(df_processed)
    df_final['Churned'] = predictions


    ### final 2 variables to be pased to frontend
    df_Churned = df_final[df_final['Churned'] == 1]
    recCount  = len(df_Churned)
    stringTobePassed = "  Based on our advanced system, "+ str(recCount) +" employees have been predicted to be at risk of churn, with an accuracy rate of "+ str(round(accuracy * 100,2)) + "%"


    ## These 2 variables are be passed to the frontend and both of them are in json formate
    df_initial = df_initial[df_initial['Churned'] == "Yes"]
    finaldata = df_initial.to_dict();
    finaldata["output"] = stringTobePassed;
    finaldata["td"] = df_Churned.to_dict();

    # {
    #     "output": stringTobePassed,
    #     "df": df_Churned.to_dict()
    # }
    # # Convert the dictionary to a JSON string
    # json_string = json.dumps(finaldata, indent=4)
    return finaldata;


if __name__ == "__main__":
    app.run(port=5001, debug=True)
