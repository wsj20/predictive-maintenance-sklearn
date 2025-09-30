import pandas as pd
import numpy as np
#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Training
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#NB: Data has no missing values as checked in DW
def load_data():    
    df = pd.read_csv('../data/ai4i2020.csv')
    print(f"Successfully loaded data into DF \n")
    return df

def data_overview(df):
    print(df.head())
    print(df.info())
    print(df.describe())

def data_preprocess(df):
    #Torque, rotation speed, tool wear
    features = [
        'Torque [Nm]',
        'Rotational speed [rpm]',
        'Tool wear [min]',
        'Type'
    ]

    X = df[features]
    y = df['Machine failure']

    X_encoded = pd.get_dummies(X, drop_first=True)
    print(X_encoded.head())

    #Stratify to ensure same proportion of failures in training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=1984, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocessing Completed \n")
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_models(X_train, y_train):
    #Logistic regression as baseline
    #Balanced to give more weight to failures as disproportionate
    logistic_reg = LogisticRegression(random_state=1984, class_weight='balanced')
    logistic_reg.fit(X_train, y_train)

    #Random Forrest Classifier:
    random_forest = RandomForestClassifier(random_state=1984, n_estimators=100)
    random_forest.fit(X_train, y_train)

    #Store in dict
    models = {
        "Logistic Regression": logistic_reg,
        "Random Forest": random_forest
    }

    print(f"All models trained \n")
    return models


if __name__ == "__main__":
    df = load_data()

    X_train, X_test, y_train, y_test = data_preprocess(df)

    print(f"\nShape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    trained_models = train_models(X_train, y_train)

    
    