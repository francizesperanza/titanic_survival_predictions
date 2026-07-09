import pandas as pd
import numpy as np


def handle_missing_values(df, age_median, fare_median, embarked_mode):

    df['AgeFilled'] = df['Age'].fillna(age_median)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['Cabin'] = df['Cabin'].fillna('U')
    return df

def encode_categorical_variables(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype('int')
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype('int')

    # Since 'T' is a rare category and is for first class passengers, we can group it 
    # with 'A' (also first class) for simplicity.

    # 'U' is already handled in missing values.
    df['CabinCat'] = df['Cabin'].str[0]
    df['CabinCat'] = df['CabinCat'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 0, 'U': 7}).astype('int')
    df['Pclass'] = df['Pclass'].astype('int')
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].astype('int')
    return df

def feature_engineering (df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['IsAlone'] = (df['FamilySize'] == 1).astype("int")
    df['IsChild'] = (df['AgeFilled'] < 18).astype("int")

    # Extract titles from names
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).astype('int')
    
    return df

def preprocess_data(df, train_df):
    age_median = train_df["Age"].median()
    fare_median = train_df["Fare"].median()
    embarked_mode = train_df["Embarked"].mode()[0]

    df = handle_missing_values(df, age_median, fare_median, embarked_mode)
    df = feature_engineering(df)
    df = encode_categorical_variables(df)
    return df