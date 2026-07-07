import pandas as pd
import numpy as np


def handle_missing_values(df):
    df['AgeFilled'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Cabin'] = df['Cabin'].fillna('U')
    return df

def encode_categorical_variables(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype('category')
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype('category')

    # Since 'T' is a rare category and is for first class passengers, we can group it 
    # with 'A' (also first class) for simplicity.

    # 'U' is already handled in missing values.
    df['CabinCat'] = df['CabinCat'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 0, 'U': 7}).astype('category')
    df['Pclass'] = df['Pclass'].astype('category')
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].astype('category')
    return df

def feature_engineering (df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['IsChild'] = (df['Age'] < 1).astype(int)
    df["TicketGroup"] = df.groupby("Ticket")["Ticket"].transform("count")

    df['CabinCat'] = df['Cabin'].str[0]

    # Extract titles from names
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).astype('category')
    return df

def preprocess_data(df):
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = encode_categorical_variables(df)
    return df