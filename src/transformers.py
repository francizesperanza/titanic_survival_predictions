from sklearn.base import BaseEstimator, TransformerMixin

class MissingValueTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.age_median_ = X["Age"].median()
        self.fare_median_ = X["Fare"].median()
        self.embarked_mode_ = X["Embarked"].mode()[0]

        self.n_features_in_ = X.shape[1] 
        return self

    def transform(self, X):
        X = X.copy()

        X["AgeFilled"] = X["Age"].fillna(self.age_median_)
        X["Fare"] = X["Fare"].fillna(self.fare_median_)
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)
        X["Cabin"] = X["Cabin"].fillna("U")

        return X
    
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] 
        return self

    def transform(self, X):
        X = X.copy()

        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["FarePerPerson"] = X["Fare"] / X["FamilySize"]
        X["IsAlone"] = (X["FamilySize"] == 1).astype("int")
        X["IsChild"] = (X["AgeFilled"] < 18).astype("int")

        X["CabinCat"] = X["Cabin"].str[0]

        X["Title"] = X["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
        X["Title"] = X["Title"].replace(
            ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],
            "Rare"
        )
        X["Title"] = X["Title"].replace(["Mlle","Ms"], "Miss")
        X["Title"] = X["Title"].replace("Mme", "Mrs")
        X['Title'] = X['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).astype("int")
        return X
    
class EncoderTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] 
        return self

    def transform(self, X):
        X = X.copy()
        X["Pclass"] = X["Pclass"].astype('int')
        X["Sex"] = X["Sex"].map({"male":0, "female":1}).astype('int')

        X["Embarked"] = X["Embarked"].map({
            "C":0,
            "Q":1,
            "S":2
        }).astype('int')

        X["CabinCat"] = X["CabinCat"].map({
            "A":0,
            "B":1,
            "C":2,
            "D":3,
            "E":4,
            "F":5,
            "G":6,
            "T":0,
            "U":7
        }).astype('int')
        
        if 'Survived' in X.columns:
            X['Survived'] = X['Survived'].astype('int')
        return X
    
class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.features = ['Pclass', 'Sex', 'AgeFilled', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone', 'FamilySize', 'CabinCat', 'FarePerPerson', 'IsChild']

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] 
        return self

    def transform(self, X):
        return X[self.features].copy()