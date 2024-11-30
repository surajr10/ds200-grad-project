import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, confusion_matrix, mean_squared_error, accuracy_score
from scipy.stats import randint
import re

class BaseModel:
    def __init__(self, data, regularization=None):
        self.data = data.copy()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regularization = regularization
    
    def preprocess(self, features):
        features.append("target")
        self.data = self.data[features]

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.drop('target', axis=1), self.data['target'], test_size=0.3, random_state=42
        )

        # Scale data
        scaler = StandardScaler()
        scaler.fit_transform(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

class LogisticRegressionModel(BaseModel):
    def __init__(self, data, regularization=None):
        super().__init__(data, regularization)
    
    def preprocess(self, features):
        # TODO: Need to handle ties
        # Add column as target for classification
        self.data['target'] = self.data['winner'].map({'model_a': 1, 'model_b': 0})
        super().preprocess(features)

    def train_model(self):
        if self.regularization == 'l1': # TODO: l1 does't work with lbfgs solver so errors currently
            self.model = LogisticRegression(penalty='l1')
        elif self.regularization == 'l2':
            self.model = LogisticRegression(penalty='l2')
        else:
            self.model = LogisticRegression(penalty=None)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, test_data=None):
        if test_data is not None:
            y_pred = self.model.predict(test_data)
            return y_pred
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def evaluate(self, y_pred):
        print("The log loss is: ", log_loss(self.y_test, y_pred))
        print("The confusion matrix is: ", confusion_matrix(self.y_test, y_pred))

# TODO: Need to create a multi-classification model for handling ties



class MultiLinearRegressionModel(BaseModel):
    def __init__(self, data, regularization=None):
        super().__init__(data, regularization)
    
    def preprocess(self, features):
        # Add column as target for classification
        self.create_avg_hardness()
        super().preprocess(features)

    def create_avg_hardness(self):
        def extract_digit(value):
            match = re.search(r"\d", str(value))
            return int(match.group()) if match else None
        for col in ["score_value_1", "score_value_2", "score_value_3"]:
            self.data[col] = self.data[col].apply(extract_digit)
        self.data['target'] = round(self.data[["score_value_1", "score_value_2", "score_value_3"]].mean(axis=1, skipna=True))

    def train_model(self):
        if self.regularization == 'l1':
            self.model = Lasso()
        elif self.regularization == 'l2':
            self.model = Ridge()
        else:
            self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, test_data=None):
        if test_data is not None:
            y_pred = self.model.predict(test_data)
            return y_pred
        y_pred = np.array(self.model.predict(self.X_test)).round()
        return y_pred

    def evaluate(self, y_pred):
        print("The MSE is: ", mean_squared_error(self.y_test, y_pred))

class RandomForestModel(BaseModel):
    def __init__(self, data):
        super().__init__(data, None)
    
    def preprocess(self, features):
        # Add column as target for classification
        self.data['target'] = self.data['winner'].map({'model_a': 1, 'model_b': 0})
        super().preprocess(features)

    def train_model(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, test_data=None):
        if test_data is not None:
            y_pred = self.model.predict(test_data)
            return y_pred
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def evaluate(self, y_pred):
        print("The log loss is: ", log_loss(self.y_test, y_pred))
        print("The confusion matrix is: ", confusion_matrix(self.y_test, y_pred))


### Below this line is not finished yet ###

# Create scaffolding for hyperparameter tuning
def tune_hyperparameters(model, data):
    for regularization in [None, 'l1', 'l2']:
        model.regularization = regularization
        model.preprocess()
        model.train_model()
        y_pred = model.predict()
        model.evaluate(y_pred)

def rf_hyperparameter_tuning(data):
    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(
        rf,
        {
            'n_estimators': randint(1, 200),
            'max_depth': randint(1, 200),
            'min_samples_split': randint(2, 200),
            'min_samples_leaf': randint(1, 200),
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        },
        n_iter=100,
        cv=3,
    )
    rand_search.fit(X_train, y_train)

    print(rand_search.best_params_)
    print(rand_search.best_score_)
    print(rand_search.best_estimator_)
    y_pred = rand_search.best_estimator_.predict(X_test)

def lr_hyperparameter_tuning(data):
    lr = LogisticRegression()
    rand_search = RandomizedSearchCV(
        lr,
        {
            'penalty': ['l1', 'l2', 'none'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        n_iter=100,
        cv=3,
    )
    rand_search.fit(X_train, y_train)

    print(rand_search.best_params_)
    print(rand_search.best_score_)
    print(rand_search.best_estimator_)
    y_pred = rand_search.best_estimator_.predict(X_test)

def mlr_hyperparameter_tuning(data):
    mlr = LinearRegression()
    params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(mlr, params, cv=5)
    rand_search = RandomizedSearchCV(
        mlr,
        {
            'alpha': np.logspace(-4, 4, 20),
            'l1_ratio': np.linspace(0, 1, 20)
        },
        n_iter=100,
        cv=3,
    )
    rand_search.fit(X_train, y_train)

    print(rand_search.best_params_)
    print(rand_search.best_score_)
    print(rand_search.best_estimator_)
    y_pred = rand_search.best_estimator_.predict(X_test)

# run cross fold validation
# def cross_fold_validation(model, data):
    
    

# Need to account for actual parameters in the regularization models


# for random forest need to tune n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap
    # use randomizedsearchcv 
# for logistic regression need to tune penalty, C, solver
# for linear regression need to tune alpha, l1_ratio