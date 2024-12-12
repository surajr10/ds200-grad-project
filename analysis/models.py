import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, log_loss, confusion_matrix, mean_squared_error, accuracy_score
from scipy.stats import randint
from statsmodels.miscmodels.ordinal_model import OrderedModel
import re
import xgboost as xgb

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
        all_features = features + ['target']
        self.data = self.data[all_features]
        self.data = self.data.dropna()

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

class XGBoostModel(BaseModel):
    def __init__(self, data):
        super().__init__(data, regularization=None)
    
    def preprocess(self, features):
        self.data['target'] = self.data['winner'].map({'model_a': 0, 'model_b': 1, 'tie': 2, 'tie (bothbad)': 3})
        super().preprocess(features)

    def train_model(self):
        self.model = xgb.XGBClassifier()
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, test_data=None):
        if test_data is not None:
            y_pred = self.model.predict(test_data)
            return y_pred
        y_pred = self.model.predict(self.X_test)
        return y_pred
    
    def evaluate(self, y_pred):
        acc = accuracy_score(self.y_test, y_pred)
        print("The accuracy is: ", acc)
        print("Classification report: ", classification_report(self.y_test, y_pred))
        return acc

class SVMModel(BaseModel):
    def __init__(self, data, decision_function_shape='ovr'):
        super().__init__(data, regularization=None)
        self.decision_function_shape = decision_function_shape
        self.C = None
    
    def preprocess(self, features):
        # Add column as target for classification
        self.data['target'] = self.data['winner']
        super().preprocess(features)

    def train_model(self):
        self.model = svm.SVC(decision_function_shape=self.decision_function_shape)
        self.tune_hyperparameters()

    def predict(self, test_data=None):
        if test_data is not None:
            y_pred = self.model.predict(test_data)
            return y_pred
        y_pred = self.model.predict(self.X_test)
        return y_pred
    
    def tune_hyperparameters(self):
        params = {'C': [0.1, 10, 100],  
              'gamma': [1, 0.1, 0.01], 
              'kernel': ['rbf']} 
        grid_search = GridSearchCV(self.model, params, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters: ", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        self.C = grid_search.best_params_['C']

    def evaluate(self, y_pred):
        acc = accuracy_score(self.y_test, y_pred)
        print("The accuracy is: ", acc)
        print("Classification report: ", classification_report(self.y_test, y_pred))
        return acc


class MultiLinearRegressionModel(BaseModel):
    def __init__(self, data, regularization=None):
        super().__init__(data, regularization)
        self.alpha = None
    
    def preprocess(self, features):
        self.data = self.data.dropna() #TODO: take this out later
        self.data['target'] = self.data['combined_hardness_score'].round().astype(int)
        super().preprocess(features)

    def train_model(self, tune_hyperparameters=False):
        if self.regularization == 'l1':
            self.model = Lasso()
        elif self.regularization == 'l2':
            self.model = Ridge()
        else:
            self.model = LinearRegression()
        if tune_hyperparameters:
            self.tune_hyperparameters()
        else:
            self.model.fit(self.X_train, self.y_train)

    def predict(self, test_data=None):
        if test_data is not None:
            y_pred = self.model.predict(test_data)
            return y_pred
        y_pred = np.array(self.model.predict(self.X_test)).round()
        return y_pred
    
    def tune_hyperparameters(self):
        if self.regularization == None:
            self.model.fit(self.X_train, self.y_train)
            return
        params = {'alpha': np.arange(0.02, 1, 0.02)}
        grid_search = GridSearchCV(self.model, params, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters: ", grid_search.best_params_)
        # TODO: Do i need to set the model to the best estimator?
        self.model = grid_search.best_estimator_
        self.alpha = grid_search.best_params_['alpha']

    def evaluate(self, y_pred):
        mse = mean_squared_error(self.y_test, y_pred)
        print("The MSE is: ", mse)
        return mse
    
# Ordinal regression model is in the works currently    
class OrdinalRegressionModel(BaseModel):
    def __init__(self, data):
        super().__init__(data, regularization=None)

    def preprocess(self, features):
        self.data = self.data.dropna() #TODO: take this out later
        self.data['target'] = self.data['combined_hardness_score'].round().astype(int)
        return super().preprocess(features)

    def train_model(self):
        model = OrderedModel(self.y_train, self.X_train, distr='probit')
        self.model = model.fit(method='bfgs')
    
    def predict(self, test_data=None):
        if test_data is not None:
            y_pred = self.model.model.predict(self.model.params, test_data)
        else:
            y_pred = self.model.model.predict(self.model.params, self.X_test)
        return y_pred.argmax(1)

    def evaluate(self, y_pred):
        mse = mean_squared_error(self.y_test, y_pred)
        print("The MSE is: ", mse)
        return mse
    
class RandomForestModel(BaseModel):
    def __init__(self, data):
        super().__init__(data, regularization=None)
    
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


# def rf_hyperparameter_tuning(data):
#     rf = RandomForestClassifier()
#     rand_search = RandomizedSearchCV(
#         rf,
#         {
#             'n_estimators': randint(1, 200),
#             'max_depth': randint(1, 200),
#             'min_samples_split': randint(2, 200),
#             'min_samples_leaf': randint(1, 200),
#             'max_features': ['auto', 'sqrt', 'log2'],
#             'bootstrap': [True, False]
#         },
#         n_iter=100,
#         cv=3,
#     )
#     rand_search.fit(X_train, y_train)

#     print(rand_search.best_params_)
#     print(rand_search.best_score_)
#     print(rand_search.best_estimator_)
#     y_pred = rand_search.best_estimator_.predict(X_test)

# def lr_hyperparameter_tuning(data):
#     lr = LogisticRegression()
#     rand_search = RandomizedSearchCV(
#         lr,
#         {
#             'penalty': ['l1', 'l2', 'none'],
#             'C': np.logspace(-4, 4, 20),
#             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#         },
#         n_iter=100,
#         cv=3,
#     )
#     rand_search.fit(X_train, y_train)

#     print(rand_search.best_params_)
#     print(rand_search.best_score_)
#     print(rand_search.best_estimator_)
#     y_pred = rand_search.best_estimator_.predict(X_test)



# run cross fold validation    
    

# Need to account for actual parameters in the regularization models


# for random forest need to tune n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap
    # use randomizedsearchcv 
# for logistic regression need to tune penalty, C, solver
# for linear regression need to tune alpha, l1_ratio