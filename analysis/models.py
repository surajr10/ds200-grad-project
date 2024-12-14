import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, log_loss, confusion_matrix, mean_squared_error, accuracy_score
from scipy.stats import randint
from statsmodels.miscmodels.ordinal_model import OrderedModel
import re
import xgboost as xgb
from collections import defaultdict
import scipy.stats as stats

class BaseModel:
    def __init__(self, data, test_data=None , regularization=None):
        self.data = data.copy()
        self.test_data = test_data.copy() if test_data is not None else None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regularization = regularization
    
    def preprocess(self, features):
        all_features = features + ['target']
        self.data = self.data[all_features]
        # self.data = self.data.dropna()

        # Split data into training and testing sets
        if self.test_data is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.data.drop('target', axis=1), self.data['target'], test_size=0.3, random_state=42
            )
        else:
            self.test_data = self.test_data[features]
            self.X_train, self.y_train = self.data.drop('target', axis=1), self.data['target']
            self.X_test = self.test_data

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
    def __init__(self, data, test_data=None, hardness=False):
        super().__init__(data, test_data=test_data, regularization=None)
        self.hardness = hardness
    
    def preprocess(self, features):
        if self.hardness:
            self.data['target'] = self.data['combined_hardness_score'].round().astype(int) - 1
        else:
            self.data['target'] = self.data['winner'].map({'model_a': 0, 'model_b': 1, 'tie': 2, 'tie (bothbad)': 3})
        super().preprocess(features)


    def train_model(self):
        # sample_weights = compute_sample_weight(
        #     class_weight='balanced',
        #     y=self.y_train #provide your own target name
        # )
        self.model = xgb.XGBClassifier(reg_alpha=0.1)
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, on_test=False):
        against = self.X_test if on_test else self.X_train
        y_pred = self.model.predict(against)
        return y_pred
    
    def evaluate(self, y_pred, on_test=False):
        against = self.y_test if on_test else self.y_train
        acc = accuracy_score(against, y_pred)
        print("The accuracy is: ", acc)
        if self.hardness:
            mse = mean_squared_error(against, y_pred)
            print("The MSE is: ", mse)
        if not self.hardness:
            print("Classification report: ", classification_report(against, y_pred))
        return acc
    
    def predict_to_outcome(self, y_pred):
        return pd.Series(y_pred).map({0: 'model_a', 1: 'model_b', 2: 'tie', 3: 'tie (bothbad)'})


class MultiLinearRegressionModel(BaseModel):
    def __init__(self, data, test_data=None, regularization=None):
        super().__init__(data, test_data, regularization)
        self.alpha = None
    
    def preprocess(self, features):
        # self.data = self.data.dropna() #TODO: take this out later
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

    def predict(self, on_test=False):
        against = self.X_test if on_test else self.X_train
        y_pred = np.array(self.model.predict(against)).round()
        y_pred = np.clip(y_pred, 1, 9)
        return y_pred
    
    def tune_hyperparameters(self):
        if self.regularization == None:
            self.model.fit(self.X_train, self.y_train)
            return
        params = {'alpha': [0.02, 0.98]}
        grid_search = GridSearchCV(self.model, params, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters: ", grid_search.best_params_)
        # TODO: Do i need to set the model to the best estimator?
        self.model = grid_search.best_estimator_
        self.alpha = grid_search.best_params_['alpha']

    def evaluate(self, y_pred, on_test=False):
        against = self.y_test if on_test else self.y_train
        mse = mean_squared_error(against, y_pred)
        print("The MSE is: ", mse)
        return mse
    
class CLogLog(stats.rv_continuous):
    def _ppf(self, q):
        return np.log(-np.log(1 - q))

    def _cdf(self, x):
        return 1 - np.exp(-np.exp(x))

    
# Ordinal regression model is in the works currently    
class OrdinalRegressionModel(BaseModel):
    def __init__(self, data, test_data=None, regularization=None):
        super().__init__(data, test_data, regularization)

    def preprocess(self, features):
        self.data['target'] = self.data['combined_hardness_score'].round().astype(int)
        return super().preprocess(features)

    def train_model(self):
        cloglog = CLogLog()
        model = OrderedModel(endog=self.y_train, exog=self.X_train, distr='probit')
        self.model = model.fit(method='bfgs')
    
    def predict(self, on_test=False):
        against = self.X_test if on_test else self.X_train
        y_pred = self.model.predict(exog=against)
        return y_pred.argmax(1)

    def evaluate(self, y_pred, on_test=False):
        against = self.y_test if on_test else self.y_train
        mse = mean_squared_error(against, y_pred)
        print("The MSE is: ", mse)
        return mse
    
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


# Following functions have credit to this notebook: https://colab.research.google.com/drive/1J2Wf7sxc9SVmGnSX_lImhT246pxNVZip?usp=sharing#scrollTo=hytEb0aXfcwm
def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in range(num_round):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]

def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T