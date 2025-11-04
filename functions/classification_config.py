from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRFClassifier

CLASSIFICATION_MODELS = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'LogisticRegression_L1': LogisticRegression(max_iter=1000, penalty='l1'),
        'LogisticRegression_L2': LogisticRegression(max_iter=1000, penalty='l2'),
        'GaussianNB': GaussianNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': XGBClassifier(),
        'RidgeClassifier': RidgeClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC(),
}

CLASSIFICATION_PARAMS_SET = {'LogisticRegression': [
    {
        'penalty': ['l1'],
        'C' : (1e-6, 1e+6, 'log-uniform'),
        'solver' : ['liblinear', 'saga'],
    },
    {
        'penalty': ['l2'],
        'C' : (1e-6, 1e+6, 'log-uniform'),
        'solver' : ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga', 'newton-cholesky'],
    }],
    'LogisticRegression_L1': {
        'C' : (1e-6, 1e+6, 'log-uniform'),
        'solver' : ['liblinear', 'saga'],
    },
    'LogisticRegression_L2': {
        'C' : (1e-6, 1e+6, 'log-uniform'),
        'solver' : ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga', 'newton-cholesky'],
    },
    'DecisionTreeClassifier': {
        'max_depth' : (1, 40),
        'criterion' : ['gini', 'entropy', 'log_loss'],
    },
    'RandomForestClassifier': {
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'max_depth' : (10, 40),
        'n_estimators' : (10, 200),
    },
    'GaussianNB': {
        'var_smoothing' : (1e-7, 1e+11, 'log-uniform'),
    },
    'GradientBoostingClassifier': {
        'learning_rate' : (0.1, 0.5, 'uniform'),
        'n_estimators' : (10, 200),
        'max_depth' : (5, 40),
    },
    'RidgeClassifier': {
        'alpha' : (1e-2, 1e+2, 'log-uniform'),
        'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    },
    'KNeighborsClassifier' : {
        'n_neighbors' : (1, 10),
        'metric' : ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'nan_euclidean'],
    },
    'SVC': [{
        'C' : (1e-3, 1e+3, 'log-uniform'),
        'kernel' : ['rbf', 'sigmoid'],
    },
    {
        'C' : (1e-3, 1e+3, 'log-uniform'),
        'degree' : (1, 3),
        'kernel' : ['poly'],
    }
    ]
    }