from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


REGRESSION_MODELS = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'XGBRegressor': XGBRegressor(),
}

# REGRESSION_PARAMS_SET = [
#     {
#         'alpha': (1e-2, 1e+2, 'log-uniform'),
#         'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
#     },
#     {
#         'alpha': (1e-2, 1e+2, 'log-uniform'),
#         'selection': ['cyclic', 'random'],
#     },
#     {
#         'alpha': (1e-2, 1e+2, 'log-uniform'),
#         'selection': ['cyclic', 'random'],
#         'l1_ratio': (0,1, 'uniform'),
#     },
#     {
#         'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#         'max_depth': (3, 20, 'uniform'),
#     },
#     [{
#         'kernel': ['poly'],
#         'C': (1e-3, 1e+3,'log-uniform'),
#         'epsilon': (0.01, 0.3,'log-uniform'),
#         'gamma': ['scale', 'auto', 0.1, 1],
#         'degree': (2, 3),
#     },
#     {
#         'C': (1e-3, 1e+3,'log-uniform'),
#         'epsilon': (0.01, 0.3,'log-uniform'),
#         'gamma': ['scale', 'auto', 0.1, 1],
#         'kernel': ['linear', 'rbf', 'sigmoid'],
#     },
#     ],
#     {
#         'n_neighbors': (1, 10),
#         'weights': ['uniform', 'distance'],
#         'algorithm': ['auto'],
#         'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2','nan_euclidean'],
#     },
#     {
#         'learning_rate' : (0.1, 0.5, 'uniform'),
#         'n_estimators' : (20, 100),
#         'max_depth' : (2, 10, 'uniform'),
#     },
# ]

REGRESSION_PARAMS_SET = {
        'Ridge': {
            'alpha': (1e-2, 1e+2, 'log-uniform'),
            'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        },
        'Lasso': {
            'alpha': (1e-2, 1e+2, 'log-uniform'),
            'selection': ['cyclic', 'random'],
        },
        'ElasticNet': {
            'alpha': (1e-2, 1e+2, 'log-uniform'),
            'selection': ['cyclic', 'random'],
            'l1_ratio': (0,1, 'uniform'),
        },
        'DecisionTreeRegressor': {
            'criterion': ['squared_error', 'friedman_mse', 'poisson'],
            'max_depth': (1, 20, 'uniform'),
        },
        'SVR': [{
            'kernel': ['poly'],
            'C': (1e-3, 1e+3,'log-uniform'),
            'epsilon': (0.01, 0.3,'log-uniform'),
            'gamma': ['scale', 'auto', 0.1, 1],
            'degree': (1, 3),
        },
        {
            'C': (1e-3, 1e+3,'log-uniform'),
            'epsilon': (0.01, 0.3,'log-uniform'),
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'sigmoid'],
        },
        ],
        'KNeighborsRegressor': {
            'n_neighbors': (1, 10),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],
            'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2','nan_euclidean'],
        },
        'XGBRegressor': {
            'learning_rate' : (0.1, 0.5, 'uniform'),
            'n_estimators' : (20, 100),
            'max_depth' : (2, 10, 'uniform'),
        },
    }
