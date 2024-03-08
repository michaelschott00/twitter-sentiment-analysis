"""Implements svms for combining the predictions from the transformers on the text data with the metadata features."""

import pandas as pd
from twitter import models
from sklearn.metrics import f1_score, mean_squared_error
from itertools import product
from tqdm import tqdm
from typing import Literal
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('data/splits/tweets_train_bert.csv', dtype={'author_id': 'category', 'type': 'category', 'possibly_sensitive': 'category', 'sentiment': 'category'})
df_val = pd.read_csv('data/splits/tweets_dev_bert.csv', dtype={'author_id': 'category', 'type': 'category', 'possibly_sensitive': 'category', 'sentiment': 'category'})

X_train = df_train.drop(columns=["sentiment", "score_compound"])
y_train_clf = df_train.sentiment
y_train_reg = df_train.score_compound

X_val = df_val.drop(columns=["sentiment", "score_compound"])
y_val_clf = df_val.sentiment
y_val_reg = df_val.score_compound

param_grid_clf = {
    'C': [0.01, 0.1, 1, 10, 25],
    'degree': [2, 3, 4],
    'kernel': ['rbf', 'poly', 'linear']
}

param_grid_reg = {
    'C': [1, 10, 25, 50, 75],
    'degree': [2, 3, 4],
    'kernel': ['rbf', 'poly', 'linear']
}


def fit_eval_svm(svm, params, X_train, y_train, X_val, y_val, metric: Literal["f1_score", "rmse"]):
    svm.set_params(**params)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_val)
    if metric == "rmse":
        return mean_squared_error(y_val, pred, squared=False)
    return f1_score(y_val, pred, average='macro')


def run_grid_search(svm, y_train, y_val, param_grid, metric: Literal["f1_score", "rmse"]):
    results = []
    for C, kernel in tqdm(product(param_grid['C'], param_grid['kernel'])):
        if kernel == 'poly':
            for degree in param_grid['degree']:
                score = fit_eval_svm(svm,
                                     {'svm__C': C, 'svm__kernel': kernel, 'svm__degree': degree},
                                     X_train, y_train,
                                     X_val, y_val,
                                     metric)
                results.append((C, kernel, degree, score))
        score = fit_eval_svm(svm,
                             {'svm__C': C, 'svm__kernel': kernel},
                             X_train, y_train,
                             X_val, y_val,
                             metric)
        results.append((C, kernel, 0, score))
    return results


result_svc = run_grid_search(models.svc_pipeline, y_train_clf, y_val_clf, param_grid_clf, "f1_score")
result_svr = run_grid_search(models.svr_pipeline, y_train_reg, y_val_reg, param_grid_reg, "rmse")

print(max(result_svc, key=lambda x: x[-1]))
print(min(result_svr, key=lambda x: x[-1]))
