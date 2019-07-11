import pandas as pd

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier


__all__ = ['get_logit_model', 'get_logit_cv_model', 'get_xgboost_model',
           'get_naiveb_model', 'get_sgd_model', 'get_svm_model']


def get_logit_model(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Train and return a logistic regression model
    """
    lr = LogisticRegression(penalty='l2',
                            solver='lbfgs',
                            fit_intercept=False,
                            intercept_scaling=False,
                            class_weight='balanced')
    lr.fit(x_train, y_train)
    return lr


def get_logit_cv_model(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegressionCV:
    """
    Train and return cross-validating logistic regression model
    """
    lr = LogisticRegressionCV(penalty='l2',
                              solver='liblinear',
                              fit_intercept=False,
                              intercept_scaling=False,
                              class_weight='balanced',
                              max_iter=500,  # didn't converge w/ 100
                              scoring='recall',
                              cv=4)
    lr.fit(x_train, y_train)
    return lr


def get_xgboost_model(x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Trains and return an XGBoost classifier (boosted tree bagging)
    """
    xgb = XGBClassifier(booster='gbtree')
    xgb.fit(x_train, y_train)
    return xgb


def get_naiveb_model(x_train: pd.DataFrame, y_train: pd.Series) -> GaussianNB:
    """
    Trains and returns a naive Bayes model
    Data must all be on the same scale in order to use naive Bayes
    """
    gnb = GaussianNB(priors=None)
    gnb.fit(x_train, y_train)
    return gnb


def get_sgd_model(x_train: pd.DataFrame, y_train: pd.Series) -> SGDClassifier:
    """
    Trains and returns a stochastic gradient descent model
    """
    sgd = SGDClassifier(loss='modified_huber',  # To get probabilities
                        fit_intercept=True,  # Default: True
                        penalty='l2',
                        class_weight='balanced',
                        max_iter=5000,  # Default: 1000
                        early_stopping=False,  # Deafult: False
                        validation_fraction=0.1,  # Default: 0.1
                        n_iter_no_change=5,  # Default: 5
                        learning_rate='adaptive',  # Default: optimal
                        eta0=0.8)
    sgd.fit(x_train, y_train)
    return sgd


def get_svm_model(x_train: pd.DataFrame, y_train: pd.Series) -> LinearSVC:
    """
    Trains and returns a linear support vector machine classifier
    Data must all be on the same scale in order to use an SVM
    This model can't predict probabilities, so it's just a 0/1 classification
    """
    lsvc = LinearSVC(penalty='l1',
                     loss='squared_hinge',
                     dual=False,
                     fit_intercept=True,
                     intercept_scaling=1,
                     class_weight='balanced')
    lsvc.fit(x_train, y_train)
    return lsvc
