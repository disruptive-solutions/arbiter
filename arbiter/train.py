import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost.sklearn import XGBClassifier
from typing import Dict

from arbiter.models import *

MODEL_FACTORIES = [get_logit_model, get_logit_cv_model, get_xgboost_model,
                   get_naiveb_model, get_sgd_model, get_svm_model]

TEST_PROPORTION = 0.30


def train_models(malware: Dict, goodware: Dict) -> Dict:
    """
    This is the primary entry point for the ``train`` module.
    Given dictionaries of known malware and known goodware,
    create, train, and return various prediction models and a scaler.
    Return format: ``{'models`: [m, ...], 'scaler': scaler}``

    :param malware: a list of known malware Path objects
    :param goodware: a list of known goodware Path objects
    :return: A dictionary of trained prediction models and a scaler
    """
    mal_df = pd.DataFrame.from_dict(malware, orient='index')
    mal_df['malware'] = 1  # Add malware boolean

    good_df = pd.DataFrame.from_dict(goodware, orient='index')
    good_df['malware'] = 0  # Add malware boolean

    full_df = pd.concat([mal_df, good_df])
    mal_bool = full_df.pop('malware')
    x_train, x_test, y_train, y_test = train_test_split(full_df, mal_bool, test_size=TEST_PROPORTION)
    scaler = RobustScaler().fit(x_train)

    x_train_scaled = scale_data(x_train, scaler)
    x_test_scaled = scale_data(x_test, scaler)

    trained_models = {'scaler': scaler}
    for model_factory in MODEL_FACTORIES:
        model = calibrate_model(model_factory, x_train_scaled, y_train,
                                x_test_scaled, y_test)
        name = getattr(model, 'base_estimator', model).__class__.__name__
        trained_models[name] = model  # Put the model in the dictionary

    return trained_models


def scale_data(data: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """
    Takes a DataFrame and returns a scaled version using sklearn's RobustScaler

    :param data: A Pandas DataFrame of the data that needs to be scaled
    :param scaler: A RobustScaler to use on the data
    :return: A Pandas DataFrame of the scaled data
    """
    scaled_x = scaler.transform(data)
    scaled_x = pd.DataFrame(scaled_x)
    scaled_x.columns = data.columns  # RobustScaler strips the column names

    return scaled_x


def calibrate_model(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
    """
    Calibrate the model if we can
    """
    trained_model = model(x_train, y_train)
    # Calibrate the probability predictions, except for XGBoost's
    if hasattr(trained_model, 'predict_proba') and not isinstance(trained_model, XGBClassifier):
        return CalibratedClassifierCV(trained_model,
                                      method='isotonic',
                                      cv='prefit').fit(x_test, y_test)
    return trained_model
