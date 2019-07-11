import pandas as pd

from typing import Dict

from arbiter import train


def get_predictions(sample: Dict, models_dict: Dict) -> Dict:
    """
    Takes in a dictionary of sample metadata and checks each entry with the provided models

    :param sample: A dictionary of sample metadata
    :param models_dict: A dictionary with the models and a 'scaler': scaler entry
    :return: A dictionary of dictionaries of model predictions
             e.g.
             {
             'file_1': {'model name': prediction, [...]},
             'file_2': {'model name': prediction, [...]}
             }
    """
    scaler = models_dict.pop('scaler')

    df = pd.DataFrame.from_dict(sample, orient='index')
    scaled_df = train.scale_data(df, scaler)

    prediction_df = pd.DataFrame()
    for name, model in models_dict.items():
        prediction_df[name] = pred_all_samples_one_model(model, scaled_df)

    return prediction_df.to_dict()


def pred_all_samples_one_model(model, data: pd.DataFrame) -> pd.Series:
    """
    Creates a column of a given model's predictions for each file

    :param model: A trained model to predict malware probability
    :param data: A pd.DataFrame of metadata from files to check
    :return: A pd.Series to put into a larger pd.DataFrame
    """
    if hasattr(model, 'predict_proba'):
        prob_preds = model.predict_proba(data)
        preds_df = pd.DataFrame(prob_preds)
        return preds_df[1]  # Only want the column predicting class 1 (malware)

    return model.predict(data)
