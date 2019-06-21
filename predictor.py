import pandas as pd
import pickle

from pefile import PEFormatError
from collections import Counter
from models import SampleData
from pathlib import Path
from sklearn.preprocessing import RobustScaler

import collector


def clean_sample(sample: SampleData, trans: RobustScaler) -> pd.DataFrame:
    """
    Takes a SampleData object,
    casts it to a pd.Series,
    adds it label-wise to a laid-out pd.DataFrame,
    drops the columns we don't need,
    casts it to a list,
    transforms it,
    casts it back to a pd.DataFrame,
    and renames the columns

    :param sample: A SampleData object with the info from pefile
    :param trans: An sklearn RobustScaler to transform the data
    :return: A 1-D Pandas DataFrame with the data scaled and organized as a
             single row with correct column names
    """

    my_cols = ['sha256', 'malware', 'debug_size', 'image_version',
               'import_rva', 'export_size', 'resource_size', 'num_sections',
               'virtual_size_2']

    data = pd.DataFrame(columns=my_cols, index=['sample'])

    # Put it in a Series and then add the things from the Series based on
    # labels instead of trusting that the order will be correct
    temp_series = pd.Series(sample.serialize())

    for label in temp_series.axes[0]:  # Add it label-wise
        data[label] = temp_series[label]

    data.drop(labels=['sha256', 'malware'], axis='columns', inplace=True)

    labels_dict = dict(enumerate(my_cols[2:]))  # To rename the columns later

    data = [list(data.loc['sample'])]  # Equivalent to NumPy's .reshape(1,-1)

    data = trans.transform(data)

    data = pd.DataFrame(data)
    data.rename(labels_dict, axis='columns', inplace=True)

    return data


def get_name(model: object) -> str:
    """
    Takes a model and builds a useful title.

    :param model: An sklearn model
    :return: The title of the model as a string
    """

    return getattr(model, 'base_estimator', model).__class__.__name__


def predict_samples(transformer: RobustScaler, files: list, models: list):
    """
    Takes the files, gets the data from them, transforms the data,
    and then prints the models' prediction.

    :param transformer: A path to a transformer to scale the data
    :param files: A list of paths to files to check
    :param models: A list of paths of pickled models to use on the data
    """
    predict_tuples = []  # Keep track of how much the models agree
    skipped_files = []
    for file_ in files:
        try:
            # Don't care about sha256 or malware columns anymore
            sample = collector.create_entry(file_, "whatever", True)
            data = clean_sample(sample, transformer)
            pos_predicts = 0
            neg_predicts = 0
            print(f"\n{'-' * 50}")
            print(f"Predictions for {file_}:\n")
        except PEFormatError:
            print(f"Couldn't get PE file data, skipping {file_}")
            skipped_files.append(str(file_))
            continue

        # Make a matrix for printing the predictions
        width = 22
        matrix = []
        matrix.append(['Model', 'Predicted Malware Probability'])
        for model in models:
            matrix.append([''])
            classifier = pickle.load(open(model, 'rb'))  # Load the saved model
            name = get_name(classifier)

            if hasattr(classifier, 'predict_proba'):
                prediction = round(classifier.predict_proba(data)[0][1] * 100, 2)
                matrix.append([f'{name}:', f'{prediction}%'])

                if prediction >= 50.0:
                    pos_predicts += 1
                elif prediction < 50.0:
                    neg_predicts += 1

            # If the model doesn't have predict_proba(), just get the classification
            else:
                prediction = classifier.predict(data)[0]

                if prediction == 1:
                    matrix.append([f'{name}:', 'Malware'])
                    pos_predicts += 1
                elif prediction == 0:
                    matrix.append([f'{name}:', 'Clean'])
                    neg_predicts += 1
                else:
                    matrix.append([f'{name}:', 'No prediction available'])

        for row in matrix:  # Print the matrix of predictions
            print(''.join(f"{col:<{width}}" for col in row))
        # Add the models' agreement to the list to keep track
        predict_tuples.append((pos_predicts, neg_predicts))

    counts = Counter(predict_tuples)
    print(f"\n{'-'*50}")
    print('(# predicting malware, # predicting clean): # of times that pairing showed up')
    print(counts)

    print("\nSkipped these files because couldn't get PE file data:")
    print(skipped_files)


def main(trans: Path, files: list, models: list):
    """
    Takes a list of files and checks them using the provided models.
    Prints the results.

    :param trans: A path to a saved transformer to use on the file
    :param files: A list of paths to PE files to check
    :param models: A list of paths to saved models to use on the file
    """
    transformer = pickle.load(open(trans, 'rb'))

    predict_samples(transformer, files, models)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('trans', type=Path, help="Path to the saved transformer to use on the file data")
    p.add_argument('-f', nargs='+', type=Path, help="Paths to the files to check with the models")
    p.add_argument('-m', nargs='+', type=Path, help="Paths to the saved models to use on the data")

    args = p.parse_args()
    main(args.trans, args.f, args.m)
