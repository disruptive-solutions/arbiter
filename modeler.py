import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier


def cleaned_data(database_path: Path) -> tuple:
    """
    Takes a path to a SQLite database of PE file metadata, scales the data,
    and returns it as a Pandas DataFrame of data and a Pandas Series of
    labels.

    :param database_path: A path to a SQLite database
    :return: tuple, (Pandas DataFrame of the data, Pandas Series of the labels)
    """
    engine = create_engine(f"sqlite:///{database_path}")
    db = pd.read_sql_query("SELECT * FROM sample", con=engine, index_col='sha256')

    y = db['malware']  # Get the column of labels
    x = db.iloc[:, 1:]  # Get the columns of data

    scaled_x = scale_data(x)

    return scaled_x, y


def get_transformer(data: pd.DataFrame) -> RobustScaler:
    """
    Break out making the transformer so that we can use it in other places too
    (i.e. pickle it for use in the predictor)

    :param data: Pandas DataFrame to use RobustScaler on
    :return: A RobustScaler that's been fit to the input data
    """
    return RobustScaler().fit(data)


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame and returns a scaled version using sklearn's RobustScaler

    :param data: A Pandas DataFrame of the data that needs to be scaled
    :return: A Pandas DataFrame of the scaled data
    """
    transformer = get_transformer(data)
    scaled_x = transformer.transform(data)
    scaled_x = pd.DataFrame(scaled_x)
    scaled_x.columns = data.columns  # RobustScaler strips the column names

    return scaled_x


def print_eval(real: pd.Series, pred: list):
    """
    Prints some evaluation metrics for a given prediction set from
    a binary classifier using some of sklearn's metrics

    :param real: The true class labels
    :param pred: The predicted class labels
    """
    metrics = precision_recall_fscore_support(real, pred)
    conf_matrix = confusion_matrix(real, pred)
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]

    accuracy = round((tp + tn) / (tp + tn + fp + fn), 5)
    precision = round(metrics[0][1], 5)
    recall = round(metrics[1][1], 5)
    false_rate = round(fn / (tn+fn), 5)
    fscore = round(metrics[2][1], 5)
    roc = round(roc_auc_score(real, pred), 5)
    num_mal = metrics[3][1]
    num_clean = metrics[3][0]

    # Set up a matrix to print out
    width = 13
    matrix = []
    matrix.append([''])
    matrix.append(['Confusion Matrix:'])
    matrix.append(['', 'Predict Clean', 'Predict Mal.'])
    matrix.append(['Actual Clean', tn, fp])
    matrix.append(['Actual Mal.', fn, tp])
    matrix.append([''])
    matrix.append(['Precision:', precision])
    matrix.append(['Recall:', recall])
    matrix.append(['FOR:', false_rate])
    matrix.append(['Accuracy:', accuracy])
    matrix.append([''])
    matrix.append(['F-Score:', fscore])
    matrix.append(['ROC Area:', roc])
    matrix.append([''])
    matrix.append(['# of Malware:', num_mal])
    matrix.append(['# of Clean:', num_clean])
    matrix.append(['-'*50])

    for row in matrix:
        print(''.join(f"{col:>{width}}" for col in row))


def plot_probs(probs: pd.Series, title: str):
    """
    Takes the array-like output from model.predict_proba() and plots a histogram
    of how confidently the model predicted malware status

    model.predict_proba() gives the predicted probability for 0 and for 1,
    but we only care about 1 for the histogram. We have to drop the 0 column
    before passing it to matplotplib.pyplot

    :param probs: array-like output from model.predict_proba
    :param title: Plot's title
    """
    probs = pd.DataFrame(probs)
    probs.drop(0, axis=1, inplace=True)

    plt.hist(probs.iloc[:, 0], bins=10, density=True)
    plt.title(title)
    plt.xlabel("Predicted Probability That a File is Malware")
    plt.ylabel("Density")
    plt.show()


def eval_models(table_of_models: list, x_test: pd.DataFrame, y_test: pd.Series, show_plots: bool):
    """
    Takes a table of trained models and the test data, prints out the
    evaluation of every model in the table, and shows plots of the models'
    predicted probabilities.

    :param table_of_models: A list of tuples with information about the models
    :param x_test: A Pandas DataFrame of the test data
    :param y_test: A Pandas Series of the test labels
    :param show_plots: bool, whether to show plots of the models' probability
                       predictions
    """

    for model, name in table_of_models:
        predicts = model.predict(x_test)
        print(f"\n{name}")
        print_eval(y_test, predicts)

        if not show_plots:  # Skip the rest if we're not showing plots
            continue

        if getattr(model, 'predict_proba', False):
            proba_predicts = model.predict_proba(x_test)
            plot_probs(proba_predicts, name)

        if 'xgboost' in name.lower():
            plot_importance(model,
                            importance_type='weight',
                            title="XGBoost Feature Importance",
                            xlabel='Number of times a feature appears in a tree')
            plt.show()


def trained_logit_model(x_train: pd.DataFrame, y_train: pd.Series) -> object:
    """
    Trains a logistic regression model and returns the trained model.

    :param x_train: A Pandas DataFrame of the training data
    :param y_train: A Pandas Series of the training labels

    :return: A trained sklearn.linear_model.LogisticRegression() model
    """
    return LogisticRegression(penalty='l2',
                              solver='liblinear',
                              fit_intercept=False,
                              intercept_scaling=False,
                              class_weight='balanced').fit(x_train, y_train)


def trained_logit_cv_model(x_train: pd.DataFrame, y_train: pd.Series) -> object:
    """
    Trains a cross-validating logistic regression model and returns the
    trained model.

    :param x_train: A Pandas DataFrame of the training data
    :param y_train: A Pandas Series of the training labels

    :return: A trained sklearn.linear_model.LogisticRegressionCV() model
    """
    return LogisticRegressionCV(penalty='l2',
                                solver='liblinear',
                                fit_intercept=False,
                                intercept_scaling=False,
                                class_weight='balanced',
                                max_iter=500,  # didn't converge w/ 100
                                scoring='recall',
                                cv=4).fit(x_train, y_train)


def trained_xgboost_model(x_train: pd.DataFrame, y_train: pd.Series) -> object:
    """
    Trains an XGBoost classifier (boosted tree bagging) and returns
    the trained model.

    :param x_train: A Pandas DataFrame of the training data
    :param y_train: A Pandas Series of the training labels

    :return: A trained xgboost.sklearn.XGBClassifier() model
    """
    return XGBClassifier().fit(x_train, y_train)


def trained_naiveb_model(x_train: pd.DataFrame, y_train: pd.Series) -> object:
    """
    Trains a naive Bayes model and returns the trained model.
    Data must all be on the same scale in order to use naive Bayes.

    :param x_train: A Pandas DataFrame of the training data
    :param y_train: A Pandas Series of the training labels

    :return: A trained sklearn.naive_bayes.GaussianNB() model
    """
    return GaussianNB(priors=None).fit(x_train, y_train)


def trained_sgd_model(x_train: pd.DataFrame, y_train: pd.Series) -> object:
    """
    Trains a stochastic gradient descent model and returns the trained model.

    :param x_train: A Pandas DataFrame of the training data
    :param y_train: A Pandas Series of the training labels

    :return: A trained sklearn.linear_model.SGDClassifier() model
    """
    return SGDClassifier(loss='modified_huber',  # To get probabilities
                         fit_intercept=False,  # Default: True
                         penalty='l2',
                         class_weight='balanced',
                         max_iter=5000,  # Default: 1000
                         early_stopping=False,  # Deafult: False
                         validation_fraction=0.2,  # Default: 0.1
                         n_iter_no_change=5,  # Default: 5
                         learning_rate='adaptive',  # Default: optimal
                         eta0=0.5).fit(x_train, y_train)


def trained_svm_model(x_train: pd.DataFrame, y_train: pd.Series) -> object:
    """
    Trains a linear support vector machine classifier and returns
    the trained model.
    Data must all be on the same scale in order to use an SVM.
    This model can't predict probabilities, so it's just a 0/1 classification.

    :param x_train: A Pandas DataFrame of the training data
    :param y_train: A Pandas Series of the training labels

    :return: A trained sklearn.svm.LinearSVC() model
    """
    return LinearSVC(penalty='l1',
                     loss='squared_hinge',
                     dual=False,
                     fit_intercept=True,
                     intercept_scaling=1,
                     class_weight='balanced').fit(x_train, y_train)


def main(db_path: Path, show_plots: bool):
    """
    Creates, trains, and evaluates several binary classifiers based on
    the data in a SQLite database.

    :param db_path: Path to an SQLite database
    :param show_plots: bool, whether to show plots of the models' predicted
                       probabilities
    """
    x, y = cleaned_data(db_path)

    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    # Train the models
    logit_model = trained_logit_model(x_train, y_train)
    logit_cv_model = trained_logit_cv_model(x_train, y_train)
    xgboost_model = trained_xgboost_model(x_train, y_train)
    naiveb_model = trained_naiveb_model(x_train, y_train)
    sgd_model = trained_sgd_model(x_train, y_train)
    svm_model = trained_svm_model(x_train, y_train)

    # Table of models
    models_table = [(logit_model, "Logit Model"),
                    (logit_cv_model, "Cross-Validated Logit Model"),
                    (xgboost_model, "XGBoost Model (boosted tree bagging)"),
                    (naiveb_model, "Naive Bayes Model"),
                    (sgd_model, "Stochastic Gradient Descent Model"),
                    (svm_model, "Linear SVM model")]

    # Print the evaluation of all the models
    eval_models(models_table, x_test, y_test, show_plots)

    # (Slightly) more complicated directions to go in:
    # Can also look into k-nearest neighbor clustering, which might be good
    # for the eventual social graph and for improving the model over time


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('database', type=Path, help="The path to a SQLite database for the model to use")
    p.add_argument('--plot', action='store_true', help="Show plots of the models' probability predictions")

    args = p.parse_args()
    main(args.database, args.plot)
