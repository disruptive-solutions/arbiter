import argparse
import json
import pickle
import sys

from pathlib import Path

from arbiter import lib


DEFAULT_MODEL_PATH = './arbiter_models.pickle'
DEFAULT_RESULTS_PATH = './arbiter_results.json'


def train_app(args: argparse.Namespace):
    from arbiter import train

    malware_data = lib.build_corpus(args.malware)
    goodware_data = lib.build_corpus(args.goodware)
    output_path = args.output

    models = train.train_models(malware_data, goodware_data)
    pickle.dump(models, open(output_path, 'wb'))


def predict_app(args: argparse.Namespace):
    from arbiter import predict

    models = pickle.load(open(args.models, 'rb'))
    sample_data = lib.build_corpus(args.sample)
    output_path = args.output

    predictions = predict.get_predictions(sample_data, models)
    json.dump(predictions, open(output_path, 'w'))


def cli_app():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers()

    train_parser = sp.add_parser('train', help='Train a set of models using malware and goodware samples. '
                                               'arbiter train -h for additional info')
    train_parser.set_defaults(func=train_app)
    train_parser.add_argument('-m', '--malware', type=Path, nargs='+',
                              help='Paths to known malware samples used for training')
    train_parser.add_argument('-g', '--goodware', type=Path, nargs='+',
                              help='Paths to known goodware samples used for training')
    train_parser.add_argument('--output', type=Path, default=Path(DEFAULT_MODEL_PATH),
                              help='The output path for storing the pickled model objects')

    predict_parser = sp.add_parser('predict', help='Predict whether files are malicious using trained models. '
                                                   'arbiter predict -h for additional info')
    predict_parser.set_defaults(func=predict_app)
    predict_parser.add_argument('-m', '--models', type=Path, default=Path(DEFAULT_MODEL_PATH),
                                help='The path to the stored models (arbiter_models.pickle)')
    predict_parser.add_argument('--output', type=Path, default=Path(DEFAULT_RESULTS_PATH),
                                help='The output path for storing the prediction results (JSON)')
    predict_parser.add_argument('sample', type=Path, nargs='+',
                                help='Paths to samples to be checked against models')

    args = p.parse_args()
    if not hasattr(args, 'func'):
        p.print_help()
        sys.exit(1)

    function = args.func
    function(args)


if __name__ == "__main__":
    cli_app()
