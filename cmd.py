import argparse, logging
from datetime import datetime

from sklearn.metrics import classification_report

from app.preprocessing import clean_data, encoding, prepare_train_test_split
from app.train import train_model


def save_model(model, path):
    """
    Save model to path

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Trained model.
    path : str
        Path to save the model.

    Returns
    -------
    None

    """
    from joblib import dump
    dump(model, path)


def load_model(path):
    """
    Load model from path

    Parameters
    ----------
    path : str
        Path to load the model.

    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Trained model.

    """
    from joblib import load
    return load(path)


def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)

    logging.info("Cleaning data")
    df_to_train = clean_data(args.path)

    logging.info("Encoding data")
    df_to_train = encoding(df_to_train)

    # Prepare train test split, because we use fixed random_state we will always get the same split (no train/test leakage)
    logging.info("Prepare train test split and targets")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_to_train)

    # Training mode
    if args.mode == "train":

        logging.info("Training model")
        model, params = train_model(X_train, y_train, args)
        logging.info("Parameters: {}".format(params))
        logging.debug(f'Train Accuracy - : {model.score(X_train, y_train):.3f}')
        logging.debug(f'Test Accuracy - : {model.score(X_test, y_test):.3f}')

        logging.info("Classification report")
        y_pred = model.predict(X_test)
        logging.info(classification_report(y_test, y_pred))

        if args.save_model:
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            full_path = args.model_path + "\ckd_model_" + dt_string + ".joblib"
            logging.info("Saving model: {}  ".format(full_path))
            save_model(model, full_path)

    # Testing mode
    elif args.mode == "test":
        logging.info("Loading model")
        model = load_model(args.model_path)

        logging.info("Classification report")
        y_pred = model.predict(X_test)
        logging.info(classification_report(y_test, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Chronic Kidney Disease ml app example.")
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    parser.add_argument(
        "--mode",
        help="mode to run the app",
        default="train",
        choices=["train", "test"]
    )
    parser.add_argument(
        "--path",
        help="path to csv data file",
        default="dataset\preprocessed_dataset_full.csv")
    parser.add_argument(
        "--tuning",
        help="hyperparameter tuning",
        action="store_true")
    parser.add_argument(
        "--save_model",
        help="save model",
        action="store_true")
    parser.add_argument(
        "--model_path",
        help="path to save model",
        default="model")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
