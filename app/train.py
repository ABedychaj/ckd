import numpy as np
from sklearn.ensemble import RandomForestClassifier  # could be different models in the future


def train_model(X_train, y_train, args):
    """
    Train the model.
    Parameters
    ----------
    X_train : pandas.DataFrame
        Train data.
    y_train : pandas.DataFrame
        Train targets.
    args : argparse.Namespace
        Arguments.

    Returns
    -------
    clf : sklearn.ensemble.RandomForestClassifier
        Trained model.
    parameters : dict
        Hyperparameters.
    """

    if args.tuning:
        from sklearn.model_selection import RandomizedSearchCV

        """ This could be extracted to some config file, and in the future we could use grid search to find the best parameters """
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=4, stop=10, num=5)]
        # Maximum number of levels in tree
        max_depth = [2, 3, 4, 5]
        # Criterion of tree creation
        criterion = ['gini', 'entropy']
        # Weight of classes
        class_weight = ["balanced", None]

        # Create the param grid
        param_grid = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'criterion': criterion,
                      'class_weight': class_weight
                      }

        # Create a based model
        rf = RandomForestClassifier()

        # Instantiate the random search model
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, cv=4, verbose=2, n_jobs=4)

        # Fit the random search to the data
        random_search.fit(X_train, y_train)
        best_ = random_search.best_estimator_

        return best_, random_search.best_params_
    else:
        clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=3, class_weight=None)
        clf.fit(X_train, y_train)
        return clf, clf.get_params()
