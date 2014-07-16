import logging
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression
import cPickle

from scorers import normalized_pcc
from scorers import balanced_accuracy


def build_model(model, X, y, X_test, output, scorer = 'roc_auc'):
    logging.info("Running cross-validation...")
    eval_model(model, X, y, scorer)

    logging.info("Fitting final model...")
    model.fit(X, y)

    # save the classifier
    if output:
        with open(output, 'wb') as fid:
            cPickle.dump(model, fid)
    return model


def eval_model(model, X, Y, scorer = 'roc_auc'):
    if not isinstance(scorer, list):
        scorer = [scorer]
    for s in scorer:
        scores = cross_val_score(model, X, Y, scoring=s, cv=5)
        logging.info(scores)
        logging.info("Average cross validation score: {}".format(scores.mean()))


def predict_remission(X, y, X_test,
                      linear = False,
                      output = None):

    rf_model = GradientBoostingClassifier(n_estimators = 2000)
    lr_model = LogisticRegression(penalty='l1')

    model = lr_model if linear else rf_model
    return build_model(model, X, y, X_test, output, scorer=['roc_auc', balanced_accuracy])


def predict_remission_length(remission_model, X, y, X_test,
                             linear = False,
                             output = None):
    rf_model = RandomForestRegressor(n_estimators = 2000)
    lr_model = LinearRegression()

    model = lr_model if linear else rf_model
    return build_model(model, X, y, X_test, output, scorer = normalized_pcc)


def predict_survival_time(X, y, X_test, linear = False,
                          output = None):
    logging.warn("Predicting survival time is not implemented")
    rf_model = RandomForestRegressor(n_estimators = 2000)
    lr_model = LinearRegression()

    model = lr_model if linear else rf_model

    return build_model(model, X, y, X_test, output, scorer = normalized_pcc)
