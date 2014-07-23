import logging
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import classification_report, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifierCV
from sklearn.grid_search import GridSearchCV
import cPickle

from scorers import normalized_pcc
from scorers import balanced_accuracy
from balancer import BalancedClassifier

classification_report_scorer = make_scorer(classification_report)

def build_model(model, X, y, output, scorer = 'roc_auc'):
    logging.info("Running cross-validation...")
    eval_model(model, X, y, scorer)

    logging.info("Fitting final model...")
    model.fit(X, y)

    # # save the classifier
    # if output:
    #     with open(output, 'wb') as fid:
    #         cPickle.dump(model, fid)
    return model


def eval_model(model, X, Y, scorer = 'roc_auc'):
    if not isinstance(scorer, list):
        scorer = [scorer]
    for s in scorer:
        scores = cross_val_score(model, X, Y, scoring=s, cv=5)
        logging.info(scores)
        logging.info("Average cross validation score: {}".format(scores.mean()))
        if s == 'roc_auc':
            X_train, X_test, y_train, y_test = train_test_split(X, Y)
            model.fit(X_train, y_train)
            print classification_report(y_test, model.predict(X_test))

def predict_remission(X, y,
                      linear = False,
                      balance = False,
                      output = None):
    rf_model = GradientBoostingClassifier(n_estimators = 4000)
    lr_model = LogisticRegression(penalty='l1')

    model = lr_model if linear else rf_model
    if balance:
        model = GridSearchCV(cv=None,
            estimator=LogisticRegression(penalty='l2'), 
            n_jobs=4, param_grid={'C': [0.001, 0.01, 0.05, 0.08, 0.1, 0.5, 1]})
        model = BalancedClassifier(base_clf = model, n_estimators = 10)
    return build_model(model, X, y, output, scorer=['roc_auc', balanced_accuracy])


def predict_remission_length(remission_model, X, y, X_test,
                             linear = False,
                             output = None):
    rf_model = RandomForestRegressor(n_estimators = 2000)
    lr_model = LinearRegression()

    model = lr_model if linear else rf_model
    return build_model(model, X, y, output, scorer = normalized_pcc)


def predict_survival_time(X, y, X_test, linear = False,
                          output = None):
    logging.warn("Predicting survival time is not implemented")
    rf_model = RandomForestRegressor(n_estimators = 2000)
    lr_model = LinearRegression()

    model = lr_model if linear else rf_model

    return build_model(model, X, y, output, scorer = normalized_pcc)
