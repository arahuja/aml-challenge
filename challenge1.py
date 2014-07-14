import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from submit import predict_submission
from transformer import Transformer

import argparse
import logging


def get_top_features(feature_names, model):
    if hasattr(model, 'coef_'):
        features = sorted(zip(model.coef_[0, :], feature_names), reverse=True)
        for x in set(features[:30] + features[-30:]):
            print x

def eval_model(model, X, Y, transformer, print_coef = False):
    scores = cross_val_score(model, X, Y, scoring='roc_auc', cv=5)
    logging.info(scores)
    logging.info("Average cross validation score: {}".format(scores.mean()))
    model.fit(X, Y)
    if print_coef:
        get_top_features(transformer.feature_names, model)

if __name__ == '__main__':

    logging.basicConfig(format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', default=False, action='store_true', dest='submit')
    parser.add_argument('--preconcat', default=False, action='store_true', dest='preconcat')

    parser.add_argument('--eval', default=False, action='store_true', dest='eval')
    parser.add_argument('--linear', default=False, action='store_true', dest='linear')
    parser.add_argument('--bin', default=False, action='store_true', dest='bin')
    
    parser.add_argument('--scale', default=False, action='store_true', dest='scale')
    parser.add_argument('--print-coef', default=False, action='store_true', dest='print_coef')
    parser.add_argument('--output', default='c1_model.pkl', dest='output')
    args = parser.parse_args()

    train_data = pd.read_csv('trainingData-release.csv')
    submit_data = pd.read_csv('scoringData-release.csv')
    data = pd.concat([train_data, submit_data], ignore_index = True)
    transformer = Transformer(include_binned = args.bin, scale = args.scale)

    X_full = transformer.fit_transform(data)
    X = X_full[:len(train_data)]
    X_test = X_full[len(train_data):]
    y = train_data['resp.simple'].map(lambda x: 1 if x == 'CR' else 0)


    rf_model = GradientBoostingClassifier(n_estimators = 500)
    lr_model = LogisticRegression(penalty='l2')

    model = lr_model if args.linear else rf_model

    if args.eval:
        logging.info("Running cross-validation...")
        eval_model(model, X, y, transformer, args.print_coef)

    if args.submit:
        logging.info("Fitting final model...")
        model.fit(X, y)
        get_top_features(transformer.feature_names, model)
        logging.info("Predicting and creating submission file...")
        import cPickle
        # save the classifier
        with open(args.output, 'wb') as fid:
            cPickle.dump(model, fid) 
        predict_submission(model, 
                           submit_data['#Patient_id'], 
                           X_test,
                           id_column = '#Patient_id',
                           prediction_column = 'CR_Confidence',
                           output_file = "challenge1_submission.csv")




