import pandas as pd
from scipy.sparse import hstack
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso, Ridge, LinearRegression

import numpy as np

import argparse
import logging

from submit import create_submission

class Transformer():
    def __init__(self, include_binned = False):
        
        self._categorical_features = [
                                      #'SEX', 
                                      #'PRIOR.MAL', # Whether the patient has previous cancer
                                      #'PRIOR.CHEMO', # Whether the patient had prior chemo
                                      #'PRIOR.XRT', # Prior radiation
                                      #'Infection' # Has infection
                                      #'cyto.cat', #  cytogenic category
                                      'ITD', # Has the ITD FLT3 mutation
                                      #'D835', # Has the D835 FLT3 mutation
                                      #'Ras.Stat' # Has the Ras.Stat mutation
                                      ]

        self._numerical_features = [
                                    'Age.at.Dx', # Age at diagnosis
                                    #'WBC',  # white blood cell count
                                    #'ABS.BLST', #  Total Myeloid blast cells
                                    #'BM.BLAST', #  Myeloid blast cells measured in bone marrow samples
                                    #'BM.MONOCYTES', # Monocyte cells in bone marrow
                                    #'BM.PROM', # Promegakarocytes measured in bone marrow
                                    #'PB.BLAST', # Myeloid blast cells in blood
                                    #'PB.MONO', # Monocytes in blood
                                    #'PB.PROM',  # Promegakarocytes in blood
                                    #'HGB', # hemoglobin count in blood
                                    #'LDH',  # lactate dehydrogenase levels measured in blood
                                    #'ALBUMIN',  # albumin levels (protein made by the liver,  body is not absorbing enough protein)
                                    #'BILIRUBIN',  # bilirubin levels (found in bile,  fluid made by the liver, can lead to jaundice)
                                    #'CREATININE', # creatinine levels (measure of kidney function, waste of creatine, should be removed by kidneys)
                                    #'FIBRINOGEN', # fibrinongen levels (protein produced by the liver)

                                    'CD34', 
                                    #'CD7',
                                    #'CD20',
                                    #'HLA.DR', 
                                    #'CD33', 
                                    #'CD10',
                                    'CD13',
                                    'CD19'
        ]

        self._proteomic = [l.strip() for l in open('proteomic_columns_used.txt')]

        self._untransformed_features = ['Age.at.Dx'] #+ self._proteomic
        self._binned_features = [   'Age.at.Dx',
                                    'CD34', 
                                    #'CD7',
                                    #'CD20',
                                    #'HLA.DR', 
                                    #'CD33', 
                                    #'CD10',
                                    'CD13',
                                    'CD19',
                                    'ALBUMIN',
                                    'FIBRINOGEN',
                                    #'BM.PROM',
                                    #'PB.MONO',
                                    #'PB.PROM',
                                    #'BM.MONOCYTES'
        ] + self._proteomic
    
        self._dv = DictVectorizer()
        self._scaler = StandardScaler()
        self._bounding_bins = {}

        self._include_binned = include_binned

    def create_bounded_features(self, data, col, splits, percentiles = False, train = True):
        binned_feature_name = col+'-binned'
        cut_func = pd.qcut if percentiles else pd.cut
        if train:
            data[col+'-binned'], bins = cut_func(data[col], splits, retbins=True) 
            self._bounding_bins[col] = bins
        else:
            data[col+'-binned'] = pd.cut(data[col], self._bounding_bins[col], retbins = False) 
        data[col+'-binned'].fillna('NA', inplace = True)


    def _fit_binned_features(self, data, train = False):
        binned_feature_names = [x + "-binned" for x in self._binned_features]
        for feature in self._binned_features:
            self.create_bounded_features(data, feature, splits = [0.0, 0.5, 1.0] , percentiles = True, train = train)
        if train:
            binnedX = self._dv.fit_transform(data[binned_feature_names].T.to_dict().values())
        else:
            binnedX = self._dv.transform(data[binned_feature_names].T.to_dict().values())
        return binnedX

    def fit(self, data):
        self.feature_names = self._untransformed_features[:]
        X = data[self._untransformed_features]
        if self._include_binned:
            binnedX = self._fit_binned_features(data, train = True)
            self.feature_names += self._dv.get_feature_names()
            X = hstack((X, binnedX)).todense()
        print X.shape
        self._scaler.fit(X)

    def transform(self, data):
        X = data[self._untransformed_features]
        if self._include_binned:
            binnedX = self._fit_binned_features(data, train = False)
            X = hstack((X, binnedX)).todense()
        self._scaler.transform(X)
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def get_top_features(feature_names, model):
    features = sorted(zip(model.coef_, feature_names), reverse=True)
    for x in set(features[:30] + features[-30:]):
        print x

def eval_model(model, X, Y, transformer):
    scores = cross_val_score(model, X, Y,cv=10)
    logging.info(scores)
    logging.info("Average cross validation score: {}".format(scores.mean()))
    model.fit(X, Y)
    get_top_features(transformer.feature_names, model)

if __name__ == '__main__':

    logging.basicConfig(format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', default=False, action='store_true', dest='submit')

    args = parser.parse_args()

    data = pd.read_csv('trainingData-release.csv')
    data = data[data['resp.simple'] == 'CR']
    transformer = Transformer(include_binned = False)

    X = transformer.fit_transform(data)
    y = np.log(data.Remission_Duration)
    print X.shape


    model = RandomForestRegressor(n_estimators = 2000)
    #model = LinearRegression()

    if args.submit:
        logging.info("Fitting final model...")
        model.fit(X, y)
        get_top_features(transformer.feature_names, model)
        logging.info("Predicting and creating submission file...")
        #import cPickle
        #c1_model = cPickle.load(open('c1_model.pkl', 'wb'))
        create_submission(model, 
                       test_file = 'scoringData-release.csv',
                       id_column = '#Patient_id',
                       prediction_column = 'Remission_Duration',
                       confidence_column = 'Confidence',
                       transformer = transformer,
                       output_file = "challenge2_submission.csv")
    else:
        logging.info("Running cross-validation...")
        eval_model(model, X, y, transformer)




