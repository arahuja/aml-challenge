import pandas as pd
from scipy.sparse import hstack
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.linear_model import LogisticRegression


import argparse
import logging

from submit import create_submission

class Transformer():
    def __init__(self):
        
        self._categorical_features = [
                                      #'SEX', 
                                      #'PRIOR.MAL', # Whether the patient has previous cancer
                                      #'PRIOR.CHEMO', # Whether the patient had prior chemo
                                      #'PRIOR.XRT', # Prior radiation
                                      #'Infection' # Has infection
                                      'cyto.cat', #  cytogenic category
                                      'ITD', # Has the ITD FLT3 mutation
                                      #'D835', # Has the D835 FLT3 mutation
                                      #'Ras.Stat' # Has the Ras.Stat mutation
                                      ]

        self._numerical_features = [
                                    #'Age.at.Dx', # Age at diagnosis
                                    #'WBC',  # white blood cell count
                                    #'ABS.BLST', #  Total Myeloid blast cells
                                    #'BM.BLAST', #  Myeloid blast cells measured in bone marrow samples
                                    'BM.MONOCYTES', # Monocyte cells in bone marrow
                                    'BM.PROM', # Promegakarocytes measured in bone marrow
                                    #'PB.BLAST', # Myeloid blast cells in blood
                                    'PB.MONO', # Monocytes in blood
                                    #'PB.PROM',  # Promegakarocytes in blood
                                    #'HGB', # hemoglobin count in blood
                                    #'LDH',  # lactate dehydrogenase levels measured in blood
                                    'ALBUMIN',  # albumin levels (protein made by the liver,  body is not absorbing enough protein)
                                    #'BILIRUBIN',  # bilirubin levels (found in bile,  fluid made by the liver, can lead to jaundice)
                                    #'CREATININE', # creatinine levels (measure of kidney function, waste of creatine, should be removed by kidneys)
                                    'FIBRINOGEN', # fibrinongen levels (protein produced by the liver)

                                    'CD34', 
                                    #'CD7',
                                    'CD20',
                                    #'HLA.DR', 
                                    #'CD33', 
                                    #'CD10',
                                    'CD13',
                                    'CD19'
        ]

        self._proteomic = [l.strip() for l in open('proteomic_columns_used.txt')]
    


        self._imputer = Imputer()
        self._dv = DictVectorizer()
        self._scaler = StandardScaler()

    def fit(self, data):
        self._cat_feat_transformers = {}
        cf = self._dv.fit_transform(data[self._categorical_features].T.to_dict().values())
        
        X = data[self._numerical_features + self._proteomic]
        self._imputer.fit(X)
        X = self._imputer.transform(X)
        self._scaler.fit(X)
        #X = hstack((X, cf))



    def transform(self, data):
        X = data[self._numerical_features + self._proteomic]

        # Impute missing values
        X = self._imputer.transform(X)
        X = self._scaler.transform(X)

        cf = self._dv.transform(data[self._categorical_features].T.to_dict().values())
        X = hstack((X, cf)).toarray()

        self.feature_names = self._numerical_features + self._proteomic + self._categorical_features

        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def get_top_features(feature_names, model):
    #print model.coef_[0, :]
    features = sorted(zip(model.coef_[0, :], feature_names), reverse=True)
    for x in features[:20]:
        print x
    for x in features[-20:]:
        print x

def eval_model(model, X, Y, transformer):
    scores = cross_val_score(model, X, Y, scoring='roc_auc', cv=10)
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
    print data.head()
    transformer = Transformer()

    X = transformer.fit_transform(data)
    y = data['resp.simple'].map(lambda x: 1 if x == 'CR' else 0)
    #print y.value_counts()


    #model = RandomForestClassifier(n_estimators = 2000)
    model = LogisticRegression(penalty='l1')

    if args.submit:
        logging.info("Fitting final model...")
        model.fit(X, y)
        get_top_features(transformer.feature_names, model)
        logging.info("Predicting and creating submission file...")
        create_submission(model, 
                       test_file = 'scoringData-release.csv',
                       id_column = '#Patient_id',
                       prediction_column = 'CR_Confidence',
                       transformer = transformer)
    else:
        logging.info("Running cross-validation...")
        eval_model(model, X, y, transformer)




