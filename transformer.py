from scipy.sparse import hstack
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer


def filter_dict_nan(d):
    for k,v in d.items():
        if v != np.nan:
            yield (k, v)


def filter_df_nan(df):
    df_dicts = df.T.to_dict().values()
    for d in df_dicts:
        yield filter_dict_nan(d)


class Transformer(object):
    def __init__(self, include_binned = False, scale = False):
        self._categorical_features = [
                                      #'SEX',
                                      #'PRIOR.MAL', # Whether the patient has previous cancer
                                      #'PRIOR.CHEMO', # Whether the patient had prior chemo
                                      #'PRIOR.XRT', # Prior radiation
                                      #'Infection' # Has infection
                                      #'cyto.cat', #  cytogenic category
                                      'ITD',  # Has the ITD FLT3 mutation
                                      #'D835', # Has the D835 FLT3 mutation
                                      #'Ras.Stat' # Has the Ras.Stat mutation
                                      ]

        self._numerical_features = [
                                    'Age.at.Dx', # Age at diagnosis
                                    #'WBC',  # white blood cell count
                                    #'ABS.BLST', #  Total Myeloid blast cells
                                    #'BM.BLAST', #  Myeloid blast cells measured in bone marrow samples
                                    'BM.MONOCYTES', # Monocyte cells in bone marrow
                                    'BM.PROM', # Promegakarocytes measured in bone marrow
                                    #'PB.BLAST', # Myeloid blast cells in blood
                                    'PB.MONO', # Monocytes in blood
                                    'PB.PROM',  # Promegakarocytes in blood
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

        self._untransformed_features = ['Age.at.Dx'] + self._proteomic
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
        self._scale = scale

    def create_bounded_features(self, data, col, splits, percentiles = False, train = True):
        binned_feature_name = col+'-binned'
        cut_func = pd.qcut if percentiles else pd.cut
        if train:
            data[binned_feature_name], bins = cut_func(data[col], splits, retbins=True) 
            self._bounding_bins[col] = bins
        else:
            data[binned_feature_name] = pd.cut(data[col], self._bounding_bins[col], retbins = False)


    def _bin_features(self, data, train = False):
        binned_feature_names = [x + "-binned" for x in self._binned_features]
        for feature in self._binned_features:
            self.create_bounded_features(data, 
                                         feature, 
                                         splits = [0.0, 0.2, 0.8, 1.0], 
                                         percentiles = True, 
                                         train = train)
        binned_vals = data[binned_feature_names].T.to_dict().values()
        if train:
            binnedX = self._dv.fit_transform(binned_vals).todense()
        else:
            binnedX = self._dv.transform(binned_vals).todense()
        binnedX = binnedX.astype(int)
        binnedX[~np.isnan(binnedX).any(axis=0)]
        return binnedX

    def fit(self, data):
        self.feature_names = self._untransformed_features[:]
        X = data[self._untransformed_features]
        if self._include_binned:
            binnedX = self._bin_features(data, train = True)
            self.feature_names += self._dv.get_feature_names()
            X = hstack((X, binnedX)).todense()
        if self._scale:
            self._scaler.fit(X)

    def transform(self, data):
        X = data[self._untransformed_features]
        if self._include_binned:
            binnedX = self._bin_features(data, train = False)
            X = hstack((X, binnedX)).todense()
        if self._scale:
            self._scaler.transform(X)
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
