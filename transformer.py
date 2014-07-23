from numpy import hstack
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
                                      'ITD',  # Has the ITD FLT3 mutation
                                      'D835', # Has the D835 FLT3 mutation
                                      'Ras.Stat' # Has the Ras.Stat mutation
                                      ]

        self._numerical_features = []

        self._proteomic = [l.strip() for l in open('proteomic_columns_min.txt')]

        self._binned_features = ['Age.at.Dx'] + self._proteomic
    
        self._cat_dv = DictVectorizer()
        self._dv = DictVectorizer()
        self._scaler = StandardScaler()
        self._bounding_bins = {}

        self._include_binned = include_binned

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
                                         splits = [0.0, 0.20, 0.8, 1.0], 
                                         percentiles = True, 
                                         train = train)
        binned_vals = data[binned_feature_names].T.to_dict().values()
        print data[binned_feature_names].head()
        if train:
            binnedX = self._dv.fit_transform(binned_vals).todense()
        else:
            binnedX = self._dv.transform(binned_vals).todense()
        binnedX = binnedX.astype(int)
        binnedX[~np.isnan(binnedX).any(axis=0)]
        return binnedX

    def fit(self, data):
        self.feature_names = self._categorical_features[:]
        X = self._cat_dv.fit_transform(data[self._categorical_features].T.to_dict().values()).todense()

        if self._include_binned:
            binnedX = np.array(self._bin_features(data, train = True))
            print self._dv.get_feature_names()
            self.feature_names += self._dv.get_feature_names()
            X = hstack((X, binnedX))

    def transform(self, data):
        X = self._cat_dv.transform(data[self._categorical_features].T.to_dict().values()).todense()
        
        if self._include_binned:
            binnedX = self._bin_features(data, train = False)
            X = hstack((X, binnedX))
        
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
