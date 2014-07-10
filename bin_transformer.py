import pandas as pd
from sklearn.feature_extraction import DictVectorizer


class BinTransformer(object):
    """
    bins: int (number of bins) or percentlile
    """
    def __init__(self, bins, percentiles):
        self._dv = DictVectorizer()
        self._bins = bins
        self._bin_boundaries = {}
        self._percentiles = percentiles
        self._feature_names = []

    def fit(self, data):
        binned_data = data.copy()
        for col in data.columns:
            cut_func = pd.qcut if self._percentiles else pd.cut
            binned_data[col], self._bin_boundaries[col] = cut_func(data[col], self._bins, retbins=True)
        self._dv.fit(binned_data.T.to_dict().values())

    def transform(self, data):
        binned_data = data.copy()
        for col in data.columns:
            binned_data[col] = pd.cut(data[col], self._bin_boundaries[col])
        binnedX = self._dv.transform(binned_data.T.to_dict().values())
        self._feature_names += self._dv.get_feature_names()
        return binnedX

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def get_feature_names(self):
        return self._feature_names()
