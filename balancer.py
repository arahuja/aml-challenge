from sklearn.base import ClassifierMixin
from sklearn import clone
import numpy as np

def subsample_indices(n, k, bootstrap = True):
    if bootstrap:
        return np.floor(np.random.rand(k) * n-1).astype(int)
    else:
        idx = np.arange(n)
        return np.random.shuffle(idx)[k]

def sample_balanced(X, Y, bootstrap = True):
    n_total = len(Y)

    true_mask = Y > 0
    X_true = X[true_mask]
    Y_true = Y[true_mask]
    n_true = len(Y_true)

    false_mask = ~true_mask
    X_false = X[false_mask]
    Y_false = Y[false_mask]
    n_false = len(Y_false)

    n_min = min(n_true, n_false)
    n_max = max(n_true, n_false)

    true_idx = subsample_indices(n_true, n_max, bootstrap)
    X_true_sub = X_true[true_idx]
    Y_true_sub = Y_true[true_idx]

    false_idx = subsample_indices(n_false, n_max, bootstrap)
    X_false_sub = X_false[false_idx]
    Y_false_sub = Y_false[false_idx]

    X_sub = np.vstack([X_true_sub, X_false_sub])
    Y_sub = np.concatenate([Y_true_sub, Y_false_sub])

    return X_sub, Y_sub

class BalancedClassifier(ClassifierMixin):
    """BalancedClassifier


    Parameters
    ----------
    base_model: scikits-learn classifier
        Some base machine learning model
    n_estimators : int
        Number of base models to create

    """

    def __init__(self, **kwargs):
        self.set_params(**kwargs)
        self._fit_models = []
        

    def get_params(self, deep=True):
        return {'n_estimators' : self.n_estimators, 'base_clf': self._base_clf}

    def set_params(self, **parameters):
        n_estimators = parameters.pop('n_estimators', 100)
        self.n_estimators = n_estimators
        self._base_clf = parameters.pop('base_clf', None)

    def fit(self, X, y):
        """Fit the model

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_models = []

        for i in xrange(self.n_estimators):
            balancedX, balancedY = sample_balanced(X, y)
            clf = clone(self._base_clf)
            clf.fit(balancedX, balancedY)
            self._fit_models.append(clf)
        return self 

    def predict_proba(self, X):
        init = True
        for clf in self._fit_models:
            if init:
                p = clf.predict_proba(X)
                init = False
            else:
                 p += clf.predict_proba(X)
        p /= len(self._fit_models)
        return p

    def predict(self, X):
        p = self.predict_proba(X)
        return p[:,1] > 0.5
