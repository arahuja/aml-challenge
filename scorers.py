from sklearn.metrics import make_scorer
from scipy.stats.stats import pearsonr

def get_top_features(feature_names, model):
    if hasattr(model, 'coef_'):
        features = sorted(zip(model.coef_[0, :], feature_names), reverse=True)
        for x in set(features[:30] + features[-30:]):
            print x

def normalized_pcc(x, y):
    c, p = pearsonr(x, y)
    return (c + 1.0) / 2

normalized_pcc_metric = make_scorer(normalized_pcc)
    