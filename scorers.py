from sklearn.metrics import make_scorer
from scipy.stats.stats import pearsonr


def normalized_pcc(x, y):
    c, p = pearsonr(x, y)
    return (c + 1.0) / 2

normalized_pcc_metric = make_scorer(normalized_pcc)
    