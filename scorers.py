from sklearn.metrics import make_scorer
from scipy.stats.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import math


def get_top_features(feature_names, model, eps = 1e-2):
    print model
    if hasattr(model, 'coef_'):
        features = zip(model.coef_[0, :], feature_names)
        features_dict = {}
        # for coef, name in features:
        #     name = name.split("-")[0]
        #     features_dict[name] = features_dict.get(name, 0) + coef
        # features = sorted([(v, k) for (k, v) in features_dict.items()], reverse=True)
        for x in features[:15] + features[-15:]:
            print x
        for v, k in features:
            if math.fabs(v) < eps:
                print k,v


def normalized_pcc_score(x, y):
    c, p = pearsonr(x, y)
    return (c + 1.0) / 2


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return float(cm[0][0]) / (cm[0][0] + cm[0][1])


def balanced_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print cm
    specificity = float(cm[0][0]) / (cm[0][0] + cm[0][1])
    recall = float(cm[1][1]) / (cm[1][0] + cm[1][1])
    return .5 * (specificity + recall)


def balanced_accuracy_auc_score(y_true, y_pred, threshold = 0.5):
    print y_pred, y_true
    bac = balanced_accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    return bac, auc_score

normalized_pcc = make_scorer(normalized_pcc_score)
balanced_accuracy = make_scorer(balanced_accuracy_score)
bac_auc = make_scorer(balanced_accuracy_auc_score)