from sklearn.metrics import make_scorer
from scipy.stats.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def get_top_features(feature_names, model):
    if hasattr(model, 'coef_'):
        features = sorted(zip(model.coef_[0, :], feature_names), reverse=True)
        for x in set(features[:30] + features[-30:]):
            print x


def normalized_pcc_score(x, y):
    c, p = pearsonr(x, y)
    return (c + 1.0) / 2


def specificity_score(y_true, y_pred, threshold = 0.5):
    y_pred = y_pred > 0.5
    cm = confusion_matrix(y_true, y_pred)
    return float(cm[0][0]) / (cm[0][0] + cm[0][1])


def balanced_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity = float(cm[0][0]) / (cm[0][0] + cm[0][1])
    recall = float(cm[1][0]) / (cm[1][0] + cm[1][1])

    return .5 * (specificity + recall)


def balanced_accuracy_auc_score(y_true, y_pred, threshold = 0.5):
    print y_pred, y_true
    bac = balanced_accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    return bac, auc_score

normalized_pcc = make_scorer(normalized_pcc_score)
balanced_accuracy = make_scorer(balanced_accuracy_score)
bac_auc = make_scorer(balanced_accuracy_auc_score)