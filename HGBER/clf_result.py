from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
def clf(H, labels):
    kf = KFold(n_splits=10, shuffle=True)
    cla_acc, clf_auc, clf_micro_f1, clf_macro_f1 = [], [], [], []
    for i in range(20):
        for train, test in kf.split(H):
            break
        xtrain, xtest, ytrain, ytest = H[train], H[test], labels[train], labels[test]

        clf = SVC(gamma=1, probability=True)
        clf.fit(xtrain, ytrain)
        pred = clf.predict(xtest)

        pre_acc = metrics.accuracy_score(ytest, pred)
        pre_clf_micro_f1 = metrics.f1_score(ytest, pred, average='micro')
        pre_clf_macro_f1 = metrics.f1_score(ytest, pred, average='macro')

        cla_acc.append(pre_acc)
        clf_micro_f1.append(pre_clf_micro_f1)
        clf_macro_f1.append(pre_clf_macro_f1)
    cla_acc, cla_std = np.average(cla_acc), np.std(cla_acc)
    clf_micro_f1, clf_micro_f1_std = np.average(clf_micro_f1), np.std(clf_micro_f1)
    clf_macro_f1, clf_macro_f1_std = np.average(clf_macro_f1), np.std(clf_macro_f1)
    return cla_acc, cla_std, clf_micro_f1, clf_micro_f1_std, clf_macro_f1, clf_macro_f1_std
def clf_lr(H, lables, size):
    X_train, X_test, Y_train, Y_test = train_test_split(H, lables, test_size=size)
    LR = LogisticRegression()
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)

    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')

    return micro_f1, macro_f1
