from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, roc_curve, f1_score, \
    confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch


def metrics_score(y_test, y_pre):
    """

    :param y_test:真实的类别标签
    :param y_pre:预测的类别标签
    :return:返回所有分数
    """

    total = []
    # Calculate metrics

    #     mse =mean_squared_error(y_test, y_pre)
    #     print("MSE: %.4f" % mse)
    #     mae = mean_absolute_error(y_test, y_pre)
    #     print("MAE: %.4f" % mae)
    #     R2 = r2_score(y_test,y_pre)
    #     print("R2: %.4f" % R2)

    acc = accuracy_score(y_test, y_pre)
    # print("ACC: %.4f" % acc)

    y2test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    y2pre = label_binarize(y_pre, classes=[0, 1, 2, 3, 4])
    #     print('y2test\n', y2test)
    #     print('y2pre\n',y2pre)
    try:
        auc = roc_auc_score(y2test, y2pre)
    except ValueError:
        auc = np.nan
    # print("AUC: %.4f" % auc)

    F1 = f1_score(y_test, y_pre, labels=None, pos_label=1, average='weighted', sample_weight=None)
    # print("F1: %.4f" % F1)

    recall = recall_score(y_test, y_pre, average='micro')
    # print("Recall: %.4f" % recall)

    precision = precision_score(y_test, y_pre, average='micro')
    # print("Precision: %.4f" % precision)

    # round() 取数原则: 4舍6入5留双. 这个算法要要优于4舍5入
    total = [round(acc, 4), round(recall, 4), round(precision, 4), round(auc, 4), round(F1, 4)]
    return total


def pred_prob2pred_label(outputs):
    # 获得类别标签
    pred_label = torch.max(outputs.cpu(), dim=1)[1]

    # 获得类别概率，主要是用来绘制ROC曲线的
    # softmax 将数值映射到了0-1之间，并且和为1
    pred_prob = torch.softmax(outputs.cpu(), dim=1)

    return pred_prob, pred_label
