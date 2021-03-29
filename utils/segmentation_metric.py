import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np


# 这个代码还有点问题，来自attentionunet的代码

def _fast_hist(label_true, label_pred, n_class):
    """

    :param label_true:
    :param label_pred:
    :param n_class:
    :return:
    """
    # print('label_true:\n{}'.format(label_true))
    # print('label_pred:\n{}'.format(label_pred))
    mask = (label_true >= 0) & (label_true < n_class)
    a = label_true[mask].astype(int)
    b = label_pred[mask].astype(int)
    c = n_class * a + b
    # print('a:\n{}'.format(a))
    # print('b:\n{}'.format(b))
    # print('c:\n{}'.format(c))
    hist = np.bincount(c, minlength=n_class**2)
    # print('hist1:\n{}'.format(hist))
    hist = hist.reshape(n_class, n_class)
    # print('hist2:\n{}'.format(hist))
    return hist

def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - 交集：np.diag取hist的对角线元素
      - 并集：hist.sum(1)和hist.sum(0)分别按两个维度相加，而对角线元素加了两次，因此减一次

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # print(hist)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {'overall_acc': acc,
            'mean_acc': acc_cls,
            'freq_w_acc': fwavacc,
            'mean_iou': mean_iu}

def dice_score_list(label_gt, label_pred, n_class):
    """

    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)

def segmentation_stats(pred_seg, target):
    n_classes = pred_seg.shape[1]
    # print('n_classes:{}'.format(n_classes))
    # pred_lbls = pred_seg.data.max(1)[1].cpu().numpy().astype(np.int16)
    pred_lbls = pred_seg.data.max(1)[0].cpu().numpy()
    gt = np.squeeze(target.data.cpu().numpy(), axis=1)
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_)
        preds.append(pred_)
    # print('preds:\n{}'.format(preds))
    # print('gts:\n{}'.format(gts))
    iou = segmentation_scores(gts, preds, n_class=n_classes)
    dice = dice_score_list(gts, preds, n_class=n_classes)
    return iou, dice

def get_segmentation_stats(prediction, target):
    seg_scores, dice_score = segmentation_stats(prediction, target)
    seg_stats = [('Overall_Acc', seg_scores['overall_acc']), ('Mean_IOU', seg_scores['mean_iou'])]
    for class_id in range(dice_score.size):
        seg_stats.append(('Class_{}'.format(class_id), dice_score[class_id]))
    return OrderedDict(seg_stats)


if __name__ == '__main__':
    target = torch.rand([2,1,4,4])
    prediction = torch.rand([2,5,4,4])

    a = get_segmentation_stats(prediction, target)
    print(a)