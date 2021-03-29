import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

from utils.visulize import plot_lr, plot_confusion_matrix, plot_roc, plot_confusion_matrix_inPrediction




def save_list2np(list_, file_name):
    a = np.array(list_)
    np.save(file_name + '_a.npy', a)  # 保存为.npy格式


def load_np2list(np_file):
    a = np.load(np_file)
    a = a.tolist()
    return a


def save_intermediate(net, save_interme, CP,
                      k, val_gt_labels, val_pred_labels,
                      val_pred_probs, cla, test_pred_probs, test_gt_labels,
                      test_pred_labels, tr_result, va_result, te_result):
    torch.save(net.state_dict(), os.path.join(save_interme, 'CP' + str(CP) + 'K' + str(k + 1) + '.pth'))

    tr_result.to_csv(os.path.join(save_interme, 'Train' + 'K' + str(k + 1) + '.csv'))
    va_result.to_csv(os.path.join(save_interme, 'Val' + 'K' + str(k + 1) + '.csv'))
    te_result.to_csv(os.path.join(save_interme, 'Test' + 'K' + str(k + 1) + '.csv'))

    # val_pred_probs, val_pred_labels, val_gt_labels
    val_cm = confusion_matrix(val_gt_labels, val_pred_labels, labels=None, sample_weight=None)
    val_con_mat = plot_confusion_matrix(val_cm, _classes=cla)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'confusion_matrix_val_k' + str(k + 1)))
    val_roc = plot_roc(val_pred_probs, val_gt_labels)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'roc_val_k' + str(k + 1)))

    # test_pred_probs, test_pred_labels, test_gt_labels
    test_cm = confusion_matrix(test_gt_labels, test_pred_labels, labels=None, sample_weight=None)
    test_con_mat = plot_confusion_matrix(test_cm, _classes=cla)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'confusion_matrix_test_k' + str(k + 1)))
    test_roc = plot_roc(test_pred_probs, test_gt_labels)
    plt.savefig(os.path.join(save_interme, 'CP' + str(CP) + 'roc_test_k' + str(k + 1)))


def save_best(k, net, val_gt_labels, val_pred_labels,
              val_pred_probs, test_gt_labels, test_pred_labels, cla,
              test_pred_probs, save_best_dir, save_dir):
    # val_pred_probs, val_pred_labels, val_gt_labels
    val_cm = confusion_matrix(val_gt_labels, val_pred_labels, labels=None, sample_weight=None)
    plot_confusion_matrix_inPrediction(val_cm)
    plt.savefig(os.path.join(save_best_dir, 'confusion_matrix2_val_k' + str(k + 1)))
    val_con_mat = plot_confusion_matrix(val_cm, _classes=cla)
    plt.savefig(os.path.join(save_best_dir, 'confusion_matrix_val_k' + str(k + 1)))
    val_roc = plot_roc(val_pred_probs, val_gt_labels)
    plt.savefig(os.path.join(save_best_dir, 'roc_val_k' + str(k + 1)))

    # confusion_matrix
    test_cm = confusion_matrix(test_gt_labels, test_pred_labels, labels=None, sample_weight=None)
    test_con_mat = plot_confusion_matrix(test_cm, _classes=cla)
    plt.savefig(os.path.join(save_best_dir, 'confusion_matrix_test_k' + str(k + 1)))
    # roc
    test_roc = plot_roc(test_pred_probs, test_gt_labels)
    plt.savefig(os.path.join(save_best_dir, 'roc_test_k' + str(k + 1)))
    # model
    torch.save(net.state_dict(), os.path.join(save_dir, 'best_linear_model_K' + str(k + 1) + '.pth'))


def save_k_final_results(net, save_dir, save_display_dir,
                         k, lr_epoch, val_gt_labels, val_pred_labels,
                         val_pred_probs, cla, test_pred_probs, test_gt_labels,
                         test_pred_labels):
    torch.save(net.state_dict(), os.path.join(save_dir, 'final_linear_model_K' + str(k + 1) + '.pth'))

    # 保存学习率衰减曲线
    plot_lr(lr_epoch)
    plt.savefig(os.path.join(save_display_dir, 'lr_' + 'K' + str(k + 1) + '.png'))

    # val_pred_probs, val_pred_labels, val_gt_labels
    val_cm = confusion_matrix(val_gt_labels, val_pred_labels, labels=None, sample_weight=None)
    plot_confusion_matrix_inPrediction(val_cm)
    plt.savefig(os.path.join(save_display_dir, 'confusion_matrix2_val_k' + str(k + 1)))
    plot_confusion_matrix(val_cm, _classes=cla)
    plt.savefig(os.path.join(save_display_dir, 'confusion_matrix_val_k' + str(k + 1)))

    plot_roc(val_pred_probs, val_gt_labels)
    plt.savefig(os.path.join(save_display_dir, 'roc_val_k' + str(k + 1)))

    # test_pred_probs, test_pred_labels, test_gt_labels
    test_cm = confusion_matrix(test_gt_labels, test_pred_labels, labels=None, sample_weight=None)
    test_con_mat = plot_confusion_matrix(test_cm, _classes=cla)
    plt.savefig(os.path.join(save_display_dir, 'confusion_matrix_test_k' + str(k + 1)))
    test_roc = plot_roc(test_pred_probs, test_gt_labels)
    plt.savefig(os.path.join(save_display_dir, 'roc_test_k' + str(k + 1)))



def pre2mask(pre):
    """
    通过预测的特征图获得掩码图，掩码图里面的数字是具体的类【0，1，2，3，...】
    :parameter pre: torch(网络里面没有sigmoid/softmax)，size = [batch_size, num_instance, h, w]
    :return: tensor, size:[h, w]， valuse:[0, 1, ..., num_instance-1]
    """

    batch_size = pre.shape[0]

    tensor_list = []
    for b in range(batch_size):
        pre_b = pre[b]
        # 经过softmax将预测的概率值映射到0-1
        pre_b = torch.softmax(pre_b, dim=0)  # 在channels维度进行softmax，使得全部channels的值相加为1
        # 在channels维度找最大值，获得的索引就是我要的mask（mask是0-255的值：0,1,2,3,4）
        max_val, max_index = torch.max(pre_b, dim=0)

        tensor_list.append(max_index)

    mask = torch.stack(tensor_list)
    return mask


def init_dir(phase, date):
    """
    初始化（新建）保存实验结果的文件夹
    :param phase:
    :param date:
    :return:
    """
    save_dir = os.path.join('results/' + phase, date)
    mkfile(save_dir)
    save_intermediate_dir = os.path.join(save_dir, 'intermediate')
    mkfile(save_intermediate_dir)
    save_display_dir = os.path.join(save_dir, 'display')
    mkfile(save_display_dir)
    save_csv_dir = os.path.join(save_dir, 'csv')
    mkfile(save_csv_dir)
    save_best_dir = os.path.join(save_dir, 'best')
    mkfile(save_best_dir)
    mkfile(save_best_dir + '/model')
    save_pre_img = os.path.join(save_dir, 'predict_img')



    return save_dir, save_intermediate_dir, save_display_dir, save_csv_dir, save_best_dir, save_pre_img


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, base_lr, total_niters, lr_power):
    lr = lr_poly(base_lr, i_iter, total_niters, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def s2t(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


def gain_index(k, seg, total_size, indices):
    """
    tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
    index: [trll,trlr],[vall,valr],[trrl,trrr]
    :param k: 第k折
    :param seg:
    :param total_size:
    :param indices:
    :return:
    """
    trll = 0
    trlr = k * seg
    vall = trlr
    valr = k * seg + seg
    trrl = valr
    trrr = total_size

    print("train indices: [%d,%d),[%d,%d), val indices: [%d,%d)"
          % (trll, trlr, trrl, trrr, vall, valr))
    train_indices = indices[trll:trlr] + indices[trrl:trrr]
    val_indices = indices[vall:valr]

    return train_indices, val_indices


# 若文件夹不存在，则创建新的文件夹
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def load_data(feature, label, name):
    """

    :param feature:
    :param label:
    :param name:
    :return:
    """
    _feature = pd.read_csv(feature, header=None)
    _label = pd.read_csv(label, header=None)
    _names = pd.read_csv(name, header=None)

    feature_list = _feature.values.tolist()
    label_list = _label.values.tolist()
    name_list = _names.values.tolist()

    feature_array = np.array(feature_list)  # (982, 512)
    class_array = np.squeeze(np.array(label_list))  # (982, 1)
    name_array = np.squeeze(np.array(name_list))  # (982, 1)

    return feature_array, class_array, name_array
    # return feature_list, label_list, name_list
