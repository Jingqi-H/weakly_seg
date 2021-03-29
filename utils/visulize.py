import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch


def tensor2img(ten):
    """
    红绿黄蓝黑分别是：sl,tm,ss,cbb,background
    :param ten:传入的是tensor，里面的数值是类别索引，e.g.[1, 2, 3, ..., num_classes-1]
    :return: 返回的变量可以直接用 image.save(name.png) 保存
    """

    img = ten.numpy()
    colors = [(0, 0, 0), (255, 0, 0), (0, 128, 0), (255, 255, 0), (0, 0, 128)]
    num_classes = 5
    seg_img = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
    for c in range(num_classes):
        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('uint8')

    # image = Image.fromarray(np.uint8(seg_img)).resize((2100, 700), Image.NEAREST)
    image = Image.fromarray(np.uint8(seg_img))
    return image


def show_image(prediction, save_dir, name):
    """
    展示模型输出的预测结果，分别保存一个bs里面的每一张图片
    :param prediction: 预测结果，[bs, num_instance, h, w]
    :param save_dir: 保存路径
    :param name: 图片名字，list
    :return:
    """
    # mkfile(save_dir)
    tensor = prediction.cpu().clone()
    max_val, max_index = torch.max(torch.softmax(tensor, dim=1), dim=1)  # 用这个函数，不需要除去bs=1的维度，有多少个bs返回多少个特征图(h*w)
    # print(max_index.shape)

    plt.cla()
    plt.close('all')
    plt.figure()
    for i, ten in enumerate(max_index):
        # print(i, ten.shape)
        img = tensor2img(ten)
        img.save(os.path.join(save_dir, name[i]))
        # plt.imshow(img)
        # plt.show()
    return img



def plot_part_loss_AucF1(file_final, file_val, file_test):
    """
    三个输入都DataFrame，用于展示训练集的分损失和验证集测试集的AUC，F1分数
    :param file_final: 最后的总输出
    :param file_val:  验证集的评价分数，包括loss	accurate	recall	precision	AUC	F1
    :param file_test: 测试集的评价分数，同上
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))

    axes[0].plot(file_final['train_loss'])
    axes[0].plot(file_final['train_seg_loss'])
    axes[0].plot(file_final['train_var_loss'])
    axes[0].plot(file_final['train_dis_loss'])
    axes[0].tick_params(labelsize=30)
    axes[0].legend(['train_loss', 'train_seg_loss', 'train_var_loss', 'train_dis_loss'], fontsize=30)
    axes[0].set_xlabel('Epoch', fontsize=30)
    axes[0].grid(linestyle=":")

    axes[1].plot(file_val['AUC'])
    axes[1].plot(file_test['AUC'])
    axes[1].plot(file_val['F1'])
    axes[1].plot(file_test['F1'])
    axes[1].tick_params(labelsize=30)
    axes[1].legend(['val_auc', 'test_auc', 'val_f1', 'test_f1'], fontsize=30)
    axes[1].set_xlabel('Epoch', fontsize=30)
    axes[1].grid(linestyle=":")


def plot_lr(lr_list):
    # print(lr_list)
    x = []
    for i in range(len(lr_list)):
        x.append(i + 1)
    plt.figure(figsize=(20, 10))
    plt.plot(x, lr_list, label='loss', linewidth=2.0)

    plt.legend()
    plt.xlabel('num', fontsize=24)
    plt.ylabel('loss', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title('name', fontsize=24)
    plt.grid(linestyle='-.')

    # plt.savefig('./test.png')
    # plt.show()


def plot_dataframe(train_file, val_file):
    """
    一张画布上有两个图，
    :param train_file:
    :param val_file:
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax1 = axes[0]
    ax2 = axes[1]

    # “.loc[:3000]”可以删掉，这个是用来限制行数的
    ax1.plot(train_file['Accurate'].loc[:3000], '-', label='train')
    ax1.plot(val_file['Accurate'].loc[:3000], '--', label='val')
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Accurate', fontsize=15)
    # ax1.legend('train', fontsize=15)
    ax1.grid(linestyle=":")
    ax1.tick_params(labelsize=12)

    ax2.plot(train_file['Loss'].loc[:3000], '-', label='train')
    ax2.plot(val_file['Loss'].loc[:3000], '--', label='val')
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)
    # ax2.legend(legend_train, fontsize=15)
    ax2.grid(linestyle=":")
    ax2.tick_params(labelsize=12)
    # plt.show()

    return fig


def plot_trainval_lossacc(dir_main32_root):
    train_file = []
    val_file = []
    legend_train, legend_val = [], []

    # 获得每个csv文件的DataFrame，存到list中
    for i in range(3):
        name_train = 'K' + str(i + 1) + 'TrainScore.csv'
        name_val = 'K' + str(i + 1) + 'ValScore.csv'
        train_file.append(pd.read_csv(os.path.join(dir_main32_root, name_train), encoding='gbk'))
        val_file.append(pd.read_csv(os.path.join(dir_main32_root, name_val), encoding='gbk'))
        legend_train.append('k' + str(i + 1) + '_train')
        legend_val.append('k' + str(i + 1) + '_val')

    # 输出k折的平均结果
    kflod_file = pd.read_csv(os.path.join(dir_main32_root, 'KfoldScore.csv'), encoding='gbk')
    print(kflod_file.describe())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax1 = axes[0]
    ax2 = axes[1]

    # “.loc[:3000]”可以删掉，这个是用来限制行数的
    ax1.plot(train_file[0]['Accurate'].loc[:3000], '-')
    ax1.plot(train_file[1]['Accurate'].loc[:3000], '-')
    ax1.plot(train_file[2]['Accurate'].loc[:3000], '-')
    ax1.plot(val_file[0]['Accurate'].loc[:3000], '--')
    ax1.plot(val_file[1]['Accurate'].loc[:3000], '--')
    ax1.plot(val_file[2]['Accurate'].loc[:3000], '--')
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Accurate', fontsize=15)
    legend_train += legend_val
    ax1.legend(legend_train, fontsize=15)
    ax1.grid(linestyle=":")
    ax1.tick_params(labelsize=12)

    ax2.plot(train_file[0]['Loss'].loc[:3000], '-')
    ax2.plot(train_file[1]['Loss'].loc[:3000], '-')
    ax2.plot(train_file[2]['Loss'].loc[:3000], '-')
    ax2.plot(val_file[0]['Loss'].loc[:3000], '--')
    ax2.plot(val_file[1]['Loss'].loc[:3000], '--')
    ax2.plot(val_file[2]['Loss'].loc[:3000], '--')
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)
    ax2.legend(legend_train, fontsize=15)
    ax2.grid(linestyle=":")
    # 设置坐标轴的数值大小
    ax2.tick_params(labelsize=12)
    # plt.show()

    return fig


def plot_loss_acc(csv_file):
    """
    传入DataFrame数据，实际上就是’linear_statistics.csv‘这个文件，根据这个文件画出loss和acc曲线
    :param csv_file:
    :return:
    """
    plt.cla()
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    plot_name = ['loss', 'acc']
    plot_phase = [['train_loss', 'val_loss', 'test_loss'], ['train_acc@1', 'val_acc@1', 'test_acc@1']]
    for i in range(len(plot_name)):
        for j in range(len(plot_phase[0])):
            axes[i].plot(csv_file[plot_phase[i][j]])
        axes[i].grid(linestyle=":")
        axes[i].tick_params(labelsize=30)
        axes[i].set_xlabel('Epoch', fontsize=30)
        axes[i].set_ylabel(plot_name[i], fontsize=30)
        axes[i].legend(plot_phase[i], fontsize=20, loc='upper left')

    return axes


def plot_confusion_matrix_inPrediction(cm, _classes=None):
    """
    和下面可视化混淆矩阵的不同在于，混淆矩阵的数值是百分号
    :param cm:
    :param _classes:
    :return:
    """
    if _classes == None:
        # classes = ['N2', 'N3', 'N4']
        classes = ['N1', 'N2', 'N3', 'N4', 'W']
    else:
        classes = _classes

    plt.cla()
    plt.close('all')
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    # 每个框里的数字
    # print(cm)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # print(y_val, x_val)
        c = cm[y_val][x_val]  # 获得混淆矩阵里的一个数
        cc = c / cm[y_val].sum()  # 或者矩阵里的这个数占这个类总数的百分之几
        if c > 0.001:
            plt.text(x_val, y_val, "%.2f" % (cc), color='black', fontsize=15, va='center', ha='center')

    #     cmap 设置混淆矩阵的颜色
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #     plt.imshow(confusion, cmap=plt.cm.Blues)
    # plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    # plt.xticks(xlocations, classes, rotation=45)
    plt.xticks(xlocations, classes, rotation=0)
    plt.yticks(xlocations, classes)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    #    show confusion matrix
    # save_name = 'K' + str(k) + 'CP' + str(epoch) + '_confusion_matrix' + '.png'
    # plt.savefig(os.path.join(save_results, save_name))
    # plt.show()
    return plt


def plot_confusion_matrix(cm, _classes=None):
    if _classes == None:
        # classes = ['N2', 'N3', 'N4']
        classes = ['N1', 'N2', 'N3', 'N4', 'W']
    else:
        classes = _classes

    plt.cla()
    plt.close('all')
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    # 每个框里的数字
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='black', fontsize=15, va='center', ha='center')

    #     cmap 设置混淆矩阵的颜色
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #     plt.imshow(confusion, cmap=plt.cm.Blues)
    # plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    # plt.xticks(xlocations, classes, rotation=45)
    plt.xticks(xlocations, classes, rotation=0)
    plt.yticks(xlocations, classes)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    #    show confusion matrix
    # save_name = 'K' + str(k) + 'CP' + str(epoch) + '_confusion_matrix' + '.png'
    # plt.savefig(os.path.join(save_results, save_name))
    # plt.show()
    return plt


def plot_roc(y_score, y_test):
    """
    根据sklearn参考文档
    :param y_score:是得到预测结果，他是概率值，并且是array
    :param y_test:是gt
    :param save_results: 保存路径
    :return:
    """

    # y_score = torch.rand([30, 5]).numpy()
    # # y = torch.tensor([1, 0, 0, 4, 1, 3, 0, 3, 4, 4, 3, 2, 0, 2, 3, 4, 1, 1, 1, 4, 3, 0, 0, 0,
    # #         1, 1, 0, 0, 2, 2])
    # y_test = torch.tensor([1, 1, 1, 4, 1, 4, 0, 3, 4, 4, 2, 2, 0, 3, 3, 3, 1, 1, 1, 4, 3, 0, 0, 0,
    #                        1, 0, 0, 0, 3, 2])

    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    # y_test = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test.shape[1]
    # print(n_classes)

    # y_test是二值，y_score是概率
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # print(i)
        # print(y_test[:, i])
        # print(y_score[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.cla()
    plt.close('all')
    plt.figure()
    # micro：多分类；macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
    plt.plot(fpr["micro"], tpr["micro"],
             label='average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['gold', '#1E90FF', '#FF6347', '#9370DB', '#228B22'])
    classes = ['N1', 'N2', 'N3', 'N4', 'W']
    # classes = ['N2', 'N3', 'N4']
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    # save_name = 'K' + str(k) + 'CP' + str(epoch) + '_roc' + '.png'
    # plt.savefig(os.path.join(save_results, save_name))
    # plt.show()
    return plt


def total_presentation(pred_probs, pred_labels, gt_labels, class_names):
    cm = confusion_matrix(gt_labels, pred_labels, labels=None, sample_weight=None)
    plt_con_mat = plot_confusion_matrix(cm, _classes=class_names)
    plt_roc = plot_roc(pred_probs, gt_labels)

    return plt_con_mat, plt_roc


# 1203 慢行厚积的代码：https://www.cnblogs.com/wanghui-garcia/p/11393076.html


if __name__ == '__main__':
    prediction = torch.rand([2, 5, 4, 4])
    save_dir = '/home/huangjq/PyCharmCode/project/2-weakly_seg/results/train_resnet50+unet+_model/today/predict_img'
    name = ['a', 'b']
    show_image(prediction, save_dir + '/EP' + str(2), name)
