import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import json
import os
import time, datetime
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.get_networks import MyNetworks
from models.get_networks_2 import MyNetworks2
from models.get_networks_3 import MyNetworks3
from dataset import augumentation
from dataset.MyImageFolder import StandarImageFolder
from utils.wheels import s2t, mkfile
from utils.metrics import pred_prob2pred_label, metrics_score
from utils.visulize import plot_confusion_matrix, plot_roc, plot_confusion_matrix_inPrediction

# 参数初始化
# 60
data_path = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v14/classification_data/img/objective_img'
model_weight_path = '/home/huangjq/PyCharmCode/4_project/7-paper/2-weakly_seg/results/train_resnet50+unet+_model/OursUnet_7_2/best/model'
# 59
# data_path = '/home/huangjq/PyCharmCode/linshi/v14/classification_data/img/objective_img'
# 88
# data_path = '/home/huangjq/PyCharmCode/project/v14/classification_data/img/objective_img'
# model_weight_path = '/home/huangjq/PyCharmCode/project/2-weakly_seg/results/train_resnet50+unet+_model/0325_OursUnet_1_2/best/model'

# save_display_dir = os.path.join('./results/predictor', model_weight_path.split('/')[-2])
# mkfile(save_display_dir)

start_time = time.time()
print('model:', model_weight_path.split('/')[-3])
print("start time:", datetime.datetime.now())

# create model
# load model weights  build_lane_network
net = MyNetworks3(n_instance=5,
                 n_classes=5, embed_dim=2,
                 branch_size=1024,
                 deep_features_size=2048,
                 backend='resnet50',
                 pretrained=False,
                 model_path='is_pre_path').cuda()

# 查看网络参数
# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print(name)
# parm = {}
# for name, parameters in net.named_parameters():
#     if name == 'binary_seg.0.bias':
#         print(name, ':', parameters.size())
#         print(name, ':', parameters)
#         parm[name] = parameters.cpu().detach().numpy()

# the same as train's

data_transform = transforms.Compose([
    transforms.Resize((128, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


test_data = StandarImageFolder(root=os.path.join(data_path, 'data_test'),
                               transform=data_transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

net.eval()
pth_list = os.listdir(model_weight_path)
for nn in pth_list:
    print(nn)
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(os.path.join(model_weight_path, nn)), strict=False)
    # print(missing_keys, unexpected_keys)

    pred_probs, pred_labels, gt_labels = 0, 0, 0
    val_acc, recall, precision, auc, f1 = 0, 0, 0, 0, 0
    with torch.no_grad():
        pred_label_all, pred_prob_all, gt_label = [], [], []
        pred_prob, pred_label = 0, 0
        for step, [img, label, name] in enumerate(test_loader, start=0):
            resnet_y, feature, em = net(img.cuda())
            pred_prob, pred_label = pred_prob2pred_label(resnet_y)

            pred_prob_all.append(pred_prob)
            pred_label_all.append(pred_label)
            gt_label.append(label)

    pred_probs = np.concatenate(pred_prob_all)
    pred_labels = np.concatenate(pred_label_all)
    gt_labels = np.concatenate(gt_label)
    val_acc, recall, precision, auc, f1 = metrics_score(gt_labels, pred_labels)
    print('test_acc:{} | auc:{} | f1:{}'.format(val_acc, auc, f1))

# cm = confusion_matrix(gt_labels, pred_labels, labels=None, sample_weight=None)
# plot_confusion_matrix(cm)
# plt.savefig(os.path.join(save_display_dir, 'test_confusion_matrix' + '.pdf'), format="pdf", bbox_inches='tight', pad_inches=0)
# plot_confusion_matrix_inPrediction(cm)
# plt.savefig(os.path.join(save_display_dir, 'test_confusion_matrix2' + '.pdf'), format="pdf", bbox_inches='tight', pad_inches=0)
# plot_roc(pred_probs, gt_labels)
# plt.savefig(os.path.join(save_display_dir, 'test_roc' + '.pdf'), format="pdf", bbox_inches='tight', pad_inches=0)

print("\nEnd time:", datetime.datetime.now())
h, m, s = s2t(time.time() - start_time)
print("Using Time: %02dh %02dm %02ds" % (h, m, s))
