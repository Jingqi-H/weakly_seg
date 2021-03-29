import torch
import numpy as np

from utils.metrics import pred_prob2pred_label
from loss.loss_total import caculate_all_loss


def val_test(per_epoch,
             is_pseudo_mask,
             net,
             val_dataset,
             data_loader,
             disc_loss,
             seg_loss,
             cla_loss,
             p_seg,
             p_discriminative,
             p_cla,
             is_val=True):
    net.eval()
    total_loss, total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        pred_label_all, pred_prob_all, gt_label = [], [], []
        pred_prob, pred_label = 0, 0
        total_num_mask = 0
        for step, data in enumerate(data_loader, start=0):
            # print(data['seg_label_mask'])
            image, cla_label = data['img'].cuda(), data['cla_label'].cuda()

            resnet_y, feature, em = net(image)
            seg_label, _mask = val_dataset.get_seg_label_batch(img_size=(data['img'].shape[2],
                                                                         data['img'].shape[3]),
                                                               mask_index=data['seg_label_mask'],
                                                               hflip=data['is_hflip'])
            _seg_loss, _cla_loss, dict_dis_loss, num_mask = caculate_all_loss(epoch=per_epoch,
                                                                              using_pseudo_mask=is_pseudo_mask,
                                                                              seg_label=seg_label,
                                                                              cla_label=cla_label,
                                                                              seg_pred=feature,
                                                                              seg_pred_after_mask=feature[_mask],
                                                                              em_after_mask=em[_mask],
                                                                              cla_pred_y=resnet_y,
                                                                              disc_loss_func=disc_loss,
                                                                              seg_loss_func=seg_loss,
                                                                              cla_loss_func=cla_loss)

            loss = p_seg * _seg_loss + p_discriminative * dict_dis_loss[
                'loss'] + p_cla * _cla_loss  # 设置3类损失的权重
            total_num_mask += num_mask
            pred_prob, pred_label = pred_prob2pred_label(resnet_y)
            pred_prob_all.append(pred_prob)
            pred_label_all.append(pred_label)
            gt_label.append(cla_label.cpu())

            total_num += image.size(0)
            total_loss += loss.item() * image.size(0)
            # .argsort: 返回按值按升序对给定维度的张量进行排序的索引
            prediction = torch.argsort(resnet_y, dim=-1, descending=True)  # 从大到小排列返回索引
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # 排序前两个的预测结果，有一个的结果是和target一直，则+1
            total_correct_2 += torch.sum(
                (prediction[:, 0:2] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        print('{} Loss: {:.4f} ACC@1: {:.2f}% ACC@2: {:.2f}%'
              .format('Val' if is_val else 'Test', total_loss / total_num,
                      total_correct_1 / total_num * 100, total_correct_2 / total_num * 100))

        pred_probs = np.concatenate(pred_prob_all)
        pred_labels = np.concatenate(pred_label_all)
        gt_labels = np.concatenate(gt_label)

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_2 / total_num * 100, pred_probs, pred_labels, gt_labels


def test(net, data_loader, criterion, is_val=False):
    net.eval()
    total_loss, total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        pred_label_all, pred_prob_all, gt_label = [], [], []
        pred_prob, pred_label = 0, 0
        for step, [data, label, name] in enumerate(data_loader, start=0):
            # print('in test:', data.shape)
            resnet_y, feature, em = net(data.cuda())
            loss = criterion(resnet_y, label.cuda())

            pred_prob, pred_label = pred_prob2pred_label(resnet_y)
            pred_prob_all.append(pred_prob)
            pred_label_all.append(pred_label)
            gt_label.append(label)

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            # .argsort: 返回按值按升序对给定维度的张量进行排序的索引
            prediction = torch.argsort(resnet_y, dim=-1, descending=True)  # 从大到小排列返回索引
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == label.cuda().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # 排序前两个的预测结果，有一个的结果是和target一直，则+1
            total_correct_2 += torch.sum(
                (prediction[:, 0:2] == label.cuda().unsqueeze(dim=-1)).any(dim=-1).float()).item()

        print('{} Loss: {:.4f} ACC@1: {:.2f}% ACC@2: {:.2f}%'
              .format('Val' if is_val else 'Test', total_loss / total_num,
                      total_correct_1 / total_num * 100, total_correct_2 / total_num * 100))

        pred_probs = np.concatenate(pred_prob_all)
        pred_labels = np.concatenate(pred_label_all)
        gt_labels = np.concatenate(gt_label)

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_2 / total_num * 100, pred_probs, pred_labels, gt_labels
