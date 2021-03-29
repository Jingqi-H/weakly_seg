import torch
import numpy as np
from utils.wheels import adjust_learning_rate
from utils.visulize import show_image
from loss.loss_total import caculate_all_loss

from utils.wheels import mkfile

def train(num_epoch,
          per_epoch,
          is_pseudo_mask,
          net,
          train_dataset,
          data_loader,
          train_optimizer,
          lr,
          disc_loss,
          seg_loss,
          cla_loss,
          p_seg,
          p_discriminative,
          p_cla,
          save_pre):
    net.train()
    total_niters = num_epoch * len(data_loader)
    lr_list = []
    adjust_lr = 0.0
    total_loss, total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0, 0

    # 每一step的loss存在这个list里面
    per_loss, per_seg_loss, per_cla_loss = [], [], []
    per_discriminative_loss, per_var_loss, per_dist_loss, per_reg_loss = [], [], [], []

    total_num = 0
    for step, data in enumerate(data_loader, start=0):
        image, cla_label = data['img'].cuda(), data['cla_label'].cuda()

        # 判读分割掩码图是否存在，如不存在，则显示图片，目前这存的是图片名字，还不能show图片
        # if data['seg_label_mask'][0] != 'None':
        #     show_pil_from_tensor(data["seg_label_mask"])

        current_idx = per_epoch * len(data_loader) + step
        train_optimizer.zero_grad()
        adjust_lr = adjust_learning_rate(train_optimizer, current_idx, base_lr=lr,
                                         total_niters=total_niters,
                                         lr_power=0.9)
        resnet_y, feature, em = net(image)
        # ---------------------- #
        # 保存预测图
        # ---------------------- #
        if (per_epoch + 1) % 20 == 0:
            # print('save predicted img.')
            save_pre_path = save_pre + '/EP' + str(per_epoch+1)
            mkfile(save_pre_path)
            show_image(prediction=feature, save_dir=save_pre_path, name=data['name'])

        seg_label, _mask = train_dataset.get_seg_label_batch(img_size=(data['img'].shape[2],
                                                                       data['img'].shape[3]),
                                                             mask_index=data['seg_label_mask'],
                                                             hflip=data['is_hflip'])

        # 计算总损失，需要用到feature，feature[mask],em[mask]
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
        loss.backward()
        train_optimizer.step()
        lr_list.append(adjust_lr)

        per_loss.append(loss.item())  # 乘了系数的
        per_cla_loss.append(_cla_loss.item())  # 完整的原始数据，没有权重相乘的
        per_seg_loss.append(_seg_loss.item())
        # 经过了权重相乘
        if num_mask != 0:  # 现在这个值是判别式损失的出现次数
            # 完整的原始数据，没有权重相乘的
            per_discriminative_loss.append(dict_dis_loss['loss'].item())
            per_var_loss.append(dict_dis_loss['loss_var'].item())
            per_dist_loss.append(dict_dis_loss['loss_dist'].item())
            per_reg_loss.append(dict_dis_loss['loss_reg'].item())

        total_num += image.size(0)
        # argsort: 返回按值按升序对给定维度的张量进行排序的索引
        # print(resnet_y, resnet_y.shape)
        prediction = torch.argsort(resnet_y, dim=-1, descending=True)  # 从大到小排列返回索引
        total_correct_1 += torch.sum((prediction[:, 0:1] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        # 排序前两个的预测结果，有一个的结果是和target一直，则+1
        total_correct_2 += torch.sum((prediction[:, 0:2] == cla_label.unsqueeze(dim=-1)).any(dim=-1).float()).item()

    print('var loss: {:.4f} | dist loss: {:.4f} | reg loss: {:.4f}'.format(np.mean(per_var_loss),
                                                                           np.mean(per_dist_loss),
                                                                           np.mean(per_reg_loss)))
    print('Train Loss: {:.4f} | seg loss: {:.4f} | discriminative loss: {:.4f} | cla loss: {:.4f}'.format(
        np.mean(per_loss),
        np.mean(per_seg_loss),
        np.mean(per_discriminative_loss),
        np.mean(per_cla_loss)))

    print(
        'Train Loss: {:.4f} ACC@1: {:.2f}% ACC@2: {:.2f}%'.format(np.mean(per_loss), total_correct_1 / total_num * 100,
                                                                  total_correct_2 / total_num * 100))

    return lr_list, np.mean(per_loss), total_correct_1 / total_num * 100, total_correct_2 / total_num * 100, [np.mean(
        per_seg_loss), np.mean(per_var_loss), np.mean(per_dist_loss), np.mean(per_reg_loss)]
