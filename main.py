import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from thop import profile, clever_format
from torch.utils.data import DataLoader
import numpy as np
import time
import datetime
import json
import random
import matplotlib.pyplot as plt

from dataset import augumentation
from dataset.MyImageFolder import StandarImageFolder, Metric_Learning_ImageFolder, ImgMaskTransform, k_fold_loader
from utils.wheels import save_intermediate, save_best, save_k_final_results, init_dir, s2t
from utils.metrics import metrics_score
from utils.visulize import plot_part_loss_AucF1, plot_loss_acc
from loss.discriminative_loss import DiscriminativeLoss_wizaron
from loss.lovase_loss import MyLovaszLoss
from trainer import train
from models.get_networks import MyNetworks
from eval import val_test, test


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(seed, k_fold, batch_size, num_epoch, continue_my_model, continue_my_model_train_path, learning_rate, num_instance,
         delta_v, delta_d, p_var, p_dist, p_reg, p_seg, p_disc, p_cla, is_pseudo_mask, is_pre, is_pre_path):
    '''1准备数据集'''
    transform = ImgMaskTransform(img_size=(256, 256))
    train_dataset = Metric_Learning_ImageFolder(root=image_folder + '/data_train', transform=transform)
    val_dataset = Metric_Learning_ImageFolder(root=image_folder + '/data_train', transform=transform)

    # {'narrow1':0, 'narrow2':1, 'narrow3':2, 'narrow4':3, 'wide':4}
    narrow_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in narrow_list.items())
    cla = []
    for key, val in narrow_list.items():
        cla.append(key)
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    indices = list(range(len(train_dataset)))
    print(len(train_dataset))
    # 打乱数据
    np.random.seed(seed)
    np.random.shuffle(indices)
    # print(indices)

    for i in range(k_fold):
        print('\n', '*' + '-' * 10, 'F{}'.format(i + 1), '-' * 10 + '*')

        '''2设置实验结果保存路径'''
        train_result = pd.DataFrame(columns=('loss', 'accurate'))
        val_result = pd.DataFrame(columns=('loss', 'accurate', 'recall', 'precision', 'AUC', 'F1'))
        test_result = pd.DataFrame(columns=('loss', 'accurate', 'recall', 'precision', 'AUC', 'F1'))

        train_len, train_loader, validation_loader = k_fold_loader(i, int(len(train_dataset) * 1 / k_fold),
                                                                   indices,
                                                                   train_dataset, val_dataset, batch_size)
        test_data = StandarImageFolder(root=os.path.join(image_folder, 'data_test'),
                                       transform=augumentation.liner_classifier_test_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        '''3初始化模型'''

        net = MyNetworks(n_instance=5,
                         n_classes=5, embed_dim=2,
                         branch_size=1024,
                         deep_features_size=2048,
                         backend='resnet50',
                         pretrained=is_pre,
                         model_path=is_pre_path).cuda()
        for param in net.extractors.conv1.parameters():
            param.requires_grad = False
        for param in net.extractors.bn1.parameters():
            param.requires_grad = False
        for param in net.extractors.layer1.parameters():
            param.requires_grad = False
        for p in net.extractors.layer2.parameters():
            p.requires_grad = False
        if continue_my_model:
            print('continue my model.')
            missing_keys, unexpected_keys = net.load_state_dict(torch.load(continue_my_model_train_path), strict=False)

        # ##########查看可以更新的参数#####################
        # 通过这个查看训练好的模型权重是否真的加载进来了
        # parm = {}
        # for name, parameters in net.named_parameters():
        #     if name == 'extractors.conv1.weight':
        #         print(name, ':', parameters.size())
        #         print(name, ':', parameters)
        #         parm[name] = parameters.cpu().detach().numpy()

        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)
        # ###############查看可以更新的参数################
        # pspnet: 53.86M FLOPs: 997.02M / lanenet34.71M FLOPs: 561.82M
        # flops, params = profile(net, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
        # flops, params = clever_format([flops, params])
        # print('# Model Params: {} FLOPs: {}'.format(params, flops))

        '''4设置优化器'''
        optimizer = optim.SGD(net.parameters(),
                              lr=learning_rate,
                              momentum=0.9,
                              dampening=0,  # 动量的抑制因子，默认为0
                              weight_decay=0.0005,  # 默认为0，有值说明用作正则化
                              nesterov=True, )  # 使用Nesterov动量，默认为False

        '''5初始化损失函数'''
        disc_criterion = DiscriminativeLoss_wizaron(num_instance=num_instance,
                                                    delta_v=delta_v,
                                                    delta_d=delta_d,
                                                    norm=2,
                                                    scale_var=p_var,
                                                    scale_dist=p_dist,
                                                    scale_reg=p_reg,
                                                    usegpu=True).cuda()
        cla_criterion = nn.CrossEntropyLoss().cuda()
        seg_criterion = MyLovaszLoss().cuda()

        results = {'train_loss': [], 'train_acc@1': [], 'train_acc@2': [],
                   'train_seg_loss': [], 'train_var_loss': [], 'train_dis_loss': [], 'train_reg_loss': [],
                   'val_loss': [], 'val_acc@1': [], 'val_acc@2': [],
                   # 'val_mask_loss': [], 'val_var_loss': [], 'val_dis_loss': [], 'val_reg_loss': [],
                   'test_loss': [], 'test_acc@1': [], 'test_acc@2': []}

        '''6开始训练'''
        best_acc, best_recall, best_precision, best_auc, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        lr_epoch = []
        for epoch in range(1, args.num_epoch + 1):
            print('\nF{} | Epoch [{}/{}]'.format(i + 1, epoch, args.num_epoch))

            # 1 train   loss_list里面放的是lovasz，var，dist，reg loss
            lr, train_loss, train_acc_1, train_acc_2, train_part_loss_list = train(num_epoch=num_epoch,
                                                                                   per_epoch=epoch - 1,
                                                                                   is_pseudo_mask=is_pseudo_mask,
                                                                                   net=net,
                                                                                   train_dataset=train_dataset,
                                                                                   data_loader=train_loader,
                                                                                   train_optimizer=optimizer,
                                                                                   lr=learning_rate,
                                                                                   disc_loss=disc_criterion,
                                                                                   seg_loss=seg_criterion,
                                                                                   cla_loss=cla_criterion,
                                                                                   p_seg=p_seg,
                                                                                   p_discriminative=p_disc,
                                                                                   p_cla=p_cla,
                                                                                   save_pre=save_pre_img)
            lr_epoch += lr
            # print('lr:', lr)

            results['train_loss'].append(train_loss)
            results['train_acc@1'].append(train_acc_1)
            results['train_acc@2'].append(train_acc_2)
            results['train_seg_loss'].append(train_part_loss_list[0])
            results['train_var_loss'].append(train_part_loss_list[1])
            results['train_dis_loss'].append(train_part_loss_list[2])
            results['train_reg_loss'].append(train_part_loss_list[3])

            train_result = train_result.append(pd.DataFrame({'loss': [train_loss],
                                                             'accurate': [train_acc_1]}), ignore_index=True)

            # 2 val
            val_loss, val_acc_1, val_acc_2, val_pred_probs, val_pred_labels, val_gt_labels = val_test(per_epoch=epoch,
                                                                                                      is_pseudo_mask=args.is_pseudo_mask,
                                                                                                      val_dataset=val_dataset,
                                                                                                      net=net,
                                                                                                      data_loader=validation_loader,
                                                                                                      disc_loss=disc_criterion,
                                                                                                      seg_loss=seg_criterion,
                                                                                                      cla_loss=cla_criterion,
                                                                                                      p_seg=p_seg,
                                                                                                      p_discriminative=p_disc,
                                                                                                      p_cla=p_cla,
                                                                                                      is_val=True)

            results['val_loss'].append(val_loss)
            results['val_acc@1'].append(val_acc_1)
            results['val_acc@2'].append(val_acc_2)

            val_acc, val_recall, val_precision, val_auc, val_f1 = metrics_score(val_gt_labels, val_pred_labels)
            val_result = val_result.append(pd.DataFrame({'loss': [val_loss],
                                                         'accurate': [val_acc],
                                                         'recall': [val_recall],
                                                         'precision': [val_precision],
                                                         'AUC': [val_auc],
                                                         'F1': [val_f1]}), ignore_index=True)
            # 3 test
            test_loss, test_acc_1, test_acc_2, test_pred_probs, test_pred_labels, test_gt_labels = test(net=net,
                                                                                                        data_loader=test_loader,
                                                                                                        criterion=cla_criterion)

            results['test_loss'].append(test_loss)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@2'].append(test_acc_2)
            test_acc, test_recall, test_precision, test_auc, test_f1 = metrics_score(test_gt_labels, test_pred_labels)
            test_result = test_result.append(pd.DataFrame({'loss': [test_loss],
                                                           'accurate': [test_acc],
                                                           'recall': [test_recall],
                                                           'Precision': [test_precision],
                                                           'AUC': [test_auc],
                                                           'F1': [test_f1]}), ignore_index=True)

            '''save statistics'''
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(os.path.join(save_dir, 'final_linear_statistics_' + 'K' + str(i + 1) + '.csv'),
                              index_label='epoch')
            total_curve = plot_loss_acc(data_frame)
            plt.savefig(os.path.join(save_dir, 'final_linear_statistics_' + 'K' + str(i + 1) + '.png'))

            # print('[Per_epoch]Val  acc:{} | auc:{} | f1:{}'.format(val_acc, val_auc, val_f1))
            # print('[Per_epoch]Test acc:{} | auc:{} | f1:{}'.format(test_acc, test_auc, test_f1))

            if val_acc_1 > best_acc:
                best_acc = val_acc_1
                # 当验证集准确率最高时，保存测试集的结果
                save_best(i, net, val_gt_labels, val_pred_labels,
                          val_pred_probs, test_gt_labels, test_pred_labels, cla,
                          test_pred_probs, save_best_dir, save_dir)
                torch.save(net.state_dict(),
                           os.path.join(save_best_dir, 'model/K' + str(i + 1) + 'EP' + str(epoch) + '.pth'))
                print('[Best]\nVal:  acc:{} | recalll:{} | precision:{} | auc:{} | f1:{}'.format(val_acc, val_recall,
                                                                                               val_precision, val_auc,
                                                                                               val_f1))
                print('Test: acc:{} | recalll:{} | precision:{} | auc:{} | f1:{}'.format(test_acc, test_recall,
                                                                                               test_precision, test_auc,
                                                                                               test_f1))

            if epoch % 100 == 0:
                # 保存中间结果
                save_intermediate(net, save_intermediate_dir, epoch,
                                  i, val_gt_labels, val_pred_labels,
                                  val_pred_probs, cla, test_pred_probs, test_gt_labels,
                                  test_pred_labels, train_result, val_result, test_result)
                print('save epoch {}!'.format(epoch))

            '''save final epoch results'''
            train_result.to_csv(os.path.join(save_csv_dir, 'Train' + 'K' + str(i + 1) + '.csv'))
            val_result.to_csv(os.path.join(save_csv_dir, 'Val' + 'K' + str(i + 1) + '.csv'))
            test_result.to_csv(os.path.join(save_csv_dir, 'Test' + 'K' + str(i + 1) + '.csv'))
        save_k_final_results(net, save_dir, save_display_dir,
                             i, lr_epoch, val_gt_labels, val_pred_labels,
                             val_pred_probs, cla, test_pred_probs, test_gt_labels,
                             test_pred_labels)

        plot_part_loss_AucF1(data_frame, val_result, test_result)
        plt.savefig(os.path.join(save_dir, 'PartLoss_AUCF1' + 'K' + str(i + 1) + '.png'))
        break


if __name__ == '__main__':
    # 60
    image_folder = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v14/classification_data/img/objective_img'
    mask_folder = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v14/classification_data/seg_mask/obj_mask'
    model_path = '/home/huangjq/PyCharmCode/4_project/frequently_used/pretrained_model/resnet50-pre.pth'
    continue_my_model_train_path = 'CP300K1.pth'

    # 88
    # 以前的数据集，判断现在的数据集有没有脏数据
    # image_folder = '/home/huangjq/PyCharmCode/project/from_yolo'
    # image_folder = '/home/huangjq/PyCharmCode/project/v14/classification_data/img/objective_img'
    # mask_folder = '/home/huangjq/PyCharmCode/project/v14/classification_data/seg_mask/obj_mask'
    # model_path = '/home/huangjq/PyCharmCode/project/pretrained_model/resnet50-pre.pth'
    # continue_my_model_train_path = '/home/huangjq/PyCharmCode/project/Metric-testing/results/train_discriminative_loss_model/0222_finetue_lane_4/intermediate/CP300K1.pth'

    # 59
    # image_folder = '/home/huangjq/PyCharmCode/linshi/v14/classification_data/img/objective_img'
    # mask_folder = '/home/huangjq/PyCharmCode/linshi/v14/classification_data/seg_mask/obj_mask'
    # model_path = '/home/huangjq/PyCharmCode/linshi/pretrained_model/resnet50-pre.pth'

    parser = argparse.ArgumentParser(description='Train Resnet50 with metric learning.')
    parser.add_argument('--phase', type=str, default='train_resnet50+unet+_model',
                        help='the type you train, classifier or self-supervised')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--k_fold', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each mini-batch')
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='The pretrained model path')

    parser.add_argument('--embedding_dim', type=int, default=2, help='the dim of the')
    parser.add_argument('--p_cla', default=1.0, type=float, help='the discriminativve loss parameter')
    parser.add_argument('--p_seg', default=1.0, type=float, help='the discriminativve loss parameter')
    parser.add_argument('--p_disc', default=1.0, type=float, help='the discriminativve loss parameter')
    parser.add_argument('--p_var', default=1.0, type=float, help='the discriminativve loss parameter')
    parser.add_argument('--p_dist', default=1.0, type=float, help='the discriminativve loss parameter')
    parser.add_argument('--p_reg', default=1e-1, type=float, help='the discriminativve loss parameter')
    parser.add_argument('--delta_v', default=0.0001, type=float, help='the discriminativve loss parameter')
    parser.add_argument('--delta_d', default=1, type=float, help='the discriminativve loss parameter')

    parser.add_argument('--num_instance', type=int, default=5, help='the num of the mask')
    parser.add_argument('--num_classes', type=int, default=5, help='the num of the mask')

    # 如果是迁移模型训练队网络，并且想在自己的模型基础上再训练网络，则continue_my_model和pre_tained都要设置为True
    parser.add_argument('--continue_my_model', action='store_true',
                        help='weather continue you haved trained model, the epoch of trained model is no huge enough')
    parser.add_argument('--is_pseudo_mask', action='store_true', help='weather use is_pseudo_mask')
    parser.add_argument('--is_pre_train', action='store_true', help='weather use ImageNet pre trained model')

    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_epoch', type=int, default=3, help='Number of sweeps over the dataset to train')
    parser.add_argument('--date', default='today', type=str, help='thr time you train your model')
    args = parser.parse_args()
    args, args_other = parser.parse_known_args()
    start_time = time.time()
    print("start time:", datetime.datetime.now())
    print(f"Running fine tune classifier with args {args}")

    setup_seed(args.seed)
    save_dir, save_intermediate_dir, save_display_dir, save_csv_dir, save_best_dir, save_pre_img = init_dir(args.phase, args.date)
    main(args.seed, args.k_fold, args.batch_size, args.num_epoch, args.continue_my_model, continue_my_model_train_path,
         args.learning_rate, args.num_instance, args.delta_v, args.delta_d, args.p_var, args.p_dist, args.p_reg, args.p_seg,
         args.p_disc, args.p_cla, args.is_pseudo_mask, args.is_pre_train, model_path)

    print("\nEnd time:", datetime.datetime.now())
    h, m, s = s2t(time.time() - start_time)
    print("Using Time: %02dh %02dm %02ds" % (h, m, s))
