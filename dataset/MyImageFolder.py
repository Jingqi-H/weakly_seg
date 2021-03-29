from torch.utils.data.sampler import SubsetRandomSampler
from utils.wheels import gain_index
from dataset import augumentation
from torchvision.datasets import ImageFolder
import torch.utils.data as DATA
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as tf
import random
import torch
import numpy as np
from torchvision import transforms
import os
from torchvision.datasets import ImageFolder
from torch import tensor
from torchvision import transforms, datasets
from dataset.augumentation import AddPepperNoise


def hold_out_loader(dataset):
    """
    留出法。没有写好，还不能用
    :param dataset:
    :return:
    """
    valid_size = 0.15
    batch_size = 8
    seed = 42
    num_workers = 4

    # obtain training indices that will be used for validation
    num_train = len(dataset)
    indices = list(range(num_train))

    np.random.seed(seed)
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    print('train_indices\n', train_idx)
    print('val_indices\n', valid_idx)
    print()

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    tr_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                           num_workers=num_workers, drop_last=True, shuffle=False)

    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                            num_workers=num_workers, drop_last=True)

    return tr_loader, val_loader


def test_loader(root, batch_size):
    """
    自监督训练的时候，用于计算准确率，判断模型是否学到有用的东西
    :param root:
    :param batch_size:
    :return:
    """
    datasets = StandarImageFolder(root=root,
                                  transform=augumentation.self_test_transform)
    loader = DATA.DataLoader(datasets,
                             drop_last=False,
                             batch_size=1,
                             shuffle=False,
                             num_workers=16)
    return loader


def k_fold_loader(i, seg, indices, tr_dataset, val_dataset, batch_size):
    """

    :param i: 第i折交叉验证
    :param seg:
    :param indices: 通过随机种子打乱之后的索引，每个索引对应一个图片
    :param tr_dataset: 训练集的dataset，有数据增强的
    :param val_dataset: 验证集的dataset，没有加数据增强
    :param batch_size:
    :return:
    """
    # 获得训练集和验证集的索引
    train_indices, val_indices = gain_index(i, seg, len(tr_dataset), indices)
    print('train_indices\n', train_indices)
    print('val_indices\n', val_indices)

    train_sampler = DATA.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = DATA.sampler.SubsetRandomSampler(val_indices)

    tr_len, val_len = len(train_sampler), len(valid_sampler)
    # print(train_len, val_len)
    print("train data: {} | val data: {}".format(tr_len, val_len))
    print()

    tr_loader = DATA.DataLoader(tr_dataset,
                                drop_last=True,
                                batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=16)
    val_loader = DATA.DataLoader(val_dataset,
                                 batch_size=1,
                                 sampler=valid_sampler,
                                 drop_last=False,
                                 num_workers=16)

    return tr_len, tr_loader, val_loader


class SimCLRImageFolder(ImageFolder):
    """
    用于对比学习（SimCLR）的ImageFolder
    可以获得batchsize的每个图片名字
    同一批图片根据不同的augumentation获得不同的input
    """

    def __init__(self, root, transform=None):
        super(SimCLRImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]  # 此时的self.imgs等于self.samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        label = self.imgs[index][1]

        img = self.loader(path)
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return pos_1, pos_2, label, path.split("/")[-1]


class StandarImageFolder(ImageFolder):
    """
    可以获得batchsize的每个图片名字的普通功能ImageFolder
    """

    def __init__(self, root, transform=None):
        super(StandarImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]  # 此时的self.imgs等于self.samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        label = self.imgs[index][1]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, path.split("/")[-1]


class ImgMaskTransform(object):
    """
    这个数据增强是特地用于Metric_learning的,根据自己的数据集算了均值和方差做了归一化
    """

    def __init__(self, PepperNoise_p=0.95, img_size=(256, 512)):
        self.img_standar_transform = transforms.Compose([
            AddPepperNoise(PepperNoise_p, p=0.5),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.37818605, 0.27225745, 0.15962803], [0.16118085, 0.13099103, 0.1116703])
        ])

        # self.mask_standar_transform = transforms.Compose([
        #     transforms.Resize(img_size),
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        self.hflip = False

    # transform 会调用该方法
    def __call__(self, img):
        # 50%的概率应用垂直，水平翻转。
        p1 = random.random()

        if p1 > 0.5:
            img = tf.hflip(img)
            self.hflip = True
        img = self.img_standar_transform(img)

        return img, self.hflip


class Metric_Learning_ImageFolder(ImageFolder):
    """
    覆写ImageFolder，用于metric learning，返回的值分别是
    name: 训练集的图片名字
    img: 输入网络的图片
    cla_label: 输入网络的图片类别标签
    is_hflip: 图片做数据增强时是否水平镜像，若是则True，不是则False，e.g. tensor([True, False, True, ...])
    seg_label_mask: 判断输入网络的图片是否有分割的掩码图，若有则是True，e.g. [False, False, False, False, False, True, ...]
    """

    def __init__(self, root, transform=None):
        super(Metric_Learning_ImageFolder, self).__init__(root, transform)
        # random.seed()
        # 图片对应的掩码图所在路径
        # 60
        self.mask_folder = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v14/classification_data/seg_mask/obj_mask'
        # 88
        # self.mask_folder = '/home/huangjq/PyCharmCode/project/v14/classification_data/seg_mask/obj_mask'
        # 59
        # self.mask_folder = '/home/huangjq/PyCharmCode/linshi/v14/classification_data/seg_mask/obj_mask'

    def __getitem__(self, index):

        img_path = self.imgs[index][0]  # 此时的self.imgs等于self.samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        img_name, img_class = img_path.split("/")[-1], img_path.split("/")[-2]
        mask_path = self.mask_folder + '/' + img_class + '/' + img_name[:-4] + ".png"

        img = self.loader(img_path)

        if self.transform is not None:
            img, is_hflip = self.transform(img)

        # print(mask_path)
        if os.path.exists(mask_path):
            # print(mask_path)
            seg_label_mask = '/' + img_class + '/' + img_name[:-4] + ".png"
        else:
            seg_label_mask = 'None'
        cla_label = self.imgs[index][1]

        output = {
            "name": img_name,
            "img": img,
            "cla_label": cla_label,
            "is_hflip": is_hflip,
            "seg_label_mask": seg_label_mask,
        }

        return output

    def get_seg_label_batch(self, img_size, mask_index, hflip):

        seg_label_list = []
        embedding_mask = []
        for k in range(len(mask_index)):
            seg_label_path = mask_index[k]
            mask_is_hflip = hflip[k].item()
            if seg_label_path != 'None':
                seg_label = Image.open(self.mask_folder + seg_label_path)
                seg_label = self.mask_tansform(img_size=img_size, hflip_index=mask_is_hflip, seg_label_=seg_label)
                seg_label_list.append(seg_label)
                embedding_mask.append(True)
            else:
                embedding_mask.append(False)
        try:
            seg_label_ten = torch.stack(seg_label_list).cuda()
        except:
            seg_label_ten = None
        return seg_label_ten, embedding_mask

    def mask_tansform(self, img_size=(256, 512), hflip_index=False, seg_label_=None):
        mask_standar_transform = transforms.Compose([
            transforms.Resize(img_size),
            # transforms.ToTensor(),   # 掩码图不能做归一化，希望他的tensor是[0, 255]de，而不是[0, 1]
        ])

        if hflip_index:
            seg_label_ = tf.hflip(seg_label_)
        seg_label_ = mask_standar_transform(seg_label_)
        seg_label_ = torch.from_numpy(np.array(seg_label_))

        return seg_label_


if __name__ == '__main__':

    """
    data_train : normMean = [0.37818605 0.27225745 0.15962803]
    data_train : normstdevs = [0.16118085 0.13099103 0.1116703 ]
    """

    # 88
    image_folder = '/home/huangjq/PyCharmCode/project/from_yolo/data_train'

    transform = ImgMaskTransform()
    dataset = Metric_Learning_ImageFolder(root=image_folder, transform=transform)

    tr_loader = DATA.DataLoader(dataset,
                                drop_last=True,
                                batch_size=16,
                                shuffle=True,
                                num_workers=16)
    for epoch in range(1):
        for step, data in enumerate(tr_loader):
            print('*' * 15, step, '*' * 15)
            print('img_name:', data['name'])
            print('img:', data['img'].shape)
            print('cla_label:', data['cla_label'])
            print('seg_label_mask:\n', data['seg_label_mask'])
            print('is_hflip:\n', data['is_hflip'])

            seg_label, em_mask = dataset.get_seg_label_batch(img_size=(data['img'].shape[2], data['img'].shape[3]),
                                                             mask_index=data['seg_label_mask'],
                                                             hflip=data['is_hflip'])
            if seg_label != None:
                print(seg_label.shape, em_mask)

            break
