from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as DATA
import os
import torch
from torchvision import transforms, datasets


# 自定义添加椒盐噪声的 transform
class AddPepperNoise(object):
    """
    增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate信噪比,是衡量噪声的比例，图像中正常像素占全部像素的占比。
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.snr
            # 椒盐噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.snr)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 椒噪声和盐噪声分别占 noise_pct 的一半
            # 1 为盐噪声，2 为 椒噪声
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            # 再转换为 image
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


self_train_transform = transforms.Compose([
    # transforms.RandomAffine(degrees=10,             # 旋转角度
    #                         translate=(0, 0.2),     # 水平偏移
    #                         scale=(0.9, 1),
    #                         shear=(6, 9),           # 裁剪
    #                         fillcolor=0),           # 图像外部填充颜色 int
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    AddPepperNoise(0.95, p=0.5),
    # transforms.Resize((300, 900)),
    # transforms.Resize((256, 512)),
    # transforms.Resize((100, 200)),
    transforms.RandomResizedCrop((100, 200)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

self_test_transform = transforms.Compose([
    transforms.Resize((100, 200)),
    transforms.ToTensor(),
    # transforms.Normalize([0.37818605, 0.27225745, 0.15962803], [0.16118085, 0.13099103, 0.1116703])
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

liner_classifier_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    AddPepperNoise(0.95, p=0.5),
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    # transforms.Normalize([0.37818605, 0.27225745, 0.15962803], [0.16118085, 0.13099103, 0.1116703])
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

liner_classifier_test_transform = transforms.Compose([
    transforms.Resize((128, 256)),
    transforms.ToTensor(),
    # transforms.Normalize([0.37818605, 0.27225745, 0.15962803], [0.16118085, 0.13099103, 0.1116703])
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

if __name__ == '__main__':

    # set_seed(1)  # 设置随机种子

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),

        # transforms.ColorJitter(brightness=0, contrast=0.5, hue=0),
        # transforms.RandomAffine(degrees=0,  # 旋转角度
        #                         translate=(0, 0.2),  # 水平偏移
        #                         scale=(0.9, 1),
        #                         shear=(6, 9),  # 裁剪
        #                         fillcolor=0),  # 图像外部填充颜色 int
        transforms.Resize((256, 512)),
        AddPepperNoise(0.98, p=0.5),
        transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v12/700_2100'

    datasets = datasets.ImageFolder(root=data_path_train,
                                    transform=data_transform)

    dataloader = DATA.DataLoader(datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4)

    for epoch in range(30):
        for step, data in enumerate(dataloader, start=0):
            images, labels = data
            #         print(images.shape)
            #         im = imgtensor2im(images)
            #         print(images.shape)
            #         print(labels)
            plt.imshow(transforms.ToPILImage()(images[0]))
            plt.savefig(os.path.join('./test', str(epoch + 1)))
            break
    #     break

    # print(transform)
    # for i in range(10):
    #     f = plt.imshow(transform(img))
    #     plt.show()
