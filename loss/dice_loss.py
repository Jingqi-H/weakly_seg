import torch.nn as nn
import torch


# ########## 说明 ################
# 参考PSPNet b站作者的代码，dice loss 的gt是onehot矩阵
# 参考链接：https://www.aiuai.cn/aifarm1159.html
# ################################


# dice系数：https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(pred, target):
    """
    是dice系数，与doceloss不同，但是有一定的相关性
    :param pred:
    :param target:
    :return:
    """
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        """

        :param logits: [bs, num_instance, h, w]
        :param targets: [bs, h, w]，里面的数值是类别索引[0, 1 , ..., num_instance]
        :return: dice loss
        """
        num = targets.size(0)
        targets, _ = self.get_one_hot(targets, logits.shape[1])

        smooth = 1
        probs = torch.sigmoid(logits)

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

    def to_one_hot(self, mask, num_instance):
        """
        Transform a mask to one hot
        change a mask to n * h* w   n is the class
        Args:
            mask:
            n_class: number of class for segmentation
        Returns:
            y_one_hot: one hot mask
        """
        y_one_hot = torch.zeros((num_instance, mask.shape[1], mask.shape[2]))
        y_one_hot = y_one_hot.scatter(0, mask, 1).long()
        return y_one_hot

    def get_one_hot(self, targets, num_instance):
        """
        这里的效率可能有问题，tensor从gpu跑到cpu
         将单通道的掩码图转成实例数通道数的掩码图
         :param label: [bs, h, w]
         :param N: num instance
         :return: [bs, num_instance, h, w], 每批数据拥有的实例种类数
         """
        n_objects = []
        ones = []
        for i in targets:
            seg_labels = self.to_one_hot(torch.unsqueeze(i, dim=0).cpu().long(), num_instance)
            ones.append(seg_labels)
            n_objects.append(len(torch.unique(i)))

        return torch.stack(ones, dim=0).cuda(), n_objects


if __name__ == '__main__':
    gt = torch.rand([1, 5, 4, 4]).cuda()
    pre = torch.randn([1, 5, 4, 4]).cuda()
    _, gt = torch.max(gt, dim=1)
    print(pre.shape)
    print(gt.shape)

    dice_loss = SoftDiceLoss()
    a = dice_loss(pre, gt)
    print(a)
