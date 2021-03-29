import numpy as np
import random
import torch
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable


class DiscriminativeLoss_wizaron(_Loss):
    def __init__(self, num_instance=5, delta_v=0.5, delta_d=1.5, norm=2, scale_var=1.0, scale_dist=1.0, scale_reg=0.001,
                 usegpu=True):
        """
        20210221:https://github.com/Wizaron/instance-segmentation-pytorch
        :param embed_dim: no use in this class function
        :param delta_v:
        :param delta_d:
        :param norm:
        :param scale_var:
        :param scale_dist:
        :param scale_reg:
        """
        super(DiscriminativeLoss_wizaron, self).__init__()
        self.num_instance = num_instance
        self.usegpu = usegpu
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.norm = norm
        self.scale_var = scale_var
        self.scale_dist = scale_dist
        self.scale_reg = scale_reg
        assert self.norm in [1, 2]

    def forward(self, embedding, segLabel):

        if segLabel is not None:
            gt_one, n_objects = self.get_one_hot(segLabel)
            var_loss, dist_loss, reg_loss = self._discriminative_loss(embedding, gt_one, n_objects, self.num_instance)
            # print('var_loss:{} | dist_loss:{} | reg_loss:{}'.format(var_loss, dist_loss, reg_loss))
        else:
            var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
            dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
            reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        loss = var_loss * self.scale_var + dist_loss * self.scale_dist + reg_loss * self.scale_reg

        output = {
            "loss_var": var_loss,
            "loss_dist": dist_loss,
            "loss_reg": reg_loss,
            "loss": loss
        }
        return output

    def _discriminative_loss(self, input, target, n_objects, max_n_objects):
        """input: bs, n_filters, fmap, fmap
           target: bs, n_instances, fmap, fmap
           n_objects: bs
        """

        bs, n_filters, height, width = input.size()
        n_instances = target.size(1)

        input = input.permute(0, 2, 3, 1).contiguous().view(
            bs, height * width, n_filters)
        target = target.permute(0, 2, 3, 1).contiguous().view(
            bs, height * width, n_instances)

        cluster_means = self.calculate_means(
            input, target, n_objects, max_n_objects, self.usegpu)

        var_term = self.calculate_variance_term(
            input, target, cluster_means, n_objects, self.delta_v, self.norm)
        dist_term = self.calculate_distance_term(
            cluster_means, n_objects, self.delta_d, self.norm, self.usegpu)
        reg_term = self.calculate_regularization_term(cluster_means, n_objects, self.norm)

        return var_term, dist_term, reg_term

    def calculate_means(self, pred, gt, n_objects, max_n_objects, usegpu):
        """
        :param pred: bs, height * width, n_filters
        :param gt: bs, height * width, n_instances
        :param n_objects: [_, _, ..., _]  这个list的长度取决于batchsize，list里面的数值取决于某一个batch里面的实例种类数
        :param max_n_objects:
        :param usegpu:
        :return:
        """

        bs, n_loc, n_filters = pred.size()
        n_instances = gt.size(2)

        pred_repeated = pred.unsqueeze(2).expand(
            bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
        # bs, n_loc, n_instances, 1
        gt_expanded = gt.unsqueeze(3)

        pred_masked = pred_repeated * gt_expanded

        means = []
        for i in range(bs):
            _n_objects_sample = n_objects[i]
            # n_loc, n_objects, n_filters
            _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
            # n_loc, n_objects, 1
            _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]

            _mean_sample = _pred_masked_sample.sum(
                0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
            if (max_n_objects - _n_objects_sample) != 0:
                n_fill_objects = int(max_n_objects - _n_objects_sample)
                _fill_sample = torch.zeros(n_fill_objects, n_filters)
                if usegpu:
                    _fill_sample = _fill_sample.cuda()
                _fill_sample = Variable(_fill_sample)
                _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
            means.append(_mean_sample)

        means = torch.stack(means)

        # means = pred_masked.sum(1) / gt_expanded.sum(1)
        # # bs, n_instances, n_filters

        return means

    def calculate_variance_term(self, pred, gt, means, n_objects, delta_v, norm=2):
        """pred: bs, height * width, n_filters
           gt: bs, height * width, n_instances
           means: bs, n_instances, n_filters"""

        bs, n_loc, n_filters = pred.size()
        n_instances = gt.size(2)

        # bs, n_loc, n_instances, n_filters
        means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
        # bs, n_loc, n_instances, n_filters
        pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
        # bs, n_loc, n_instances, n_filters
        gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)

        _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
                            delta_v, min=0.0) ** 2) * gt[:, :, :, 0]

        var_term = 0.0
        for i in range(bs):
            _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
            _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

            var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
        var_term = var_term / bs

        return var_term

    def calculate_distance_term(self, means, n_objects, delta_d, norm=2, usegpu=True):
        """means: bs, n_instances, n_filters"""

        bs, n_instances, n_filters = means.size()

        dist_term = 0.0
        for i in range(bs):
            _n_objects_sample = int(n_objects[i])

            if _n_objects_sample <= 1:
                continue

            _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
            means_1 = _mean_sample.unsqueeze(1).expand(
                _n_objects_sample, _n_objects_sample, n_filters)
            means_2 = means_1.permute(1, 0, 2)

            diff = means_1 - means_2  # n_objects, n_objects, n_filters

            _norm = torch.norm(diff, norm, 2)

            margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
            if usegpu:
                margin = margin.cuda()
            margin = Variable(margin)

            _dist_term_sample = torch.sum(
                torch.clamp(margin - _norm, min=0.0) ** 2)
            _dist_term_sample = _dist_term_sample / \
                                (_n_objects_sample * (_n_objects_sample - 1))
            dist_term += _dist_term_sample

        dist_term = dist_term / bs

        return dist_term

    def calculate_regularization_term(self, means, n_objects, norm):
        """means: bs, n_instances, n_filters"""

        bs, n_instances, n_filters = means.size()

        reg_term = 0.0
        for i in range(bs):
            _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
            _norm = torch.norm(_mean_sample, norm, 1)
            reg_term += torch.mean(_norm)
        reg_term = reg_term / bs

        return reg_term

    def to_one_hot(self, mask):
        """
        Transform a mask to one hot
        change a mask to n * h* w   n is the class
        Args:
            mask:
            n_class: number of class for segmentation
        Returns:
            y_one_hot: one hot mask
        """
        y_one_hot = torch.zeros((self.num_instance, mask.shape[1], mask.shape[2]))
        y_one_hot = y_one_hot.scatter(0, mask, 1).long()
        return y_one_hot

    def get_one_hot(self, label):
        """
        这里的效率可能有问题，tensor从gpu跑到cpu
         将单通道的掩码图转成实例数通道数的掩码图
         :param label: [bs, h, w]
         :param N: num instance
         :return: [bs, num_instance, h, w], 每批数据拥有的实例种类数
         """
        n_objects = []
        ones = []
        for i in label:
            seg_labels = self.to_one_hot(torch.unsqueeze(i, dim=0).cpu().long())
            ones.append(seg_labels)
            n_objects.append(len(torch.unique(i)))

        return torch.stack(ones, dim=0).cuda(), n_objects

    # def get_one_hot(self, label):
    #     """
    #     这里的效率可能有问题，tensor从gpu跑到cpu
    #      将单通道的掩码图转成实例数通道数的掩码图
    #      :param label: [bs, h, w]
    #      :param N: num instance
    #      :return: [bs, num_instance, h, w], 每批数据拥有的实例种类数
    #      """
    #     ones = []
    #     n_objects = []
    #     for i in label:
    #         n_objects.append(len(torch.unique(i)))
    #         print('i:', i.shape)
    #         print(torch.unique(i))
    #         seg_labels = torch.eye(self.num_instance)[i.view([-1])]
    #         seg_labels = seg_labels.reshape((int(i.shape[0]), int(i.shape[1]), self.num_instance))
    #         # print('seg_labels', seg_labels)
    #         # print(seg_labels.argmax(-1) == i) # 判断转换的one hot矩阵是否有误，没错的话全都是true
    #         ones.append(seg_labels.permute(2, 0, 1))
    #
    #     return torch.stack(ones, dim=0).cuda(), n_objects


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(42)

    em_dim = 2
    loss = DiscriminativeLoss_wizaron(num_instance=5,
                                      delta_v=0.5,
                                      delta_d=1.5,
                                      norm=2,
                                      scale_var=1.0,
                                      scale_dist=1.0,
                                      scale_reg=0.001,
                                      usegpu=True).cuda()

    pre = torch.rand([3, 5, 10, 10])
    em = torch.rand([3, em_dim, 10, 10])
    seg_gt = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                           [4, 4, 3, 3, 3, 3, 3, 3, 3, 3],
                           [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                           [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                           ])
    segLabel = torch.stack([seg_gt, seg_gt, seg_gt], dim=0)  # torch.Size([3, 10, 10])

    output = loss(em.cuda(), segLabel.cuda())

    print(output['loss'])
    print(output['loss_var'])
    print(output['loss_dist'])
    print(output)
