import torch
from utils.wheels import pre2mask


def caculate_all_loss(epoch,
                      using_pseudo_mask,
                      seg_label,
                      cla_label,
                      seg_pred,
                      seg_pred_after_mask,
                      em_after_mask,
                      cla_pred_y,
                      disc_loss_func,
                      seg_loss_func,
                      cla_loss_func):
    """

    :param epoch: 当前的epoch
    :param using_pseudo_mask: 是否使用伪标签
    :param seg_label: 真实分割标签
    :param cla_label: 真实分类标签
    :param seg_pred: 所有预测分割结果
    :param seg_pred_after_mask:部分预测分割结果，剔除了没有真实分割标签的预测分割结果
    :param em_after_mask: 部分预测嵌入结果，剔除了没有真实分割标签的预测嵌入结果
    :param cla_pred_y: 预测分类结果
    :param disc_loss_func:
    :param seg_loss_func:
    :param cla_loss_func:
    :return: 分割损失，分类损失，判别式损失，有掩码图的数量
    """
    num_seg_label = 0

    if not seg_label is None:  # 如果seg_label不是None，则执行
        num_seg_label = seg_label.shape[0]  # 他是计算判别式损失的计数器

    # #############################
    # 只有lovasz_softmax需要用到伪标签，判别式损失不需要，因此只有lovasz_softmax需要做判断
    # 共有三个判别条件：seg_label是否存在；是否要使用pseudo_label；epoch是否大于阈值
    # #############################
    seg_loss = torch.tensor(0, dtype=seg_pred.dtype, device=seg_pred.device)
    if not seg_label is None:
        # print('seg_label is not none')
        # print(seg_pred_after_mask.dtype, seg_label.dtype)  # torch.float32 torch.uint8
        # print(seg_label.to(dtype=torch.int64).dtype)   # torch.int64
        # print(torch.unique(seg_label))
        # print(seg_pred_after_mask.shape, seg_label.shape)
        seg_loss = seg_loss_func(seg_pred_after_mask, seg_label.to(dtype=torch.int64))
    else:
        # print('seg_label is none')
        if using_pseudo_mask and (epoch + 1) > 150:
            # print('using pseudo_label.')
            pseudo_label = pre2mask(seg_pred)  # 这个函数应该还可以优化：通过与各的概率图获得伪标签
            seg_loss = seg_loss_func(seg_pred, pseudo_label)

    dict_loss = disc_loss_func(embedding=em_after_mask, segLabel=seg_label)
    # print(cla_pred_y.shape, cla_label.cuda().shape)
    c_loss = cla_loss_func(cla_pred_y, cla_label.cuda())

    return seg_loss, c_loss, dict_loss, num_seg_label
