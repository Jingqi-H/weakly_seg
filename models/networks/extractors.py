import torch.nn as nn
import torch
from torchvision.models import resnet50


# ########## 说明 ################
# 被注释掉的代码是会议投稿论文用到的特征提取器代码，resnet部分的makelayer有点疑问
# 打算用的这个代码是b站作者霹雳吧啦的
# 这个做实验中的特征提取器
# ################################

class BasicBlock(nn.Module):
    # 针对18层和34层的残差结构
    expansion = 1  # 每个残差层内部的卷积核关系

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 若下采样是none的话，对应的是实线的残差结构，不执行下面的
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 针对50层、101和152层的残差结构
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    block: 选择是哪一个类型的残差结构，有BasicBlock，Bottleneck两种，分别对应两种类别的残差结构
    blocks_num：使用的残差结构的数目，是个列表，比如34层的，对应的是[3, 4, 6, 3]
    num_classes=1000：分类个数
    include_top=True
    """

    def __init__(self, block, blocks_num):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        x1 = self.layer1(x)
        # print(x.shape)
        x2 = self.layer2(x1)
        # print(x.shape)
        x3 = self.layer3(x2)
        # print(x.shape)
        x4 = self.layer4(x3)
        # print(x.shape)

        return x1, x2, x3, x4


def resnet50(pre_trained=False, model_weight_path='./path/resnet50.pth'):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pre_trained:
        model.load_state_dict(torch.load(model_weight_path), strict=False)
    return model


if __name__ == '__main__':
    net = resnet50(pre_trained=False)
    for param in net.layer1.parameters():
        param.requires_grad = False
    for p in net.layer2.parameters():  # 将fine-tuning 的参数的 requires_grad 设置为 True
        p.requires_grad = False
    for p in net.layer3.parameters():  # 将fine-tuning 的参数的 requires_grad 设置为 True
        p.requires_grad = False
    for p in net.layer4.parameters():  # 将fine-tuning 的参数的 requires_grad 设置为 True
        p.requires_grad = False
    net.cuda()
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    a = torch.rand([1, 3, 256, 512])
    b1, b2, b3, b4 = net(a.cuda())
    print(b1.shape, b2.shape, b3.shape, b4.shape)