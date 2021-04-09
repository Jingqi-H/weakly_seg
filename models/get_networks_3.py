import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.extractors import resnet50


# ########## 说明 ################
# 将特征提取，上采样，分类器拼接起来
# 用了博客里的代码，删掉了注意力模块： https://blog.csdn.net/qq_21997625/article/details/86572942
#
# ################################


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        # self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, e=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            # print('e', e.shape)
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv(x), inplace=True)
        # print('x_new',x.size())
        return x


class MyNetworks3(nn.Module):
    def __init__(
            self, final_dim=8, n_instance=5, n_classes=5, embed_dim=4, branch_size=256,
            deep_features_size=2048, backend='resnet50',
            pretrained=False, model_path='the path of ImageNet_weights'):
        """
        decoder from https://blog.csdn.net/qq_21997625/article/details/86572942
        github project:https://github.com/liaopeiyuan/ml-arsenal-public/blob/master/models/TGS_salt/SEResnextUnet50_OC_scSE_hyper.py
        :param n_instance:
        :param n_classes:
        :param embed_dim:
        :param branch_size:
        :param deep_features_size:
        :param backend:
        :param pretrained:
        :param model_path:
        """
        super(MyNetworks3, self).__init__()
        self.n_instance = n_instance
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.branch_size = branch_size
        self.deep_features_size = deep_features_size
        self.backend = backend
        self.pretrained = pretrained
        self.model_path = model_path
        self.final_dim = final_dim

        self.extractors = resnet50(self.pretrained, self.model_path)

        self.conv1 = nn.Sequential(
            self.extractors.conv1,
            self.extractors.bn1,
            self.extractors.relu,
        )
        self.encoder2 = self.extractors.layer1  # 256
        self.encoder3 = self.extractors.layer2  # 512
        self.encoder4 = self.extractors.layer3  # 1024
        self.encoder5 = self.extractors.layer4  # 2048

        self.center = nn.Sequential(
            # ConvBn2d(2048, 2048, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            ConvBn2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(1024 + 2048, 2048)
        self.decoder4 = Decoder(2048 + 1024, 1024)
        self.decoder3 = Decoder(1024 + 512, 512)
        self.decoder2 = Decoder(512 + 256, 256)
        # self.decoder1 = Decoder(256 + 64, 64)
        self.decoder1 = Decoder(256, 64)

        self.logit = nn.Sequential(
            # nn.Conv2d(1344, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Conv2d(256, 128, 3, padding=4, dilation=4, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Conv2d(128, 64, 3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.final_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.final_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.deep_features_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.n_classes),
        )

        # ----------------- embedding -----------------
        self.embedding = nn.Sequential(
            nn.Conv2d(self.final_dim, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, self.embed_dim, 1)
        )

        # ----------------- binary segmentation -----------------
        self.segmenting = nn.Sequential(
            nn.Conv2d(self.final_dim, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, self.n_instance, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

    def forward(self, x):
        e1 = self.conv1(x)  # 64
        # print('e1:', e1.shape)
        e2 = self.encoder2(e1)  # 256
        # print('e2:', e2.shape)
        e3 = self.encoder3(e2)  # 512
        # print('e3:', e3.shape)
        e4 = self.encoder4(e3)
        # print('e4:', e4.shape)
        e5 = self.encoder5(e4)
        # print('e5:', e5.shape)

        f = self.center(e5)
        # print('f:', f.shape)
        d5 = self.decoder5(f, e5)
        # print('d5:', d5.shape)
        d4 = self.decoder4(d5, e4)
        # print('d4:', d4.shape)
        d3 = self.decoder3(d4, e3)
        # print('d3:', d3.shape)
        d2 = self.decoder2(d3, e2)
        # print('d2:', d2.shape)
        d1 = self.decoder1(d2)
        # print('d1:', d1.shape)
        # f = torch.cat((
        #             F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=False),
        #             d1,
        #             F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
        #             F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
        #             F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
        #             F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        #         ), 1)
        # print(f.shape)
        # logit = self.logit(f)

        logit = self.logit(d1)
        # # 具体任务：分割/嵌入
        y_seg = self.segmenting(logit)
        y_em = self.embedding(logit)

        y_cla = self.avgpool(e5)
        y_cla = torch.flatten(y_cla, 1)
        y_cla = self.classifier(y_cla)

        return y_cla, y_seg, y_em

    def network_name(self):
        return 'ResNet+UNet+more_tricks'


if __name__ == '__main__':
    net = MyNetworks3(n_instance=5,
                      n_classes=5, embed_dim=2,
                      branch_size=1024,
                      deep_features_size=2048,
                      backend='resnet50',
                      pretrained=False,
                      model_path='model_weight_path').cuda()
    img = torch.rand([7, 3, 128, 256])
    segLabel = torch.rand([7, 1, 128, 256])
    cla, seg, emb = net(img.cuda())
    print(cla.shape, seg.shape, emb.shape)

    # e1: torch.Size([7, 64, 64, 128])
    # e2: torch.Size([7, 256, 64, 128])
    # e3: torch.Size([7, 512, 32, 64])
    # e4: torch.Size([7, 1024, 16, 32])
    # e5: torch.Size([7, 2048, 8, 16])
    # f: torch.Size([7, 1024, 4, 8])
    # d5: torch.Size([7, 256, 8, 16])
    # d4: torch.Size([7, 256, 16, 32])
    # d3: torch.Size([7, 256, 32, 64])
    # d2: torch.Size([7, 256, 64, 128])
    # d1: torch.Size([7, 256, 128, 256])
    # torch.Size([7, 5]) torch.Size([7, 5, 128, 256]) torch.Size([7, 2, 128, 256])
