import torch
import torch.nn as nn

from models.networks.extractors import resnet50
from models.networks.auxiliary_module import MyUpsample


# ########## 说明 ################
# 将特征提取，上采样，分类器拼接起来
# ################################

class MyNetworks(nn.Module):
    def __init__(
            self, n_instance=5, n_classes=5, embed_dim=4, branch_size=256,
            deep_features_size=2048, backend='resnet50',
            pretrained=False, model_path='the path of ImageNet_weights'):
        super(MyNetworks, self).__init__()
        self.n_instance = n_instance
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.branch_size = branch_size
        self.deep_features_size = deep_features_size
        self.backend = backend
        self.pretrained = pretrained
        self.model_path = model_path

        self.extractors = resnet50(self.pretrained, self.model_path)
        self.upsample = MyUpsample(bilinear=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
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
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, self.embed_dim, 1)
        )

        # ----------------- binary segmentation -----------------
        self.segmenting = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, self.n_instance, 1)
        )

    def forward(self, x):
        # feature extractor
        # torch.Size([1, 256, 64, 128]) torch.Size([1, 512, 32, 64]) torch.Size([1, 1024, 16, 32]) torch.Size([1, 2048, 8, 16])
        x1, x2, x3, x4 = self.extractors(x)
        # upsample
        x_up = self.upsample(x1, x2, x3, x4)

        # 具体任务：分割/嵌入
        y_seg = self.segmenting(x_up)
        y_em = self.embedding(x_up)

        y_cla = self.avgpool(x4)
        y_cla = torch.flatten(y_cla, 1)
        y_cla = self.classifier(y_cla)

        return y_cla, y_seg, y_em

    def network_name(self):
        return 'ResNet+UNet'


if __name__ == '__main__':
    net = MyNetworks(n_instance=5,
                     n_classes=5, embed_dim=2,
                     branch_size=1024,
                     deep_features_size=2048,
                     backend='resnet50',
                     pretrained=False,
                     model_path='model_weight_path').cuda()
    img = torch.rand([7, 3, 256, 512])
    segLabel = torch.rand([7, 1, 256, 512])
    cla, seg, emb = net(img.cuda())
    print(cla.shape, seg.shape, emb.shape)
