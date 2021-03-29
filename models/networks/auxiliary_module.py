import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3, 3), padding_size=(1, 1)):
        super(Conv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding=1),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class up_concat(nn.Module):
    def __init__(self, in_size, out_size, bilinear=True):
        super(up_concat, self).__init__()
        # Upsample通过插值方法完成上采样。所以不需要训练参数
        # ConvTranspose2d可以理解为卷积的逆过程。所以可以训练参数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_size, in_size, 2, stride=2)
        self.conv = Conv(in_size + out_size, out_size)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))


class MyUpsample(nn.Module):
    def __init__(self, bilinear=True):
        super(MyUpsample, self).__init__()

        self.up_concat3 = up_concat(2048, 1024, bilinear)
        self.up_concat2 = up_concat(1024, 512, bilinear)
        self.up_concat1 = up_concat(512, 256, bilinear)
        self.final = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

    def forward(self, x_1, x_2, x_3, x_4):
        up3 = self.up_concat3(x_3, x_4)
        up2 = self.up_concat2(x_2, up3)
        up1 = self.up_concat1(x_1, up2)
        final_x = self.final(up1)

        return final_x

if __name__ == '__main__':
    x1, x2, x3, x4 = torch.rand([1, 256, 64, 128]), torch.rand([1, 512, 32, 64]), torch.rand(
        [1, 1024, 16, 32]), torch.rand([1, 2048, 8, 16])
    net = MyUpsample(bilinear=False)
    x = net(x1, x2, x3, x4)
    print(x.shape)
