"""
sorces:
https://github.com/LeeJunHyun/Image_Segmentation
https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
"""
import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_conv2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.upscale_factor = 2
        self.up = nn.Sequential(
            nn.PixelShuffle(upscale_factor=self.upscale_factor),
            nn.ConvTranspose2d(
                ch_in // self.upscale_factor**2,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(ch_out),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class Recurrent_block2(nn.Module):
    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.GELU(),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t), Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class RRCNN_block2(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block2(ch_out, t=t), Recurrent_block2(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Attention_block2(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.act = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        # x4 ([1, 512, 32, 36])
        # d5 [1, 512, 32, 36])
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class U_Net2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        """
                | Name                | Type                                  | Params
        ------------------------------------------------------------------------------
        0 | criterionCycle      | L1Loss                                | 0
        1 | criterionGAN        | MSELoss                               | 0
        2 | criterionCanny      | CannyLoss                             | 0
        3 | criterionVgg        | LearnedPerceptualImagePatchSimilarity | 2.5 M
        4 | criterionClothMask  | L1Loss                                | 0
        5 | model_generator     | U_Net2                                | 29.8 M
        6 | model_discriminator | ACGPNDiscriminator                    | 2.8 M
        7 | last_activation     | Hardtanh                              | 0
        ------------------------------------------------------------------------------
        32.6 M    Trainable params
        2.5 M     Non-trainable params
        35.1 M    Total params
        70.108    Total estimated model params size (MB)
        """
        super().__init__()

        self.Maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.Conv1 = conv_block2(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block2(ch_in=64, ch_out=128)
        self.Conv3 = conv_block2(ch_in=128, ch_out=256)
        self.Conv4 = conv_block2(ch_in=256, ch_out=512)
        self.Conv5 = conv_block2(ch_in=512, ch_out=1024)

        self.Up5 = up_conv2(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block2(ch_in=1024, ch_out=512)

        self.Up4 = up_conv2(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block2(ch_in=512, ch_out=256)

        self.Up3 = up_conv2(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block2(ch_in=256, ch_out=128)

        self.Up2 = up_conv2(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block2(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.dropout(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.dropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.dropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.dropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        # x4 ([1, 512, 32, 36])
        # x5 ([1, 512, 16, 18])
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # d1 = self.last_activation(d1)

        return d1


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super().__init__()

        self.Maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.RRCNN1 = RRCNN_block2(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block2(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block2(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block2(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block2(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv2(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block2(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv2(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block2(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv2(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block2(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv2(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block2(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return d1


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.Maxpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block2(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block2(ch_in=64, ch_out=128)
        self.Conv3 = conv_block2(ch_in=128, ch_out=256)
        self.Conv4 = conv_block2(ch_in=256, ch_out=512)
        self.Conv5 = conv_block2(ch_in=512, ch_out=1024)

        self.Up5 = up_conv2(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block2(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block2(ch_in=1024, ch_out=512)

        self.Up4 = up_conv2(ch_in=512, ch_out=256)
        self.Att4 = Attention_block2(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block2(ch_in=512, ch_out=256)

        self.Up3 = up_conv2(ch_in=256, ch_out=128)
        self.Att3 = Attention_block2(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block2(ch_in=256, ch_out=128)

        self.Up2 = up_conv2(ch_in=128, ch_out=64)
        self.Att2 = Attention_block2(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block2(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # x 3

        x2 = self.Maxpool(x1)  # x1 64
        x2 = self.Conv2(x2)  # x2 64
        x2 = self.dropout(x2)  # x2 128

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.dropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.dropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.dropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super().__init__()

        self.Maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block2(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block2(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block2(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block2(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block2(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv2(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block2(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block2(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv2(ch_in=512, ch_out=256)
        self.Att4 = Attention_block2(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block2(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv2(ch_in=256, ch_out=128)
        self.Att3 = Attention_block2(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block2(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv2(ch_in=128, ch_out=64)
        self.Att2 = Attention_block2(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block2(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return d1


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        reps,
        strides=1,
        start_with_relu=True,
        grow_first=True,
    ):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(
                in_filters, out_filters, 1, stride=strides, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []

        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(
                SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False)
            )
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class ACGPNDiscriminator(nn.Module):
    def __init__(self, in_channels=21):
        super().__init__()
        self.in_channels = in_channels

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class ACGPNDiscriminatorSingleImage(nn.Module):
    def __init__(self, in_channels=21):
        super().__init__()
        self.in_channels = in_channels

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_input):
        return self.model(img_input)


if __name__ == "__main__":
    input_tensor = torch.rand(1, 3, 512, 512)
    model = AttU_Net2()
    output_tensor = model.forward(input_tensor)
    print(output_tensor)
