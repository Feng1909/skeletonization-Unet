import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers

class DoubleConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AttentionGroup(nn.Layer):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = nn.Conv2D(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2D(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2D(num_channels, 3, kernel_size=1)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        # s = paddle.nn.Softmax(self.conv_1x1(x), axis=1)
        s = self.softmax(self.conv_1x1(x))

        att = s[:,0,:,:].unsqueeze(1) * x1 + s[:,1,:,:].unsqueeze(1) * x2 \
            + s[:,2,:,:].unsqueeze(1) * x3

        return x + att

class UNet(nn.Layer):
    """
    The UNet implementation based on PaddlePaddle.
    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).
    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
                 align_corners=False,
                 use_deconv=False,
                 in_channels=3,
                 pretrained=None):
        super().__init__()

        self.encode = Encoder(in_channels)
        self.decode = Decoder()

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        # x, short_cuts = self.encode(x)
        # x = self.decode(x, short_cuts)
        out1, out2, out3, out4, x = self.encode(x)
        x, aux_128, aux_64, aux_32 = self.decode(out1, out2, out3, out4, x)

        return x.squeeze(), aux_128.squeeze(), aux_64.squeeze(), aux_32.squeeze()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

class UpConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.Conv2DTranspose(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveAvgPool2D(1)

        self.fc = nn.Sequential(nn.Conv2D(in_planes, in_planes // 16, 1),
                                nn.ReLU(),
                                nn.Conv2D(in_planes // 16, in_planes, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = DoubleConv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = DoubleConv2d(512, 1024, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2D(kernel_size=2)

        self.att1 = AttentionGroup(64)
        self.att2 = AttentionGroup(128)
        self.att3 = AttentionGroup(256)
        self.att4 = AttentionGroup(512)
        self.att5 = AttentionGroup(1024)
        # self.double_conv = nn.Sequential(
        #     layers.ConvBNReLU(in_channels, 64, 3), layers.ConvBNReLU(64, 64, 3))
        # down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        # self.down_sample_list = nn.LayerList([
        #     self.down_sampling(channel[0], channel[1])
        #     for channel in down_channels
        # ])

    # def down_sampling(self, in_channels, out_channels):
    #     modules = []
    #     modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
    #     modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
    #     modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
    #     return nn.Sequential(*modules)

    def forward(self, x):
        # short_cuts = []
        # x = self.double_conv(x)
        # for down_sample in self.down_sample_list:
        #     short_cuts.append(x)
        #     x = down_sample(x)
        
        out1 = self.conv1(x)
        out1 = self.att1(out1)

        out2 = self.conv2(self.pooling(out1))
        out2 = self.att2(out2)

        out3 = self.conv3(self.pooling(out2))
        out3 = self.att3(out3)

        out4 = self.conv4(self.pooling(out3))
        out4 = self.att4(out4)

        out5 = self.conv5(self.pooling(out4))
        out5 = self.att5(out5)
        return out1, out2, out3, out4, out5


class Decoder(nn.Layer):
    def __init__(self):
        super().__init__()
        self.upconv1 = UpConv2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = UpConv2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = UpConv2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = UpConv2d(128, 64, kernel_size=2, stride=2)

        self.conv1 = DoubleConv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(128, 64, kernel_size=3, padding=1)

        self.conv1x1 = nn.Conv2D(64, 1, kernel_size=1, stride=1, padding=0)
        self.aux_conv_128 = nn.Conv2D(128, 1, kernel_size=1, stride=1, padding=0)
        self.aux_conv_64 = nn.Conv2D(256, 1, kernel_size=1, stride=1, padding=0)
        self.aux_conv_32 = nn.Conv2D(512, 1, kernel_size=1, stride=1, padding=0)

        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()

        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()

        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()

        self.ca4 = ChannelAttention(64)
        self.sa4 = SpatialAttention()
        # up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        # self.up_sample_list = nn.LayerList([
        #     UpSampling(channel[0], channel[1], align_corners, use_deconv)
        #     for channel in up_channels
        # ])

    def forward(self, out1, out2, out3, out4, x):
        # for i in range(len(short_cuts)):
        #     x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        # return x
        x = self.upconv1(x)
        x = paddle.concat([x, out4], axis=1)
        x = self.conv1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        aux_32 = self.aux_conv_32(x)

        x = self.upconv2(x)
        x = paddle.concat([x, out3], axis=1)
        x = self.conv2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        aux_64 = self.aux_conv_64(x)

        x = self.upconv3(x)
        x = paddle.concat([x, out2], axis=1)
        x = self.conv3(x)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        aux_128 = self.aux_conv_128(x)

        x = self.upconv4(x)
        x = paddle.concat([x, out1], axis=1)
        x = self.conv4(x)
        x = self.ca4(x) * x
        x = self.sa4(x) * x
        x = self.conv1x1(x)

        return x, aux_128, aux_64, aux_32


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                paddle.shape(short_cut)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x