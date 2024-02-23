import torch.nn as nn
from retinanet.utils import conv1x1, conv3x3

class PyramidFeatureNetwork(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatureNetwork, self).__init__()

        ###################################################################
        # TODO: Please substitute the "?" with specific numbers
        ##################################################################

# upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest') #scale_factor为缩放大小
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        
#         # upsample C5 to get P5 from the FPN paper
#         self.P5_1 = conv1x1(C5_size, feature_size)
#         # self.P5_upsampled = nn.Upsample(scale_factor="?", mode='nearest')
#         self.P5_upsampled = nn.Upsample(scale_factor="2", mode='nearest')
#         self.P5_2 = conv3x3(feature_size, feature_size, stride="1")
#         # self.P5_2 = conv3x3(feature_size, feature_size, stride="?")

#         # add P5 elementwise to C4
#         self.P4_1 = conv1x1(C4_size, feature_size)
#         self.P4_upsampled = nn.Upsample(scale_factor="2", mode='nearest')
#         self.P4_2 = conv3x3(feature_size, feature_size, stride="1")

#         # add P4 elementwise to C3
#         self.P3_1 = conv1x1(C3_size, feature_size)
#         self.P3_2 = conv3x3(feature_size, feature_size, stride="1")

#         # "P6 is obtained via a 3x3 stride-2 conv on C5"
#         self.P6 = conv3x3(C5_size, feature_size, stride="2")

#         # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
#         self.P7_1 = nn.ReLU()
#         self.P7_2 = conv3x3(feature_size, feature_size, stride="2")

# #     self.P5_1使用1x1卷积将C5的通道数降维到特征尺寸（feature_size）。

# # self.P5_upsampled使用上采样来放大P5的特征图。这里的scale_factor应该是2，因为在特征金字塔中，每一层的分辨率是上一层的一半。

# # self.P5_2使用3x3卷积进一步处理P5的特征图，stride应该是1，因为我们不需要改变特征图的尺寸。

# # self.P4_1使用1x1卷积将C4的通道数降维到特征尺寸。

# # self.P4_upsampled使用上采样来放大P4的特征图，scale_factor同样是2。

# # self.P4_2使用3x3卷积进一步处理P4的特征图，stride是1。

# # self.P3_1使用1x1卷积将C3的通道数降维到特征尺寸。

# # self.P3_2使用3x3卷积处理P3的特征图，stride是1。

# # self.P6是直接在C5上使用3x3卷积得到的，其stride应该是2，用来降低分辨率。

# # self.P7_1是一个ReLU激活函数。

# # self.P7_2在P6上使用3x3卷积得到P7，其stride也是2。

# # 在填写TODO的地方，你需要根据FPN的结构来指定正确的scale_factor和stride：

# # scale_factor是上采样倍数，根据FPN的结构通常是2。
# # stride是卷积操作中的步长，用于降低特征图的分辨率，通常在金字塔的下一层是上一层的1/2，所以stride通常是2。

        ##################################################################

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
