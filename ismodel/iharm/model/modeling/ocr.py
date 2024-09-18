import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution. #根据HRNet预测的特征分布？？？
        Employ the soft-weighted method to aggregate the context.  ##软加权？
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)  # b c h*w
        feats = feats.view(batch_size, feats.size(1), -1) # b c h*w
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw  # dim=2 参数表示在第二个维度上进行 softmax 操作
        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3).contiguous()  # batch x k x c
        """
        matmul(probs, feats) 概率分布矩阵 probs 与特征矩阵 feats 进行矩阵乘法
        permute(0, 2, 1)：这是张量的维度变换操作，通过 .permute() 方法对张量的维度进行重新排列。
        .unsqueeze() 方法在指定的位置插入一个新的维度。在上一步维度变换的结果上，在第四个维度（索引为3）插入一个新的维度
        .contiguous()：这是张量的连续性操作，通过 .contiguous() 方法确保张量在内存中是连续存储的
        """
        return ocr_context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 norm_layer=nn.BatchNorm2d,
                 align_corners=True):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale,
                                                           norm_layer, align_corners)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.Sequential(norm_layer(out_channels), nn.ReLU(inplace=True)), #norm+ReLU
            nn.Dropout2d(dropout) #对输入进行随机失活操作，以减少模型的过拟合
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class ObjectAttentionBlock2D(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 norm_layer=nn.BatchNorm2d,
                 align_corners=True):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.align_corners = align_corners

        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True))
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True))
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True))
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sequential(norm_layer(self.in_channels), nn.ReLU(inplace=True))
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map ##(self.key_channels ** -.5)计算了一个缩放因子，进行归一化，以平衡相似度矩阵中的数值范围
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        #x.size()[2:] 表示从第三个维度开始（索引为2），取出剩余的维度大小。
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w),
                                    mode='bilinear', align_corners=self.align_corners) #是否对齐插值的角点

        return context
