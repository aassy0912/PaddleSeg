# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.backbones.strideformer import ConvBNAct


@manager.MODELS.add_component
class GCMobileSeg(nn.Layer):
    """
    The GC_MobileSeg implementation based on PP_MobileSeg.

    """

    def __init__(self,
                 num_classes,
                 backbone,
                 head_use_dw=True,
                 align_corners=False,
                 pretrained=None,
                 upsample='intepolate'):
        super().__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.num_classes = num_classes

        self.gcb = GlobalContextBlock(
            in_channels=backbone.feat_channels[0], scale=8)  # test

        self.decode_head = PPMobileSegHead(
            num_classes=num_classes,
            in_channels=backbone.feat_channels[0],
            use_dw=head_use_dw,
            align_corners=align_corners)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_hw = x.shape[2:]
        x = self.backbone(x)

        # 融合 Global Context Block 。
        x = self.gcb(x)

        x = self.decode_head(x)
        if self.upsample == 'intepolate' or self.training or self.num_classes < 30:
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)
        elif self.upsample == 'vim':
            labelset = paddle.unique(paddle.argmax(x, 1))
            x = paddle.gather(x, labelset, axis=1)
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)

            pred = paddle.argmax(x, 1)
            pred_retrieve = paddle.zeros(pred.shape, dtype='int32')
            for i, val in enumerate(labelset):
                pred_retrieve[pred == i] = labelset[i].cast('int32')

            x = pred_retrieve
        else:
            raise NotImplementedError(self.upsample, " is not implemented")

        return [x]

# Copy from pp_mobileseg.py
class PPMobileSegHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 use_dw=False,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.last_channels = in_channels

        self.linear_fuse = ConvBNAct(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1,
            act=nn.ReLU)
        self.dropout = nn.Dropout2D(dropout_ratio)
        self.conv_seg = nn.Conv2D(
            self.last_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x


class GlobalContextBlock(nn.Layer):
    def __init__(self, in_channels, scale=8):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2D(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(axis=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2D(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = paddle.shape(x)

        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).reshape(
            [b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
        query = x.reshape([b, c, h * w])

        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = paddle.matmul(query, key)
        concate_QK = concate_QK.reshape([b, c, 1, 1])

        value = self.Conv_value(concate_QK)
        out = x + value
        return out
