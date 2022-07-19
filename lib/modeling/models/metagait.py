import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper

import math

from einops import rearrange, reduce, repeat

def conv3d_sample_by_sample(x, weight, oup, inp, ksize, stride,padding,groups):
    batch_size = x.shape[0]
    if batch_size == 1:
        out = F.conv3d(
            x,
            weight=weight.view(oup, inp, ksize[0], ksize[1],ksize[2]),
            stride=stride,
            padding=padding,
            groups=groups,
        )
    else:

        out = F.conv3d(
            x.view(1, -1, x.shape[2], x.shape[3],x.shape[4]),
            weight.view(batch_size * oup, inp, ksize[0], ksize[1],ksize[2]),
            stride=stride,
            padding=padding,
            groups=groups * batch_size,
        )
        out = out.view(batch_size, oup, out.shape[2], out.shape[3], out.shape[4])
    return out

class MHN(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride,padding, M=2, G=2):
        super().__init__()
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.oup = oup
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding

        self.wn_fc1 = nn.Conv3d(inp_gap, M * oup, 1, 1, 0, groups=1, bias=True)

        self.wn_fc2 = nn.Conv3d(
            M * oup, oup * inp * kernel_size[0] * kernel_size[1] * kernel_size[2], 1, 1, 0, groups=G * oup, bias=False
        )
        self.reduce = nn.Conv3d(inp, max(16, inp // 16), kernel_size=1)

    def forward(self, x):
        x_gap = self.reduce(x.mean(dim=[2,3,4], keepdim=True))
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
     
        return conv3d_sample_by_sample(
           x, x_w, self.oup, self.inp, self.ksize, self.stride, self.padding, 1
        )

class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x):
        # n c s h w
        y = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)

        y = self.fc(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * y

class LocalAttention(nn.Module):
 
    def __init__(self, channel, k_size=3, gamma=2, b=1):
        super(LocalAttention, self).__init__()
        
        t = int(abs((math.log(channel,2)+b)/gamma))
        k_size = t if t%2 else t+1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, padding=(1 - 1) // 2, bias=False) 
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.conv5 = nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dynamic_conv = MHN(128, 3, (1,1,1), (1,1,1), (0,0,0))
    def forward(self, x):
        # feature descriptor on the global spatial information
        # n, c, s, h, w = x.size()
        # y = rearrange(x, 'n c s h w -> n s c h w')
        # n c s
        weight = self.avg_pool(x)
        weight = self.dynamic_conv(weight)
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = rearrange(y, 'n c s -> n s c')

        # Two different branches of ECA module
        y1 = self.conv1(y)
        y3 = self.conv3(y)
        y5 = self.conv5(y)


        y1 = rearrange(y1, 'n s c -> n c s').unsqueeze(-1).unsqueeze(-1)
        y3 = rearrange(y3, 'n s c -> n c s').unsqueeze(-1).unsqueeze(-1)
        y5 = rearrange(y5, 'n s c -> n c s').unsqueeze(-1).unsqueeze(-1)

        

         # n c s
        # Multi-scale information fusion
        y1 = self.sigmoid(y1)
        y3 = self.sigmoid(y3)
        y5 = self.sigmoid(y5)

        weight1 = weight[:,0,:,:,:].unsqueeze(1)
        weight2 = weight[:,1,:,:,:].unsqueeze(1)
        weight3 = weight[:,2,:,:,:].unsqueeze(1)

        return x * y1 * weight1 + x * y5 * weight2 + x * y3 * weight3

        


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class MetaGait(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(MetaGait, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )
        self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.MaxPool0 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])
        self.Bn = nn.BatchNorm1d(in_c[-1])
        self.Head1 = SeparateFCs(64, in_c[-1], class_num)
        self.eca = LocalAttention(128)
        self.se = SELayer(128)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]
        outs1 = self.eca(outs)
        outs2 = self.se(outs)
        outs = outs1 + outs2

        outs = self.TP(outs, dim=2, seq_dim=2, seqL=seqL)[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]
        outs = outs.permute(2, 0, 1).contiguous()  # [p, n, c]

        gait = self.Head0(outs)  # [p, n, c]
        gait = gait.permute(1, 2, 0).contiguous()  # [n, c, p]
        bnft = self.Bn(gait)  # [n, c, p]
        logi = self.Head1(bnft.permute(2, 0, 1).contiguous())  # [p, n, c]

        gait = gait.permute(0, 2, 1).contiguous()  # [n, p, c]
        bnft = bnft.permute(0, 2, 1).contiguous()  # [n, p, c]
        logi = logi.permute(1, 0, 2).contiguous()  # [n, p, c]

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': bnft, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': bnft
            }
        }
        return retval
