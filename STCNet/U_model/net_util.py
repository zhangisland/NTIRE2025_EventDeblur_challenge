
import torch as th
import torch.nn.functional as F
from torch import nn

from .size_adapter import SizeAdapter
import torch
from .arches import *
from einops import rearrange
import numbers

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out=self.sigmoid(out)
        return out



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1,kernel_size=(3,3), padding=(1,1), bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)

        return out





# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res






class shallow_cell(nn.Module):
    def __init__(self,inChannels):
        super(shallow_cell, self).__init__()
        self.n_feats = 64
        act = nn.ReLU(inplace=True)
        bias = False
        reduction = 4
        self.shallow_feat = nn.Sequential(conv(inChannels, self.n_feats, 3, bias=bias),
                                           CAB(self.n_feats, 3, reduction, bias=bias, act=act))

    def forward(self,x):
        feat = self.shallow_feat(x)
        return feat







class EN_Block(nn.Module):

    def __init__(self, in_planes, planes,kernel_size=3, reduction=4, bias=False):
        super(EN_Block, self).__init__()

        act = nn.ReLU(inplace=True)
        self.down = DownSample(in_planes, planes)
        self.encoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder = nn.Sequential(*self.encoder)


    def forward(self, x):
        x = self.down(x)
        x=self.encoder(x)
        return x


class DE_Block(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, reduction=4, bias=False):
        super(DE_Block, self).__init__()
        act = nn.ReLU(inplace=True)

        self.up=SkipUpSample(in_planes, planes)
        self.decoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder = nn.Sequential(*self.decoder)
        self.skip_attn = CAB(planes, kernel_size, reduction, bias=bias, act=act)


    def forward(self, x, skpCn):

        x = self.up(x, self.skip_attn(skpCn))
        x = self.decoder(x)

        return x


##########################################################################
## Layer Norm
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class Cross_Modal_Aggregation(nn.Module):

    def __init__(self, inChannels):
        super(Cross_Modal_Aggregation, self).__init__()
        self.conv = nn.Conv2d(inChannels*2, 2, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.ChannelAttention=ChannelAttention(inChannels)
        self.SpatialAttention=SpatialAttention()

        self.inChannels=inChannels

    def forward(self, inp1,inp2):
        x = torch.cat([inp1, inp2], dim=1)
        x=self.conv(x)
        x_i,x_e=x[:, 0:1, :, :], x[:, 1:2, :, :]
        x_i=self.sigmoid(x_i)
        x_e=self.sigmoid(x_e)
        x_i=self.pool(x_i)
        x_e=self.pool(x_e)
        g_inp1=x_i*inp1
        g_inp2=x_e*inp2

        inp1_CA=self.ChannelAttention(g_inp1)*g_inp1
        inp1_CA_SA=self.SpatialAttention(inp1_CA)*inp1_CA
        fuse=inp1_CA_SA+g_inp2

        return fuse




class Cross_Modal_Calibration(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Modal_Calibration, self).__init__()
        self.num_heads = num_heads
        self.temperature= nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, dm, input):
        assert dm.shape == input.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = input.shape
        q = self.q(dm)
        k = self.k(input)
        v = self.v(input)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn=(q @ k.transpose(-2, -1)) * self.temperature
        out=(attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out



class Spatio_Temporal_Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Spatio_Temporal_Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, inp1, inp2):
        #####inp1--current_frame
        #####inp2--reference_frame
        assert inp1.shape == inp2.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = inp1.shape

        q = self.q(inp2)  # image
        k = self.k(inp1)  # event
        v = self.v(inp1)  # event

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        return attn,v


class Multimodal_Coordinate_Attention(nn.Module):
    def __init__(self, dim, reduction=32):
        super(Multimodal_Coordinate_Attention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, dim // reduction)
        self.conv = nn.Conv2d(dim, mip, kernel_size=1, stride=1, padding=0)
        self.act =nn.Sigmoid()

        self.conv_h = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)


    def forward(self, inp1, inp2):

        assert inp1.shape == inp2.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = inp1.shape

        inp1_h = self.pool_h(inp1)
        inp1_w = self.pool_w(inp1).permute(0, 1, 3, 2)

        inp2_h = self.pool_h(inp2)
        inp2_w = self.pool_w(inp2).permute(0, 1, 3, 2)

        x=torch.cat([inp1_h, inp1_w,inp2_h,inp2_w], dim=2)
        x=self.conv(x)

        x_inp1_h, x_inp1_w,x_inp2_h, x_inp2_w = torch.split(x, [h, w,h, w], dim=2)
        x_inp1_w = x_inp1_w.permute(0, 1, 3, 2)
        x_inp2_w = x_inp2_w.permute(0, 1, 3, 2)

        x_inp1_h= self.act(self.conv_h(x_inp1_h))
        x_inp1_w= self.act(self.conv_w(x_inp1_w))
        x_inp2_h= self.act(self.conv_h(x_inp2_h))
        x_inp2_w= self.act(self.conv_w(x_inp2_w))

        y_inp1=inp1*x_inp1_h*x_inp1_w
        y_inp2=inp2*x_inp2_h*x_inp2_w
        fuse=y_inp1+y_inp2

        return fuse


class Multimodal_Coordinate_Attention_SE(nn.Module):
    def __init__(self, dim, reduction=32):
        super(Multimodal_Coordinate_Attention_SE, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, dim // reduction)
        self.conv = nn.Conv2d(dim, mip, kernel_size=1, stride=1, padding=0)
        self.act =nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化比例为 0.5
        self.conv_h = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)
        self.se = SELayer(256) 
        
    def forward(self, inp1, inp2):

        assert inp1.shape == inp2.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = inp1.shape

        inp1_h = self.pool_h(inp1)
        inp1_w = self.pool_w(inp1).permute(0, 1, 3, 2)

        inp2_h = self.pool_h(inp2)
        inp2_w = self.pool_w(inp2).permute(0, 1, 3, 2)

        x=torch.cat([inp1_h, inp1_w,inp2_h,inp2_w], dim=2)
        x=self.conv(x)

        x_inp1_h, x_inp1_w,x_inp2_h, x_inp2_w = torch.split(x, [h, w,h, w], dim=2)
        x_inp1_w = x_inp1_w.permute(0, 1, 3, 2)
        x_inp2_w = x_inp2_w.permute(0, 1, 3, 2)

        x_inp1_h= self.act(self.conv_h(x_inp1_h))
        x_inp1_w= self.act(self.conv_w(x_inp1_w))
        x_inp2_h= self.act(self.conv_h(x_inp2_h))
        x_inp2_w= self.act(self.conv_w(x_inp2_w))

        y_inp1=inp1*x_inp1_h*x_inp1_w
        y_inp2=inp2*x_inp2_h*x_inp2_w
        fuse=y_inp1+y_inp2
        
        
       
        se_fuse = self.se(fuse)

        output = fuse + self.alpha * se_fuse


        return output
    
class SELayer(nn.Module):
    def __init__(self, channel=256, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y




