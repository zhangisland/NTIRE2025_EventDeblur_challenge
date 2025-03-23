import torch
import torch.nn as nn
from basicsr.models.archs.resnet_3D import r3d_18, Conv_3d, upConv3D, identity


class UNet3d_18(nn.Module):
    def __init__(self, channels=[32,64,96,128], bn=True):
        super().__init__()
        growth = 2 # since concatenating previous outputs
        upmode = "transpose" # use transposeConv to upsample

        self.channels = channels

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.encoder = r3d_18(bn=bn, channels=channels)

        self.decoder = nn.Sequential(
            Conv_3d(channels[::-1][0], channels[::-1][1] , kernel_size=3, padding=1, bias=True),
            upConv3D(channels[::-1][1]*growth, channels[::-1][2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode),
            upConv3D(channels[::-1][2]*growth, channels[::-1][3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode),
            Conv_3d(channels[::-1][3]*growth, channels[::-1][3] , kernel_size=3, padding=1, bias=True),
            upConv3D(channels[::-1][3]*growth , channels[::-1][3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode)
        )

        self.feature_fuse = nn.Sequential(
            *([nn.Conv2d(channels[::-1][3]*2, channels[::-1][3], kernel_size=1, stride=1, bias=False)] + \
              [nn.BatchNorm2d(channels[::-1][3]) if bn else identity()])
        )

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels[::-1][3], 3 , kernel_size=7, stride=1, padding=0) 
        )
    
    def forward(self, img0, img1):
        images = torch.stack((img0, img1) , dim=2)

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = torch.cat([dx_3 , x_3], dim=1)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = torch.cat([dx_2 , x_2], dim=1)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = torch.cat([dx_1 , x_1], dim=1)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = torch.cat([dx_0 , x_0], dim=1)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)

        return out
