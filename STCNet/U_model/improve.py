import torch
import torch.nn as nn
import torch.nn.functional as F

class Modality_Spatial_Collaboration(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Modality_Spatial_Collaboration, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)

        self.CMC = Cross_Modal_Calibration(dim, num_heads, bias)
        self.CMA = Cross_Modal_Aggregation(dim)

        self.project_out = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)

       
        
        self.gamma_B = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(dim // 4, 1, 1),
                nn.Sigmoid()
            )
        self.gamma_E = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(dim // 4, 1, 1),
                nn.Sigmoid()
            )
       

    def compute_DM(self, image, event):
        """ 计算 Differential Modality (DM) """
        norm_image = image / (image.norm(dim=(2, 3), keepdim=True) + 1e-6)
        norm_event = event / (event.norm(dim=(2, 3), keepdim=True) + 1e-6)

        
        gamma_B = self.gamma_B(image)  # MLP 计算权重
        gamma_E = self.gamma_E(event)
       

        DM = gamma_B * norm_image - gamma_E * norm_event
        return DM

    def forward(self, image, event):
        """
        :param image: (b, c, h, w)
        :param event: (b, c, h, w)
        :return: (b, c, h, w)
        """
        assert image.shape == event.shape, 'The shape of image doesn\'t match event'
        
        # 计算 DM
        F_dm = self.compute_DM(image, event)

        # Cross-Modal Calibration (CMC)
        CMC_img = self.CMC(self.norm1_image(F_dm), self.norm1_event(image))
        CMC_event = self.CMC(self.norm1_image(F_dm), self.norm1_event(event))

        # Cross-Modal Aggregation (CMA)
        fuse_img_to_event = self.CMA(CMC_img, event)
        fuse_event_to_img = self.CMA(CMC_event, image)

        # Concatenation and Projection
        fuse_cat = torch.cat((fuse_img_to_event, fuse_event_to_img), 1)
        fused = self.project_out(fuse_cat)

        return fused


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
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