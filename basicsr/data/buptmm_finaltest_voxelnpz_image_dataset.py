from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
import os.path as osp
from glob import glob
from pathlib import Path
import random
import numpy as np
import torch

from basicsr.data.event_util import voxel_norm
from basicsr.utils import FileClient, imfrombytes, img2tensor, get_root_logger
from torch.utils.data.dataloader import default_collate


class FinalTestVoxelnpzPngSingleDeblurDataset(data.Dataset):
    """Paired vxoel(npz) and blurry image (png) dataset for event-based single image deblurring.
    --HighREV
    |----train
    |    |----blur
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |    |----voxel
    |    |    |----SEQNAME_%5d.npz
    |    |    |----...
    |    |----sharp
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |----val
    ...

    
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.num_bins = opt['voxel_bins']        
        
        self.split = 'test'  # train/val/test
        self.norm_voxel = opt['norm_voxel']
        self.dataPath = []

        blur_frames = sorted(glob(osp.join(self.dataroot, 'blur', '*.png')))
        if self.num_bins == 6:
            event_frames = sorted(glob(osp.join(self.dataroot, 'voxel', '*.npz')))
        else:
            event_frames = sorted(glob(osp.join(self.dataroot, f'voxel_bin{self.num_bins}', '*.npz')))
        
        if len(event_frames) == 0:
            raise ValueError(f"No event frames found in {self.dataroot} of num_bins={self.num_bins}.")
        
        assert len(blur_frames) == len(event_frames), f"Mismatch in blur ({len(blur_frames)}) and event ({len(event_frames)}) frame counts."

        for i in range(len(blur_frames)):
            self.dataPath.append({
                'blur_path': blur_frames[i],
                'event_paths': event_frames[i],
            })
        logger = get_root_logger()
        logger.info(f"Dataset initialized with {len(self.dataPath)} samples.")

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        image_path = self.dataPath[index]['blur_path']
        event_path = self.dataPath[index]['event_paths']

        # get LQ
        img_bytes = self.file_client.get(image_path)  # 'lq'
        img_lq = imfrombytes(img_bytes, float32=True)
        voxel = np.load(event_path)['voxel']

        ## Data augmentation
        # voxel shape: h,w,c

        img_lq = img2tensor(img_lq) # hwc -> chw
        voxel = img2tensor(voxel) # hwc -> chw

        ## Norm voxel
        if self.norm_voxel:
            voxel = voxel_norm(voxel)

        origin_index = os.path.basename(image_path).split('.')[0]
        seq = '_'.join(origin_index.split('_')[:-1])

        return {'frame': img_lq, 'voxel': voxel, 'image_name': origin_index, 'seq': seq}

    def __len__(self):
        return len(self.dataPath)
