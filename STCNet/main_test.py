
import torch


import os
from config import Config
import re



torch.backends.cudnn.benchmark = True
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import utils
from dataset_RGB import *
from U_model import unet,se_unet,Detail_unet,Detail_se_unet
from warmup_scheduler import GradualWarmupScheduler


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
import argparse


parser = argparse.ArgumentParser(description="Train STCNet with checkpoints")

parser.add_argument('--config', type=str, default="finetune_training.yml", help="Path to yml")
parser.add_argument('--result_dir', type=str, default="./STCNet_output", help="Path to res directory")
parser.add_argument('--method', type=str, default="unet", help="method")
args = parser.parse_args()
opt = Config(args.config)
opt.result_dir = args.result_dir
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

@torch.no_grad()
def main():
    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)


    
    if args.method == 'unet':
        model_restoration = unet.Restoration(3, 6, 3,opt)
    elif args.method == 'Detail_unet':
        model_restoration = Detail_unet.Restoration(3, 6, 3,opt)
    elif args.method == 'Detail_se_unet':
        model_restoration = Detail_se_unet.Restoration(3, 6, 3,opt)
    else:
        model_restoration = se_unet.Restoration(3, 6, 3,opt)
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    
    if opt.TESTING.RESUME:
        path_chk_rest = opt.TESTING.RESUME_PATH
        if path_chk_rest is None:
            raise ValueError(f"Please provide a valid checkpoint path ({path_chk_rest}) for resuming training")
        
        print(f"choose model: {path_chk_rest}")
        
        utils.load_checkpoint(model_restoration, path_chk_rest)
        
    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    
    test_files=sorted(os.listdir(os.path.join(opt.father_test_path_npz, 'blur')))
    test_files_dirs = []
    for test_file in test_files:
        prefix = "_".join(test_file.split("_")[:-1])  
        if prefix not in test_files_dirs:
            test_files_dirs.append(prefix)


    print(f'test_files_dirs: {test_files_dirs}')
    print('===> Loading datasets')


    model_restoration.eval()

    for test_file in test_files_dirs:
        out_path = os.path.join(opt.result_dir)
        isExists = os.path.exists(out_path)
        if not isExists:
            os.makedirs(out_path)

        val_dataset = DataLoaderTestNoSharp_npz(opt.father_test_path_npz, opt.father_test_voxel_path, test_file, opt)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                                pin_memory=True)
        print(f"len val dataloader: {len(val_loader)}")
        for ii, data_val in enumerate(val_loader):
            input_img = data_val[0].cuda()
            print(f'input_img.shape: {input_img.shape}')
            input_event = data_val[1].cuda()
            name = data_val[3]

            print(f'name: {name}')
            with torch.no_grad():
                restored = model_restoration(input_img, input_event)  

            output = restored[0, :, :, :] * 255
            output.clamp_(0.0, 255.0)
            output = output.byte()
            output = output.cpu().numpy()
            output = output.transpose([1, 2, 0])  

            cv2.imwrite((os.path.join(out_path,name[1][0])), output)


if __name__ == '__main__':
    main()


