import torch


import os
from config import Config
from loguru import logger  
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from dataset_RGB import *
from U_model import unet,se_unet,Detail_unet,Detail_se_unet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import re
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

import heapq
import argparse


parser = argparse.ArgumentParser(description="Train STCNet with checkpoints")
parser.add_argument('--checkpoints', type=str, default="./n_checkpoints", help="Path to checkpoint directory")
parser.add_argument('--config', type=str, default="training.yml", help="Path to yml")
parser.add_argument('--log_dir', type=str, default="logs", help="Path to res directory")
parser.add_argument('--method', type=str, default="unet", help="method")
args = parser.parse_args()

opt = Config(args.config)
from datetime import datetime
log_dir = args.log_dir
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
log_filename = f"{args.log_dir}/training_{timestamp}.log"  

logger.add(log_filename, rotation="1 week", retention="2 weeks", compression="zip")

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
logger.info(opt)  
max_best_models = 3
best_models = []  
def save_best_model(epoch, model, optimizer, epoch_loss,model_dir):
    model_path = os.path.join(model_dir, f"model_{epoch_loss:.6f}_{epoch:06d}.pth")

    
    if len(best_models) < max_best_models:
        heapq.heappush(best_models, (epoch_loss, model_path))
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   model_path)
    else:
        
        worst_loss, worst_model = best_models[0]  
        if epoch_loss < worst_loss:
            
            os.remove(worst_model)
            heapq.heapreplace(best_models, (epoch_loss, model_path))
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       model_path)
def main():
    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION
    writer = SummaryWriter(log_dir=args.log_dir)
    result_dir = os.path.join(args.checkpoints, 'results', session)
    model_dir = os.path.join(args.checkpoints, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    


    
    opt.VGGLayers = [int(layer) for layer in list(opt.VGGLayers)]
    opt.VGGLayers.sort()
    logger.info(opt.VGGLayers)  

    if opt.VGGLayers[0] < 1 or opt.VGGLayers[-1] > 4:
        raise Exception("Only support VGG Loss on Layers 1 ~ 4")
    opt.VGGLayers = [layer - 1 for layer in list(opt.VGGLayers)]  

    if opt.w_VGG > 0:
        
        from vgg_networks.vgg import Vgg16
        VGG = Vgg16(requires_grad=False)
        VGG = VGG.cuda()

    
    if args.method == 'unet':
        model_restoration = unet.Restoration(3, 6, 3,opt)
    elif args.method == 'Detail_unet':
        model_restoration = Detail_unet.Restoration(3, 6, 3,opt)
    elif args.method == 'Detail_se_unet':
        model_restoration = Detail_se_unet.Restoration(3, 6, 3,opt)
    else:
        model_restoration = se_unet.Restoration(3, 6, 3,opt)
    model_restoration.cuda()
    logger.info(f"select method: {args.method}")  
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")  

    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    
    if opt.TRAINING.RESUME:

        checkpoint_dir = f"./{args.checkpoints}/models/STCNet/"

        
        model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth") and "latest"  in f]

        
        if model_files:
            best_model = model_files[0]
            path_chk_rest = os.path.join(checkpoint_dir, best_model)
        else:
            path_chk_rest = './ckpt/STCNet_model_best.pth'
      
        logger.info(f"Selected model:{path_chk_rest}")
        
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        if path_chk_rest == './ckpt/STCNet_model_best.pth':
            start_epoch = 1
        utils.load_optim(optimizer, path_chk_rest)
        '''
        path_chk_rest = utils.get_last_path(model_dir, '_best_psnr.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest[0])
        start_epoch = utils.load_start_epoch(path_chk_rest[0]) + 1

        utils.load_optim(optimizer, path_chk_rest[0])
        '''
        for i in range(0, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info(f"Resuming Training with learning rate: {new_lr}")  
    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    
    criterion_char = losses.CharbonnierLoss()


    
    train_dataset = DataLoaderTrain_npz(opt.father_train_path_npz,opt.father_train_voxel_path, opt)
    print("len:",len(train_dataset))
    train_loader= create_data_loader(train_dataset, opt)
    val_dataset = DataLoaderVal_npz(opt.father_val_path_npz, opt.father_val_voxel_path,opt)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                            pin_memory=True)
    logger.info(f'Start Epoch {start_epoch} End Epoch {opt.OPTIM.NUM_EPOCHS + 1}')  
    logger.info('Loading datasets')  
    
    best_psnr_models = []
    
    best_ssim_models = []
   
    
    max_best_models = 3
    best_psnr = 0
    best_epoch = 0
    best_ssim = 0
    
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        model_restoration.train()
        for iteration, data in enumerate(tqdm(train_loader), 1):
            
            for param in model_restoration.parameters():
                param.grad = None
            input_img = data[0].cuda()
            input_event = data[1].cuda()
            input_target = data[2].cuda()
            restored = model_restoration(input_img, input_event)
            input_target=input_target[:,1,:,:,:]
            loss_char = criterion_char(restored, input_target)
            loss = loss_char
            loss.backward(retain_graph=False)

            torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), 20)

            optimizer.step()
            epoch_loss += loss.item()
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        save_best_model(epoch=epoch,model=model_restoration,optimizer=optimizer,epoch_loss=epoch_loss,model_dir=model_dir)

        
        
        logger.info(f'Start testing epoch: {epoch}!')  
        logger.info(f'{epoch+1 % opt.TRAINING.VAL_AFTER_EVERY}')  
        
        if (epoch) % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(tqdm(val_loader), 0):
                
                input_img = data_val[0].cuda()
                input_event = data_val[1].cuda()
                input_target = data_val[2].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_img, input_event)

                input_target = input_target[:, 1, :, :, :]

                for res, tar in zip(restored, input_target):
                    res = torch.clamp(res, 0, 1)
                    input1 = res.cpu().numpy().transpose([1, 2, 0])
                    input2 = tar.cpu().numpy().transpose([1, 2, 0])
                   
                    ssim_rgb = SSIM(input1, input2, channel_axis=-1,data_range=1.0)
                    ssim_val_rgb.append(ssim_rgb)

                    psnr_rgb = PSNR(input1, input2)
                    psnr_val_rgb.append(psnr_rgb)

            ssim_val_rgb = np.mean(ssim_val_rgb)
            psnr_val_rgb = np.mean(psnr_val_rgb)
            psnr_model_path = os.path.join(model_dir, f"model_psnr_{psnr_val_rgb:.4f}_epoch_{epoch}.pth")
            ssim_model_path = os.path.join(model_dir, f"model_ssim_{ssim_val_rgb:.4f}_epoch_{epoch}.pth")
            logger.info('Start saving model!')  
            
            if len(best_psnr_models) < max_best_models:
                
                heapq.heappush(best_psnr_models, (psnr_val_rgb, psnr_model_path))
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()},
                        psnr_model_path)
            else:
                
                if psnr_val_rgb > best_psnr_models[0][0]:
                    
                    worst_model = heapq.heappop(best_psnr_models)
                    os.remove(worst_model[1])  

                    
                    heapq.heappush(best_psnr_models, (psnr_val_rgb, psnr_model_path))
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()},
                            psnr_model_path)

                with open(model_dir + '/BEST_PSNR.txt', 'a') as f:
                    f.write('Epoch:' + str(epoch) + ' PSNR:' + str(psnr_val_rgb) + ' ' + 'SSIM: ' + str(
                        ssim_val_rgb) + "\n")
            
            
            if len(best_ssim_models) < max_best_models:
                
                heapq.heappush(best_ssim_models, (ssim_val_rgb, ssim_model_path))
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()},
                        ssim_model_path)
            else:
                
                if ssim_val_rgb > best_ssim_models[0][0]:
                    
                    worst_model = heapq.heappop(best_ssim_models)
                    os.remove(worst_model[1])  

                    
                    heapq.heappush(best_ssim_models, (ssim_val_rgb, ssim_model_path))
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()},
                            ssim_model_path)

                with open(model_dir + '/BEST_PSNR.txt', 'a') as f:
                    f.write('Epoch:' + str(epoch) + ' PSNR:' + str(psnr_val_rgb) + ' ' + 'SSIM: ' + str(
                        ssim_val_rgb) + "\n")

            logger.info(f"[epoch {epoch} PSNR: {psnr_val_rgb:.4f} SSIM: {ssim_val_rgb:.4f} --- best_epoch {best_epoch} Best_PSNR {best_psnr:.4f} Best_SSIM {best_ssim:.4f}]")  
            with open(model_dir + '/BEST.txt', 'a') as f:
                f.write('Epoch:' + str(epoch) + ' PSNR:' + str(psnr_val_rgb) + ' ' + 'SSIM: ' + str(
                    ssim_val_rgb) + "\n")
            
        scheduler.step()

        logger.info(f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.4f}\tLoss: {epoch_loss:.4f}\tLearningRate {scheduler.get_lr()[0]:.6f}")  
        writer.add_scalar('Learning Rate', scheduler.get_lr()[0], epoch)
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))

    writer.close()
if __name__ == '__main__':
    main()