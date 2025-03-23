import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import os, sys, math, random, glob, cv2, h5py, logging, random
import utils
import torch.utils.data as data
from dist_util import *
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader




def binary_events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0
    

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]

    pols[pols == 0] = -1  

    tis = ts.astype(np.int)
    dts = ts - tis

    vals_left = pols * (1.0 - dts)

    vals_right = pols * dts


    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    def set_epoch(self, epoch):
        
        self.seed = epoch

def create_data_loader(data_set, opts, mode='train'):

    total_samples = 1800


    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    
    sampler = SubsetSequentialSampler(indices)
    data_loader = DataLoader(dataset=data_set, num_workers=4,
                             batch_size=opts.OPTIM.BATCH_SIZE, sampler=sampler, pin_memory=True,shuffle=False,drop_last=False)

    return data_loader


class DataLoaderTrain_npz(Dataset):
    def __init__(self, rgb_dir, voxel_dir,unrolling_len,ps,bin_nums):
        super(DataLoaderTrain_npz, self).__init__()
        self.unrolling_len = unrolling_len
        self.TRAIN_PS = ps
        self.voxel_dir = voxel_dir
        
        self.bin_nums = bin_nums
        if self.bin_nums == 6:
            self.event_img_path = os.path.join(self.voxel_dir, f'voxel')
        else:
            self.event_img_path = os.path.join(self.voxel_dir, f'voxel_{self.bin_nums}')
        self.blur_img_path = os.path.join(rgb_dir, 'blur')
       
        self.sharp_img_path = os.path.join(rgb_dir, 'sharp')

        
        self._load_data_by_scene()

        
        self._build_index_map()

    def _load_data_by_scene(self):
        """按场景前缀组织数据"""
        
        blur_files = sorted(glob.glob(os.path.join(self.blur_img_path, '*.png')))
        event_files = sorted(glob.glob(os.path.join(self.event_img_path, '*.npz')))
        sharp_files = sorted(glob.glob(os.path.join(self.sharp_img_path, '*.png')))

        
        assert len(blur_files) == len(event_files) == len(sharp_files), "文件数量不匹配"

        
        self.image_data = {}
        for b, e, s in zip(blur_files, event_files, sharp_files):
            scene_prefix = os.path.basename(b).rsplit('_', 1)[0]  
            if scene_prefix not in self.image_data:
                self.image_data[scene_prefix] = {
                    'blur': [], 'voxel': [], 'sharp': []
                }
            self.image_data[scene_prefix]['blur'].append(b)
            self.image_data[scene_prefix]['voxel'].append(e)
            self.image_data[scene_prefix]['sharp'].append(s)

        
        self.scene_list = list(self.image_data.keys())

    def _build_index_map(self):
        """预计算全局索引到场景的映射表"""
        self.index_map = []
        for scene_idx, scene in enumerate(self.scene_list):
            num_frames = len(self.image_data[scene]['blur'])
            valid_samples = num_frames - self.unrolling_len + 1
            for start_frame in range(valid_samples):
                self.index_map.append((scene_idx, start_frame))

    def __len__(self):
        return len(self.index_map)  

    def __getitem__(self, index):
        
        scene_idx, start_frame = self.index_map[index]
        scene_prefix = self.scene_list[scene_idx]

        
        blur_paths = self.image_data[scene_prefix]['blur']
        event_paths = self.image_data[scene_prefix]['voxel']
        sharp_paths = self.image_data[scene_prefix]['sharp']

        
        blur_imgs, sharp_imgs, event_imgs = list(), list(), list()

        for t in range(self.unrolling_len):
            frame_idx = start_frame + t
            
            blur_img = cv2.imread(blur_paths[frame_idx])
            blur_img = np.float32(blur_img) / 255.0
            blur_img = blur_img.transpose(2, 0, 1)  

            
            event_data = np.load(event_paths[frame_idx])
            event_frame = np.float32(event_data['voxel'])
            event_frame = np.transpose(event_frame, (2, 0, 1))
            
            sharp_img = cv2.imread(sharp_paths[frame_idx])
            sharp_img = np.float32(sharp_img) / 255.0
            sharp_img = sharp_img.transpose(2, 0, 1)  

            
            
            blur_img, event_frame, sharp_img = utils.image_proess(blur_img, event_frame, sharp_img,
                                                                self.TRAIN_PS)

            event_frame = event_frame.unsqueeze(0)
            sharp_img = sharp_img.unsqueeze(0)
            blur_img = blur_img.unsqueeze(0)
            

            blur_imgs.append(blur_img)
            sharp_imgs.append(sharp_img)
            event_imgs.append(event_frame)
            

        
        data = (torch.cat(blur_imgs, dim=0), torch.cat(event_imgs, dim=0), torch.cat(sharp_imgs, dim=0))
        

        return data


class DataLoaderVal_npz(Dataset):
    def __init__(self, rgb_dir,voxel_dir, unrolling_len, bin_nums):
        self.rgb_dir = rgb_dir
        self.voxel_dir = voxel_dir
        self.bin_nums = bin_nums
        self.unrolling_len = unrolling_len
        blur_img_path = os.path.join(rgb_dir, 'blur')
        if self.bin_nums == 6:
            event_img_path = os.path.join(self.voxel_dir, f'voxel')
        else:
            event_img_path = os.path.join(self.voxel_dir, f'voxel_{self.bin_nums}')
        sharp_img_path = os.path.join(rgb_dir, 'sharp')

        
        self.DVS_stream_height = 1224
        self.DVS_stream_width = 1632

        
        blur_files = sorted(os.listdir(blur_img_path))  
        event_files = sorted(os.listdir(event_img_path))
        sharp_files = sorted(os.listdir(sharp_img_path))
        self.length = len(blur_files)

        
        file_groups = {}  
        for file_name in blur_files:
            prefix = "_".join(file_name.split("_")[:-1])  
            if prefix not in file_groups:
                file_groups[prefix] = {"blur": [], "voxel": [], "sharp": []}
            file_groups[prefix]["blur"].append(os.path.join(blur_img_path, file_name))

        for file_name in event_files:
            prefix = "_".join(file_name.split("_")[:-1])
            if prefix in file_groups:  
                file_groups[prefix]["voxel"].append(os.path.join(event_img_path, file_name))

        for file_name in sharp_files:
            prefix = "_".join(file_name.split("_")[:-1])
            if prefix in file_groups:
                file_groups[prefix]["sharp"].append(os.path.join(sharp_img_path, file_name))

        
        for key in file_groups:
            file_groups[key]["blur"].sort()
            file_groups[key]["voxel"].sort()
            file_groups[key]["sharp"].sort()

        
        self.seqs_info = {}
        self.length = 0
        seq_idx = 0

        for key, files in file_groups.items():
            seq_info = {
                "blur": files["blur"],
                "voxel": files["voxel"],
                "sharp": files["sharp"],
                "length": len(files["blur"])
            }
            self.length += seq_info["length"]
            self.seqs_info[seq_idx] = seq_info
            seq_idx += 1

        self.seqs_info["length"] = self.length
        self.seqs_info["num"] = len(file_groups)

    def __len__(self):
        return self.seqs_info["length"] - (self.unrolling_len - 1) * self.seqs_info["num"]

    def __getitem__(self, idx):
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs, event_imgs = [], [], []

        for i in range(self.seqs_info["num"]):
            seq_length = self.seqs_info[i]["length"] - self.unrolling_len + 1
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length

        for i in range(self.unrolling_len):
            blur_img = cv2.imread(self.seqs_info[seq_idx]["blur"][frame_idx + i])
            blur_img = np.float32(blur_img) / 255.0
            blur_img = blur_img.transpose([2, 0, 1])

            
            event = np.load(self.seqs_info[seq_idx]["voxel"][frame_idx + i])
            event_frame = np.float32(event["voxel"])  
            event_frame = np.transpose(event_frame, (2, 0, 1))  


            sharp_img = cv2.imread(self.seqs_info[seq_idx]["sharp"][frame_idx + i])
            sharp_img = np.float32(sharp_img) / 255.0
            sharp_img = sharp_img.transpose([2, 0, 1])
            
            blur_img = torch.from_numpy(blur_img).unsqueeze(0)
            sharp_img = torch.from_numpy(sharp_img).unsqueeze(0)
            event_frame = torch.from_numpy(event_frame).unsqueeze(0)

            blur_imgs.append(blur_img)
            sharp_imgs.append(sharp_img)
            event_imgs.append(event_frame)

        data = (torch.cat(blur_imgs, dim=0), torch.cat(event_imgs, dim=0), torch.cat(sharp_imgs, dim=0))
        return data


class DataLoaderTest_npz(Dataset):
    def __init__(self, rgb_dir, file_dir,voxel_dir, unrolling_len,bin_nums):
        self.rgb_dir = rgb_dir
        self.bin_nums = bin_nums
        blur_img_path = os.path.join(rgb_dir, 'blur')
        if self.bin_nums == 6:
            event_img_path = os.path.join(self.voxel_dir, f'voxel')
        else:
            event_img_path = os.path.join(self.voxel_dir, f'voxel_{self.bin_nums}')
        sharp_img_path = os.path.join(rgb_dir, 'sharp')
        self.unrolling_len = unrolling_len
       
        self.DVS_stream_height = 1224
        self.DVS_stream_width = 1632

        
        blur_files = sorted(os.listdir(blur_img_path))  
        event_files = sorted(os.listdir(event_img_path))
        sharp_files = sorted(os.listdir(sharp_img_path))

        
        file_groups = {}
        for file_name in blur_files:
            prefix = "_".join(file_name.split("_")[:-1])  
            if prefix not in file_groups:
                file_groups[prefix] = {"blur": [], "voxel": [], "sharp": []}
            file_groups[prefix]["blur"].append(os.path.join(blur_img_path, file_name))

        for file_name in event_files:
            prefix = "_".join(file_name.split("_")[:-1])
            if prefix in file_groups:
                file_groups[prefix]["voxel"].append(os.path.join(event_img_path, file_name))

        for file_name in sharp_files:
            prefix = "_".join(file_name.split("_")[:-1])
            if prefix in file_groups:
                file_groups[prefix]["sharp"].append(os.path.join(sharp_img_path, file_name))

        
        for key in file_groups:
            file_groups[key]["blur"].sort()
            file_groups[key]["voxel"].sort()
            file_groups[key]["sharp"].sort()

        
        self.seqs_info = {}
        self.length = 0
        seq_idx = 0

        if file_dir in file_groups:
            seq_info = {
                "blur": file_groups[file_dir]["blur"],
                "voxel": file_groups[file_dir]["voxel"],
                "sharp": file_groups[file_dir]["sharp"],
                "length": len(file_groups[file_dir]["blur"])
            }
            self.length += seq_info["length"]
            self.seqs_info[seq_idx] = seq_info
            seq_idx += 1

        self.seqs_info["length"] = self.length
        self.seqs_info["num"] = len(self.seqs_info)

    def __len__(self):
        
        return self.seqs_info["length"]

    def __getitem__(self, idx):
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs, event_imgs = [], [], []
        blur_names, sharp_names, event_names = [], [], []  

        for i in range(self.seqs_info["num"]):
            seq_length = self.seqs_info[i]["length"]
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length
        if frame_idx == 0:
            frame_indices = [0, 0, 1]  
        elif frame_idx == seq_length - 1:
            frame_indices = [seq_length - 1, seq_length - 1, seq_length - 2]  
        else:
            frame_indices = [max(0, frame_idx - 1), frame_idx, min(seq_length - 1, frame_idx + 1)]  

        
        for i in range(self.unrolling_len):
            
            fill_idx = frame_indices[i] 
            blur_img = cv2.imread(self.seqs_info[seq_idx]["blur"][fill_idx])
            sharp_img = cv2.imread(self.seqs_info[seq_idx]["sharp"][fill_idx])
            event_frame = np.load(self.seqs_info[seq_idx]["voxel"][fill_idx])["voxel"]
            blur_name = self.seqs_info[seq_idx]["blur"][fill_idx].split("/")[-1]  
            sharp_name = self.seqs_info[seq_idx]["sharp"][fill_idx].split("/")[-1]
            event_name = self.seqs_info[seq_idx]["voxel"][fill_idx].split("/")[-1]
           
            blur_img = np.float32(blur_img) / 255.0
            blur_img = blur_img.transpose([2, 0, 1])

            event_frame = np.float32(event_frame)
            event_frame = np.transpose(event_frame, (2, 0, 1))

            sharp_img = np.float32(sharp_img) / 255.0
            sharp_img = sharp_img.transpose([2, 0, 1])

            
            blur_img = torch.from_numpy(blur_img).unsqueeze(0)
            sharp_img = torch.from_numpy(sharp_img).unsqueeze(0)
            event_frame = torch.from_numpy(event_frame).unsqueeze(0)

            blur_imgs.append(blur_img)
            sharp_imgs.append(sharp_img)
            event_imgs.append(event_frame)
            
            
            blur_names.append(blur_name)
            sharp_names.append(sharp_name)
            event_names.append(event_name)

        
        return torch.cat(blur_imgs, dim=0), torch.cat(event_imgs, dim=0), torch.cat(sharp_imgs, dim=0), blur_names



class DataLoaderTestNoSharp_npz(Dataset):
    def __init__(self, rgb_dir, voxel_dir, file_dir, args):
        self.rgb_dir = rgb_dir
        self.voxel_dir = voxel_dir
        blur_img_path = os.path.join(rgb_dir, 'blur')

        event_img_path = os.path.join(self.voxel_dir, 'voxel')
        
        self.args = args
        self.DVS_stream_height = 1224
        self.DVS_stream_width = 1632

        
        blur_files = sorted(os.listdir(blur_img_path))  
        event_files = sorted(os.listdir(event_img_path))
        

        
        file_groups = {}
        for file_name in blur_files:
            prefix = "_".join(file_name.split("_")[:-1])  
            if prefix not in file_groups:
                file_groups[prefix] = {"blur": [], "voxel": [], "sharp": []}
            file_groups[prefix]["blur"].append(os.path.join(blur_img_path, file_name))

        for file_name in event_files:
            prefix = "_".join(file_name.split("_")[:-1])
            if prefix in file_groups:
                file_groups[prefix]["voxel"].append(os.path.join(event_img_path, file_name))
        
        for key in file_groups:
            file_groups[key]["blur"].sort()
            file_groups[key]["voxel"].sort()
                
        self.seqs_info = {}
        self.length = 0
        seq_idx = 0

        if file_dir in file_groups:
            seq_info = {
                "blur": file_groups[file_dir]["blur"],
                "voxel": file_groups[file_dir]["voxel"],
                "length": len(file_groups[file_dir]["blur"])
            }
            self.length += seq_info["length"]
            self.seqs_info[seq_idx] = seq_info
            seq_idx += 1

        self.seqs_info["length"] = self.length
        self.seqs_info["num"] = len(self.seqs_info)

    def __len__(self):
        return self.seqs_info["length"]

    def __getitem__(self, idx):
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs, event_imgs = [], [], []
        blur_names, sharp_names, event_names = [], [], []  

        for i in range(self.seqs_info["num"]):
            seq_length = self.seqs_info[i]["length"]
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length
        if frame_idx == 0:
            frame_indices = [0, 0, 1]  
        elif frame_idx == seq_length - 1:
            frame_indices = [seq_length - 1, seq_length - 1, seq_length - 2]  
        else:
            frame_indices = [max(0, frame_idx - 1), frame_idx, min(seq_length - 1, frame_idx + 1)]  

        
        for i in range(self.args.unrolling_len):
            
            fill_idx = frame_indices[i] 
            blur_img = cv2.imread(self.seqs_info[seq_idx]["blur"][fill_idx])
           
            event_frame = np.load(self.seqs_info[seq_idx]["voxel"][fill_idx])["voxel"]
            blur_name = self.seqs_info[seq_idx]["blur"][fill_idx].split("/")[-1]  
          
            event_name = self.seqs_info[seq_idx]["voxel"][fill_idx].split("/")[-1]
           
            blur_img = np.float32(blur_img) / 255.0
            blur_img = blur_img.transpose([2, 0, 1])

            event_frame = np.float32(event_frame)
            event_frame = np.transpose(event_frame, (2, 0, 1))

            blur_img = torch.from_numpy(blur_img).unsqueeze(0)
         
            event_frame = torch.from_numpy(event_frame).unsqueeze(0)

            blur_imgs.append(blur_img)
            event_imgs.append(event_frame)
                        
            blur_names.append(blur_name)         
            event_names.append(event_name)

        return torch.cat(blur_imgs, dim=0), torch.cat(event_imgs, dim=0), event_name, blur_names