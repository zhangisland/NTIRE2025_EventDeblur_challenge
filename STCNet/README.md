# STCNet - Image Deblurring with Event Cameras
This repository contains the implementation of STCNet, a deep learning model for image deblurring using event cameras.

## Installation

To set up the environment and install dependencies, follow these steps:


```bash
conda create -n STCNet python=3.8.5
conda activate STCNet
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm loguru tensorboard lpips einops torchinfo
cd pytorch-gradual-warmup-lr;python setup.py install;cd ..
```

## Dataset Setup
Same dataset structure as EFNet.
1. (preferable) Create symbolic links to the dataset directories.
```bash
cd datasets
ln -s /path/to/HighREV HighREV 
cd ..
```
2. or update the dataset paths in `config.py` as follows:
```python
self.father_train_path_npz="path/to/datasets/HighREV/train/"
self.father_val_path_npz="path/to/datasets/HighREV/val/"
self.father_test_path_npz="path/to/datasets/HighREV/val/"
self.father_train_voxel_path = "path/to/datasets/HighREV_voxel/train/"
self.father_val_voxel_path = "/path/to/datasets/HighREV_voxel/val/"
self.father_test_voxel_path = "/path/to/datasets/HighREV_voxel/val/"
```

## Training
```bash 
python main_train.py --checkpoints finetune_checkpoints --config finetune_training.yml --log_dir finetune_logs --method unet
```
## Testing
```bash
python main_test.py --result_dir STCNet_output --config testing.yml
```

## Citations

```
@inproceedings{sun2023event,
  title={Event-based frame interpolation with ad-hoc deblurring},
  author={Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Sun, Peng and Cao, Jiezhang and Zhang, Kai and Jiang, Qi and Wang, Kaiwei and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18043--18052},
  year={2023}
}

@inproceedings{sun2022event,
  title={Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
  author={Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Gool, Luc Van},
  booktitle={European Conference on Computer Vision},
  pages={412--428},
  year={2022},
  organization={Springer}
}
```

## Citations
```
@inproceedings{wengEventBasedBlurry2023,
  title = {Event-Based Blurry Frame Interpolation Under Blind Exposure},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  author = {Weng, Wenming and Zhang, Yueyi and Xiong, Zhiwei},
  year = {2023},
  pages = {1588--1598}
}
```
