# [NTIRE 2025 the First Challenge on Event-Based Deblurring](https://codalab.lisn.upsaclay.fr/competitions/21498) @ [CVPR 2025](https://cvlai.net/ntire/2025/)

## Team BUPTMM

| TeamName |  PSNR | SSIM |
|:--------:|:-----:|:----:|
|  BUPTMM  | 40.204 | 0.92 |

### Checkpoints
- Pretrained model download command: `wget https://1drv.ms/u/c/ea2a370299726471/EcDVUzTQwuNLozRtZUHvpCUB8zJyWhiOaY-PbMQM4dFDtw?e=I2gAHG`
After downloading, put it to `model_zoo/`

### Results
- Result of PSNR40.204 download command: `wget https://1drv.ms/u/c/ea2a370299726471/EcBi4zfln8dOsmerhs8kr9YBlsAlW3EpZNtbMyrRadB2-g?e=qBDhSd`


## Dataset Structure
The structure of the HighREV dataset with raw events is as following:

```
    --HighREV
    |----train
    |    |----blur
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |    |----event
    |    |    |----SEQNAME_%5d_%2d.npz
    |    |    |----...
    |    |----voxel
    |    |    |----SEQNAME_%5d.npz
    |    |    |----...
    |    |----voxel_bin24
    |    |    |----SEQNAME_%5d.npz
    |    |    |----...
    |    |----sharp
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |----val
    ...
```


### Converting events to voxel
We use `./basicsr/utils/npz2voxel.py` script to convert the raw events to voxel grids (bin=24) for better performance in EFNet training process.

#### Dataset codes:
- `basicsr/data/buptmm_voxelnpz_image_dataset.py` for processing voxel grids.
- `basicsr/data/buptmm_finaltest_voxelnpz_image_dataset.py` for processing voxel grids.






## EFNet

### Installation
```bash
cd NTIRE2025_EventDeblur_challenge
conda create -n efnet python=3.8.5  # necessary to create a new env for basicsr
conda activate efnet
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python setup.py develop --no_cuda_ext  # necessary when adding new functions or moving files, --no_cuda_ext will not install cuda extension
```

### How to start training?
Single GPU training:
```bash
python basicsr/train.py -opt options/train/HighREV/buptmm_EFNet_bin32_tunebin24.yml
```
Multi-GPU training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/HighREV/buptmm_EFNet_bin32_tunebin24.yml --launcher pytorch
```


### How to start testing?
```bash
python basicsr/test_demo.py -opt /root/work/NTIRE2025_EventDeblur_challenge/options/test/HighREV/BUPTMM_EFNet_test_highrev_bin24.yml
```

Calculating flops:
set ``print_flops`` to ``true`` and set your input shapes in ``flops_input_shape`` in the test yml file.
Example:
```
print_flops: true 
flops_input_shape: 
  - [3, 256, 256] # image shape
  - [6, 256, 256] # event shape
```
## STCNet
Please refer to the [STCNet folder](STCNet/README.md)


## Reproduce Results
After testing of EFNet and STCNet, use `merge.py` to get the final results.
Put results of EFNet to `EFNet_output` folder, and put results of STCNet to `STCNet_output` folder. 
Then run: `python merge.py`


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

@article{yangMotionDeblurring2024,
  title = {Motion Deblurring via Spatial-Temporal Collaboration of Frames and Events},
  author = {Yang, Wen and Wu, Jinjian and Ma, Jupo and Li, Leida and Shi, Guangming},
  year = {2024},
  month = mar,
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {38},
  number = {7},
  pages = {6531--6539}
}

@inproceedings{wengEventBasedBlurry2023,
  title = {Event-Based Blurry Frame Interpolation Under Blind Exposure},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  author = {Weng, Wenming and Zhang, Yueyi and Xiong, Zhiwei},
  year = {2023},
  pages = {1588--1598}
}
```
