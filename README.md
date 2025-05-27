<h1 align="center">Every Angle Is Worth A Second Glance: Mining Kinematic Skeletal Structures from Multi-view Joint Cloud</h1> 
<h3 align="center">IEEE Transactions on Visualization and Computer Graphics</h3> 
<h3 align="center">
<a href="https://arxiv.org/abs/2502.02936">Paper</a> | <a href="https://jjkislele.github.io/pages/projects/jcsatMocap/">Project Page</a> 
</h3> 

## Installation

To get started, please follow the steps as listed:

```bash
# clone this repository and init the submodules
git clone --recursive https://github.com/HKBU-VSComputing/2025_TVCG_JointCloud-Mocap.git

# create a conda environment (python==3.10)
conda env create -f environment.yml
conda activate jcsat_mocap

# install a specific pytorch version and cuda support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

The code has been tested on Windows/Linux under the environment:
```bash
print("torch.version:", torch.__version__)
# torch.version: 2.6.0+cu126
print("CUDA.version:", torch.version.cuda)
# CUDA.version: 12.6
print("cuDNN.version:", torch.backends.cudnn.version())
# cuDNN.version: 90501
```

## Data and pre-trained model

All have been uploaded to [Google Drive](https://drive.google.com/drive/folders/104lxYpTD9v1zZUF_gQGoW1mL6POyj30G?usp=drive_link). Please download them and put them into the correct location.

```
./data/prepare/
├── shelf_data/
│   ├── camera_params.npy
│   ├── shelf_3d_eval_4dassoc.npy
│   ├── shelf_with_gt_in_shelf14.npy
│   └── shelf_with_gt_in_skel19.npy
└── shelf_test/
│   └── shelf_preprocess_v4_0310.npy
./pretrained/
└── shelf_best_checkpoint-119.pth
```

### Prepare Joint Cloud

```bash
python -m data.prepare_joint_cloud
```

## Training

```bash
python train.py
```

## Evaluation

```bash
python inference.py \
--model_path ./pretrained/shelf_best_checkpoint-119.pth \
--dataset_name shelf
```

```bash
python evaluation.py \
--prediction_path ./pretrained/shelf_best_checkpoint-119_results \
--dataset_name shelf
```

The results are stored in `pretrained/shelf_best_checkpoint-119_results/pred.log` and the metrics are stored in `pretrained/shelf_best_checkpoint-119_results/eval.log`

## BibTeX

If you find our work helpful or use our code, please consider citing:

```bibtex
@article{jiang2025every,
  title={Every Angle Is Worth A Second Glance: Mining Kinematic Skeletal Structures from Multi-view Joint Cloud},
  author={Jiang, Junkun and Chen, Jie and Au, Ho Yin and Chen, Mingyuan and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2502.02936},
  year={2025}
}
```

## Acknowledgement

We would like to thank the following contributors whose remarkable work has served as the foundation for our code:

- [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [Optimal Transport Kernel Embedding](https://github.com/claying/OTK)
- [3DETR: An End-to-End Transformer Model for 3D Object Detection](https://github.com/facebookresearch/3detr)