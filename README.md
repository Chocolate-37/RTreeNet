# RTreeNet
A Fine-Scale Segmentation Method for Individual Rubber Trees Based on UAV LiDAR Point Cloud

## Install

```bash
# step 1. clone this repo
git clone https://github.com/ma-xu/pointMLP-pytorch.git
cd pointMLP-pytorch

# step 2. create a conda virtual environment and activate it
conda env create
conda activate pointmlp
```

```bash
# Optional solution for step 2: install libs step by step
conda create -n pointmlp python=3.7 -y
conda activate pointmlp
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
# if you are using Ampere GPUs (e.g., A100 and 30X0), please install compatible Pytorch and CUDA versions, like:
# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```
