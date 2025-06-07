# RTreeNet
A Fine-Scale Segmentation Method for Individual Rubber Trees Based on UAV LiDAR Point Cloud

## Install

```bash
# install libs step by step
conda create -n rtreenet python=3.7 -y
conda activate rtreenet
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
# if you are using Ampere GPUs (e.g., A100 and 30X0), please install compatible Pytorch and CUDA versions, like:
# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```
### Tree Fine-Scale Segmentation

- Train RTreeNet
```bash
cd part_segmentation
python main.py --model RTreeNet
# please add other paramemters as you wish.
```

- Test RTreeNet
```bash
cd part_segmentation
# Suppose your weight file is actually located at: checkpoints/Demo1/best_insiou_model.pth
python main_CPA_per_recall_f1.py --eval True --exp_name Demo1 --model_type insiou --model RTreeNet
# please add other paramemters as you wish.
```

