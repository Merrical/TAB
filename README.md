# Transformer-based Annotation Bias-aware Medical Image Segmentation (TAB)
#### [Project Page](https://github.com/Merrical/TAB)

This repo contains the official implementation of our paper: Transformer-based Annotation Bias-aware Medical Image Segmentation, which highlights the issue of annotator-related biases existed in medical image segmentation tasks.
<p align="center"><img src="https://raw.githubusercontent.com/Merrical/TAB/master/TAB_overview.png" width="90%"></p>

#### [Paper](https://arxiv.org/abs/2306.01340.pdf)

### Requirements
This repo was tested with Ubuntu 20.04.4 LTS, Python 3.8, PyTorch 1.7.1, and CUDA 10.1.
We suggest using virtual env to configure the experimental environment.

1. Clone this repo:

```bash
git clone https://github.com/Merrical/TAB.git
```

2. Create experimental environment using virtual env:

```bash
virtualenv .env --python=3.8 # create
source .env/bin/activate # activate
pip install -r requirements.txt
```
### Dataset

The dataset details and the download link can be found in the [here](https://github.com/jiwei0921/MRNet).

### Training 

```bash
python main.py --dataset RIGA --phase train --net_arch TAB --masks --no_aux_loss --num_worker 8 \
--learning_rate 5e-5 --weight_decay 0.0 --num_epoch 300 \
--lambda_ 1.0 --rank 7 --loop 0
```

### Inference

```bash
python main.py --dataset RIGA --phase test --net_arch TAB --masks --no_aux_loss --num_worker 8 \
--learning_rate 5e-5 --weight_decay 0.0 --num_epoch 300 \
--lambda_ 1.0 --rank 7 --loop 0
```
### Bibtex
```
@inproceedings{Liao2023TAB,
  title={Transformer-based Annotation Bias-aware Medical Image Segmentation},
  author={Liao, Zehui and Hu, Shishuai and Xie, Yutong and Xia, Yong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2023},
  organization={Springer}
}
```

### Contact Us
If you have any questions, please contact us ( merrical@mail.nwpu.edu.cn ).
