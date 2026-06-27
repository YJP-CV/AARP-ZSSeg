# AARP: Attention Adaptation with Reinforced Prompt for Zero-Shot Semantic Segmentation

This repository contains the PyTorch implementation of **AARP: Attention Adaptation with Reinforced Prompt for Zero-Shot Semantic Segmentation**.

AARP is a zero-shot semantic segmentation framework built upon the Segment Anything Model (SAM). It aims to improve SAM-based segmentation under challenging domain shifts, such as low-light scenes, rainy weather, reflection, and complex traffic scenarios.

## Overview

Zero-shot semantic segmentation enables models to segment object categories that are not seen during training. Although SAM has shown strong generalization ability, its performance may degrade in challenging real-world scenarios due to domain gaps and prompt sensitivity.

To address these issues, AARP introduces two key modules:

* **Expert Attention Adaptation (EAA)**
  An SVD-driven Mixture-of-Experts adaptation module that learns adaptive perturbations in the self-attention layers of the SAM image encoder.

* **Reinforced Point Selection (RPS)**
  A reinforcement-based point prompt generation and selection module that dynamically selects structure-aware point prompts for accurate segmentation.

The overall framework decouples feature adaptation and prompt generation, improving both domain robustness and object separation.

## Main Contributions

* We propose **AARP**, a SAM-based zero-shot semantic segmentation framework for challenging scenarios.
* We introduce **Expert Attention Adaptation (EAA)** to adapt self-attention features through SVD-based expert perturbation.
* We design **Reinforced Point Selection (RPS)** to generate and select optimal point prompts using a point-query quadtree and reinforcement learning.
* Extensive experiments demonstrate strong performance on multiple datasets, including Hangzhou Traffic, GM Traffic, and MSCOCO.

## Framework

The AARP pipeline consists of the following components:

1. Input image is processed by the SAM image encoder.
2. EAA is inserted into the self-attention blocks to adapt image features.
3. RPS generates candidate point prompts and selects optimal prompts.
4. Selected prompts are encoded by the SAM prompt encoder.
5. The SAM mask decoder predicts the final segmentation masks.

## Installation

```bash
git clone https://github.com/your-username/AARP-ZSSeg.git
cd AARP-ZSSeg

conda create -n aarp python=3.9 -y
conda activate aarp

pip install -r requirements.txt
```

## Requirements

```text
python >= 3.8
pytorch >= 1.10
torchvision
opencv-python
numpy
tqdm
matplotlib
segment-anything
```

## Dataset Preparation

Please organize the datasets as follows:

```text
datasets/
├── Hangzhou_Traffic/
│   ├── train/
│   ├── val/
│   └── test/
├── GM_Traffic/
│   ├── train/
│   ├── val/
│   └── test/
└── MSCOCO/
    ├── train/
    ├── val/
    └── test/
```

The dataset path can be specified in the configuration file:

```yaml
DATASET:
  name: Hangzhou_Traffic
  root: ./datasets/Hangzhou_Traffic
```

## Training

To train AARP on the Hangzhou Traffic dataset:

```bash
python train.py \
  --config configs/aarp_hangzhou.yaml \
  --dataset Hangzhou_Traffic \
  --sam_checkpoint checkpoints/sam_vit_h.pth
```

To train on the GM Traffic dataset:

```bash
python train.py \
  --config configs/aarp_gm.yaml \
  --dataset GM_Traffic \
  --sam_checkpoint checkpoints/sam_vit_h.pth
```

## Evaluation

To evaluate the trained model:

```bash
python test.py \
  --config configs/aarp_hangzhou.yaml \
  --checkpoint outputs/aarp_hangzhou/best_model.pth \
  --dataset Hangzhou_Traffic
```

The evaluation metrics include:

* mean Intersection over Union, mIoU
* mean Accuracy, mAcc

## Results

| Method         | Hangzhou mIoU | Hangzhou mAcc | GM mIoU | GM mAcc | MSCOCO mIoU | MSCOCO mAcc |
| -------------- | ------------: | ------------: | ------: | ------: | ----------: | ----------: |
| SAM-h Baseline |          66.8 |          77.5 |    68.4 |    79.4 |           - |           - |
| AARP           |          76.6 |          84.9 |    76.8 |    85.2 |        77.5 |        85.9 |

## Ablation Study

| EAA | RPS | Hangzhou mIoU | Hangzhou mAcc | GM mIoU | GM mAcc |
| --- | --- | ------------: | ------------: | ------: | ------: |
| ✗   | ✗   |          66.8 |          77.5 |    68.4 |    79.4 |
| ✓   | ✗   |          74.7 |          83.9 |    75.3 |    84.6 |
| ✗   | ✓   |          71.2 |          81.2 |    71.9 |    81.9 |
| ✓   | ✓   |          76.6 |          84.9 |    76.8 |    85.2 |

## Repository Structure

```text
AARP-ZSSeg/
├── configs/
│   ├── aarp_hangzhou.yaml
│   ├── aarp_gm.yaml
│   └── aarp_mscoco.yaml
├── datasets/
├── models/
│   ├── eaa.py
│   ├── rps.py
│   └── aarp.py
├── tools/
│   ├── train.py
│   ├── test.py
│   └── visualize.py
├── utils/
├── checkpoints/
├── outputs/
├── requirements.txt
└── README.md
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhuo2026aarp,
  title={AARP: Attention Adaptation with Reinforced Prompt for Zero-Shot Semantic Segmentation},
  author={Zhuo, Yue and Ye, Jinpeng and Zhou, Di and Xu, Pengpeng and Tian, Yan},
  journal={},
  year={2026}
}
```

## Acknowledgement

This project is built upon the Segment Anything Model. We thank the authors of SAM and related zero-shot semantic segmentation methods for their excellent work.

## License

This project is released for academic research purposes only. Please refer to the license file for more details.
