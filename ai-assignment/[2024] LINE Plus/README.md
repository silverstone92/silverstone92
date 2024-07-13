# [LINE Plus] VOOM AI lab - AI Engineer 과제 전형
LINE Plus VOOM AI lab - AI Engineer 포지션의 과제 전형에 참여할 기회를 주신 것에 감사드립니다.

# Introduction
LINE Plus VOOM AI lab - AI Engineer 포지션의 과제 전형에 참여할 기회를 주신 것에 감사드립니다. <br>
본 Repo에서는 과제 전형 (TYPE B) - “InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning” 논문의 실험을 재현합니다. <br>
해당 논문에서 제안하는 InfoBatch 기법을 사용했을시 Baseline 모델 대비 학습 속도는 개선 되면서 성능은 유지됨을 확인하는 것이 본 Repo의 목표입니다. <br>
컨텐츠는 다음과 같습니다:

## Contents
* [1. Getting Started](#1.-getting-started) <br>
* [2. Data Preparation](#2.-data-preparation) <br>
* [3. Training](#3.-training) <br>
* [4. Testing](#4.-testing) <br>
* [5. Experimental Results](#5.-experimental-results) <br>
* [6. Citation](#5.-experimental-results) <br>

## 1. Getting Started
### 1) Experimental Settings
##### Hardware specifications <br>
- ```GPU```: NVIDIA T4 Tensor Core GPU x 4
##### Requirements <br>
본 과제는 torch 2.0.1+cu117 & torchvision 0.16.2+cu118을 이용하여 수행되었습니다. <br>
터미널에서 아래 명령어를 입력하여 필요한 라이브러리를 설치할 수 있습니다.
``` 
$ pip install -r requirements.txt
```

## 3.

## 4.

## 5. Experimental Results

### CIFAR-10 Dataset (ResNet-18)
|Method|Best Accuracy (%)|Total Training Time (Min)|
|:---:|:---:|:---:|
|Full Dataset|94.98|166.75|
|InfoBatch (r=0.3)|94.96|137.92|
|InfoBatch (r=0.5)|94.79|111.83|
|InfoBatch (r=0.7)|94.34|86.49|

## Citation
```bibtex
@inproceedings{
  title={InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning},
  author={Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Zhaopan Xu and Zhou, Daquan and Lei Shang and Baigui Sun and Xuansong Xie and You, Yang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=C61sk5LsK6}
}
```
