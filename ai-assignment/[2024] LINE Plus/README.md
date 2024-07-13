# [LINE Plus] VOOM AI lab - AI Engineer 과제 전형
LINE Plus VOOM AI lab - AI Engineer 포지션의 과제 전형에 참여할 기회를 주신 것에 감사드립니다.

# Introduction
Welcome to the Suprema face recognition(FR) framework!
This manual will help you getting started with the Suprema FR framework.
The table of contents is as follow:

# Contents
* [1. Getting Started](#1.-getting-started) <br>
* [2. Data Preparation](#2.-data-preparation) <br>
* [3. Training](#3.-training) <br>
* [4. Testing](#4.-testing) <br>
* [5. Model Weights](#5.-model-weights) <br>

## 1. Getting Started
#### 1) Experimental Settings

- **Hardware specifications** <br>
```GPU```: NVIDIA T4 Tensor Core GPU x 4

- **Requirements** <br>
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
