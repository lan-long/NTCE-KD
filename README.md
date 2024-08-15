# NTCE-KD
the official implementation of “NTCE-KD: Non-Target-Class-Enhanced Knowledge Distillation”

### Framework

<div style="text-align:center"><img src="pic/ntce-kd.png" width="80%" ></div>


### Main Benchmark Results

On CIFAR-100:


| Teacher <br> Student |ResNet56 <br> ResNet20|ResNet110 <br> ResNet32| ResNet32x4 <br> ResNet8x4| WRN-40-2 <br> WRN-16-2| WRN-40-2 <br> WRN-40-1 | VGG13 <br> VGG8|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|
| KD | 70.66 | 73.08 | 73.33 | 74.92 | 73.54 | 72.98 |
| **MKL** | **72.16** | **74.41** | **76.91** | **76.58** | **74.92** | **74.89** |
| **NTCE-KD** | **73.46** | **75.71** | **78.66** | **77.65** | **76.44** | **76.33** |


| Teacher <br> Student |ResNet32x4 <br> ShuffleNet-V1|WRN-40-2 <br> ShuffleNet-V1| VGG13 <br> MobileNet-V2| ResNet50 <br> MobileNet-V2| ResNet32x4 <br> MobileNet-V2|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|
| KD | 74.07 | 74.83 | 67.37 | 67.35 | 74.45 |
| **MKL** | **76.81** | **77.01** | **70.13** | **70.52** | **77.10** |
| **NTCE-KD** | **78.43** | **78.66** | **72.06** | **72.85** | **79.43** |


On ImageNet:

| Teacher <br> Student |ResNet34 <br> ResNet18|ResNet50 <br> MobileNet-V1|
|:---------------:|:-----------------:|:-----------------:|
| KD | 70.66 | 68.58 | 
| **MKL** | **72.07** | **72.79** |
| **NTCE-KD** | **73.12** | **74.90** |


### Getting started

1. Evaluation

- You can evaluate the performance of our models or models trained by yourself.

- Our models are downloaded from <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>, please download the checkpoints to `./download_ckpts`

- If test the models on ImageNet, please download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  python3 tools/eval.py -m resnet32x4 # resnet32x4 on cifar100
  python3 tools/eval.py -m ResNet34 -d imagenet # ResNet34 on imagenet
  ```

2. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # for our MKL method.
  python3 tools/train.py --cfg configs/cifar100/ntce/res32x4_res8x4.yaml

  # for our NTCE-KD method.
  python3 tools/train.py --cfg configs/cifar100/ntce/res32x4_res8x4.yaml NTCE.ISAUG Ture
  ```

3. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # for our MKL method.
  python3 tools/train.py --cfg configs/imagenet/r34_r18/ntce.yaml

  
  # for our NTCE-KD method.
  python3 tools/train.py --cfg configs/imagenet/r34_r18/ntce.yaml NTCE.ISAUG Ture
  ```



# Acknowledgement

- Thanks for DKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller).
