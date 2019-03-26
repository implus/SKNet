# SKNet: Selective Kernel Networks <sub>([paper](https://arxiv.org/pdf/1903.06586.pdf))</sub>
By Xiang Li<sup>[1,2]</sup>, Wenhai Wang<sup>[3,2]</sup>, Xiaolin Hu<sup>[4]</sup> and Jian Yang<sup>[1]</sup>

[PCALab, Nanjing University of Science and Technology]<sup>[1]</sup> [Momenta](https://momenta.ai/)<sup>[2]</sup> [Nanjing University]<sup>[3]</sup> [Tsinghua University]<sup>[4]</sup>.

## Approach
<div align="center">
  <img src="https://github.com/implus/SKNet/blob/master/figures/sknet.jpg">
</div>
<p align="center">
  Figure 1: The Diagram of a Selective Kernel Convolution module.
</p>


## Implementation
In this repository, all the models are implemented by [Caffe](https://github.com/BVLC/caffe).
 
We use the data augmentation strategies with [SENet](https://github.com/hujie-frank/SENet). 

There are two new layers introduced for efficient training and inference, these are *Axpy* and *CuDNNBatchNorm* layers.  
+ The [*Axpy*](https://github.com/hujie-frank/SENet/blob/master/src/caffe/layers/) layer is already implemented in [SENet](https://github.com/hujie-frank/SENet).
+ The [*CuDNNBatchNorm*] is mainly borrowed from [GENet](https://github.com/hujie-frank/GENet).

## Trained Models
Table 2. Single crop validation error on ImageNet-1k (center 224x224/320x320 crop from resized image with shorter side = 256). 

| Model | Top-1 224x | Top-1 320x | #P | GFLOPs | 
|:-:|:-:|:-:|:-:|:-:|
|ResNeXt-50        |22.23|21.05|25.0M|4.24|
|AttentionNeXt-56  |21.76|–    |31.9M|6.32|
|InceptionV3       |–    |21.20|27.1M|5.73|
|ResNeXt-50 + BAM  |21.70|20.15|25.4M|4.31|
|ResNeXt-50 + CBAM |21.40|20.38|27.7M|4.25|
|SENet-50          |21.12|19.71|27.7M|4.25|
|SKNet-50          |20.79|19.32|27.5M|4.47|
|ResNeXt-101       |21.11|19.86|44.3M|7.99|
|Attention-92      | –   |19.50|51.3M|10.43|
|DPN-92            |20.70|19.30|37.7M|6.50|
|DPN-98            |20.20|18.90|61.6M|11.70|
|InceptionV4       | –   |20.00|42.0M|12.31|
|Inception-ResNetV2| –   |19.90|55.0M|13.22|
|ResNeXt-101 + BAM |20.67|19.15|44.6M|8.05|
|ResNeXt-101 + CBAM|20.60|19.42|49.2M|8.00|
|SENet-101         |20.58|18.61|49.2M|8.00|
|SKNet-101         |20.19|18.40|48.9M|8.46|

Download:

|Model|caffe model|
|:-:|:-:|
|SKNet-50|[GoogleDrive](https://drive.google.com/file/d/1EKanqFkqoU3L6vgSLW3GjPciesZ2rrUH/view?usp=sharing)|
|SKNet-101|[GoogleDrive](https://drive.google.com/file/d/1NEYIYeSXQeinLGU5GH2hf8T6MJytzt0U/view?usp=sharing)|

20190323_Update: SKNet-101 model is deleted by mistake. We are retraining a model and it will come soon in 2-3 days.
20190326_Update: SKNet-101 model is ready.

## Attention weights correspond to object scales in low/middle layers
We look deep into the selection distributions from the perspective of classes on SK_2_3 (low), SK_3_4 (middle), SK_5_3 (high) layers:
<div align="center">
  <img src="https://github.com/implus/SKNet/blob/master/figures/cls_attention_diff.jpg">
</div>
<p align="center">
  Figure 2: Average mean attention difference (mean attention value of kernel 5x5 minus that of kernel 3x3) on SK units of SKNet-50, for each of 1,000 categories using all validation samples on ImageNet. On low or middle level SK units (e.g., SK\_2\_3, SK\_3\_4), 5x5 kernels are clearly imposed with more emphasis if the target object becomes larger (1.0x -> 1.5x).
</p>

More details of attention distributions on specific images are as follows:
<div align="center">
  <img src="https://github.com/implus/SKNet/blob/master/figures/pics_attention_3_scales.png">
</div>



## Citation

If you use Selective Kernel Convolution in your research, please cite the paper:
    
    @inproceedings{li2019selective,
      title={Selective Kernel Networks},
      author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Yang, Jian},
      journal={IEEE Conference on Computer Vision and Pattern Recognition},
      year={2019}
    }
