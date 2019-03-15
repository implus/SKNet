# SKNet: Selective Kernel Networks <sub>([paper]())</sub>
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
Coming soon!

## Citation

If you use Selective Kernel Convolution in your research, please cite the paper:
    
