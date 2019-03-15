/*
 * CuDNNBatchNorm Layer
 * mainly borrow from https://github.com/hujie-frank/GENet/blob/master/src/caffe/layers/cudnn_batch_norm_layer.cu
 */

#ifdef USE_CUDNN

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* bias_data = this->blobs_[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);

  if (this->phase_ == TEST || this->frozen_) {
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        Caffe::cudnn_handle(), CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_data,
        bottom_desc_, top_data,
        scale_bias_mean_var_desc_, scale_data, bias_data,
        this->blobs_[2]->gpu_data(),  // mean
        this->blobs_[3]->gpu_data(),  // variance
        epsilon));
  } else {
    Dtype* save_mean = save_mean_.mutable_gpu_data();
    Dtype* save_inv_var = save_inv_var_.mutable_gpu_data();
    if (inplace_) {
      caffe_copy(bottom[0]->count(), bottom_data, input_dup_.mutable_gpu_data()); 
    }
    // Call Batch normalization forward
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        Caffe::cudnn_handle(), mode_,
        cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_data,
        bottom_desc_, top_data,
        scale_bias_mean_var_desc_, scale_data, bias_data,
        1 - this->momentum_,
        this->blobs_[2]->mutable_gpu_data(),  // mean
        this->blobs_[3]->mutable_gpu_data(),  // variance
        epsilon, save_mean, save_inv_var));
  }
}

template <typename Dtype>
__global__ void combine_scale(const int count, const double eps, const Dtype* scale,
    const Dtype* var, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = scale[index] / sqrt(var[index] + eps);
  }
}

template <typename Dtype>
__global__ void cudnnBatchNormalizationBackwardFrozen(const int count, 
    const int c, const int dim, const Dtype* scale, const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, count) {
    const int id = index / dim % c; 
    bottom_diff[index] = scale[id] * top_diff[index];
  }
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = inplace_? input_dup_.gpu_data() : bottom[0]->gpu_data();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();

  double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);

  if (frozen_) {
    if (!combined_) {
      const int count = this->blobs_[0]->count();
      combine_scale<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, epsilon,
          scale_data, this->blobs_[3]->gpu_data(), combined_scale_.mutable_gpu_data());
      combined_ = true;
    } 
    const int count = bottom[0]->count();
    cudnnBatchNormalizationBackwardFrozen<Dtype><<<CAFFE_GET_BLOCKS(count), 
        CAFFE_CUDA_NUM_THREADS>>>(count, bottom[0]->shape(1), 
        bottom[0]->count(2), combined_scale_.gpu_data(), top_diff, bottom_diff);
  } else {
    const Dtype* save_mean = save_mean_.gpu_data();
    const Dtype* save_inv_var = save_inv_var_.gpu_data();
    Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();

    // call Batch Normalization Backward
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        Caffe::cudnn_handle(), mode_,
        cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
#if CUDNN_VERSION >= 4005
        cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::one,
#endif
        bottom_desc_, bottom_data,
        bottom_desc_, top_diff,
        bottom_desc_, bottom_diff,
        scale_bias_mean_var_desc_,
        scale_data, scale_diff, bias_diff,
        epsilon, save_mean, save_inv_var));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);

}  // namespace caffe

#endif
