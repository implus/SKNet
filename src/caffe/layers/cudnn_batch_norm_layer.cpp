/*
 * CuDNNBatchNorm Layer
 * mainly borrow from https://github.com/hujie-frank/GENet/blob/master/src/caffe/layers/cudnn_batch_norm_layer.cpp
 */

#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  momentum_ = param.momentum();
  eps_ = param.eps();
  frozen_ = param.frozen();
  inplace_ = (bottom[0] == top[0]);

#if CUDNN_VERSION_MIN(7, 0, 0)
  mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  mode_ = CUDNN_BATCHNORM_SPATIAL;      // only SPATIAL mode is supported
#endif

  int channels = bottom[0]->channels();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    this->blobs_[0].reset(new Blob<Dtype>(1, channels, 1, 1));
    this->blobs_[1].reset(new Blob<Dtype>(1, channels, 1, 1));
    this->blobs_[2].reset(new Blob<Dtype>(1, channels, 1, 1));
    this->blobs_[3].reset(new Blob<Dtype>(1, channels, 1, 1));

    shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(param.scale_filler()));
    scale_filler->Fill(this->blobs_[0].get());

    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(param.bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());

    caffe_set(this->blobs_[2]->count(), Dtype(0),
        this->blobs_[2]->mutable_cpu_data());

    caffe_set(this->blobs_[3]->count(), frozen_ ? Dtype(1) : Dtype(0),
        this->blobs_[3]->mutable_cpu_data());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // runing average stats does not use weight decay and learning rate
  while (this->layer_param_.param_size() < 4) {
    this->layer_param_.mutable_param()->Add();
  }
  this->layer_param_.mutable_param(2)->set_lr_mult(Dtype(0));
  this->layer_param_.mutable_param(2)->set_decay_mult(Dtype(0));

  this->layer_param_.mutable_param(3)->set_lr_mult(Dtype(0));
  this->layer_param_.mutable_param(3)->set_decay_mult(Dtype(0));

  // shutdown scale and bias update in frozen mode
  if (frozen_) {
    // slope
    this->layer_param_.mutable_param(0)->set_lr_mult(Dtype(0));
    this->layer_param_.mutable_param(0)->set_decay_mult(Dtype(0));

    // bias
    this->layer_param_.mutable_param(1)->set_lr_mult(Dtype(0));
    this->layer_param_.mutable_param(1)->set_decay_mult(Dtype(0));

    LOG(INFO) << "BatchNorm frozen mode: force to set all parameters lr_mult and decay_mult to 0!";
  }

  // Initialize desc.
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&scale_bias_mean_var_desc_);

  handles_setup_ = true;
  combined_ = false;

  LOG(INFO) << "Using CuDNN BN engine.";
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);

  // set up main tensors
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());

  // aux tensors for caching mean & invVar from fwd to bwd pass
  int C = bottom[0]->channels();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
#if CUDNN_VERSION_MIN(7, 0, 0)
  if (mode_ == CUDNN_BATCHNORM_SPATIAL || mode_ == CUDNN_BATCHNORM_SPATIAL_PERSISTENT) {
#else
  if (mode_ == CUDNN_BATCHNORM_SPATIAL) {
#endif
    save_mean_.Reshape(1, C, 1, 1);
    save_inv_var_.Reshape(1, C, 1, 1);
    combined_scale_.Reshape(1, C, 1, 1);
  } else if (mode_ == CUDNN_BATCHNORM_PER_ACTIVATION) {
    save_mean_.Reshape(1, C, H, W);
    save_inv_var_.Reshape(1, C, H, W);
    combined_scale_.Reshape(1, C, H, W);
  } else {
    LOG(FATAL) << "Unknown cudnnBatchNormMode_t";
  }
  if (inplace_) { 
    input_dup_.ReshapeLike(*bottom[0]);
  }
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc_,
      bottom_desc_, mode_));
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
CuDNNBatchNormLayer<Dtype>::~CuDNNBatchNormLayer() {
  if (!handles_setup_) return;

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_);
}

INSTANTIATE_CLASS(CuDNNBatchNormLayer);
REGISTER_LAYER_CLASS(CuDNNBatchNorm);
}  // namespace caffe

#endif
