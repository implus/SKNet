/*
 * CuDNNBatchNorm Layer
 *
 * mainly borrow from https://github.com/hujie-frank/GENet/blob/master/include/caffe/layers/cudnn_batch_norm_layer.hpp
 */

#ifndef CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNBatchNormLayer : public Layer<Dtype> {
 public:
  explicit CuDNNBatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNBatchNormLayer();

  virtual inline const char* type() const { return "CuDNNBatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    return DiagonalAffineMap<Dtype>::identity(2);
  }

  inline Blob<Dtype>* save_mean() { return &save_mean_; }
  inline Blob<Dtype>* save_inv_var() { return &save_inv_var_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // cuDNN descriptors / handles
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;
  cudnnBatchNormMode_t mode_;

  Blob<Dtype> save_mean_;
  Blob<Dtype> save_inv_var_;
  Blob<Dtype> input_dup_;
  bool handles_setup_;

  Blob<Dtype> combined_scale_;
  bool combined_;

  Dtype momentum_;
  Dtype eps_;
  // whether or not using moving average for inference
  bool frozen_;
  bool inplace_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
