#ifndef CAFFE_REMAP_LAYER_HPP_
#define CAFFE_REMAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RemapLayer : public Layer<Dtype> {
 public:
  explicit RemapLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Remap"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  //virtual inline int MinBottomBlobs() const { return 3; }
  //virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}
#endif  // CAFFE_REMAP_LAYER_HPP_
