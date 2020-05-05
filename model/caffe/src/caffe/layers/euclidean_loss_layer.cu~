#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void clip_kernel(const int n, Dtype* a, Dtype threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    if (a[index] > threshold) a[index] = threshold;
    else if (a[index] < -threshold) a[index] = -threshold;
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  /*int N  = bottom[0]->num();
  int C  = bottom[0]->channels();
  int W = bottom[0]->width();
  int H = bottom[0]->height();
  Dtype* v = bottom[0]->mutable_cpu_data();
  const Dtype* v2 = bottom[1]->cpu_data();
  for ( int n = 0; n < N; n++ ) {
    for ( int c = 0; c < C; c++ ) {		
      for ( int h = 0; h < H; h++ ) {
        for ( int w = 0; w < W; w++ ) {
          if (v[((n*C+c)*H+h)*W+w] == 0) v[((n*C+c)*H+h)*W+w] = v2[((n*C+c)*H+h)*W+w];
        }
      }
    }
  }*/
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  ////////////////// robust clip ///////////////
  Dtype threshold = this->layer_param_.threshold_param().threshold();
  if (threshold != 0)
    clip_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.mutable_gpu_data(), threshold);
  //////////////////////////////////////////////
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
  /*int N  = bottom[0]->num();
  int C  = bottom[0]->channels();
  int W = bottom[0]->width();
  int H = bottom[0]->height();
  Dtype* d = bottom[0]->mutable_cpu_diff();
  const Dtype* v = bottom[0]->cpu_data();
  for ( int n = 0; n < N; n++ ) {
    for ( int c = 0; c < C; c++ ) {		
      for ( int h = 0; h < H; h++ ) {
        for ( int w = 0; w < W; w++ ) {
          if (v[((n*C+c)*H+h)*W+w] == 0) d[((n*C+c)*H+h)*W+w] = 0;
        }
      }
    }
  }*/
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
