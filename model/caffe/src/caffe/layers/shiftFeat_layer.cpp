#include <vector>
#include <iostream>
#include "caffe/layers/shiftFeat_layer.hpp"

namespace caffe {

template<typename Dtype>
void ShiftFeatLayer< Dtype >::LayerSetUp(const vector< Blob< Dtype >* >& bottom,
                                      const vector< Blob< Dtype >* >& top) {
	// Nothing to setup for
}

template<typename Dtype>
void ShiftFeatLayer< Dtype >::Reshape(const vector< Blob< Dtype >* >& bottom,
                                   const vector< Blob< Dtype >* >& top) {
	const ShiftParameter& shift_param = this->layer_param_.shift_param();  
  	res = shift_param.res();
	step = shift_param.step();
	
	CHECK_EQ(bottom[0]->channels() % 3, 0) << "Remap requires input blob (bottom[0]) to have only 3 * views channels";
	top[0]->Reshape(bottom[0]->num(), res*2,
		bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ShiftFeatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    								const vector<Blob<Dtype>*>& top) {
	// Not implemented
}

template <typename Dtype>
void ShiftFeatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    								const vector<bool>& propagate_down,
    								const vector<Blob<Dtype>*>& bottom) {
	// Not implemented	
}

#ifdef CPU_ONLY
STUB_GPU(ShiftFeatLayer);
#endif
INSTANTIATE_CLASS(ShiftFeatLayer);
REGISTER_LAYER_CLASS(ShiftFeat);

}  // namespace caffe
