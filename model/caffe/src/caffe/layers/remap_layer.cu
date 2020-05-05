#include <vector>
#include <iostream>
#include "caffe/layers/remap_layer.hpp"
#include <curand.h>
#include <curand_kernel.h>

namespace caffe {

template<typename Dtype>
__device__ inline Dtype getvalue(Dtype* V, int x, int y, int c, int n, int C, int W, int H) {
	x = max(0, min(W-1, x));
	y = max(0, min(H-1, y));
	c = max(0, min(C-1, c));
	return V[((n*C+c)*H+y)*W+x];
}

template<typename Dtype>
__device__ inline void addvalue(Dtype* V, int x, int y, int c, int n, int C, int W, int H, Dtype v) {
	if ( x < 0 || x > W - 1 || y < 0 || y > H - 1 || c < 0 || c > C - 1) 
		return;
	V[((n*C+c)*H+y)*W+x] += v;
}


template <typename Dtype>
__global__ void RemapForward(const int nthreads,
	const Dtype* const Vb, const Dtype* const coords, 
	Dtype* const Vt,
	const int C, const int H0, const int H1,
	const int W0, const int W1) {
    
   // nthreads = N * C * H1 * W1
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / W1 / H1 / C;
		const int c = (index / W1 / H1) % C;
		const int h = (index / W1) % H1;
		const int w = index % W1;
		Dtype x = w + coords[((n*2+0)*H1+h)*W1+w];
		Dtype y = h + coords[((n*2+1)*H1+h)*W1+w];
		int x0 = floor(x); int x1 = x0 + 1; Dtype wx = x - x0;
		int y0 = floor(y); int y1 = y0 + 1; Dtype wy = y - y0;		
		Dtype w00 = (1 - wx) * (1 - wy), w01 = (1 - wx) * wy, w10 = wx * (1 - wy), w11 = wx * wy;
		Dtype v00 = getvalue(Vb, x0, y0, c, n, C, W0, H0);
		Dtype v01 = getvalue(Vb, x0, y1, c, n, C, W0, H0);
		Dtype v10 = getvalue(Vb, x1, y0, c, n, C, W0, H0);
		Dtype v11 = getvalue(Vb, x1, y1, c, n, C, W0, H0);
		Vt[index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
	} 
}

template <typename Dtype>
void RemapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    								const vector<Blob<Dtype>*>& top) {
	const Dtype* Vb = bottom[0]->gpu_data();
	const Dtype* coords = bottom[1]->gpu_data();
	Dtype* Vt = top[0]->mutable_gpu_data();
	int N  = bottom[0]->num();
	int C  = bottom[0]->channels();
	int W0 = bottom[0]->width();
	int H0 = bottom[0]->height();
	int W1 = bottom[1]->width();
	int H1 = bottom[1]->height();	

	int count = N * C * H1 * W1;
	RemapForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      	count, Vb, coords, Vt, C, H0, H1, W0, W1);
}

template <typename Dtype>
__global__ void RemapBackwardValue(const int nthreads,
const Dtype* const coords,
const Dtype* const top_diff, Dtype* const b0_diff,
const int C, const int H0, const int H1,
const int W0, const int W1) {
	// nthreads = N * C * H1 * W1
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / W1 / H1 / C;
		const int c = (index / W1 / H1) % C;
      const int h = (index / W1) % H1;
      const int w = index % W1;

      Dtype x = w + coords[((n*2+0)*H1+h)*W1+w];
      Dtype y = h + coords[((n*2+1)*H1+h)*W1+w];
      int x0 = floor(x); int y0 = floor(y);
      int x1 = x0 + 1;   int y1 = y0 + 1;
      Dtype wx = x - x0, wy = y - y0;

		Dtype dv00 = (1-wx) * (1-wy);
		Dtype dv01 = (1-wx) * wy;
		Dtype dv10 = wx * (1-wy);
		Dtype dv11 = wx * wy;
		// Backprop for bottom[0] (values)
		addvalue(b0_diff, x0, y0, c, n, C, W0, H0, dv00 * top_diff[index]);
		addvalue(b0_diff, x0, y1, c, n, C, W0, H0, dv01 * top_diff[index]);
		addvalue(b0_diff, x1, y0, c, n, C, W0, H0, dv10 * top_diff[index]);
		addvalue(b0_diff, x1, y1, c, n, C, W0, H0, dv11 * top_diff[index]);
	}
}

template <typename Dtype>
__global__ void RemapBackwardFlow(const int nthreads,
const Dtype* const Vb, const Dtype* const coords, 
const Dtype* const top_diff, Dtype* const b1_diff,
const int C, const int H0, const int H1,
const int W0, const int W1) {

    // nthreads = N * H1 * W1
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / W1 / H1;
		const int h = (index / W1) % H1;
		const int w = index % W1;

		for ( int c = 0; c < C; c++ ) {
			Dtype x = w + coords[((n*2+0)*H1+h)*W1+w];
			Dtype y = h + coords[((n*2+1)*H1+h)*W1+w];
			int x0 = floor(x); int y0 = floor(y);
			int x1 = x0 + 1;   int y1 = y0 + 1;
			Dtype wx = x - x0, wy = y - y0;			
			Dtype dx, dy;
			Dtype v00 = getvalue(Vb, x0, y0, c, n, C, W0, H0);
			Dtype v01 = getvalue(Vb, x0, y1, c, n, C, W0, H0);
			Dtype v10 = getvalue(Vb, x1, y0, c, n, C, W0, H0);
			Dtype v11 = getvalue(Vb, x1, y1, c, n, C, W0, H0);

			dx = (wy-1)*v00 - wy*v01 + (1-wy)*v10 + wy*v11;
			dy = (wx-1)*v00 - wx*v10 + (1-wx)*v01 + wx*v11;			
			
			// Backprop for bottom[1] (coords)
			addvalue(b1_diff, w, h, 0, n, 2, W1, H1, dx * top_diff[((n*C+c)*H1+h)*W1+w]);
			addvalue(b1_diff, w, h, 1, n, 2, W1, H1, dy * top_diff[((n*C+c)*H1+h)*W1+w]);
		}
	}
}

template <typename Dtype>
void RemapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    								const vector<bool>& propagate_down,
    								const vector<Blob<Dtype>*>& bottom) {

	const Dtype* Vb = bottom[0]->gpu_data();
	const Dtype* coords = bottom[1]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* b0_diff = bottom[0]->mutable_gpu_diff();
	Dtype* b1_diff = bottom[1]->mutable_gpu_diff();

	if (propagate_down[0])
   	caffe_gpu_set(bottom[0]->count(), Dtype(0), b0_diff);
	if (propagate_down[1])
		caffe_gpu_set(bottom[1]->count(), Dtype(0), b1_diff);
	
	int N = bottom[0]->num();
	int C = bottom[0]->channels();
	int W0 = bottom[0]->width();
	int H0 = bottom[0]->height();
	int W1 = bottom[1]->width();
	int H1 = bottom[1]->height();

	if (propagate_down[0]) {
		int count = N * C * W1 *H1;
		RemapBackwardValue<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      	count, coords, top_diff, b0_diff, C, H0, H1, W0, W1);
	}
	if (propagate_down[1]) {
		int count = N * W1 *H1;
		RemapBackwardFlow<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      	count, Vb, coords, top_diff, b1_diff, C, H0, H1, W0, W1);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(RemapLayer);

}  // namespace caffe
