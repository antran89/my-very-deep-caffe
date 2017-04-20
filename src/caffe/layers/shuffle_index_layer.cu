#include <algorithm>
#include <utility>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "caffe/layers/shuffle_index_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void ShuffleIndexForward(const int num_axes, const int* d_bottom_shape, const int* d_top_shape,
                                    const int* d_new_axes, const int bottom_count,
                                    const Dtype* bottom_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, bottom_count) {
        // dynamic array allocation on gpu is too expensive, must roll
        // out a loopy shuffling index computation.
        // This loopy shuffling index implementation is cheap because
        // num_axes is usually small.
        int top_offset = 0, index_cnt, bot_axis, top_index;
        for (int i = 0; i < num_axes; i++) {
            bot_axis = d_new_axes[i];
            index_cnt = index;
            for (int k = num_axes-1; k > bot_axis; k--) {
                index_cnt /= d_bottom_shape[i];
            }
            top_index = index_cnt % d_bottom_shape[i];

            // accumulate top_offset
            top_offset *= d_top_shape[i];
            top_offset += top_index;
        }

        // moving data elements
        top_data[top_offset] = bottom_data[index];
    }
}

template<typename Dtype>
void ShuffleIndexLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int bottom_count = bottom[0]->count();
    vector<int> bottom_shape = bottom[0]->shape();
    vector<int> top_shape = top[0]->shape();
    int num_axes = bottom_shape.size();
    if (bottom_count == 0) return;

    // allocate memory for bottom_shape and top_shape on global GPU memory
    int *d_bottom_shape, *d_top_shape, *d_new_axes;
    cudaMalloc(&d_bottom_shape, num_axes);
    cudaMemcpy(d_bottom_shape, &bottom_shape[0], num_axes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_top_shape, num_axes);
    cudaMemcpy(d_top_shape, &top_shape[0], num_axes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_new_axes, num_axes);
    cudaMemcpy(d_new_axes, &new_axes_[0], num_axes, cudaMemcpyHostToDevice);

    ShuffleIndexForward<Dtype> <<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
    num_axes, d_bottom_shape, d_top_shape, d_new_axes, bottom_count, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
    cudaFree(d_bottom_shape);
    cudaFree(d_top_shape);
    cudaFree(d_new_axes);
}

template<typename Dtype>
__global__ void ShuffleIndexBackward(const int num_axes, const int* d_bottom_shape, const int* d_top_shape,
                                    const int* d_new_axes, const int bottom_count,
                                    const Dtype* top_diff, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, bottom_count) {
        // dynamic array allocation on gpu is too expensive, must roll
        // out a loopy shuffling index computation.
        // This loopy shuffling index implementation is cheap because
        // num_axes is usually small.
        int top_offset = 0, index_cnt, bot_axis, top_index;
        for (int i = 0; i < num_axes; i++) {
            bot_axis = d_new_axes[i];
            index_cnt = index;
            for (int k = num_axes-1; k > bot_axis; k--) {
                index_cnt /= d_bottom_shape[i];
            }
            top_index = index_cnt % d_bottom_shape[i];

            // accumulate top_offset
            top_offset *= d_top_shape[i];
            top_offset += top_index;
        }

        // moving data elements
        bottom_diff[index] = top_diff[top_offset];
    }
}

template<typename Dtype>
void ShuffleIndexLayer<Dtype>::Backward_gpu(
        const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
        const vector<Blob<Dtype> *> &bottom) {
    if (!propagate_down[0]) {
      return;
    }
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int bottom_count = bottom[0]->count();
    vector<int> bottom_shape = bottom[0]->shape();
    vector<int> top_shape = top[0]->shape();
    int num_axes = bottom_shape.size();
    if (bottom_count == 0) return;

    // allocate memory for bottom_shape and top_shape on global GPU memory
    int *d_bottom_shape, *d_top_shape, *d_new_axes;
    cudaMalloc(&d_bottom_shape, num_axes);
    cudaMemcpy(d_bottom_shape, &bottom_shape[0], num_axes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_top_shape, num_axes);
    cudaMemcpy(d_top_shape, &top_shape[0], num_axes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_new_axes, num_axes);
    cudaMemcpy(d_new_axes, &new_axes_[0], num_axes, cudaMemcpyHostToDevice);

    ShuffleIndexBackward<Dtype> <<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
    num_axes, d_bottom_shape, d_top_shape, d_new_axes, bottom_count, top_diff, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    cudaFree(d_bottom_shape);
    cudaFree(d_top_shape);
    cudaFree(d_new_axes);
}

INSTANTIATE_LAYER_GPU_FUNCS(ShuffleIndexLayer);

}  // namespace caffe
