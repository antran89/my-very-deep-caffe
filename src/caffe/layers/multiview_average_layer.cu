#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multiview_average_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiviewAverageForward(const int nthreads, const Dtype* const bottom_data,
                                        const int channels, const int height,
                                        const int width, const int num_views, Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % width;
        const int h = (index / width) % height;
        const int c = (index / width / height) % channels;
        const int n = index / width / height / channels;
        Dtype aveval = 0;
        for (int v = 0; v < num_views; v++) {
            int bottom_index = (((n * num_views + v) * channels + c) * height + h) * width + w;
            aveval += bottom_data[bottom_index];
        }
        top_data[index] = aveval / num_views;
    }
}

template <typename Dtype>
void MultiviewAverageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    const int num_views = CAFFE_NUM_TEST_VIEWS;
    MultiviewAverageForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, bottom_data, channels_, height_, width_, num_views, top_data);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiviewAverageLayer);

}  // namespace caffe
