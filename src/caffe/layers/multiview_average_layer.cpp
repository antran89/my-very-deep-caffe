#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multiview_average_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiviewAverageLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK(this->phase_ == caffe::TEST) << "MultiviewAverageLayer only available in "
                                        "TEST phase.";
  int num = bottom[0]->shape(0);
  CHECK_EQ(num % CAFFE_NUM_TEST_VIEWS, 0)
          << "first dimension of bottom blob must be divisible by number of views";
  // reshape the top blob
  vector<int> bottom_shape = bottom[0]->shape();
  if (bottom_shape.size() > 1)
      channels_ = bottom_shape[1];
  else
      channels_ = 1;
  if (bottom_shape.size() > 2) {
      height_ = bottom_shape[2];
      width_ = bottom_shape[3];
  }
  else
  {
      height_ = 1;
      width_ = 1;
  }
  bottom_shape[0] = num / CAFFE_NUM_TEST_VIEWS;
  top[0]->Reshape(bottom_shape);
}

template <typename Dtype>
void MultiviewAverageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int top_count = top[0]->count();
    const int num_views = CAFFE_NUM_TEST_VIEWS;
    int top_num = top[0]->shape(0);

    // the main loop
    int top_index = 0;
    for (int n = 0; n < top_num; n++)
        for (int c = 0; c < channels_; c++)
            for (int h = 0; h < height_; h++)
                for (int w = 0; w < width_; w++) {
                    Dtype aveval = 0;
                    for (int v = 0; v < num_views; v++)
                    {
                        int bottom_index = (((n * num_views + v)
                                         * channels_ + c) * height_ + h) * width_ + w;
                        aveval += bottom_data[bottom_index];
                    }
                    top_data[top_index++] = aveval / num_views;
                }
    CHECK_EQ(top_index, top_count) << "the end top index must be equal to top_count";
}

#ifdef CPU_ONLY
STUB_GPU(MultiviewAverageLayer);
#endif

INSTANTIATE_CLASS(MultiviewAverageLayer);
REGISTER_LAYER_CLASS(MultiviewAverage);

}  // namespace caffe
