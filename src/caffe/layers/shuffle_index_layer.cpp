#include <vector>

#include "caffe/layers/shuffle_index_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ShuffleIndexLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
    new_axes_.clear();
    for (int i = 0; i < this->layer_param_.shuffle_index_param().new_index_size(); i++)
        new_axes_.push_back(this->layer_param_.shuffle_index_param().new_index(i));
    CHECK_EQ(new_axes_.size(), bottom[0]->num_axes())
            << "New index should have same dimensions as input blob dimension.";
}

template<typename Dtype>
void ShuffleIndexLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    // new shape
    vector<int> new_shape;
    for (int i = 0; i < new_axes_.size(); i++)
        new_shape.push_back(bottom[0]->shape(new_axes_[i]));
    top[0]->Reshape(new_shape);

    CHECK_EQ(bottom[0]->count(), top[0]->count())
            << "Input and output blob shold have same number of elements.";
}

template<typename Dtype>
vector<int> ShuffleIndexLayer<Dtype>::ConvertIndex2VectorIndex(const vector<int>& blob_shape,
                                                               const int blob_count, const int index)
{
    int index_cnt = index;
    int num_axes = blob_shape.size();
    vector<int> vec_index(num_axes);
    for(int i = num_axes-1; i >= 0; i--) {
        vec_index[i] = index_cnt % blob_shape[i];
        index_cnt /= blob_shape[i];
    }
    return vec_index;
}

template<typename Dtype>
void ShuffleIndexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int bottom_count = bottom[0]->count();
    vector<int> bottom_shape = bottom[0]->shape();

    for (int index = 0; index < bottom_count; index++) {
        // index of a vector
        vector<int> vec_index = ConvertIndex2VectorIndex(bottom_shape, bottom_count, index);
        vector<int> new_vec_index(bottom_shape.size());
        for (int i = 0; i < bottom_shape.size(); i++)
            new_vec_index[i] = vec_index[new_axes_[i]];

        // move data element
        int top_offset = top[0]->offset(new_vec_index);
        top_data[top_offset] = bottom_data[index];
    }
}

template<typename Dtype>
void ShuffleIndexLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int bottom_count = bottom[0]->count();
    vector<int> bottom_shape = bottom[0]->shape();

    for (int index = 0; index < bottom_count; index++) {
        // index of a vector
        vector<int> vec_index = ConvertIndex2VectorIndex(bottom_shape, bottom_count, index);
        vector<int> new_vec_index(bottom_shape.size());
        for (int i = 0; i < bottom_shape.size(); i++)
            new_vec_index[i] = vec_index[new_axes_[i]];

        // move data element
        int top_offset = top[0]->offset(new_vec_index);
        bottom_diff[index] = top_diff[top_offset];
    }
}

#ifdef CPU_ONLY
STUB_GPU(ShuffleIndexLayer);
#endif

INSTANTIATE_CLASS(ShuffleIndexLayer);
REGISTER_LAYER_CLASS(ShuffleIndex);

}  // namespace caffe
