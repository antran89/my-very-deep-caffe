#ifndef CAFFE_SHUFFLEINDEX_LAYER_HPP_
#define CAFFE_SHUFFLEINDEX_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Shuffle index of the input blob.
 *
 * All elements inside blob are also shuffled.
 */

template <typename Dtype>
class ShuffleIndexLayer : public Layer<Dtype> {
public:
    explicit ShuffleIndexLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "ShuffleIndex"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    /**
     * @param bottom input blob
     * @param top output shuffled blob
     */
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    /**
     * @param top
     * @param propagate_down
     * @param bottom
     */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    /**
     * @brief ConvertIndex2VectorIndex
     *        Convert index to a vector of index according the shape of a blob.
     *
     * @param blob_shape input blob shape
     * @param blob_count input blob count
     * @param index int index
     * @return an index in vector form
     */
    vector<int> ConvertIndex2VectorIndex(const vector<int>& blob_shape, const int blob_count, const int index);

    vector<int> new_axes_;
};

}  // namespace caffe

#endif  // CAFFE_SHUFFLEREINDEX_LAYER_HPP_
