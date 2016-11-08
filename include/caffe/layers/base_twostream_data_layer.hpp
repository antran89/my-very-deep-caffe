#ifndef CAFFE_TWOSTREAM_DATA_LAYERS_HPP_
#define CAFFE_TWOSTREAM_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for two-stream data layers that feed blobs to the Net.
 * The order of blobs is rgb blob, flow blob, and an optional label blob.
 * In the prototxt file, we need to specify in this order.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseTwostreamDataLayer : public Layer<Dtype> {
public:
    explicit BaseTwostreamDataLayer(const LayerParameter& param);
    // LayerSetUp: implements common data layer setup functionality, and calls
    // DataLayerSetUp to do special data layer setup for individual layer types.
    // This method may not be overridden except by the BasePrefetchingDataLayer.
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    // Data layers should be shared by multiple solvers in parallel
    virtual inline bool ShareInParallel() const { return true; }
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {}
    // Data layers have no bottoms, so reshaping is trivial.
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top) {}

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

protected:
    TransformationParameter transform_param_;
    shared_ptr<DataTransformer<Dtype> > data_transformer_;
    bool output_labels_;
};

template <typename Dtype>
class TwostreamBatch {
public:
    Blob<Dtype> flow_data_, rgb_data_, label_;
};

template <typename Dtype>
class BasePrefetchingTwostreamDataLayer :
        public BaseTwostreamDataLayer<Dtype>, public InternalThread {
public:
    explicit BasePrefetchingTwostreamDataLayer(const LayerParameter& param);
    // LayerSetUp: implements common data layer setup functionality, and calls
    // DataLayerSetUp to do special data layer setup for individual layer types.
    // This method may not be overridden.
    void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                    const vector<Blob<Dtype>*>& top);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    // Prefetches batches (asynchronously if to GPU memory)
    static const int PREFETCH_COUNT = 6;

protected:
    virtual void InternalThreadEntry();
    virtual void load_batch(TwostreamBatch<Dtype>* batch) = 0;

    TwostreamBatch<Dtype> prefetch_[PREFETCH_COUNT];
    BlockingQueue<TwostreamBatch<Dtype>*> prefetch_free_;
    BlockingQueue<TwostreamBatch<Dtype>*> prefetch_full_;

};

}  // namespace caffe

#endif  // CAFFE_TWOSTREAM_DATA_LAYERS_HPP_
