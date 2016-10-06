#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_twostream_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseTwostreamDataLayer<Dtype>::BaseTwostreamDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseTwostreamDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
    if (top.size() < 3 ) {
        output_labels_ = false;
    } else {
        output_labels_ = true;
    }
    data_transformer_.reset(
                new DataTransformer<Dtype>(transform_param_, this->phase_));
    data_transformer_->InitRand();
    // The subclasses should setup the size of bottom and top
    DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingTwostreamDataLayer<Dtype>::BasePrefetchingTwostreamDataLayer(
        const LayerParameter& param)
    : BaseTwostreamDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_free_.push(&prefetch_[i]);
    }
}

template <typename Dtype>
void BasePrefetchingTwostreamDataLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    BaseTwostreamDataLayer<Dtype>::LayerSetUp(bottom, top);
    // Before starting the prefetch thread, we make cpu_data and gpu_data
    // calls so that the prefetch thread does not accidentally make simultaneous
    // cudaMalloc calls when the main thread is running. In some GPUs this
    // seems to cause failures if we do not so.
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].rgb_data_.mutable_cpu_data();
        prefetch_[i].flow_data_.mutable_cpu_data();
        if (this->output_labels_) {
            prefetch_[i].label_.mutable_cpu_data();
        }
    }
#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU) {
        for (int i = 0; i < PREFETCH_COUNT; ++i) {
            prefetch_[i].rgb_data_.mutable_gpu_data();
            prefetch_[i].flow_data_.mutable_gpu_data();
            if (this->output_labels_) {
                prefetch_[i].label_.mutable_gpu_data();
            }
        }
    }
#endif
    DLOG(INFO) << "Initializing prefetch";
    this->data_transformer_->InitRand();
    StartInternalThread();
    DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingTwostreamDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
    cudaStream_t stream;
    if (Caffe::mode() == Caffe::GPU) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
#endif

    try {
        while (!must_stop()) {
            Batch<Dtype>* batch = prefetch_free_.pop("Waiting for free prefetch batch");
            load_batch(batch);
#ifndef CPU_ONLY
            if (Caffe::mode() == Caffe::GPU) {
                batch->rgb_data_.data().get()->async_gpu_push(stream);
                batch->flow_data_.data().get()->async_gpu_push(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
#endif
            prefetch_full_.push(batch);
        }
    } catch (boost::thread_interrupted&) {
        // Interrupted exception is expected on shutdown
    }
#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
#endif
}

template <typename Dtype>
void BasePrefetchingTwostreamDataLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->rgb_data_);
    top[1]->ReshapeLike(batch->flow_data_);
    // Copy the data
    caffe_copy(batch->rgb_data_.count(), batch->rgb_data_.cpu_data(),
               top[0]->mutable_cpu_data());
    caffe_copy(batch->flow_data_.count(), batch->flow_data_.cpu_data(),
               top[1]->mutable_cpu_data());
    DLOG(INFO) << "Prefetch copied";
    if (this->output_labels_) {
        // Reshape to loaded labels.
        top[2]->ReshapeLike(batch->label_);
        // Copy the labels.
        caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
                   top[2]->mutable_cpu_data());
    }

    prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingTwostreamDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseTwostreamDataLayer);
INSTANTIATE_CLASS(BasePrefetchingTwostreamDataLayer);

}  // namespace caffe
